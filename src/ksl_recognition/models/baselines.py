"""
baselines.py
-------------------
Classic recurrent baselines (vanilla RNN, LSTM, BiLSTM, GRU, BiGRU) for the
paper's comparison table against MultiHeadSignModel (sign_model.py's
mask-aware GCN + Transformer).

These are deliberately drop-in replacements for MultiHeadSignModel: same
constructor shape (in_ch, head_classes, **kwargs), same forward signature
(feats, visibility, frame_mask, head=...), same .encode()/.add_head()/.heads.
That means the *entire* rest of the pipeline -- extraction/utils.py's landmarks,
features.assemble_features(), dataset.py's KSLLandmarkDataset/PadCollate,
augmentations.py, and train.ipynb's training loop, optimizer, LR schedule,
and checkpoint format -- is shared unchanged between the GCN+Transformer
model and these baselines. Only the architecture differs, which is what you
want for a fair "does the graph+attention structure actually help" ablation
in the paper.

What these baselines deliberately do NOT have, unlike MultiHeadSignModel:
  * No anatomical graph -- landmarks are flattened to one flat per-frame
    vector (V*C), so the model has to learn any spatial joint structure from
    scratch instead of getting it for free from the skeleton adjacency.
  * No mask-aware aggregation -- visibility is just another input channel
    (already baked into `feats` by assemble_features()), not something that
    reweights an aggregation step.
  * No attention / CLS token -- the sequence is pooled by a masked mean over
    the recurrent outputs (see RecurrentEncoder.forward), which is the
    standard way to turn an RNN's per-timestep outputs into one clip-level
    vector while still respecting padding.
That gap (structure-free baseline vs. structure-aware model) is exactly the
point of a comparison table -- these are meant to be weaker, not tuned to
win.

Tensor contract (identical to sign_model.py):
  feats      : (B, T, V, C_in)  x / velocity / flag channels
  visibility : (B, T, V)        1 = real landmark, 0 = absent (unused directly
                                 here -- already folded into feats' last
                                 channel by assemble_features(); kept as a
                                 parameter only so the call signature matches
                                 MultiHeadSignModel exactly)
  frame_mask : (B, T)           1 = usable frame, 0 = padding / total failure
"""

from __future__ import annotations
import torch
import torch.nn as nn

NUM_NODES = 51          # pose(9) + left_hand(21) + right_hand(21), see sign_model.py

_RNN_CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


# --------------------------------------------------------------------------- #
# Shared recurrent encoder -- one class, parametrised by cell type +
# bidirectional, covers RNN / LSTM / GRU / BiLSTM / BiGRU.
# --------------------------------------------------------------------------- #
class RecurrentEncoder(nn.Module):
    """Flatten (B,T,V,C) -> (B,T,V*C) per frame, run it through a stacked
    RNN/LSTM/GRU (optionally bidirectional), then masked-mean-pool over time
    into one (B, out_dim) clip vector.

    Uses pack_padded_sequence so padded frames never get folded into the
    recurrence itself (not just masked out afterwards) -- important since,
    unlike the Transformer's key_padding_mask, a plain nn.RNN/LSTM/GRU has no
    way to ignore padding on its own.
    """

    def __init__(self, in_ch: int, cell: str = "lstm", hidden: int = 256,
                 num_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.3, num_nodes: int = NUM_NODES):
        super().__init__()
        cell = cell.lower()
        if cell not in _RNN_CELLS:
            raise ValueError(f"cell must be one of {list(_RNN_CELLS)}, got {cell!r}")
        self.cell_name = cell
        self.input_dim = num_nodes * in_ch

        rnn_cls = _RNN_CELLS[cell]
        rnn_kwargs = dict(
            input_size=self.input_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,   # nn.RNN warns/no-ops otherwise
        )
        if cell == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"
        self.rnn = rnn_cls(**rnn_kwargs)

        self.out_dim = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, feats: torch.Tensor, visibility: torch.Tensor,
                frame_mask: torch.Tensor) -> torch.Tensor:
        B, T, V, C = feats.shape
        x = feats.reshape(B, T, V * C)

        lengths = frame_mask.sum(dim=1).clamp_min(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T)          # (B, T, out_dim)

        # Masked mean over time. Chosen over "take the last real timestep"
        # because it's identical in spirit to how SpatialGCN's node pooling
        # and works the same way regardless of direction (last-hidden for a
        # BiRNN needs care to grab each direction's correct end).
        m = frame_mask.unsqueeze(-1)                                # (B, T, 1)
        summed = (out * m).sum(dim=1)
        count = m.sum(dim=1).clamp_min(1.0)
        pooled = summed / count
        return self.norm(self.dropout(pooled))


# --------------------------------------------------------------------------- #
# Full model with swappable per-language heads -- same shape as
# sign_model.MultiHeadSignModel so training/eval code doesn't change.
# --------------------------------------------------------------------------- #
class MultiHeadRecurrentModel(nn.Module):
    """
    cell: "rnn" | "lstm" | "gru".  bidirectional=True + cell="lstm" == BiLSTM,
    bidirectional=True + cell="gru" == BiGRU, bidirectional=False + cell="rnn"
    == vanilla RNN, etc. -- covers the whole classic-baseline family from one
    class instead of five near-duplicate ones.
    """

    def __init__(self, in_ch: int, head_classes: dict[str, int], cell: str = "lstm",
                 hidden: int = 256, num_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.3, num_nodes: int = NUM_NODES, **_ignored):
        super().__init__()
        self.encoder = RecurrentEncoder(
            in_ch, cell=cell, hidden=hidden, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout, num_nodes=num_nodes,
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.encoder.out_dim, n) for name, n in head_classes.items()}
        )

    def add_head(self, name: str, num_classes: int):
        self.heads[name] = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, feats, visibility, frame_mask, head: str):
        z = self.encoder(feats, visibility, frame_mask)     # (B, out_dim)
        return self.heads[head](z)                          # (B, n_classes)

    def encode(self, feats, visibility, frame_mask):
        return self.encoder(feats, visibility, frame_mask)


# --------------------------------------------------------------------------- #
# Named presets for the paper's comparison table -- kwargs for
# MultiHeadRecurrentModel(...). hidden/num_layers roughly size-matched to
# sign_model's default D_MODEL=128/256 range so "ours is bigger" isn't the
# reason it wins.
# --------------------------------------------------------------------------- #
BASELINE_PRESETS = {
    "rnn":    dict(cell="rnn",  bidirectional=False),
    "lstm":   dict(cell="lstm", bidirectional=False),
    "bilstm": dict(cell="lstm", bidirectional=True),
    "gru":    dict(cell="gru",  bidirectional=False),
    "bigru":  dict(cell="gru",  bidirectional=True),
}


def build_baseline(name: str, in_ch: int, head_classes: dict[str, int],
                   hidden: int = 256, num_layers: int = 2, dropout: float = 0.3,
                   num_nodes: int = NUM_NODES) -> MultiHeadRecurrentModel:
    """name: one of BASELINE_PRESETS' keys ("rnn", "lstm", "bilstm", "gru", "bigru")."""
    if name not in BASELINE_PRESETS:
        raise ValueError(f"unknown baseline {name!r}, choose from {list(BASELINE_PRESETS)}")
    return MultiHeadRecurrentModel(
        in_ch, head_classes, hidden=hidden, num_layers=num_layers,
        dropout=dropout, num_nodes=num_nodes, **BASELINE_PRESETS[name],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
