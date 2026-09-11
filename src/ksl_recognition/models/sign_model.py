"""
Khmer Sign Language recognition model.

Pipeline:  (B, T, V=51, C) landmarks  ->  mask-aware GCN (spatial)
           ->  temporal Transformer encoder (+ CLS)  ->  classifier head

Design decisions baked in (from the design discussion):
  * Node-level visibility drives MASK-AWARE aggregation, not just a feature flag.
    Absent landmarks contribute zero to their neighbours instead of leaking a
    fake (0,0,0) into the graph.
  * Frame mask means "frame has NO usable info" (padding / total detection
    failure). A missing hand alone does NOT mask the frame -> one-handed signs
    survive.
  * Anatomical adjacency: pose bones + per-hand finger bones + bridge edges
    that anchor each hand's wrist to the corresponding pose wrist.
  * Small enough to train on a laptop (~1M params at the default config).

Tensor contract for forward():
  feats      : (B, T, V, C_in)  x / velocity / flag channels
  visibility : (B, T, V)        1 = real landmark, 0 = absent (used for masking)
  frame_mask : (B, T)           1 = usable frame, 0 = padding / total failure
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Landmark layout + anatomical adjacency
# --------------------------------------------------------------------------- #
# MediaPipe Holistic, trimmed to upper body: pose(9) + left_hand(21) +
# right_hand(21) = 51 nodes. Pose keeps only nose/shoulders/elbows/wrists/hips
# (see extraction.utils.UPPER_BODY_POSE_IDX) -- legs and feet carry no signing
# information and were never wired into the adjacency below anyway.
POSE = 9
HAND = 21
NUM_NODES = POSE + 2 * HAND          # 51
LH_OFFSET = POSE                      # left  hand nodes:  9..29
RH_OFFSET = POSE + HAND               # right hand nodes: 30..50

# Upper-body pose edges (compact indices 0..8):
# 0 nose, 1 L-shoulder, 2 R-shoulder, 3 L-elbow, 4 R-elbow,
# 5 L-wrist, 6 R-wrist, 7 L-hip, 8 R-hip.
_POSE_EDGES = [
    (1, 2),            # shoulders
    (1, 3), (3, 5),    # left arm
    (2, 4), (4, 6),    # right arm
    (1, 7), (2, 8), (7, 8),  # torso
    (0, 1), (0, 2),    # head -> shoulders
]

# MediaPipe hand skeleton (local indices 0..20, wrist = 0).
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # knuckle bridge (palm rigidity)
    (0, 5), (0, 17),                       # palm base
]

# Pose wrist indices, to bridge into the dedicated hand graphs.
_POSE_LEFT_WRIST = 5
_POSE_RIGHT_WRIST = 6


def build_adjacency() -> torch.Tensor:
    """Return a symmetric (51, 51) binary adjacency (no self-loops)."""
    A = torch.zeros(NUM_NODES, NUM_NODES)

    def add(i, j):
        A[i, j] = 1.0
        A[j, i] = 1.0

    for i, j in _POSE_EDGES:
        add(i, j)
    for i, j in _HAND_EDGES:
        add(LH_OFFSET + i, LH_OFFSET + j)
        add(RH_OFFSET + i, RH_OFFSET + j)

    # Bridge edges: pose wrist <-> hand-graph wrist (local node 0).
    add(_POSE_LEFT_WRIST, LH_OFFSET + 0)
    add(_POSE_RIGHT_WRIST, RH_OFFSET + 0)
    return A


# --------------------------------------------------------------------------- #
# 2. Mask-aware graph convolution
# --------------------------------------------------------------------------- #
class MaskedGraphConv(nn.Module):
    """
    Graph conv with visibility-weighted aggregation.

    Standard GCN aggregates  A_norm @ (X W).  Here the adjacency is masked
    per-frame by node visibility BEFORE row-normalisation, so an absent node
    contributes exactly zero to its neighbours and the normalisation divides
    only by visible neighbours (no fake (0,0,0) leakage).

    A learnable edge-importance mask (ST-GCN style) lets the model up/down-
    weight individual anatomical edges. It is multiplicative on the base graph,
    so it can never invent non-anatomical edges.
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)                       # (V, V) base graph
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.lin = nn.Linear(in_ch, out_ch)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, Cin)   vis: (B, T, V)
        B, T, V, _ = x.shape
        h = self.lin(x)                                    # (B, T, V, Cout)

        # Base graph + self-loops, modulated by learnable edge importance.
        A = self.A * self.edge_importance
        A = A + torch.eye(V, device=A.device)              # self-loops
        A = A.expand(B, T, V, V).clone()                   # (B, T, V, V)

        # Mask columns (sources) by visibility: absent node j cannot send.
        A = A * vis.unsqueeze(2)                            # zero out column j
        # Row-normalise by the visible degree.
        deg = A.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        A = A / deg

        out = torch.einsum("btij,btjc->btic", A, h)        # aggregate
        return out


# --------------------------------------------------------------------------- #
# 3. Spatial encoder: stack of graph convs + mask-aware node pooling
# --------------------------------------------------------------------------- #
class SpatialGCN(nn.Module):
    def __init__(self, in_ch: int, hidden=(64, 128), A: torch.Tensor | None = None):
        super().__init__()
        if A is None:
            A = build_adjacency()
        chs = [in_ch, *hidden]
        self.layers = nn.ModuleList(
            [MaskedGraphConv(chs[i], chs[i + 1], A) for i in range(len(hidden))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(c) for c in hidden])
        # Skip connections: project when channel dims change, identity otherwise.
        self.skips = nn.ModuleList([
            nn.Linear(chs[i], chs[i + 1], bias=False) if chs[i] != chs[i + 1]
            else nn.Identity()
            for i in range(len(hidden))
        ])
        self.out_dim = hidden[-1]

    def forward(self, x, vis):
        # x: (B, T, V, Cin)  vis: (B, T, V)
        for conv, norm, skip in zip(self.layers, self.norms, self.skips):
            x = F.relu(norm(conv(x, vis) + skip(x)))
        # Mask-aware mean over nodes: divide by visible node count, not V.
        w = vis.unsqueeze(-1)                              # (B, T, V, 1)
        summed = (x * w).sum(dim=2)                        # (B, T, C)
        count = w.sum(dim=2).clamp_min(1.0)               # (B, T, 1)
        return summed / count                              # (B, T, C)


# --------------------------------------------------------------------------- #
# 4. Temporal transformer encoder with CLS token
# --------------------------------------------------------------------------- #
class TemporalTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4,
                 dim_ff=256, dropout=0.2, max_len=128):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))  # +1 for CLS
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, frame_mask):
        # x: (B, T, d)   frame_mask: (B, T)  (1 = keep)
        B, T, d = x.shape
        cls = self.cls.expand(B, 1, d)
        x = torch.cat([cls, x], dim=1)                      # (B, T+1, d)
        x = x + self.pos[:, : T + 1]

        # key_padding_mask: True = ignore. CLS is always attended (False).
        cls_keep = torch.ones(B, 1, device=frame_mask.device)
        keep = torch.cat([cls_keep, frame_mask], dim=1)     # (B, T+1)
        key_padding_mask = keep == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x[:, 0])                           # CLS -> (B, d)


# --------------------------------------------------------------------------- #
# 5. Full model with swappable per-language heads
# --------------------------------------------------------------------------- #
class SignEncoder(nn.Module):
    """Shared encoder: SpatialGCN -> TemporalTransformer -> (B, d) CLS vector."""

    def __init__(self, in_ch: int, d_model=128, gcn_hidden=(64, 128),
                 nhead=4, num_layers=4, dim_ff=256, dropout=0.2, max_len=128):
        super().__init__()
        A = build_adjacency()
        self.node_type = nn.Parameter(torch.zeros(NUM_NODES, in_ch))  # per-joint bias
        nn.init.trunc_normal_(self.node_type, std=0.02)

        self.gcn = SpatialGCN(in_ch, hidden=gcn_hidden, A=A)
        self.proj = nn.Linear(self.gcn.out_dim, d_model)
        self.temporal = TemporalTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, dropout=dropout, max_len=max_len,
        )
        self.out_dim = d_model

    def forward(self, feats, visibility, frame_mask):
        # feats: (B, T, V, Cin)  visibility: (B, T, V)  frame_mask: (B, T)
        x = feats + self.node_type                          # broadcast per-joint type
        x = self.gcn(x, visibility)                         # (B, T, gcn_out)
        x = self.proj(x)                                    # (B, T, d)
        return self.temporal(x, frame_mask)                 # (B, d)


class MultiHeadSignModel(nn.Module):
    """
    Shared encoder + a ModuleDict of linear heads.

    Stage 1 (pretraining): heads = {"asl": n_asl, "rsl": n_rsl, ...}; call
        forward(..., head="asl").
    Stage 2 (KSL): add_head("ksl", 100), then full fine-tune with a smaller LR
        on the encoder than on the new head (do NOT freeze the encoder).
    """

    def __init__(self, in_ch: int, head_classes: dict[str, int], **enc_kwargs):
        super().__init__()
        self.encoder = SignEncoder(in_ch, **enc_kwargs)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.encoder.out_dim, n) for name, n in head_classes.items()}
        )

    def add_head(self, name: str, num_classes: int):
        self.heads[name] = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, feats, visibility, frame_mask, head: str):
        z = self.encoder(feats, visibility, frame_mask)     # (B, d)
        return self.heads[head](z)                          # (B, n_classes)

    def encode(self, feats, visibility, frame_mask):
        return self.encoder(feats, visibility, frame_mask)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)