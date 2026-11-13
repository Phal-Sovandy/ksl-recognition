# Model Architecture

Diagram of the model actually implemented in [scripts/model_claude.py](scripts/model_claude.py)
(`MultiHeadSignModel`), as trained by [scripts/training/train.ipynb](scripts/training/train.ipynb).
For the higher-level project rationale (why this pipeline, dataset strategy, training
stages) see [ksl_sign_model_design.md](ksl_sign_model_design.md); this document is the
literal architecture, grounded in the current code.

**Tensor contract** (`SignEncoder.forward` / `MultiHeadSignModel.forward`):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `feats` | `(B, T, V=51, C_in=5)` | per-node `[x, y, dx, dy, vis]` |
| `visibility` | `(B, T, 51)` | 1 = real landmark, 0 = absent → drives mask-aware GCN aggregation |
| `frame_mask` | `(B, T)` | 1 = usable frame, 0 = padding/total detection failure → drives Transformer attention masking |

51 nodes = 9 upper-body pose + 21 left-hand + 21 right-hand (legs/feet dropped at
extraction — no signing information).

---

## 1. End-to-end pipeline

```mermaid
flowchart TD
    subgraph INPUT["Raw extracted clip (.npz)"]
        A1["coords (T, 51, 2)<br/>pixel-space x,y"]
        A2["vis (T, 51)<br/>1 = landmark detected"]
    end

    subgraph FEAT["features.py — per-clip preprocessing (data side, not inside the model)"]
        direction TB
        B1["normalize_clip()<br/>shoulder-centred + shoulder-width scaled;<br/>each hand additionally re-centred on its own wrist"]
        B2["add_velocity()<br/>append frame-to-frame Δx, Δy"]
        B3["concat visibility as the last channel"]
        B1 --> B2 --> B3
    end
    A1 --> B1
    A2 --> B1

    subgraph COLLATE["dataset.py: PadCollate (batches variable-length clips)"]
        C1["feats (B, T, 51, 5)<br/>[x, y, dx, dy, vis]"]
        C2["visibility (B, T, 51)"]
        C3["frame_mask (B, T) = frame_usable(vis)"]
    end
    B3 --> C1
    A2 --> C2 --> C3

    subgraph ENCODER["SignEncoder — shared across every language/gloss head"]
        D0["x = feats + node_type<br/>learnable per-joint bias, shape (51, 5)"]

        subgraph GCN["SpatialGCN — 2× MaskedGraphConv over the 51×51 anatomical adjacency (§2)"]
            direction TB
            G1["MaskedGraphConv 5 → 128<br/>+ skip-proj + LayerNorm + ReLU"]
            G2["MaskedGraphConv 128 → 256<br/>+ skip-proj + LayerNorm + ReLU"]
            G1 --> G2
        end

        D1["mask-aware mean-pool over the 51 nodes<br/>(divide by VISIBLE node count, not 51)"]
        D2["Linear proj 256 → 256 (d_model)"]

        subgraph TRANS["TemporalTransformer"]
            direction TB
            T0["prepend learnable CLS token<br/>+ learnable positional embedding (max_len+1, 256)"]
            T1["TransformerEncoderLayer × 6  (§3)<br/>nhead=8, dim_ff=512, GELU, dropout=0.25<br/>src_key_padding_mask = NOT frame_mask (CLS always attended)"]
            T2["take CLS token's output → LayerNorm"]
            T0 --> T1 --> T2
        end

        D0 --> GCN --> D1 --> D2 --> T0
    end
    C1 --> D0
    C2 -.mask.-> G1
    C2 -.mask.-> G2
    C3 -.key_padding_mask.-> T1

    subgraph HEAD["Swappable per-language/gloss head"]
        H1["heads[name] = Linear(256 → num_classes)<br/>e.g. heads['ksl'], heads['arabic'], ..."]
    end
    T2 --> H1 --> OUT["logits (B, num_classes)"]
```

**Why a shared encoder + swappable heads:** the GCN+Transformer backbone (`SignEncoder`)
never changes shape between languages — only which `nn.Linear` head sits on top does. That's
what lets a Stage-1 checkpoint pretrained on e.g. Arabic/RSL warm-start KSL fine-tuning
(`model.load_state_dict(ckpt, strict=False)` copies every `encoder.*` weight and silently
drops the mismatched `heads.*` keys — see the co-training / fine-tuning discussion elsewhere
in this repo's history) and what lets `multilingual_pretrain` train several heads on one
encoder at once, round-robin.

---

## 2. Anatomical adjacency (the GCN's graph)

`build_adjacency()` returns a fixed, symmetric 51×51 binary matrix — **not learned** — built
from real skeletal bones, not a fully-connected or attention-derived graph. A learnable
`edge_importance` weight (one scalar per matrix entry) can up/down-weight an existing edge
but can never invent a non-anatomical one (it multiplies the base graph, never adds to it).

```mermaid
flowchart LR
    subgraph POSE["Pose (9 nodes)"]
        nose((nose))
        lsh((L-shoulder))
        rsh((R-shoulder))
        lel((L-elbow))
        rel((R-elbow))
        lwr((L-wrist))
        rwr((R-wrist))
        lhip((L-hip))
        rhip((R-hip))
        nose --- lsh
        nose --- rsh
        lsh --- rsh
        lsh --- lel --- lwr
        rsh --- rel --- rwr
        lsh --- lhip
        rsh --- rhip
        lhip --- rhip
    end

    subgraph LHAND["Left hand (21 nodes) — MediaPipe hand skeleton"]
        lw0((wrist))
        lw0 --- lt1 --- lt2 --- lt3 --- lt4
        lw0 --- li1 --- li2 --- li3 --- li4
        lw0 --- lm1 --- lm2 --- lm3 --- lm4
        lw0 --- lr1 --- lr2 --- lr3 --- lr4
        lw0 --- lp1 --- lp2 --- lp3 --- lp4
        li1 --- lm1 --- lr1 --- lp1
        lt1(thumb1); lt2(thumb2); lt3(thumb3); lt4(thumb4)
        li1(index1); li2(index2); li3(index3); li4(index4)
        lm1(middle1); lm2(middle2); lm3(middle3); lm4(middle4)
        lr1(ring1); lr2(ring2); lr3(ring3); lr4(ring4)
        lp1(pinky1); lp2(pinky2); lp3(pinky3); lp4(pinky4)
    end

    subgraph RHAND["Right hand (21 nodes) — mirror of the left"]
        rw0((wrist))
    end

    lwr -. bridge edge .- lw0
    rwr -. bridge edge .- rw0
```

(Right hand collapsed to its wrist node here for readability — it's an exact mirror of the
left hand's 21-node/25-edge finger topology: thumb/index/middle/ring/pinky chains + 3
knuckle-bridge edges + palm-base edges.) The two **bridge edges** (pose wrist ↔ that hand's
own wrist node) are what let a hand-shape signal reach the arm/body graph at all — without
them the hands and pose would be two disconnected components.

**Mask-aware aggregation** (`MaskedGraphConv`, not shown as its own diagram — it's a small
edit to the standard `A_norm @ (XW)` GCN update):

1. `A = base_adjacency * edge_importance` (learnable weights, anatomical structure fixed)
2. add self-loops, broadcast to `(B, T, V, V)`
3. **zero out column j** wherever node `j` is not visible in that frame — an absent joint
   contributes exactly 0 to every neighbour instead of leaking a fake `(0,0)`
4. row-normalise by the **visible** degree only (not the fixed graph degree)
5. aggregate: `einsum("btij,btjc->btic", A, XW)`

A one-handed sign therefore never gets a phantom "hand at the origin" signal — the other
hand's nodes simply drop out of every neighbour's average for that frame.

---

## 3. One `TransformerEncoderLayer` (× 6, `nn.TransformerEncoder` — PyTorch stock layer)

`model_claude.py` constructs `nn.TransformerEncoderLayer` without passing `norm_first`,
so it uses PyTorch's default — **Post-LN** (normalize *after* each residual add, not
before):

```mermaid
flowchart TD
    IN["input (B, T+1, 256)<br/>token 0 = CLS"] --> MHA["Multi-Head Self-Attention<br/>nhead = 8 → 32-dim/head<br/>src_key_padding_mask blocks padded frames"]
    MHA --> ADD1((+))
    IN --> ADD1
    ADD1 --> LN1["LayerNorm"]
    LN1 --> FF["Feed-Forward<br/>Linear 256→512 → GELU → Dropout(0.25) → Linear 512→256"]
    FF --> ADD2((+))
    LN1 --> ADD2
    ADD2 --> LN2["LayerNorm"]
    LN2 --> OUT["output (B, T+1, 256)"]
```

Padded frames are masked out of attention entirely (`src_key_padding_mask`), not just
zero-valued — they contribute to no query's weighted sum. The CLS token's padding-mask entry
is always `False` (always attended to), and after all 6 layers, `TemporalTransformer` reads
**only the CLS token's final vector** (`x[:, 0]`) as the whole clip's representation, then
applies one more `LayerNorm` before it reaches the classifier head.

---

## 4. Configured vs. default hyperparameters

`SignEncoder`/`SpatialGCN`/`TemporalTransformer` all ship small defaults in
`model_claude.py` itself; `train.ipynb` §2/§7 overrides every one of them for actual
training runs. The diagrams above use the **train.ipynb** column.

| Hyperparameter | `model_claude.py` default | `train.ipynb` (current) |
| --- | --- | --- |
| `in_ch` (`C_in`) | *(required, no default)* | 5 — `[x, y, dx, dy, vis]` |
| `gcn_hidden` | `(64, 128)` | `(128, 256)` |
| `d_model` | 128 | 256 |
| `nhead` | 4 | 8 |
| `num_layers` (Transformer) | 4 | 6 |
| `dim_ff` | 256 | 512 |
| `dropout` | 0.2 | 0.25 |
| `max_len` | 128 | 128 |

At the `train.ipynb` configuration, the shared backbone (`SignEncoder`) is **~4M
parameters**; each per-language head adds only `256 × num_classes + num_classes`
(e.g. ~50K params for KSL's 194 classes) — negligible next to the shared encoder,
which is the whole point of the multi-head design.
