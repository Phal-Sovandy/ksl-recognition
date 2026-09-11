"""
train.py
--------
Train the KSL classifier on extracted landmarks.

Run as a module (requires `pip install -e .` from the repo root, see README):

From scratch (no multilingual pretraining yet):
    python -m ksl_recognition.training.train --data data/processed_landmarks/ksl --epochs 60

Fine-tune from a Stage-1 pretrained encoder checkpoint:
    python -m ksl_recognition.training.train --data data/processed_landmarks/ksl \
        --pretrained checkpoints/stage1.pt --enc-lr 1e-4 --head-lr 1e-3
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ksl_recognition.data.augmentations import AugmentPipeline
from ksl_recognition.data.dataset import KSLLandmarkDataset, scan_landmarks, scan_val_landmarks, stratified_split, PadCollate
from ksl_recognition.models.sign_model import MultiHeadSignModel, count_parameters

HEAD = "ksl"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")          # Apple GPU (M3)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model, enc_lr, head_lr, weight_decay):
    """Layer-wise LR: smaller on the (possibly pretrained) encoder, larger on the head."""
    head_ids = {id(p) for p in model.heads[HEAD].parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    enc_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    return torch.optim.AdamW(
        [{"params": enc_params, "lr": enc_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for feats, vis, fmask, labels in loader:
        feats, vis, fmask, labels = (t.to(device) for t in (feats, vis, fmask, labels))
        logits = model(feats, vis, fmask, head=HEAD)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root of extracted landmarks")
    ap.add_argument("--val-data",      default=None,
                    help="root of a dedicated validation set (same <class>/<clip>.npz "
                         "layout). If omitted, val is split out of --data via --val-frac")
    ap.add_argument("--epochs",        type=int,   default=100)
    ap.add_argument("--batch-size",    type=int,   default=32)
    ap.add_argument("--max-len",       type=int,   default=128)
    ap.add_argument("--val-frac",      type=float, default=0.2,
                    help="ignored when --val-data is given")
    ap.add_argument("--enc-lr",        type=float, default=1e-3)
    ap.add_argument("--head-lr",       type=float, default=1e-3)
    ap.add_argument("--weight-decay",  type=float, default=1e-5)
    ap.add_argument("--warmup-epochs", type=int,   default=5,
                    help="linear LR warm-up before plateau scheduler kicks in")
    ap.add_argument("--sched-patience",type=int,   default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="applied to the TRAIN loss only; val loss stays plain "
                         "cross-entropy so it remains a faithful overfit signal")
    # model architecture
    ap.add_argument("--d-model",       type=int,   default=256)
    ap.add_argument("--num-layers",    type=int,   default=6)
    ap.add_argument("--dropout",       type=float, default=0.1)
    ap.add_argument("--workers",       type=int,   default=8)
    ap.add_argument("--pretrained",    default=None,
                    help="Stage-1 encoder checkpoint (optional)")
    ap.add_argument("--out",           default="checkpoints/ksl_best.pt")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--no-augment",    action="store_true",
                    help="disable training augmentation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print("device:", device)

    items, class_to_idx = scan_landmarks(args.data)
    num_classes = len(class_to_idx)
    if args.val_data:
        train_items = items
        val_items = scan_val_landmarks(args.val_data, class_to_idx)
        print(f"{len(train_items)} train clips (from {args.data}) | "
              f"{len(val_items)} val clips (from {args.val_data}) | {num_classes} classes")
    else:
        train_items, val_items = stratified_split(items, args.val_frac, args.seed)
        print(f"{len(items)} clips | {num_classes} classes | "
              f"train {len(train_items)} / val {len(val_items)}")

    augmenter = None if args.no_augment else AugmentPipeline.default()
    print("augmentation:", "off" if augmenter is None else "on (default pipeline)")

    collate = PadCollate(use_velocity=True)
    train_ds = KSLLandmarkDataset(train_items, class_to_idx, args.max_len,
                                  augment=augmenter)
    val_ds = KSLLandmarkDataset(val_items, class_to_idx, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.workers)

    in_ch = 5  # x, y, dx, dy, vis  (use_velocity=True on 2D coords)
    d     = args.d_model
    model = MultiHeadSignModel(
        in_ch=in_ch, head_classes={HEAD: num_classes},
        d_model=d, gcn_hidden=(d // 2, d), nhead=max(1, d // 32),
        num_layers=args.num_layers, dim_ff=d * 2,
        dropout=args.dropout, max_len=args.max_len,
    ).to(device)

    # MPS doesn't implement the TransformerEncoder nested-tensor fast path (only
    # triggered in eval mode). Disable it so eval stays on the GPU. Same results.
    for m in model.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            m.use_nested_tensor = False

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded pretrained (missing={len(missing)}, unexpected={len(unexpected)})")
    print("params:", f"{count_parameters(model):,}")

    optim = build_optimizer(model, args.enc_lr, args.head_lr, args.weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=args.sched_patience, min_lr=1e-6
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for feats, vis, fmask, labels in train_loader:
            feats, vis, fmask, labels = (t.to(device) for t in (feats, vis, fmask, labels))
            logits = model(feats, vis, fmask, head=HEAD)
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item() * labels.numel()
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_total += labels.numel()

        tr_loss, tr_acc = run_loss / run_total, run_correct / run_total
        val_loss, val_acc = evaluate(model, val_loader, device)
        if epoch <= args.warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_acc)
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "val_acc": val_acc,
                "arch": dict(in_ch=5, d_model=d, gcn_hidden=(d // 2, d),
                             nhead=max(1, d // 32), num_layers=args.num_layers,
                             dim_ff=d * 2, dropout=args.dropout,
                             max_len=args.max_len),
            }, args.out)
            flag = "  <- best, saved"
        print(f"epoch {epoch:>3} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}{flag}")

    print(f"\nBest val acc: {best_acc:.3f}  ->  {args.out}")


if __name__ == "__main__":
    main()
