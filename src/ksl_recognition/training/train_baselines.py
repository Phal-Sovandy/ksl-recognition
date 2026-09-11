"""
train_baselines.py
-------------------
Trains one or more classic recurrent baselines (RNN / LSTM / BiLSTM / GRU /
BiGRU, from baselines.py) on KSL and writes a comparison table against
sign_model.MultiHeadSignModel -- for the paper's "does the GCN+Transformer
structure actually help" table.

Reuses the exact same data pipeline as train.ipynb's ksl_finetune stage
(scan_landmarks -> stratified_split -> KSLLandmarkDataset -> PadCollate,
same AdamW + linear-warmup + ReduceLROnPlateau schedule, same label
smoothing / grad clipping) so the only thing that differs between rows of
the resulting table is architecture. Kept as a plain script rather than a
notebook cell so all baselines can be run back-to-back unattended (a
30-minute paper run instead of babysitting five notebook re-runs).

Run (all 5 baselines, defaults matching train.ipynb's ksl_finetune):
    python -m ksl_recognition.training.train_baselines

Run a subset with custom hyperparameters:
    python -m ksl_recognition.training.train_baselines --models lstm bilstm \\
        --epochs 100 --hidden 256 --num-layers 2

Stage-2 fine-tuning from a Stage-1 pretrained encoder (same idea as train.py's
--pretrained / train.ipynb's ksl_finetune stage): --pretrained points at a
checkpoint produced by train_multilingual.py --model <same architecture>.
Only meaningful for one architecture at a time (a pretrained LSTM encoder
can't warm-start a GRU), so --pretrained requires exactly one --models entry:

    python -m ksl_recognition.training.train_multilingual --model lstm \\
        --languages <winning combo> --out checkpoints/multilingual_pretrain_lstm.pt
    python -m ksl_recognition.training.train_baselines --models lstm \\
        --pretrained checkpoints/multilingual_pretrain_lstm.pt

Output:
    checkpoints/baselines/<model>_best.pt   -- best-val-acc checkpoint per model
    checkpoints/baselines/results.csv       -- one row per model: params,
                                                best val acc, top-3/top-5 acc,
                                                and the --pretrained source (or
                                                "scratch") for provenance
"""
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ksl_recognition.data.augmentations import AugmentPipeline
from ksl_recognition.data.dataset import KSLLandmarkDataset, PadCollate, scan_landmarks, scan_val_landmarks, stratified_split
from ksl_recognition.models.baselines import BASELINE_PRESETS, build_baseline, count_parameters


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model, enc_lr, head_lr, weight_decay):
    """Mirrors train.ipynb's build_optimizer(): smaller LR on the encoder,
    larger on the head. For a from-scratch baseline this split matters less
    than it does for a pretrained encoder, but keeping it identical avoids
    "different optimizer setup" being a confound in the comparison."""
    head_ids = {id(p) for head in model.heads.values() for p in head.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW(
        [{"params": enc_params, "lr": enc_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate(model, loader, device, head, k_list=(1, 3, 5)):
    model.eval()
    correct = {k: 0 for k in k_list}
    total = 0
    loss_sum = 0.0
    for feats, vis, fmask, labels in loader:
        feats, vis, fmask, labels = (t.to(device) for t in (feats, vis, fmask, labels))
        logits = model(feats, vis, fmask, head=head)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        n_classes = logits.shape[1]
        for k in k_list:
            kk = min(k, n_classes)
            topk = logits.topk(kk, dim=1).indices
            correct[k] += (topk == labels.unsqueeze(1)).any(1).sum().item()
        total += labels.numel()
    acc = {k: c / total for k, c in correct.items()}
    return loss_sum / total, acc


def train_one(name, args, train_loader, val_loader, num_classes, in_ch, device, ckpt_dir):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    model = build_baseline(
        name, in_ch=in_ch, head_classes={"ksl": num_classes},
        hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"params: {n_params:,}")

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        ckpt_model_type = ckpt.get("model_type")
        if ckpt_model_type and ckpt_model_type != name:
            raise ValueError(
                f"--pretrained checkpoint was trained with --model {ckpt_model_type!r}, "
                f"can't warm-start a {name!r} encoder with it")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # strict=False only tolerates missing/unexpected KEYS (e.g. the
        # checkpoint's own language heads, none named "ksl") -- a genuine
        # shape mismatch (wrong architecture) still raises, so this doesn't
        # silently eat a real mistake.
        print(f"loaded pretrained ({args.pretrained})  "
             f"missing={len(missing)}  unexpected={len(unexpected)}")

    optim = build_optimizer(model, args.enc_lr, args.head_lr, args.weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=args.sched_patience, min_lr=1e-6)

    best_acc, best_topk = 0.0, {}
    best_path = ckpt_dir / f"{name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for feats, vis, fmask, labels in train_loader:
            feats, vis, fmask, labels = (t.to(device) for t in (feats, vis, fmask, labels))
            logits = model(feats, vis, fmask, head="ksl")
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            run_loss += loss.item() * labels.numel()
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_total += labels.numel()

        tr_loss, tr_acc = run_loss / run_total, run_correct / run_total
        val_loss, val_topk = evaluate(model, val_loader, device, head="ksl")
        val_acc = val_topk[1]

        if epoch <= args.warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_acc)
        lr = optim.param_groups[0]["lr"]

        flag = ""
        if val_acc > best_acc:
            best_acc, best_topk = val_acc, val_topk
            torch.save({"model": model.state_dict(), "val_acc": val_acc,
                       "epoch": epoch, "arch": name}, best_path)
            flag = "  <- best"

        if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            print(f"epoch {epoch:>3}/{args.epochs}  lr {lr:.2e}  "
                 f"train loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
                 f"val loss {val_loss:.3f} acc {val_acc:.3f}{flag}")

    print(f"{name}: best val acc {best_acc:.4f}  "
         f"(top3={best_topk.get(3, float('nan')):.4f} top5={best_topk.get(5, float('nan')):.4f})")
    return {"model": name, "params": n_params, "val_acc": best_acc,
           "val_top3": best_topk.get(3), "val_top5": best_topk.get(5),
           "pretrained": args.pretrained or "scratch"}


def main():
    ap = argparse.ArgumentParser(description="Train classic RNN-family baselines on KSL.")
    ap.add_argument("--models", nargs="+", default=list(BASELINE_PRESETS),
                    choices=list(BASELINE_PRESETS),
                    help="which baselines to train (default: all of rnn/lstm/bilstm/gru/bigru)")
    ap.add_argument("--data-dir", default="data/processed_landmarks/ksl")
    ap.add_argument("--val-dir", default=None, help="dedicated val root; omit to split --data-dir")
    ap.add_argument("--out-dir", default="checkpoints/baselines")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=67)

    # Architecture -- hidden/num_layers roughly size-matched to sign_model's
    # D_MODEL=128..256 range so parameter count isn't a confound.
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)

    # Training -- same defaults as train.ipynb's ksl_finetune stage.
    ap.add_argument("--epochs", type=int, default=170)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--enc-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--sched-patience", type=int, default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--pretrained", default=None,
                    help="Stage-1 checkpoint from train_multilingual.py --model <arch> "
                         "to warm-start the encoder (strict=False -- its language heads "
                         "are discarded, only the shared encoder loads). Requires exactly "
                         "one --models entry, matching the checkpoint's architecture.")
    args = ap.parse_args()

    if args.pretrained and len(args.models) != 1:
        raise SystemExit("--pretrained requires exactly one --models entry "
                         "(a pretrained encoder only fits its own architecture)")

    torch.manual_seed(args.seed)
    device = pick_device()
    print("device:", device)

    items, class_to_idx = scan_landmarks(args.data_dir)
    if args.val_dir:
        train_items = items
        val_items = scan_val_landmarks(args.val_dir, class_to_idx)
    else:
        train_items, val_items = stratified_split(items, val_frac=args.val_frac, seed=args.seed)
    num_classes = len(class_to_idx)
    print(f"{len(train_items)} train / {len(val_items)} val clips  |  {num_classes} classes")

    collate = PadCollate(use_velocity=True)
    train_ds = KSLLandmarkDataset(train_items, class_to_idx, args.max_len,
                                  augment=AugmentPipeline.default())
    val_ds = KSLLandmarkDataset(val_items, class_to_idx, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.workers, drop_last=True)
    print(f"train batches: {len(train_loader)}  |  val batches: {len(val_loader)}")

    in_ch = 5   # x, y, dx, dy, vis -- see assemble_features(); matches sign_model's IN_CH
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.models:
        results.append(train_one(name, args, train_loader, val_loader,
                                 num_classes, in_ch, device, out_dir))

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "params", "val_acc", "val_top3",
                                          "val_top5", "pretrained"])
        w.writeheader()
        w.writerows(results)

    print(f"\n{'=' * 60}\nsummary ({csv_path})\n{'=' * 60}")
    print(f"{'model':<10}{'params':>12}{'val_acc':>10}{'top3':>10}{'top5':>10}  pretrained")
    for r in results:
        print(f"{r['model']:<10}{r['params']:>12,}{r['val_acc']:>10.4f}"
             f"{r['val_top3']:>10.4f}{r['val_top5']:>10.4f}  {r['pretrained']}")


if __name__ == "__main__":
    main()
