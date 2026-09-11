"""
train_multilingual.py
----------------------
Stage-1 JOINT pretraining across every available sign-language dataset at
once: one shared encoder (SpatialGCN -> TemporalTransformer), one classifier
head per language, trained by round-robin interleaving batches from each
language's own loader every epoch.

This is deliberately NOT the same as train.py's single-dataset --pretrained
flow (that trains one language, save, done). Training jointly means the
shared encoder gets gradient signal from every language throughout, instead
of overwriting itself language-by-language -- see DRAFT_NOTES.md for why
sequential pretraining risks catastrophic forgetting and why merging all
languages' classes into one flat head is actively wrong (false friends:
visually similar signs across languages can mean different things, and nothing
in skeleton-only input signals which language a clip belongs to).

--model selects the architecture being pretrained: "gcn" (default,
sign_model.MultiHeadSignModel) or one of baselines.BASELINE_PRESETS
("rnn"/"lstm"/"bilstm"/"gru"/"bigru") -- for the paper's architecture-
comparison table, run this once per baseline with the SAME --languages subset
used for the GCN's winning pretrain combo (see DRAFT_NOTES.md's Table 1
ablation), then feed each resulting checkpoint into train_baselines.py's
--pretrained for Stage 2. Every architecture shares the exact same round-robin
loop / optimizer / schedule below since MultiHeadRecurrentModel matches
MultiHeadSignModel's forward(feats, vis, fmask, head=...) contract exactly --
only model construction and the saved "arch" dict branch on --model.

At the end, only encoder.state_dict() matters -- every language head here is
disposable, exactly like the single-language rsl_pretrain checkpoint was.
Feed the resulting checkpoint into train.py's --pretrained for Stage 2 (KSL):
loading with strict=False already discards every head here (none of them are
named "ksl"), keeping only the shared encoder weights.

Languages auto-discovered from LANGUAGES below: any entry whose --data path
doesn't exist on disk yet is skipped with a warning, not an error, so this
script runs today with whatever's extracted and picks up more automatically
as more extract_*_landmarks.py runs finish.

Run as a module (requires `pip install -e .` from the repo root, see README):

    python -m ksl_recognition.training.train_multilingual --epochs 120 --steps-per-lang 40

Resume (same idea as train.ipynb's per-stage resume cell):
    python -m ksl_recognition.training.train_multilingual --epochs 200 --resume
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ksl_recognition.data.augmentations import AugmentPipeline
from ksl_recognition.data.dataset import (KSLLandmarkDataset, PadCollate, scan_landmarks,
                     scan_val_landmarks, stratified_split)
from ksl_recognition.models.sign_model import MultiHeadSignModel, count_parameters
from ksl_recognition.models.baselines import BASELINE_PRESETS, MultiHeadRecurrentModel

# --------------------------------------------------------------------------- #
# Language registry -- add a new language here once it's extracted; entries
# whose --data path doesn't exist yet are skipped automatically (see main()).
# "val_data" = dedicated held-out split (like Slovo's train/test); languages
# without one fall back to a stratified split of --data via VAL_FRAC.
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "rsl":       dict(data="data/processed_landmarks/slovo/train", val_data="data/processed_landmarks/slovo/test"),
    "arabic":    dict(data="data/processed_landmarks/arabic",  val_data=None),
    "wlasl":     dict(data="data/processed_landmarks/wlasl",    val_data=None),
    "indian":   dict(data="data/processed_landmarks/indian",  val_data=None),
    "srilanka":  dict(data="data/processed_landmarks/srilanka", val_data=None),
    "argentina": dict(data="data/processed_landmarks/argentina",val_data=None),
    "vietnam":   dict(data="data/processed_landmarks/vietnam",  val_data=None),
}
VAL_FRAC = 0.15   # for languages without a dedicated val_data split


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cycle_loader(loader):
    """Infinite generator over a DataLoader -- reshuffles every full pass
    (each `for batch in loader` call re-triggers loader.__iter__())."""
    while True:
        for batch in loader:
            yield batch


def build_optimizer(model, enc_lr, head_lr, weight_decay):
    """Layer-wise LR: all heads (however many languages) get head_lr, the
    shared encoder gets enc_lr."""
    head_ids = {id(p) for head in model.heads.values() for p in head.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    enc_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    return torch.optim.AdamW(
        [{"params": enc_params, "lr": enc_lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )


def load_language(name, cfg, args, augmenter, collate):
    """Scan + split + build (train_loader, val_loader, num_classes, class_to_idx)
    for one language, or None if its data isn't extracted yet."""
    data_dir = Path(cfg["data"])
    if not data_dir.exists():
        print(f"  [{name}] SKIPPED -- {data_dir} not found (not extracted yet?)")
        return None

    items, class_to_idx = scan_landmarks(data_dir)
    if cfg.get("val_data"):
        train_items = items
        val_items = scan_val_landmarks(cfg["val_data"], class_to_idx)
    else:
        train_items, val_items = stratified_split(items, VAL_FRAC, args.seed)

    train_ds = KSLLandmarkDataset(train_items, class_to_idx, args.max_len, augment=augmenter)
    val_ds = KSLLandmarkDataset(val_items, class_to_idx, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.workers)

    print(f"  [{name:10s}] {len(class_to_idx):4d} classes | "
          f"{len(train_items):6d} train / {len(val_items):5d} val | "
          f"{len(train_loader):4d} batches/epoch")
    return {
        "class_to_idx": class_to_idx,
        "num_classes": len(class_to_idx),
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


@torch.no_grad()
def evaluate_all(model, langs, device):
    """Run evaluate() per language against its own val loader/head.
    Returns {lang: (loss, acc)}."""
    model.eval()
    results = {}
    for name, lang in langs.items():
        correct = total = 0
        loss_sum = 0.0
        for feats, vis, fmask, labels in lang["val_loader"]:
            feats, vis, fmask, labels = (t.to(device) for t in (feats, vis, fmask, labels))
            logits = model(feats, vis, fmask, head=name)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
        results[name] = (loss_sum / max(total, 1), correct / max(total, 1))
    return results


def main():
    ap = argparse.ArgumentParser(description="Joint multilingual Stage-1 pretraining.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--steps-per-lang", type=int, default=40,
                    help="batches drawn from EACH active language per epoch "
                         "(round-robin interleaved, not all-of-one-then-next). "
                         "Total steps/epoch = steps_per_lang * num_active_languages.")
    ap.add_argument("--enc-lr", type=float, default=5e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=4e-4)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--sched-patience", type=int, default=12)
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="applied to TRAIN loss only; eval stays plain cross-entropy "
                         "so val loss stays a trustworthy overfit signal")
    # model architecture -- MUST match what Stage 2 (--pretrained) expects to load
    ap.add_argument("--model", default="gcn", choices=["gcn", *BASELINE_PRESETS],
                    help="'gcn' = sign_model.MultiHeadSignModel (default); "
                         "otherwise one of baselines.BASELINE_PRESETS "
                         "(rnn/lstm/bilstm/gru/bigru)")
    ap.add_argument("--d-model", type=int, default=256, help="gcn only")
    ap.add_argument("--hidden", type=int, default=256, help="baseline models only")
    ap.add_argument("--num-layers", type=int, default=6,
                    help="transformer layers (gcn) or stacked RNN layers (baseline) -- "
                         "gcn default (6) is deliberately deep; drop this for baselines "
                         "(e.g. --num-layers 2), their default depth in train_baselines.py")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=2,
                    help="per-language DataLoader workers -- total processes "
                         "spawned scale with number of ACTIVE languages")
    ap.add_argument("--out", default=None,
                    help="default: checkpoints/multilingual_pretrain.pt for --model gcn, "
                         "checkpoints/multilingual_pretrain_<model>.pt otherwise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--languages", default=None,
                    help="comma-separated subset of LANGUAGES to use (default: all "
                         "with data on disk) -- e.g. --languages rsl,arabic for a "
                         "quick smoke test before committing to a full run")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out's parent>/multilingual_pretrain_resume.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print("device:", device)

    wanted = set(args.languages.split(",")) if args.languages else set(LANGUAGES)
    augmenter = None if args.no_augment else AugmentPipeline.default()
    collate = PadCollate(use_velocity=True)

    print(f"\nScanning {len(wanted)} requested language(s):")
    langs = {}
    for name in sorted(wanted):
        if name not in LANGUAGES:
            print(f"  [{name}] SKIPPED -- not in LANGUAGES registry")
            continue
        result = load_language(name, LANGUAGES[name], args, augmenter, collate)
        if result is not None:
            langs[name] = result

    if not langs:
        print("\nNo languages available -- extract at least one before running this.")
        return
    print(f"\n{len(langs)} active language(s): {sorted(langs)}\n")

    in_ch = 5  # x, y, dx, dy, vis
    head_classes = {name: l["num_classes"] for name, l in langs.items()}

    if args.model == "gcn":
        d = args.d_model
        model = MultiHeadSignModel(
            in_ch=in_ch, head_classes=head_classes,
            d_model=d, gcn_hidden=(d // 2, d), nhead=max(1, d // 32),
            num_layers=args.num_layers, dim_ff=d * 2,
            dropout=args.dropout, max_len=args.max_len,
        ).to(device)

        def arch_cfg():
            return dict(in_ch=in_ch, d_model=d, gcn_hidden=(d // 2, d), nhead=max(1, d // 32),
                       num_layers=args.num_layers, dim_ff=d * 2, dropout=args.dropout,
                       max_len=args.max_len)
    else:
        model = MultiHeadRecurrentModel(
            in_ch=in_ch, head_classes=head_classes, hidden=args.hidden,
            num_layers=args.num_layers, dropout=args.dropout,
            **BASELINE_PRESETS[args.model],
        ).to(device)

        def arch_cfg():
            return dict(in_ch=in_ch, model=args.model, hidden=args.hidden,
                       num_layers=args.num_layers, dropout=args.dropout)

    # MPS doesn't implement the TransformerEncoder nested-tensor fast path
    # (only triggered in eval mode). Disable it so eval stays on the GPU.
    # No-op for baseline models -- they have no nn.TransformerEncoder submodule.
    for m in model.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            m.use_nested_tensor = False

    print(f"model: {args.model}  |  params: {count_parameters(model):,}")

    optim = build_optimizer(model, args.enc_lr, args.head_lr, args.weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=args.sched_patience, min_lr=1e-6)

    out_arg = args.out or (
        "checkpoints/multilingual_pretrain.pt" if args.model == "gcn"
        else f"checkpoints/multilingual_pretrain_{args.model}.pt"
    )
    out_path = Path(out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Namespaced by --model so a gcn run and a baseline run sharing --out's
    # parent dir don't clobber each other's resume checkpoint. Kept unsuffixed
    # for gcn to not break any in-flight resume of an existing run.
    resume_name = ("multilingual_pretrain_resume.pt" if args.model == "gcn"
                   else f"multilingual_pretrain_{args.model}_resume.pt")
    resume_path = out_path.parent / resume_name

    start_epoch, best_avg_acc, history = 1, 0.0, {"avg_val_acc": []}
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        for pg in optim.param_groups:            # let a changed --weight-decay take effect
            pg["weight_decay"] = args.weight_decay
        warmup_sched.load_state_dict(ckpt["warmup"])
        plateau_sched.load_state_dict(ckpt["plateau"])
        start_epoch = ckpt["epoch"] + 1
        best_avg_acc = ckpt["best_avg_acc"]
        history = ckpt["history"]
        print(f"resumed from epoch {ckpt['epoch']}  best_avg_acc={best_avg_acc:.3f}")
    elif args.resume:
        print(f"--resume given but {resume_path} not found -- starting from scratch")

    lang_names = sorted(langs)                    # fixed round-robin order
    train_iters = {name: cycle_loader(l["train_loader"]) for name, l in langs.items()}
    steps_per_epoch = args.steps_per_lang * len(lang_names)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        run = {name: {"loss": 0.0, "correct": 0, "total": 0} for name in lang_names}

        for step in range(steps_per_epoch):
            name = lang_names[step % len(lang_names)]
            feats, vis, fmask, labels = next(train_iters[name])
            feats, vis, fmask, labels = (t.to(device) for t in (feats, vis, fmask, labels))

            logits = model(feats, vis, fmask, head=name)
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            run[name]["loss"] += loss.item() * labels.numel()
            run[name]["correct"] += (logits.argmax(1) == labels).sum().item()
            run[name]["total"] += labels.numel()

        val_results = evaluate_all(model, langs, device)
        avg_val_acc = sum(acc for _, acc in val_results.values()) / len(val_results)

        if epoch <= args.warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(avg_val_acc)
        current_lr = optim.param_groups[0]["lr"]

        history["avg_val_acc"].append(avg_val_acc)
        flag = ""
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": {name: l["class_to_idx"] for name, l in langs.items()},
                "avg_val_acc": avg_val_acc,
                "per_language_val": val_results,
                "epoch": epoch,
                "arch": arch_cfg(),
                "model_type": args.model,
            }, out_path)
            flag = "  <- best, saved"

        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optim.state_dict(),
            "warmup": warmup_sched.state_dict(), "plateau": plateau_sched.state_dict(),
            "best_avg_acc": best_avg_acc, "history": history,
        }, resume_path)

        per_lang_str = "  ".join(
            f"{name}={run[name]['correct']/max(run[name]['total'],1):.2f}/"
            f"{val_results[name][1]:.2f}"
            for name in lang_names)
        print(f"epoch {epoch:>3} | lr {current_lr:.2e} | avg_val_acc {avg_val_acc:.3f}{flag}")
        print(f"    (train/val acc per language)  {per_lang_str}")

    print(f"\nBest avg val acc: {best_avg_acc:.3f}  ->  {out_path}")


if __name__ == "__main__":
    main()
