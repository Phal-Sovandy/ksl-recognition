"""
eda_landmarks.py
-----------------
Exploratory data analysis over every extracted dataset in
data/processed_landmarks/*, without touching any training code. Standalone
by design (only numpy/pandas/matplotlib, + optional scipy for the density
plots, scikit-learn for t-SNE, and/or umap-learn for UMAP) so it runs even
in an environment that doesn't have torch/cv2.

Per dataset (auto-discovered, same "flat root/<class>/*.npz vs pre-split
root/{train,test}/<class>/*.npz" layouts train_multilingual.py/dataset.py
already handle -- see DRAFT_NOTES.md's Kaggle-export section):
  * class count, total clip count
  * clips-per-class distribution: min / max / mean / median / std
  * clip-length-in-frames distribution: min / max / mean / median / std
    (from a random sample per dataset -- opening every .npz in e.g. arabic2's
    76k clips just to read a shape would be slow; use --sample-size 0 for an
    exhaustive pass if you want exact numbers)
  * mean landmark-visibility ratio (how much of the skeleton is actually
    detected, on average) -- a cheap data-quality signal
  * optional per-dataset 2D embedding (t-SNE or UMAP, see --method) of a
    simple per-clip pose descriptor (mean + std of shoulder-normalised
    (x, y) per node), coloured by class
  * optional cross-language embedding (a sample of clips from every
    dataset, same descriptor), coloured by language -- shows whether sign
    languages separate in raw pose-statistics space before any model is
    trained. UMAP is the better choice here (see DRAFT_NOTES.md discussion):
    t-SNE only preserves local neighbourhoods, not inter-cluster distances,
    so it can't honestly answer "is language A closer to B than to C" --
    UMAP preserves more of that global structure and scales far better to
    the larger datasets (arabic2 ~76k clips, wlasl ~12.5k).

Output goes to --out-dir (default plots/eda/): summary.csv, one
<lang>_class_counts.csv and one <lang>_frame_lengths.csv (per-clip class,
path, frame count, visibility ratio -- the same sampled clips the
frame-length plot and summary.csv stats are computed from) per dataset,
plus *.png plots -- including combined_frame_length_density.png and
combined_visibility_ratio_density.png, one KDE line per language overlaid
on a single shared plot per metric (needs scipy; skipped with a message if
it's not installed).

If a dataset's class folders were renamed to opaque IDs (e.g.
ksl_recognition.export.encode_ksl_classes's sign_001, sign_002, ... for KSL), a
`sign_id_map.xlsx` sitting at that dataset's root is picked up automatically
and every class name in every output above (CSVs, plots, embedding legends)
shows the real class name instead of the ID -- no flag needed, and datasets
without such a file behave exactly as before (see load_class_name_map()).

Run from the project root:
    python -m ksl_recognition.analysis.eda_landmarks
    python -m ksl_recognition.analysis.eda_landmarks --languages ksl,arabic2 --tsne --combined-tsne
    python -m ksl_recognition.analysis.eda_landmarks --method umap --combined-tsne --no-plots
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this is a script, not a notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

METHOD_REQUIREMENT = {
    "tsne": (HAS_SKLEARN, "pip install scikit-learn"),
    "umap": (HAS_UMAP, "pip install umap-learn"),
}

SPLIT_NAMES = {"train", "test", "val"}
NAMED_BAR_MAX_CLASSES = 50   # above this, class names on a bar chart are unreadable

# Written by ksl_recognition.export.encode_ksl_classes (sign_id -> gloss) when a
# dataset's class folders were renamed to opaque IDs (sign_001, ...). If a
# dataset root has one of these, every class name below is translated back
# to its real name automatically -- no per-dataset flag needed, and datasets
# that were never encoded (no such file) behave exactly as before.
CLASS_NAME_MAP_FILENAME = "sign_id_map.xlsx"


def load_class_name_map(root: Path) -> dict[str, str]:
    """{sign_id: gloss} from root/sign_id_map.xlsx, or {} if that file
    doesn't exist (the common case -- most datasets aren't encoded) or can't
    be read (e.g. openpyxl not installed -- falls back to raw folder names
    rather than failing the whole EDA run)."""
    path = root / CLASS_NAME_MAP_FILENAME
    if not path.exists():
        return {}
    try:
        df = pd.read_excel(path)
        return dict(zip(df["sign_id"].astype(str), df["gloss"].astype(str)))
    except Exception as e:
        print(f"  [{root.name}] could not read {path.name} ({e}) -- using raw folder names")
        return {}


# --------------------------------------------------------------------------- #
# Discovery -- mirrors dataset.py/train_multilingual.py's two known layouts.
# --------------------------------------------------------------------------- #
def discover_classes(root: Path):
    """Returns (entries, is_pre_split).
    entries: list of (class_name, class_dir_path, split_or_None). class_name
    is resolved through root's sign_id_map.xlsx when present (see
    load_class_name_map) -- e.g. ksl's "sign_041" folder shows up as its real
    Khmer gloss everywhere downstream (CSVs, plots, embeddings) instead of
    the opaque encoded ID."""
    name_map = load_class_name_map(root)
    resolved = lambda n: name_map.get(n, n)

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    names = {p.name for p in subdirs}
    is_pre_split = bool(subdirs) and names <= SPLIT_NAMES and all(
        any(c.is_dir() for c in p.iterdir()) for p in subdirs
    )
    if is_pre_split:
        entries = [
            (resolved(class_dir.name), class_dir, split_dir.name)
            for split_dir in subdirs
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        ]
    else:
        entries = [(resolved(p.name), p, None) for p in subdirs]
    return entries, is_pre_split


def list_datasets(root: Path, wanted: set[str] | None):
    for p in sorted(root.iterdir()):
        if p.is_dir() and (wanted is None or p.name in wanted):
            yield p.name, p


# --------------------------------------------------------------------------- #
# Per-clip descriptor for t-SNE: shoulder-normalised (resolution/position/
# scale invariant), mean+std per node over visible frames only. Node indices
# match features.py's upper-body layout (POSE=9, L/R shoulder = 1/2), but
# this file deliberately doesn't import features.py/torch -- keeps the EDA
# tool dependency-light and runnable outside the ksl conda env if needed.
# --------------------------------------------------------------------------- #
def normalize_clip(coords: np.ndarray, vis: np.ndarray) -> np.ndarray:
    both = (vis[:, 1] > 0) & (vis[:, 2] > 0)
    if both.any():
        center = ((coords[both, 1] + coords[both, 2]) * 0.5).mean(axis=0)
        scale = np.linalg.norm(coords[both, 1] - coords[both, 2], axis=-1).mean()
    elif (vis > 0).any():
        center = coords[vis > 0].mean(axis=0)
        scale = 1.0
    else:
        center, scale = np.zeros(2), 1.0
    return (coords - center) / max(scale, 1e-3)


def clip_descriptor(coords: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(T, V, 2), (T, V) -> flat (V*2*2,) vector: per-node mean & std of
    normalised (x, y) across visible frames; nodes never visible -> 0."""
    norm = normalize_clip(coords, vis)
    mask = vis > 0
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A node that's never visible in this clip (e.g. an occluded hand)
        # gives an all-NaN slice -- expected, nan_to_num below turns it into
        # a clean 0.0, but nanmean/nanstd warn about it every time otherwise.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        masked = np.where(mask[..., None], norm, np.nan)
        mean = np.nan_to_num(np.nanmean(masked, axis=0), nan=0.0)
        std = np.nan_to_num(np.nanstd(masked, axis=0), nan=0.0)
    return np.concatenate([mean.ravel(), std.ravel()])


def load_npz(path: Path):
    d = np.load(path)
    return d["coords"], d["vis"]


# --------------------------------------------------------------------------- #
# Per-dataset stats
# --------------------------------------------------------------------------- #
def dataset_stats(name: str, root: Path, sample_size: int, rng: random.Random):
    entries, is_pre_split = discover_classes(root)

    per_class_count: dict[str, int] = {}
    per_split_count: dict[str, int] = {}
    all_clip_paths: list[tuple[str, Path]] = []   # (class_name, npz_path) -- for sampling
    for class_name, class_dir, split in entries:
        npzs = sorted(class_dir.glob("*.npz"))
        per_class_count[class_name] = per_class_count.get(class_name, 0) + len(npzs)
        if split:
            per_split_count[split] = per_split_count.get(split, 0) + len(npzs)
        all_clip_paths.extend((class_name, p) for p in npzs)

    counts = np.array(list(per_class_count.values()))
    total_clips = int(counts.sum())

    # Sample for frame-length / visibility stats (opening files is the slow part).
    # frame_rows is kept (not just the aggregate arrays below) so the CSV export
    # and the frame-length plot can reuse this exact sample instead of each
    # re-opening every file with their own separate sample.
    pool = all_clip_paths if sample_size <= 0 else rng.sample(
        all_clip_paths, k=min(sample_size, len(all_clip_paths))
    )
    frame_rows, errors = [], 0
    for cname, path in pool:
        try:
            coords, vis = load_npz(path)
            frame_rows.append((cname, str(path), int(coords.shape[0]), float((vis > 0).mean())))
        except Exception:
            errors += 1
    lengths = np.array([r[2] for r in frame_rows]) if frame_rows else np.array([0])
    vis_ratios = np.array([r[3] for r in frame_rows]) if frame_rows else np.array([0.0])

    summary = dict(
        dataset=name,
        n_classes=len(per_class_count),
        total_clips=total_clips,
        clips_per_class_min=int(counts.min()) if len(counts) else 0,
        clips_per_class_max=int(counts.max()) if len(counts) else 0,
        clips_per_class_mean=float(counts.mean()) if len(counts) else 0.0,
        clips_per_class_median=float(np.median(counts)) if len(counts) else 0.0,
        clips_per_class_std=float(counts.std()) if len(counts) else 0.0,
        frame_len_min=int(lengths.min()),
        frame_len_max=int(lengths.max()),
        frame_len_mean=float(lengths.mean()),
        frame_len_median=float(np.median(lengths)),
        frame_len_std=float(lengths.std()),
        mean_visibility_ratio=float(vis_ratios.mean()),
        sample_size_used=len(pool),
        read_errors=errors,
        is_pre_split=is_pre_split,
        **{f"split_{k}_clips": v for k, v in per_split_count.items()},
    )
    return summary, per_class_count, all_clip_paths, frame_rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_class_distribution(name: str, per_class_count: dict, out_dir: Path):
    counts = sorted(per_class_count.values(), reverse=True)
    n = len(counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, n + 1), counts, marker=".")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("class rank (most -> least clips)")
    axes[0].set_ylabel("clips (log scale)")
    axes[0].set_title(f"{name}: class-frequency rank ({n} classes)")

    axes[1].hist(counts, bins=min(30, max(5, n // 2)))
    axes[1].set_xlabel("clips per class")
    axes[1].set_ylabel("number of classes")
    axes[1].set_title(f"{name}: clips-per-class histogram")

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_class_distribution.png", dpi=130)
    plt.close(fig)

    if n <= NAMED_BAR_MAX_CLASSES:
        items = sorted(per_class_count.items(), key=lambda kv: kv[1], reverse=True)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.22 * n)))
        ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1])
        ax.set_xlabel("clips")
        ax.set_title(f"{name}: clips per class")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_class_counts_named.png", dpi=130)
        plt.close(fig)


def plot_frame_lengths(name: str, frame_rows, out_dir: Path):
    """frame_rows: the (class, path, frame_length, vis_ratio) rows dataset_stats()
    already sampled -- reused here instead of re-sampling/re-reading the files."""
    lengths = [r[2] for r in frame_rows]
    if not lengths:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=40)
    ax.set_xlabel("frames per clip")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: clip length distribution (n={len(lengths)} sampled)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_frame_lengths.png", dpi=130)
    plt.close(fig)


def run_embedding(descriptors: np.ndarray, seed: int, method: str,
                  n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """2D embedding via t-SNE or UMAP -- same (N, 2) output shape either way,
    so callers don't need to branch on `method`."""
    if method == "umap":
        n_neighbors = min(n_neighbors, len(descriptors) - 1)
        return umap.UMAP(
            n_components=2, n_neighbors=max(2, n_neighbors), min_dist=min_dist,
            random_state=seed,
        ).fit_transform(descriptors)
    perplexity = min(30, max(5, len(descriptors) // 4))
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        random_state=seed, learning_rate="auto",
    ).fit_transform(descriptors)


def plot_dataset_embedding(name: str, per_class_count: dict, all_clip_paths, args, rng, out_dir: Path):
    ok, install_hint = METHOD_REQUIREMENT[args.method]
    if not ok:
        print(f"  [{name}] skipping {args.method} -- not installed ({install_hint} in the ksl env)")
        return
    top_classes = [c for c, _ in sorted(per_class_count.items(), key=lambda kv: kv[1], reverse=True)
                   [:args.tsne_classes]]
    by_class: dict[str, list[Path]] = {}
    for cname, path in all_clip_paths:
        if cname in top_classes:
            by_class.setdefault(cname, []).append(path)

    labels, descriptors = [], []
    for cname in top_classes:
        paths = by_class.get(cname, [])
        chosen = rng.sample(paths, k=min(args.tsne_per_class, len(paths)))
        for path in chosen:
            try:
                coords, vis = load_npz(path)
                descriptors.append(clip_descriptor(coords, vis))
                labels.append(cname)
            except Exception:
                pass
    if len(descriptors) < 10:
        print(f"  [{name}] skipping {args.method} -- not enough readable clips sampled")
        return

    emb = run_embedding(np.stack(descriptors), args.seed, args.method,
                        args.umap_neighbors, args.umap_min_dist)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20" if len(top_classes) <= 20 else "gist_ncar")
    for i, cname in enumerate(top_classes):
        idx = [j for j, l in enumerate(labels) if l == cname]
        if idx:
            ax.scatter(emb[idx, 0], emb[idx, 1], s=14, color=cmap(i / max(1, len(top_classes) - 1)),
                       label=cname if len(top_classes) <= 20 else None)
    ax.set_title(f"{name}: {args.method.upper()} of top {len(top_classes)} classes "
                 f"({args.tsne_per_class}/class, pose descriptor)")
    if len(top_classes) <= 20:
        ax.legend(fontsize=6, markerscale=1.5, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_{args.method}.png", dpi=140)
    plt.close(fig)


def plot_combined_embedding(datasets: dict, args, rng, out_dir: Path):
    ok, install_hint = METHOD_REQUIREMENT[args.method]
    if not ok:
        print(f"skipping combined {args.method} -- not installed ({install_hint} in the ksl env)")
        return
    labels, descriptors = [], []
    for name, (_, _, all_clip_paths, _) in datasets.items():
        chosen = rng.sample(all_clip_paths, k=min(args.combined_per_dataset, len(all_clip_paths)))
        for _, path in chosen:
            try:
                coords, vis = load_npz(path)
                descriptors.append(clip_descriptor(coords, vis))
                labels.append(name)
            except Exception:
                pass
    if len(descriptors) < 10:
        print(f"skipping combined {args.method} -- not enough readable clips sampled")
        return

    emb = run_embedding(np.stack(descriptors), args.seed, args.method,
                        args.umap_neighbors, args.umap_min_dist)
    names = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(names):
        idx = [j for j, l in enumerate(labels) if l == name]
        ax.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.7,
                   color=cmap(i / max(1, len(names) - 1)), label=name)
    ax.set_title(f"Cross-language {args.method.upper()} "
                 f"({args.combined_per_dataset} clips/dataset, pose descriptor)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_name = f"combined_{args.method}_by_language.png"
    fig.savefig(out_dir / out_name, dpi=140)
    plt.close(fig)
    print(f"  saved {out_name} ({len(descriptors)} clips, {len(names)} languages)")


def _kde_line(values, n_points: int = 200):
    """1D array -> (x, density) for a smooth KDE curve, or (None, None) if
    there aren't enough samples (or no variation at all -- e.g. every value
    identical) to fit a density from."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return None, None
    kde = gaussian_kde(values)
    lo, hi = values.min(), values.max()
    pad = (hi - lo) * 0.1 or 1.0   # a little headroom past the data's own range
    x = np.linspace(lo - pad, hi + pad, n_points)
    return x, kde(x)


def plot_combined_density(datasets: dict, out_dir: Path):
    """One density-line plot per metric (frame length, visibility ratio),
    every dataset's frame_rows sample overlaid as its own line on a single
    shared axes -- answers "how does clip length / detection quality compare
    across languages" at a glance, reusing the exact same sample each
    dataset's *_frame_lengths.csv was built from (no extra file I/O)."""
    if not HAS_SCIPY:
        print("skipping density plots -- scipy not installed (pip install scipy in the ksl env)")
        return

    # (metric key, column index into a frame_rows row, x-axis label)
    metrics = [("frame_length", 2, "frames per clip"),
              ("visibility_ratio", 3, "visibility ratio")]
    names = sorted(datasets)
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")

    for metric_name, col_idx, xlabel in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for i, name in enumerate(names):
            frame_rows = datasets[name][3]
            x, y = _kde_line([r[col_idx] for r in frame_rows])
            if x is None:
                continue
            ax.plot(x, y, label=name, color=cmap(i / max(1, len(names) - 1)), linewidth=1.6)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(f"{metric_name.replace('_', ' ')} density by language")
        ax.legend(fontsize=8, loc="best", ncol=2 if len(names) > 8 else 1)
        fig.tight_layout()
        out_name = f"combined_{metric_name}_density.png"
        fig.savefig(out_dir / out_name, dpi=140)
        plt.close(fig)
        print(f"  saved {out_name} ({plotted} languages)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/processed_landmarks")
    ap.add_argument("--out-dir", default="plots/eda")
    ap.add_argument("--languages", default=None,
                    help="comma-separated subset of dataset folder names (default: all found)")
    ap.add_argument("--sample-size", type=int, default=500,
                    help="clips sampled per dataset for frame-length/visibility stats; 0 = exhaustive")
    ap.add_argument("--no-plots", action="store_true", help="only compute/save summary.csv, skip PNGs")
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne",
                    help="embedding method for --tsne/--combined-tsne (default: tsne). "
                         "umap needs 'pip install umap-learn' in the ksl env; scales better "
                         "to large datasets and preserves inter-cluster distance better than t-SNE")
    ap.add_argument("--tsne", action="store_true", help="per-dataset 2D embedding of top classes")
    ap.add_argument("--tsne-classes", type=int, default=15)
    ap.add_argument("--tsne-per-class", type=int, default=20)
    ap.add_argument("--combined-tsne", action="store_true", help="cross-language 2D embedding, colored by dataset")
    ap.add_argument("--combined-per-dataset", type=int, default=150)
    ap.add_argument("--umap-neighbors", type=int, default=15, help="umap only: n_neighbors")
    ap.add_argument("--umap-min-dist", type=float, default=0.1, help="umap only: min_dist")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    wanted = set(args.languages.split(",")) if args.languages else None
    datasets = {}   # name -> (summary, per_class_count, all_clip_paths, frame_rows)

    print(f"Scanning datasets under {root}/ ...")
    for name, path in list_datasets(root, wanted):
        print(f"  [{name}] ...", end=" ", flush=True)
        summary, per_class_count, all_clip_paths, frame_rows = dataset_stats(
            name, path, args.sample_size, rng)
        datasets[name] = (summary, per_class_count, all_clip_paths, frame_rows)
        print(f"{summary['n_classes']} classes, {summary['total_clips']} clips")

        pd.DataFrame(
            sorted(per_class_count.items(), key=lambda kv: kv[1], reverse=True),
            columns=["class", "clip_count"],
        ).to_csv(out_dir / f"{name}_class_counts.csv", index=False)

        # Per-clip frame length + visibility ratio, for the same sample used
        # in summary.csv's frame_len_*/mean_visibility_ratio and (if plotted
        # below) the frame-length histogram -- written regardless of --no-plots,
        # same as class_counts.csv above.
        pd.DataFrame(
            frame_rows, columns=["class", "clip_path", "frame_length", "visibility_ratio"],
        ).to_csv(out_dir / f"{name}_frame_lengths.csv", index=False)

        if not args.no_plots:
            plot_class_distribution(name, per_class_count, out_dir)
            plot_frame_lengths(name, frame_rows, out_dir)
        if args.tsne:
            plot_dataset_embedding(name, per_class_count, all_clip_paths, args, rng, out_dir)

    if not datasets:
        print("No datasets found -- check --root/--languages.")
        return

    summary_df = pd.DataFrame([s for s, _, _, _ in datasets.values()]).set_index("dataset")
    summary_df.to_csv(out_dir / "summary.csv")
    print("\n" + summary_df.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved summary.csv + per-dataset CSVs/plots -> {out_dir.resolve()}")

    if not args.no_plots:
        plot_combined_density(datasets, out_dir)

    if args.combined_tsne:
        plot_combined_embedding(datasets, args, rng, out_dir)


if __name__ == "__main__":
    main()
