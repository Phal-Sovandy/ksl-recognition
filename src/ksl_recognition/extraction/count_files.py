"""
count_files.py
---------------
Quick sanity check on extraction progress: for each dataset under
data/raw_data/, count raw video files vs. .npz landmark files under the
matching data/processed_landmarks/ subfolder, so you can see at a glance
what's done and what's left without re-running extraction.

Raw-data folder layout varies wildly per dataset (flat, nested by class,
nested by class/signer, train/test splits, ...) so this doesn't try to
mirror structure -- it just recursively counts video files (same VIDEO_EXTS
extraction/utils.py uses) under each dataset root and recursively counts .npz
files under the corresponding processed folder.

One dataset breaks that pattern: "Arabic SL (2)" ships as pre-extracted JPG
frame folders, not videos (see extract_arabic2_landmarks.py) -- one "clip"
there is one leaf folder of .jpg frames, not one video file. FRAME_DATASETS
below counts leaf image-folders instead of VIDEO_EXTS for those.

The raw_data <-> processed_landmarks name mapping isn't 1:1 (e.g. "Arabic SL
(2)" -> "arabic2", "ksl_dataset" -> "ksl"), so it's hardcoded below -- update
DATASET_MAP if a new dataset is added.

Run:
    python -m ksl_recognition.extraction.count_files
    python -m ksl_recognition.extraction.count_files --raw data/raw_data --processed data/processed_landmarks
"""
import argparse
from pathlib import Path

from ksl_recognition.extraction.utils import VIDEO_EXTS

# raw_data folder name -> processed_landmarks folder name
DATASET_MAP = {
    "Arabic SL (2)":  "arabic2",
    "Argentina SL":   "argentina",
    "Indian SL (2)":  "indian2",
    "Sri Lanka":      "srilanka",
    "Vietnam SL":     "vietnam",
    "WLASL":          "wlasl",
    "ksl_dataset":    "ksl",
    "slovo":          "slovo",
}

# datasets shipped as frame-image folders instead of video files -- count
# leaf image-folders ("clips") instead of VIDEO_EXTS for these
FRAME_DATASETS = {"Arabic SL (2)"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_IGNORED_NAMES = {".DS_Store"}


def _iter_files(root: Path):
    """All files under root, skipping hidden/junk files and __pycache__."""
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file() or p.name in _IGNORED_NAMES or "__pycache__" in p.parts:
            continue
        yield p


def count_videos(root: Path) -> int:
    return sum(1 for p in _iter_files(root) if p.suffix.lower() in VIDEO_EXTS)


def count_frame_clips(root: Path) -> int:
    """One "clip" = one leaf directory that directly contains image files."""
    if not root.exists():
        return 0
    count = 0
    for d in root.rglob("*"):
        if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir() if p.is_file()):
            count += 1
    return count


def count_npz(root: Path) -> int:
    return sum(1 for p in _iter_files(root) if p.suffix.lower() == ".npz")


def main():
    ap = argparse.ArgumentParser(
        description="Count raw videos vs. extracted .npz landmarks per dataset.")
    ap.add_argument("--raw", default="data/raw_data")
    ap.add_argument("--processed", default="data/processed_landmarks")
    args = ap.parse_args()

    raw_root, proc_root = Path(args.raw), Path(args.processed)

    rows = []
    for raw_name, proc_name in DATASET_MAP.items():
        raw_dir, proc_dir = raw_root / raw_name, proc_root / proc_name
        n_raw = count_frame_clips(raw_dir) if raw_name in FRAME_DATASETS else count_videos(raw_dir)
        n_proc = count_npz(proc_dir)
        rows.append((raw_name, n_raw, proc_name, n_proc))

    # anything sitting in either root that DATASET_MAP doesn't know about
    known_raw = {r.lower() for r in DATASET_MAP}
    known_proc = {p.lower() for p in DATASET_MAP.values()}
    extra_raw = [d.name for d in (raw_root.iterdir() if raw_root.exists() else [])
                 if d.is_dir() and d.name.lower() not in known_raw]
    extra_proc = [d.name for d in (proc_root.iterdir() if proc_root.exists() else [])
                  if d.is_dir() and d.name.lower() not in known_proc]

    name_w = max(len(r) for r, *_ in rows) + 2
    pname_w = max(len(p) for *_, p, _ in rows) + 2
    header = (f"{'raw dataset':<{name_w}}{'videos':>8}   "
              f"{'processed dataset':<{pname_w}}{'npz':>8}   {'done %':>7}")
    print(header)
    print("-" * len(header))

    tot_raw = tot_proc = 0
    for raw_name, n_raw, proc_name, n_proc in rows:
        pct = f"{(100 * n_proc / n_raw):.1f}%" if n_raw else "n/a"
        print(f"{raw_name:<{name_w}}{n_raw:>8}   {proc_name:<{pname_w}}{n_proc:>8}   {pct:>7}")
        tot_raw += n_raw
        tot_proc += n_proc

    print("-" * len(header))
    pct = f"{(100 * tot_proc / tot_raw):.1f}%" if tot_raw else "n/a"
    print(f"{'TOTAL':<{name_w}}{tot_raw:>8}   {'':<{pname_w}}{tot_proc:>8}   {pct:>7}")

    if extra_raw:
        print(f"\nNote: dirs under {raw_root} not in DATASET_MAP (not counted): {extra_raw}")
    if extra_proc:
        print(f"Note: dirs under {proc_root} not in DATASET_MAP (not counted): {extra_proc}")


if __name__ == "__main__":
    main()
