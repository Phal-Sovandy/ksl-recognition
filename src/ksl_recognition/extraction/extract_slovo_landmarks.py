"""
extract_slovo_landmarks.py
--------------------------
SLOVO dataset -> MediaPipe Holistic -> per-clip (coords, vis) arrays
in the same format as extract_landmarks.py (KSL pipeline).

Shared extraction core (MediaPipe processing, worker pool, progress bar,
summary) lives in extraction/utils.py -- this file only implements the one
thing specific to SLOVO's layout: reading annotations.csv to find
(video_path, out_path) pairs.

SLOVO layout:
    slovo/
        annotations.csv        (attachment_id, text, user_id, height, width, length, train)
        train/<uuid>.mp4
        test/<uuid>.mp4

Output mirrors the KSL landmark format that features.assemble_features() expects:
    coords : float32  (T, 51, 2)   pixel-space x, y
    vis    : float32  (T, 51)      per-node visibility

Node layout (same as extract_landmarks.py / features.py / sign_model.py):
    [ 0.. 8]  upper-body pose (9)   [ 9..29]  left hand (21)   [30..50]  right hand (21)

Output directory tree:
    <output>/
        train/<label>/<uuid>.npz
        test/<label>/<uuid>.npz

Run:
    python -m ksl_recognition.extraction.extract_slovo_landmarks \\
        --slovo  data/raw_data/slovo \\
        --output data/processed_landmarks/slovo \\
        --workers 8

    # Only train split, skip already-done files:
    python -m ksl_recognition.extraction.extract_slovo_landmarks \\
        --slovo data/raw_data/slovo --output data/processed_landmarks/slovo \\
        --splits train --workers 8
"""
import argparse
from pathlib import Path

import pandas as pd

from ksl_recognition.extraction.utils import VIDEO_EXTS, add_common_args, run_extraction


def _load_annotations(slovo_dir):
    """Return DataFrame with columns: attachment_id, text, split (train/test)."""
    csv_path = slovo_dir / "annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"annotations.csv not found at {csv_path}")
    df = pd.read_csv(csv_path, sep="\t")
    # Normalise the train column -> split string
    df["split"] = df["train"].apply(lambda x: "train" if str(x).strip().lower() == "true" else "test")
    return df[["attachment_id", "text", "split"]]


def _find_video(slovo_dir, split, attachment_id):
    """Return Path to video or None if not found (try all known extensions)."""
    base = slovo_dir / split / attachment_id
    for ext in VIDEO_EXTS:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None


def discover_pairs(slovo_dir, output_dir, splits=("train", "test")):
    slovo_dir = Path(slovo_dir)
    df = _load_annotations(slovo_dir)
    df = df[df["split"].isin(splits)].reset_index(drop=True)

    pairs, missing = [], 0
    for _, row in df.iterrows():
        uid, label, split = row["attachment_id"], row["text"], row["split"]
        video_path = _find_video(slovo_dir, split, uid)
        if video_path is None:
            missing += 1
            continue
        out_path = output_dir / split / label / f"{uid}.npz"
        pairs.append((video_path, out_path))

    if missing:
        print(f"Warning: {missing} entries in annotations.csv have no video file on disk.")
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks from SLOVO dataset.")
    ap.add_argument("--slovo", required=True,
                    help="root of SLOVO dataset (contains annotations.csv, train/, test/)")
    ap.add_argument("--output", required=True, help="root dir for .npz landmark files")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "test"], help="which splits to process (default: both)")
    add_common_args(ap)
    args = ap.parse_args()

    output_dir = Path(args.output)
    pairs = discover_pairs(args.slovo, output_dir, tuple(args.splits))
    if not pairs:
        print("No videos found to process. Check --slovo path and split directories.")
        return

    run_extraction(pairs, desc="SLOVO landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
