"""
extract_vietnam_landmarks.py
------------------------------
Vietnamese Sign Language -> MediaPipe Holistic -> per-clip (coords, vis)
arrays in the same format as extract_landmarks.py (KSL pipeline).

Shared extraction core lives in extraction/utils.py -- this file only implements
this dataset's layout: a CSV label file instead of folder-per-class.

    Vietnam SL/
        Videos/
            D0001N.mp4
            D0001T.mp4
            D0002.mp4
            ...
        Labels/
            label.csv        columns: ID,VIDEO,LABEL
                              1,D0001N.mp4,dia chi
                              2,D0001T.mp4,dia chi
                              3,D0001B.mp4,dia chi
                              4,D0002.mp4,tinh

Multiple videos legitimately share one LABEL (different takes/signers of the
same gloss, e.g. the D0001N/T/B trio above) -- they all land in that gloss's
output subfolder. Labels are Vietnamese text with diacritics; used as-is for
the output folder name (UTF-8 filenames, same approach as the Khmer KSL
class folders already in this project) except for a literal '/' -- some
glosses contain one (dates like "8/3", synonym pairs like "một ít/ một
chút") which extraction.utils.sanitize_class_name replaces before it can be
mistaken for a path separator.

Run:
    python -m ksl_recognition.extraction.extract_vietnam_landmarks \\
        --videos "data/raw_data/Vietnam SL/Videos" \\
        --labels "data/raw_data/Vietnam SL/Labels/label.csv" \\
        --output data/processed_landmarks/vietnam \\
        --workers 8
"""
import argparse
import csv
from pathlib import Path

from ksl_recognition.extraction.utils import add_common_args, run_extraction, sanitize_class_name


def _load_labels(csv_path):
    """label.csv -> list of (video_filename, sanitized_label) rows, in file order."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = (row.get("VIDEO") or row.get("video") or "").strip()
            label = (row.get("LABEL") or row.get("label") or "").strip()
            if video and label:
                rows.append((video, sanitize_class_name(label)))
    return rows


def discover_pairs(videos_dir, labels_csv, output_dir):
    videos_dir = Path(videos_dir)
    rows = _load_labels(labels_csv)
    pairs, missing = [], []
    for video_name, label in rows:
        video_path = videos_dir / video_name
        if not video_path.exists():
            missing.append(video_name)
            continue
        out_path = output_dir / label / f"{video_path.stem}.npz"
        pairs.append((video_path, out_path))
    return pairs, missing


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks for Vietnamese SL (CSV label layout).")
    ap.add_argument("--videos", default="data/raw_data/Vietnam SL/Videos",
                    help="dir containing the flat pool of video files")
    ap.add_argument("--labels", default="data/raw_data/Vietnam SL/Labels/label.csv",
                    help="CSV with columns ID,VIDEO,LABEL mapping filenames to glosses")
    ap.add_argument("--output", default="data/processed_landmarks/vietnam",
                    help="root dir for .npz landmark files")
    add_common_args(ap)
    args = ap.parse_args()

    output_dir = Path(args.output)
    pairs, missing = discover_pairs(args.videos, args.labels, output_dir)
    if missing:
        print(f"Warning: {len(missing)} rows in {args.labels} have no matching file "
              f"under {args.videos}, e.g. {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if not pairs:
        print("No videos found to process. Check --videos and --labels paths.")
        return

    run_extraction(pairs, desc="Vietnam SL landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
