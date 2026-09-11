"""
extract_argentina_landmarks.py
--------------------------------
Argentina Sign Language (LSA64) -> MediaPipe Holistic -> per-clip (coords, vis)
arrays in the same format as extract_landmarks.py (KSL pipeline).

Shared extraction core lives in extraction/utils.py -- this file only implements
this dataset's layout: no folders at all, class/signer/repetition encoded in
the filename.

    Argentina SL/
        001_001_001.mp4   <- class 001, signer 001, repetition 001
        001_001_002.mp4
        ...
        064_010_005.mp4   <- class 064, signer 010, repetition 005
    (64 classes x 10 signers x 5 repetitions = 3200 clips)

IMPORTANT -- class naming: this script does NOT hardcode LSA64's 64 gloss
names, because guessing them wrong would silently mislabel every clip. By
default the output class folders are just the zero-padded numeric IDs
("001", "002", ...). If you have LSA64's official id->gloss list, pass it
via --label-map as a 2-column CSV (id,label -- header row optional) and the
real gloss names will be used instead.

Run:
    python -m ksl_recognition.extraction.extract_argentina_landmarks \\
        --input  "data/raw_data/Argentina SL" \\
        --output data/processed_landmarks/argentina \\
        --workers 8

    # with the official gloss names:
    python -m ksl_recognition.extraction.extract_argentina_landmarks \\
        --input "data/raw_data/Argentina SL" --output data/processed_landmarks/argentina \\
        --label-map lsa64_glosses.csv --workers 8
"""
import argparse
import csv
import re
from pathlib import Path

from ksl_recognition.extraction.utils import (VIDEO_EXTS, add_common_args, run_extraction,
                           sanitize_class_name)

_FILENAME_RE = re.compile(r"^(\d{3})_(\d{3})_(\d{3})$")   # class_signer_rep


def _load_label_map(path):
    """2-column CSV: id,label (header row optional). Returns {int id: label}."""
    mapping = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                cls_id = int(row[0].strip())
            except ValueError:
                continue                    # header row, e.g. "id,label"
            mapping[cls_id] = sanitize_class_name(row[1].strip())
    return mapping


def discover_pairs(input_dir, output_dir, label_map):
    input_dir = Path(input_dir)
    videos = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    pairs, unparsed = [], []
    for v in videos:
        m = _FILENAME_RE.match(v.stem)
        if not m:
            unparsed.append(v.name)
            continue
        cls_id, signer_id, rep_id = (int(g) for g in m.groups())
        cls = label_map.get(cls_id, f"{cls_id:03d}")
        pairs.append((v, output_dir / cls / f"{v.stem}.npz"))
    return pairs, unparsed


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks for Argentina SL (LSA64).")
    ap.add_argument("--input", default="data/raw_data/Argentina SL",
                    help="root dir of classID_signerID_rep.mp4 files (flat, no subfolders)")
    ap.add_argument("--output", default="data/processed_landmarks/argentina",
                    help="root dir for .npz landmark files")
    ap.add_argument("--label-map", default=None,
                    help="optional 2-column CSV (id,label) mapping LSA64 numeric class "
                         "IDs to real gloss names; omit to use zero-padded IDs as class names")
    add_common_args(ap)
    args = ap.parse_args()

    label_map = _load_label_map(args.label_map) if args.label_map else {}
    if args.label_map:
        print(f"loaded {len(label_map)} class names from {args.label_map}")
    else:
        print("no --label-map given -- class folders will be zero-padded numeric IDs")

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs, unparsed = discover_pairs(input_dir, output_dir, label_map)
    if unparsed:
        print(f"Warning: {len(unparsed)} files didn't match the "
              f"'classID_signerID_rep.ext' pattern, skipped: {unparsed[:5]}"
              f"{' ...' if len(unparsed) > 5 else ''}")
    if not pairs:
        print(f"Found 0 usable videos under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="LSA64 (Argentina SL) landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
