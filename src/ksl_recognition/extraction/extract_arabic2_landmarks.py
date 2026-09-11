"""
extract_arabic2_landmarks.py
------------------------------
Second Arabic Sign Language dataset ("Arabic SL (2)") -> MediaPipe Holistic ->
per-clip (coords, vis) arrays, same output format as extract_landmarks.py.

Different from every other dataset here: clips are pre-extracted RGB FRAME
SEQUENCES (folders of .jpg files), not video files -- KArSL-style layout:

    Arabic SL (2)/
        01/01/train/0001/01_01_0001_(08_12_16_15_35_44)_c/*.jpg
        01/01/test/0389/03_01_0389_(21_09_17_22_48_20)_c/*.jpg

Class = the 4-digit folder directly under train/ or test/ (e.g. "0001").
No label/gloss file ships with this dataset (checked: zero non-.jpg files
anywhere in the tree) -- class folders are used as-is, NOT translated to
real gloss names, same defensive call as LSA64. Pass --label-map (2-col
CSV: id,label) if you obtain the official KArSL class list.

train/test folders are NOT treated as a split: train has 332 unique class
IDs, test has 503, only 331 overlap. That asymmetry means they don't share
a class set the way Slovo's train/test do, so treating "test" as a held-out
val set would silently evaluate on classes the model never trained on (or
drop 172 classes from training entirely). All clips from both folders are
extracted together as one flat dataset; use stratified_split() at training
time instead.

Run:
    python -m ksl_recognition.extraction.extract_arabic2_landmarks \\
        --input  "data/raw_data/Arabic SL (2)" \\
        --output data/processed_landmarks/arabic2 \\
        --workers 8
"""
import argparse
import csv
from pathlib import Path

from ksl_recognition.extraction.utils import add_common_args, process_one_from_frames, run_extraction, sanitize_class_name

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _sorted_frames(clip_dir):
    """Sort a clip's frame files by trailing _NNNN frame index; falls back to
    plain name sort if a file doesn't match that pattern."""
    files = [p for p in clip_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    def key(p):
        tail = p.stem.rsplit("_", 1)[-1]
        return (int(tail), p.stem) if tail.isdigit() else (10**9, p.stem)

    return sorted(files, key=key)


def _load_label_map(path):
    """2-column CSV: id,label (header row optional). Returns {str id: label}."""
    mapping = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip().isdigit():
                continue                    # skip header / malformed rows
            mapping[row[0].strip()] = sanitize_class_name(row[1].strip())
    return mapping


def discover_pairs(input_dir, output_dir, label_map=None):
    """Class = the folder directly under any train/ or test/ dir found
    anywhere in the tree. One "clip" = one frame-folder below that."""
    label_map = label_map or {}
    input_dir = Path(input_dir)
    pairs = []
    for split_dir in sorted(p for p in input_dir.rglob("*") if p.is_dir()):
        if split_dir.name not in ("train", "test"):
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls_id = cls_dir.name
            cls = label_map.get(cls_id, sanitize_class_name(cls_id))
            for clip_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
                frames = _sorted_frames(clip_dir)
                if not frames:
                    continue
                out_path = output_dir / cls / f"{clip_dir.name}.npz"
                pairs.append((frames, out_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks for Arabic SL (2) (KArSL-style frame folders).")
    ap.add_argument("--input", default="data/raw_data/Arabic SL (2)",
                    help="root dir of the nested .../train|test/<class>/<clip>/*.jpg tree")
    ap.add_argument("--output", default="data/processed_landmarks/arabic2",
                    help="root dir for .npz landmark files")
    ap.add_argument("--label-map", default=None,
                    help="optional 2-column CSV (id,label) mapping the 4-digit class "
                         "IDs to real gloss names; omit to use the IDs as class names")
    add_common_args(ap)
    args = ap.parse_args()

    label_map = _load_label_map(args.label_map) if args.label_map else {}
    if args.label_map:
        print(f"loaded {len(label_map)} class names from {args.label_map}")
    else:
        print("no --label-map given -- class folders will be the raw 4-digit IDs")

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs = discover_pairs(input_dir, output_dir, label_map)
    if not pairs:
        print(f"Found 0 clips under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="Arabic SL (2) landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite,
                   worker_fn=process_one_from_frames)


if __name__ == "__main__":
    main()
