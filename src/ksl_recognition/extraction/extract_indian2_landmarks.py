"""
extract_indian2_landmarks.py
------------------------------
Indian Sign Language (source 2, "Indian SL (2)") -> MediaPipe Holistic ->
per-clip (coords, vis) arrays in the same format as extract_landmarks.py
(KSL pipeline).

Shared extraction core lives in extraction/utils.py -- this file only implements
this dataset's layout: nested, numbered category folders.

This is a DIFFERENT download from "India SL" (see extract_india_landmarks.py)
and has a nested, numbered layout instead of a flat one:

    Indian SL (2)/
        Adjectives_1of8/
            Adjectives/
                4. sad/*.MOV
                2. quiet/*.MOV
                ...
        Animals_1of2/
            Animals/
                ...
        README.md            <- not a class, ignored (no video extension)
        download_data.sh     <- not a class, ignored

The "N. " numeric prefix on each leaf folder (ordering artifact from the
source download, not part of the gloss) is stripped to get the class name --
"4. sad" -> "sad" (extraction.utils.strip_numeric_prefix). The class is always
the video file's IMMEDIATE parent folder, so this works regardless of how
deep the "_XofY" / category nesting goes above it.

Output filenames encode the full source path (category/part/original stem),
since generic camera filenames like MVI_9568.MOV can repeat across different
leaf folders and would otherwise collide once regrouped by class
(extraction.utils.unique_relative_stem).

Run:
    python -m ksl_recognition.extraction.extract_indian2_landmarks \\
        --input  "data/raw_data/Indian SL (2)" \\
        --output data/processed_landmarks/indian2 \\
        --workers 8
"""
import argparse
from pathlib import Path

from ksl_recognition.extraction.utils import (VIDEO_EXTS, add_common_args, run_extraction,
                           strip_numeric_prefix, unique_relative_stem)


def discover_pairs(input_dir, output_dir):
    input_dir = Path(input_dir)
    videos = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    pairs = []
    for v in videos:
        cls = strip_numeric_prefix(v.parent.name)
        out_path = output_dir / cls / f"{unique_relative_stem(v, input_dir)}.npz"
        pairs.append((v, out_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks for Indian SL (2) (nested layout).")
    ap.add_argument("--input", default="data/raw_data/Indian SL (2)",
                    help="root dir of the nested Category_XofY/Category/N. class/*.mov tree")
    ap.add_argument("--output", default="data/processed_landmarks/indian2",
                    help="root dir for .npz landmark files")
    add_common_args(ap)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs = discover_pairs(input_dir, output_dir)
    if not pairs:
        print(f"Found 0 videos under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="Indian SL (2) landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
