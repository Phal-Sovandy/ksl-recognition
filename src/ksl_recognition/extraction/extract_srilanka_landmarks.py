"""
extract_srilanka_landmarks.py
-------------------------------
Sri Lankan Sign Language -> MediaPipe Holistic -> per-clip (coords, vis)
arrays in the same format as extract_landmarks.py (KSL pipeline).

Shared extraction core lives in extraction/utils.py -- this file only implements
this dataset's layout: nested part-of-speech category folders.

    Sri Lanka/
        Greetings/
            Alright/*.MOV
            Hello/*.MOV
        Numbers/
            7. seven/*.MOV
            Subtraction/*.mov
        Nouns/
            Cat/*.MOV
            Tree/*.MOV
        ...

The category folder (Greetings, Numbers, Nouns, ...) is discarded -- only the
video's IMMEDIATE parent folder becomes the class, with any "N. " numeric
ordering prefix stripped ("7. seven" -> "seven", extraction.utils.strip_numeric_prefix).
This works regardless of nesting depth, and Numbers/Subtraction (no numeric
prefix) passes through unchanged.

Output filenames encode the full source path (category/class/original stem),
so identically-numbered clips like "..._001.mov" that recur under every
class can't collide once regrouped by class name
(extraction.utils.unique_relative_stem).

Run:
    python -m ksl_recognition.extraction.extract_srilanka_landmarks \\
        --input  "data/raw_data/Sri Lanka" \\
        --output data/processed_landmarks/srilanka \\
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
        description="Extract MediaPipe Holistic landmarks for Sri Lankan SL (nested layout).")
    ap.add_argument("--input", default="data/raw_data/Sri Lanka",
                    help="root dir of the nested Category/[N. ]class/*.ext tree")
    ap.add_argument("--output", default="data/processed_landmarks/srilanka",
                    help="root dir for .npz landmark files")
    add_common_args(ap)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs = discover_pairs(input_dir, output_dir)
    if not pairs:
        print(f"Found 0 videos under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="Sri Lanka SL landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
