"""
extract_wlasl_landmarks.py
---------------------------
WLASL (Word-Level American Sign Language) -> MediaPipe Holistic -> per-clip
(coords, vis) arrays in the same format as extract_landmarks.py (KSL pipeline).

Shared extraction core lives in extraction/utils.py -- this file only implements
WLASL's dataset layout: a flat folder per gloss word.

    WLASL/
        a/*.mp4
        about/*.mp4
        accident/*.mp4
        ...
    (~2000 classes, ~12.5k clips)

Run:
    python -m ksl_recognition.extraction.extract_wlasl_landmarks \\
        --input  data/raw_data/WLASL \\
        --output data/processed_landmarks/wlasl \\
        --workers 8
"""
import argparse
from pathlib import Path

from ksl_recognition.extraction.utils import VIDEO_EXTS, add_common_args, run_extraction


def discover_pairs(input_dir, output_dir):
    """root/<class>/<clip>.ext -> [(video_path, output_dir/<class>/<clip>.npz), ...]"""
    input_dir = Path(input_dir)
    videos = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return [(v, output_dir / v.relative_to(input_dir).with_suffix(".npz"))
            for v in videos]


def main():
    ap = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks for WLASL (ASL).")
    ap.add_argument("--input", default="data/raw_data/WLASL",
                    help="root dir of videos (one subfolder per gloss word)")
    ap.add_argument("--output", default="data/processed_landmarks/wlasl",
                    help="root dir for .npz landmark files")
    add_common_args(ap)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs = discover_pairs(input_dir, output_dir)
    if not pairs:
        print(f"Found 0 videos under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="WLASL landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
