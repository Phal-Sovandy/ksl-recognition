"""
extract_landmarks.py
--------------------
Video -> MediaPipe Holistic -> per-clip (coords, vis) arrays for the KSL pipeline.

Shared extraction core (MediaPipe processing, worker pool, progress bar,
summary) lives in extraction/utils.py -- this file only implements the one
thing specific to KSL's dataset layout: how to find (video_path, out_path)
pairs.

Output per clip (what features.assemble_features(coords, vis) expects):
    coords : float32  (T, 51, 2)   pixel-space x, y
    vis    : float32  (T, 51)      per-node visibility; vis > 0 means "trust this node"

Node layout (MUST match features.py / the adjacency in sign_model.py):
    [ 0 ..  8]  upper-body pose (9)   [ 9 .. 29]  left hand (21)   [30 .. 50]  right hand (21)

KSL layout (flat, one folder per class):
    ksl_dataset/
        <class>/*.mp4

Run:
    python -m ksl_recognition.extraction.extract_landmarks --input data/raw_data/ksl_dataset --output data/processed_landmarks/ksl --workers 8
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
    ap = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks for KSL.")
    ap.add_argument("--input", required=True, help="root dir of videos (class subfolders)")
    ap.add_argument("--output", required=True, help="root dir for .npz landmark files")
    add_common_args(ap)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs = discover_pairs(input_dir, output_dir)
    if not pairs:
        print(f"Found 0 videos under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="KSL landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
