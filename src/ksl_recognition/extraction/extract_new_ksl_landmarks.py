"""
extract_new_ksl_landmarks.py
------------------------------
New KSL gloss batch ("ksl-new_words") -> MediaPipe Holistic -> per-clip
(coords, vis) arrays, same output format as extract_landmarks.py.

Different from the main KSL dataset (extract_landmarks.py): clips here ship
as pre-extracted RGB FRAME SEQUENCES (folders of .jpg files), not video
files, and every gloss folder is Khmer script or a raw number rather than a
transliterated class name:

    ksl-new_words/
        <gloss>/
            action_1/<take>_frame_<NNNNN>.jpg
            action_2/<take>_frame_<NNNNN>.jpg
            ...
        50/
            action_1/*.jpg
            ...

Class = the gloss folder name, used as-is (Khmer text and bare numbers are
both valid directory names; sanitize_class_name only guards against a
literal '/' inside a gloss). One "clip" = one action_N folder underneath it.

Frame filenames are NOT consistent across this dataset -- two schemes coexist,
sometimes as sibling clip folders under the very same gloss:
  - "<take>_frame_<NNNNN>.jpg"  -- NNNNN is a running counter, monotonic
    across the whole clip folder regardless of <take>.
  - "<take>_<idx>.jpg"          -- idx resets to 0 at the start of every new
    <take> within the folder (e.g. "...,5_42.jpg,5_43.jpg,6_0.jpg,6_1.jpg,...").
Sorting on the trailing number alone (fine for the first scheme) silently
scrambles the second, since a later take's early frames (idx small) would
sort before an earlier take's late frames. Frames are therefore sorted by
(take, trailing index) so the take number always dominates, which orders
both schemes correctly.

This dataset is meant to grow over time as new glosses are recorded --
run_extraction() already skips any clip whose .npz output exists, so
re-running this script after dropping in new gloss folders (or new action_N
takes under an existing gloss) only extracts what's new; pass --overwrite to
force a full re-extraction instead.

Run:
    python -m ksl_recognition.extraction.extract_new_ksl_landmarks \\
        --input  "data/raw_data/ksl-new_words" \\
        --output data/processed_landmarks/new_ksl_gloss \\
        --workers 8
"""
import argparse
import re
from pathlib import Path

from ksl_recognition.extraction.utils import (add_common_args, process_one_from_frames, run_extraction,
                           sanitize_class_name)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Matches both naming schemes found in this dataset: "<take>_frame_<idx>.jpg"
# and "<take>_<idx>.jpg".
_FRAME_RE = re.compile(r"^(\d+)_(?:frame_)?(\d+)$")


def _sorted_frames(clip_dir):
    """Sort a clip's frame files by (take, trailing index) rather than the
    trailing index alone -- see the take-local-vs-global counter note in the
    module docstring for why the index alone isn't safe to sort on. Files
    that don't match either naming scheme sort last, alphabetically."""
    files = [p for p in clip_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    def key(p):
        m = _FRAME_RE.match(p.stem)
        if m:
            take, idx = m.groups()
            return (0, int(take), int(idx), p.stem)
        return (1, 0, 0, p.stem)

    return sorted(files, key=key)


def discover_pairs(input_dir, output_dir):
    """root/<gloss>/action_N/*.jpg -> [(frame_paths, output_dir/<gloss>/action_N.npz), ...]"""
    input_dir = Path(input_dir)
    pairs = []
    for gloss_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        cls = sanitize_class_name(gloss_dir.name)
        for clip_dir in sorted(p for p in gloss_dir.iterdir() if p.is_dir()):
            frames = _sorted_frames(clip_dir)
            if not frames:
                continue
            out_path = output_dir / cls / f"{clip_dir.name}.npz"
            pairs.append((frames, out_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks for the new KSL gloss "
                    "frame-image dataset.")
    ap.add_argument("--input", default="data/raw_data/ksl-new_words",
                    help="root dir of the <gloss>/action_N/*.jpg tree")
    ap.add_argument("--output", default="data/processed_landmarks/new_ksl_gloss",
                    help="root dir for .npz landmark files")
    add_common_args(ap)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    pairs = discover_pairs(input_dir, output_dir)
    if not pairs:
        print(f"Found 0 clips under {input_dir} -- check the --input path.")
        return

    run_extraction(pairs, desc="New KSL gloss landmarks", workers=args.workers,
                   model_complexity=args.model_complexity,
                   min_visibility=args.min_visibility,
                   max_frames=args.max_frames, overwrite=args.overwrite,
                   worker_fn=process_one_from_frames)


if __name__ == "__main__":
    main()
