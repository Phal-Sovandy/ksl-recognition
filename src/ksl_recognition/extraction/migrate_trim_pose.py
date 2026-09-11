"""
migrate_trim_pose.py
---------------------
One-time (but idempotent/re-runnable) migration for .npz landmark files that
were extracted BEFORE pose was trimmed to upper-body-only (see
extraction.utils.UPPER_BODY_POSE_IDX). Old files store the full MediaPipe
75-node layout (pose 33 + hand 21 + hand 21); this rewrites them in place to
the new 51-node layout (pose 9 + hand 21 + hand 21) by slicing out the
already-computed leg/foot columns -- no MediaPipe re-run needed, since the
kept landmarks were already extracted, just at different array positions.

Old layout : [ 0..32] pose(33)  [33..53] left hand(21)  [54..74] right hand(21)
New layout : [ 0.. 8] pose(9)   [ 9..29] left hand(21)  [30..50] right hand(21)
Kept old pose indices, in order: 0 (nose), 11/12 (shoulders), 13/14 (elbows),
15/16 (wrists), 23/24 (hips) -- must match extraction.utils.UPPER_BODY_POSE_IDX.

Safe to re-run: any file whose coords are already 51-wide is left untouched.
Writes to a temp file and replaces atomically so a crash mid-run can't leave
a truncated .npz behind.

Run:
    python -m ksl_recognition.extraction.migrate_trim_pose --root data/processed_landmarks
    python -m ksl_recognition.extraction.migrate_trim_pose --root data/processed_landmarks --dry-run
"""
import argparse
import os
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

OLD_POSE = 33
NEW_POSE_OLD_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]   # must match extraction.utils
HAND = 21
OLD_NUM_NODES = OLD_POSE + 2 * HAND    # 75
NEW_NUM_NODES = len(NEW_POSE_OLD_IDX) + 2 * HAND  # 51

# Old-layout index list to keep -> gives the new 51-node layout directly,
# since the old hand blocks (33..53, 54..74) are already contiguous and in
# the right relative order.
KEEP_IDX = NEW_POSE_OLD_IDX + list(range(OLD_POSE, OLD_POSE + HAND)) + \
           list(range(OLD_POSE + HAND, OLD_POSE + 2 * HAND))


def migrate_file(path: Path) -> str:
    """Returns 'migrated', 'skipped' (already new layout), or 'unexpected'."""
    with np.load(path) as d:
        coords, vis = d["coords"], d["vis"]

    if coords.shape[1] == NEW_NUM_NODES:
        return "skipped"
    if coords.shape[1] != OLD_NUM_NODES:
        return f"unexpected shape {coords.shape}"

    new_coords = coords[:, KEEP_IDX, :]
    new_vis = vis[:, KEEP_IDX]

    tmp_path = path.with_suffix(".npz.tmp")
    np.savez_compressed(tmp_path, coords=new_coords, vis=new_vis)
    os.replace(tmp_path, path)   # atomic on the same filesystem
    return "migrated"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="root dir to walk for .npz files")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and report counts without writing anything")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.npz"))
    print(f"found {len(files)} .npz files under {root}")

    counts = {"migrated": 0, "skipped": 0, "error": 0}
    errors = []
    for path in tqdm(files, desc="migrating"):
        try:
            if args.dry_run:
                with np.load(path) as d:
                    shape1 = d["coords"].shape[1]
                if shape1 == NEW_NUM_NODES:
                    counts["skipped"] += 1
                elif shape1 == OLD_NUM_NODES:
                    counts["migrated"] += 1   # "would migrate"
                else:
                    counts["error"] += 1
                    errors.append((path, f"unexpected shape[1]={shape1}"))
                continue

            result = migrate_file(path)
            if result == "migrated":
                counts["migrated"] += 1
            elif result == "skipped":
                counts["skipped"] += 1
            else:
                counts["error"] += 1
                errors.append((path, result))
        except Exception as e:
            counts["error"] += 1
            errors.append((path, str(e)))

    verb = "would migrate" if args.dry_run else "migrated"
    print(f"\n{verb}: {counts['migrated']}   already new layout: {counts['skipped']}   "
          f"errors: {counts['error']}")
    if errors:
        print("\nfirst errors:")
        for path, msg in errors[:20]:
            print(f"  {path}: {msg}")


if __name__ == "__main__":
    main()
