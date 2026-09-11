#!/usr/bin/env python3
"""
encode_ksl_classes.py
----------------------
Copy the processed KSL landmark dataset into a new tree with every class
(gloss) folder renamed to an opaque ID -- "sign_001", "sign_002", ... --
instead of the Khmer gloss text, and write an Excel file mapping each ID
back to its original gloss.

Non-destructive, like ksl_recognition.export.rename_class_folders: reads from
data/processed_landmarks/ksl/<gloss>/*.npz and COPIES into
data/encoded_ksl/<sign_id>/*.npz -- the original dataset is never touched
or modified. Clip filenames inside each class folder are copied as-is
(only the class/gloss name is encoded); .npz contents are untouched.

Output:
    data/encoded_ksl/
        sign_001/*.npz
        sign_002/*.npz
        ...
        sign_id_map.xlsx    <- columns: sign_id, gloss, num_clips

Run:
    python -m ksl_recognition.export.encode_ksl_classes
    python -m ksl_recognition.export.encode_ksl_classes --dry-run
    python -m ksl_recognition.export.encode_ksl_classes --input data/processed_landmarks/ksl \\
        --output data/encoded_ksl --overwrite
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "data" / "processed_landmarks" / "ksl"
DEFAULT_DST = REPO_ROOT / "data" / "encoded_ksl"


def sign_id(i: int, count: int) -> str:
    """0 -> 'sign_001', ...  Width is at least 3 digits (matches the
    "sign_001, sign_002, ..." naming) but grows automatically past 999
    classes instead of colliding (sign_1000, not wrapping back to sign_000)."""
    width = max(3, len(str(max(count - 1, 0) + 1)))
    return f"sign_{i + 1:0{width}d}"


def discover_classes(src: Path) -> list[Path]:
    classes = sorted(d for d in src.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subfolders under {src}")
    return classes


def copy_class(src_dir: Path, dst_dir: Path, dry_run: bool) -> int:
    """Copy every .npz clip from src_dir into dst_dir, filenames unchanged.
    Returns the number of clips copied (or that would be, on --dry-run)."""
    clips = sorted(f for f in src_dir.iterdir() if f.is_file() and f.suffix == ".npz")
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in clips:
            shutil.copy2(f, dst_dir / f.name)
    return len(clips)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Encode KSL gloss class folders as opaque sign_NNN IDs "
                    "(non-destructive copy) + an Excel sign_id<->gloss map.")
    ap.add_argument("--input", default=str(DEFAULT_SRC),
                    help="root dir of the processed KSL landmarks (<gloss>/*.npz)")
    ap.add_argument("--output", default=str(DEFAULT_DST),
                    help="destination root for the encoded copy + sign_id_map.xlsx")
    ap.add_argument("--overwrite", action="store_true",
                    help="wipe an existing non-empty --output dir first "
                         "(the SOURCE dataset is never touched either way)")
    ap.add_argument("--dry-run", action="store_true",
                    help="preview the id assignment without writing/copying anything")
    args = ap.parse_args()

    src, dst = Path(args.input), Path(args.output)
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")

    if dst.exists() and any(dst.iterdir()):
        if args.dry_run:
            pass
        elif args.overwrite:
            print(f"--overwrite: removing existing {dst}")
            shutil.rmtree(dst)
        else:
            raise SystemExit(
                f"{dst} already exists and is non-empty -- pass --overwrite to "
                f"replace it, or --output to pick a different destination.")

    classes = discover_classes(src)
    print(f"{len(classes)} gloss classes found under {src}")

    rows = []
    for i, class_dir in enumerate(classes):
        sid = sign_id(i, len(classes))
        n_clips = copy_class(class_dir, dst / sid, args.dry_run)
        rows.append({"sign_id": sid, "gloss": class_dir.name, "num_clips": n_clips})
        action = "would copy" if args.dry_run else "copied"
        print(f"  {action}: {class_dir.name!r} -> {sid}  ({n_clips} clips)")

    map_df = pd.DataFrame(rows)
    if args.dry_run:
        print(f"\n[dry-run] would write {len(map_df)}-row map -- nothing written to disk")
        return

    xlsx_path = dst / "sign_id_map.xlsx"
    map_df.to_excel(xlsx_path, index=False)
    total_clips = map_df["num_clips"].sum()
    print(f"\nDone. {len(classes)} classes, {total_clips} clips copied -> {dst}")
    print(f"sign_id <-> gloss map written -> {xlsx_path}")


if __name__ == "__main__":
    main()
