#!/usr/bin/env python3
"""Copy processed_landmarks into a Kaggle-ready tree with plain
zero-padded numeric names for both class folders (0000, 0001, ...) and
the .npz clip files inside them (0000.npz, 0001.npz, ...).

Non-destructive: reads from data/processed_landmarks/<language>/... and
COPIES into data/processed_landmarks_kaggle/<language>/... -- the
originals are never touched. Numeric IDs sidestep transliteration
entirely (Khmer/ksl class names, Cyrillic/slovo clip UUIDs, spaces and
parens in indian/srilanka filenames, Khmer characters in ksl filenames,
...), so this works uniformly for every language.

Writes two map files per language for traceability:
  - `class_index_map.json`: {class_id: original_class_folder_name}
  - `clip_index_map.json`:  {class_id: {clip_id: original_filename}}

Usage:
    python -m ksl_recognition.export.rename_class_folders                # all languages
    python -m ksl_recognition.export.rename_class_folders --dry-run       # preview only
    python -m ksl_recognition.export.rename_class_folders --languages ksl vietnam slovo
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3] / "data" / "processed_landmarks"
DST_ROOT = Path(__file__).resolve().parents[3] / "data" / "processed_landmarks_kaggle"

# slovo ships pre-split (root/{train,test}/<class>); everything else is flat
# (root/<class>).
PRESPLIT_LANGUAGES = {"slovo"}


def numeric_id(i: int, count: int) -> str:
    width = max(4, len(str(max(count - 1, 0))))
    return str(i).zfill(width)


def copy_class_dir(src: Path, target: Path, dry_run: bool) -> dict[str, str]:
    """Copy one class folder's .npz clips into `target`, renamed 0000.npz, ...

    Returns {clip_id: original_filename}.
    """
    clips = sorted(f for f in src.iterdir() if f.is_file() and f.suffix == ".npz")
    clip_map: dict[str, str] = {}
    for i, f in enumerate(clips):
        clip_id = numeric_id(i, len(clips))
        clip_map[clip_id] = f.name
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target / f"{clip_id}.npz")
    return clip_map


def copy_flat(src: Path, dst: Path, dry_run: bool) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    class_map: dict[str, str] = {}
    clip_maps: dict[str, dict[str, str]] = {}
    for i, d in enumerate(class_dirs):
        cid = numeric_id(i, len(class_dirs))
        class_map[cid] = d.name
        target = dst / cid
        action = "[dry-run] would copy" if dry_run else "copying"
        clip_map = copy_class_dir(d, target, dry_run)
        print(f"  {action}: {d.name!r} -> {cid} ({len(clip_map)} clips)")
        clip_maps[cid] = clip_map
    return class_map, clip_maps


def copy_presplit(src: Path, dst: Path, dry_run: bool) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    # Assign one numeric ID per class name, shared across splits, so the
    # same class gets the same ID in train/ and test/.
    splits = sorted(d for d in src.iterdir() if d.is_dir())
    class_names = sorted({d.name for split in splits for d in split.iterdir() if d.is_dir()})
    id_by_name = {name: numeric_id(i, len(class_names)) for i, name in enumerate(class_names)}
    clip_maps: dict[str, dict[str, str]] = {}

    for split in splits:
        for d in sorted(x for x in split.iterdir() if x.is_dir()):
            cid = id_by_name[d.name]
            target = dst / split.name / cid
            action = "[dry-run] would copy" if dry_run else "copying"
            clip_map = copy_class_dir(d, target, dry_run)
            print(f"  {action}: {split.name}/{d.name!r} -> {split.name}/{cid} ({len(clip_map)} clips)")
            clip_maps.setdefault(cid, {}).update(clip_map)

    class_map = {cid: name for name, cid in id_by_name.items()}
    return class_map, clip_maps


def process_language(lang_src: Path, dry_run: bool) -> None:
    print(f"[{lang_src.name}]")
    lang_dst = DST_ROOT / lang_src.name
    if lang_dst.exists() and any(lang_dst.iterdir()):
        print(f"  skipping: {lang_dst} already exists and is non-empty")
        return

    if lang_src.name in PRESPLIT_LANGUAGES:
        class_map, clip_maps = copy_presplit(lang_src, lang_dst, dry_run)
    else:
        class_map, clip_maps = copy_flat(lang_src, lang_dst, dry_run)

    if dry_run:
        return

    (lang_dst / "class_index_map.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2, sort_keys=True)
    )
    (lang_dst / "clip_index_map.json").write_text(
        json.dumps(clip_maps, ensure_ascii=False, indent=2, sort_keys=True)
    )
    n_clips = sum(len(v) for v in clip_maps.values())
    print(f"  wrote class_index_map.json ({len(class_map)} classes), clip_index_map.json ({n_clips} clips)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Subset of language folder names under data/processed_landmarks (default: all discovered)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview copies without touching disk")
    args = parser.parse_args()

    if args.languages:
        lang_dirs = [SRC_ROOT / name for name in args.languages]
        for d in lang_dirs:
            if not d.is_dir():
                raise SystemExit(f"not a directory: {d}")
    else:
        lang_dirs = sorted(d for d in SRC_ROOT.iterdir() if d.is_dir())

    if not args.dry_run:
        DST_ROOT.mkdir(exist_ok=True)

    for lang_dir in lang_dirs:
        process_language(lang_dir, args.dry_run)


if __name__ == "__main__":
    main()
