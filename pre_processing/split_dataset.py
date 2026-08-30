"""Split each task's images into train / val / test.

Rewritten to fix four defects in the original:

1. **No deduplication.** The image collection contains byte-identical copies:
   the `name (1).jpg` / `name (2).jpg` pattern that downloads and file managers
   produce. Splitting at random scattered those copies across splits, so the
   same photograph appeared in both train and test. That is data leakage: it
   flatters every held-out metric without the model having learned anything.
   Files are now grouped by SHA-256 of their contents, and a group is assigned
   to exactly one split.

2. **No seed.** `random.shuffle` was unseeded, so the split was different on
   every run and could never be reproduced. Now seeded and configurable.

3. **Not idempotent.** The task loop read every directory under `base`, which
   includes `train/`, `val/` and `test/` once they exist, so a second run would
   have treated `train/` as a task and produced `train/train/...`. The split
   directories are now excluded by name, and the script refuses to overwrite
   existing splits unless `--force` is passed.

4. **Hardcoded Windows path.** `D:\\PD_LAB\\dataset` meant the script only ran on
   one machine. The path is now an argument, defaulting to the repo layout.

Usage:
    python split_dataset.py --dry-run        # report only, touch nothing
    python split_dataset.py                  # write the splits
    python split_dataset.py --seed 7 --force # reproducible re-split
"""

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)

# Extensions torchvision's ImageFolder will actually load. Anything else is
# copied but silently ignored at training time, so it is worth reporting.
LOADABLE = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def file_hash(path: Path, chunk_size: int = 1 << 16) -> str:
    """SHA-256 of the file's bytes. Identical content gives an identical key."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sizes(n: int) -> tuple[int, int, int]:
    """Largest-remainder allocation, so the three parts always sum to n."""
    exact = [n * r for r in RATIOS]
    counts = [int(x) for x in exact]
    for i in sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True):
        if sum(counts) < n:
            counts[i] += 1
    return tuple(counts)


def split_class(class_dir: Path, rng: random.Random):
    """Group a class's files by content, shuffle the groups, and allocate them.

    Returns (assignment, stats). Duplicate groups move as a unit, which is what
    keeps a photograph out of two splits at once.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    unloadable: list[Path] = []

    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in LOADABLE:
            unloadable.append(path)
        groups[file_hash(path)].append(path)

    keys = sorted(groups)
    rng.shuffle(keys)

    n_train, n_val, _ = split_sizes(len(keys))
    assignment = {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }

    stats = {
        "files": sum(len(v) for v in groups.values()),
        "unique": len(groups),
        "duplicate_files": sum(len(v) - 1 for v in groups.values() if len(v) > 1),
        "duplicate_groups": sum(1 for v in groups.values() if len(v) > 1),
        "unloadable": unloadable,
    }
    return assignment, groups, stats


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(repo_root / "dataset"),
                        help="dataset root holding the per-task folders")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite existing splits")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        print(f"error: no such directory: {base}")
        return 2

    existing = [s for s in SPLITS if (base / s).exists()]
    if existing and not (args.force or args.dry_run):
        print(f"error: {', '.join(existing)} already exist under {base}.")
        print("Re-splitting invalidates any model trained on the current split and every")
        print("metric reported from it. Pass --force if that is what you intend, or")
        print("--dry-run to see what would change.")
        return 1

    tasks = sorted(
        d for d in base.iterdir() if d.is_dir() and d.name not in SPLITS
    )
    if not tasks:
        print(f"error: no task directories under {base}")
        return 2

    rng = random.Random(args.seed)
    print(f"base={base}  seed={args.seed}"
          f"{'  [DRY RUN: nothing will be written]' if args.dry_run else ''}\n")

    for task in tasks:
        print(f"{task.name}/")
        for class_dir in sorted(d for d in task.iterdir() if d.is_dir()):
            assignment, groups, stats = split_class(class_dir, rng)

            if stats["files"] == 0:
                print(f"  {class_dir.name:<14} empty, skipped")
                continue

            counts = {s: sum(len(groups[k]) for k in keys) for s, keys in assignment.items()}
            print(f"  {class_dir.name:<14} {stats['files']:>4} files, "
                  f"{stats['unique']:>4} unique  ->  "
                  f"train {counts['train']:>4}  val {counts['val']:>3}  test {counts['test']:>3}")

            if stats["duplicate_files"]:
                print(f"    deduplicated: {stats['duplicate_files']} duplicate file(s) in "
                      f"{stats['duplicate_groups']} group(s), kept with their originals")
            for path in stats["unloadable"]:
                print(f"    warning: {path.name} has an extension ImageFolder cannot load; "
                      f"it will be silently skipped during training")

            if args.dry_run:
                continue

            for split, keys in assignment.items():
                dest = base / split / task.name / class_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for key in keys:
                    for src in groups[key]:
                        shutil.copy2(src, dest / src.name)
        print()

    print(
        "Dry run complete. No files written."
        if args.dry_run
        else "Dataset splitting completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
