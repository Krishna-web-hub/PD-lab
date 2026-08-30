"""Copy images from Images/ into per-task, per-class folders.

Two changes from the original:

* The hardcoded `D:\\PD_LAB` base is now `--base`, defaulting to this repo.
* **The AI-generated images were silently dropped.** The original matched with
  `if "AI_gen_food" in root`, but the folder on disk is named `AI_gen _food`,
  with a space before `_food`. The test never matched, all 340 AI-generated
  images were skipped, and the script still printed "completed". Source folders
  are now matched on a normalised key (lowercased, non-alphanumerics removed),
  so spacing and punctuation cannot break the mapping, and any folder that
  matches nothing is reported instead of ignored.

A per-destination summary is printed at the end, so a destination receiving zero
images is visible rather than silent.

    python organise_dataset.py --dry-run
    python organise_dataset.py
"""

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# normalised source folder -> [(task, class), ...]
ROUTES = {
    "aigenfood": [("ai_detection", "ai_generated")],
    "goodfood": [("ai_detection", "real_food"), ("spoilage_detection", "fresh")],
    "badfood": [("spoilage_detection", "spoiled")],
}


def normalise(name: str) -> str:
    """'AI_gen _food' and 'AI_gen_food' both become 'aigenfood'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(REPO), help="repo root holding Images/ and dataset/")
    parser.add_argument("--source", default=None, help="override the Images/ folder")
    parser.add_argument("--dry-run", action="store_true", help="report only; copy nothing")
    args = parser.parse_args()

    base = Path(args.base)
    source = Path(args.source) if args.source else base / "Images"
    if not source.is_dir():
        print(f"error: no such directory: {source}")
        return 2

    written: Counter[str] = Counter()
    unmatched: list[str] = []

    for class_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        routes = ROUTES.get(normalise(class_dir.name))
        files = [p for p in sorted(class_dir.iterdir()) if p.is_file()]

        if routes is None:
            unmatched.append(f"{class_dir.name} ({len(files)} files)")
            continue

        for task, label in routes:
            dest = base / "dataset" / task / label
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                for src in files:
                    shutil.copy2(src, dest / src.name)
            written[f"{task}/{label}"] += len(files)
        print(f"{class_dir.name:<16} {len(files):>4} files -> "
              + ", ".join(f"{t}/{c}" for t, c in routes))

    print("\nper-destination totals:")
    for task, label in sorted({r for rs in ROUTES.values() for r in rs}):
        key = f"{task}/{label}"
        count = written[key]
        flag = "   <-- NOTHING MATCHED THIS DESTINATION" if count == 0 else ""
        print(f"  {key:<32} {count:>5}{flag}")

    if unmatched:
        print("\nsource folders that matched no route (nothing was copied from them):")
        for name in unmatched:
            print(f"  {name}")

    print("\nDry run complete. No files copied." if args.dry_run
          else "\nDataset organization completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
