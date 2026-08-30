"""Check every image decodes, and optionally write a resized copy.

Two changes from the original:

* The hardcoded `D:\\...\\Images` path is now `--source`, defaulting to the
  `Images/` folder in this repo.
* The original only *reported* corrupted files despite the README describing a
  resize step. Pass `--out` to actually write the resized copies; without it the
  script reports and writes nothing, which is what it did before.

Uses Pillow rather than OpenCV. Pillow arrives with torchvision, so this runs in
the same environment as the rest of the project; OpenCV was an extra dependency
that is not installed here.

    python cleaning.py                      # report corrupted images only
    python cleaning.py --out ../processed_images
"""

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO = Path(__file__).resolve().parents[1]
SIZE = (224, 224)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(REPO / "Images"),
                        help="folder of class subdirectories to check")
    parser.add_argument("--out", default=None,
                        help="if given, write resized copies here (mirrors the class folders)")
    parser.add_argument("--size", type=int, default=SIZE[0], help="output edge length")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"error: no such directory: {source}")
        return 2

    out = Path(args.out) if args.out else None
    if out and out.exists() and any(out.iterdir()):
        print(f"error: {out} already exists and is not empty. Refusing to overwrite.")
        print("Delete it or choose another --out if you mean to regenerate it.")
        return 1

    checked = corrupted = written = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        checked += 1
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                if out:
                    dest = out / path.parent.relative_to(source)
                    dest.mkdir(parents=True, exist_ok=True)
                    rgb.resize((args.size, args.size), Image.LANCZOS).save(
                        dest / f"{path.stem}.jpg", quality=95
                    )
                    written += 1
        except (UnidentifiedImageError, OSError) as exc:
            corrupted += 1
            print(f"Corrupted: {path}  ({exc})")

    print(f"\nchecked {checked} file(s), {corrupted} corrupted, {written} written")
    if not out:
        print("(no --out given, so nothing was written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
