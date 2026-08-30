"""Quick check that a folder loads as an ImageFolder dataset.

The hardcoded `D:\\PD_LAB\\processed_images` path is now `--data-dir`, defaulting
to the `processed_images/` folder in this repo.

Also reports the per-class counts and any files ImageFolder skipped, since a
silently ignored file is the failure mode this check exists to catch.

    python testing.py
    python testing.py --data-dir ../dataset/train/spoilage_detection
"""

import argparse
from collections import Counter
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(REPO / "processed_images"),
                        help="folder of class subdirectories")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: no such directory: {data_dir}")
        return 2

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(str(data_dir), transform=transform)

    print(f"Directory   : {data_dir}")
    print(f"Classes     : {dataset.classes}")
    print(f"Total images: {len(dataset)}")

    counts = Counter(dataset.classes[t] for t in dataset.targets)
    for name in dataset.classes:
        print(f"  {name:<16} {counts[name]:>5}")

    # Files on disk that ImageFolder did not pick up, usually an extension it
    # does not recognise (.avif, .heic), which it skips without any warning.
    on_disk = {p for p in data_dir.rglob("*") if p.is_file()}
    loaded = {Path(p) for p, _ in dataset.samples}
    skipped = sorted(on_disk - loaded)
    if skipped:
        print(f"\n{len(skipped)} file(s) on disk were NOT loaded:")
        for p in skipped:
            print(f"  {p.relative_to(data_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
