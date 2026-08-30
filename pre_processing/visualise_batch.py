"""Plot a grid of images with their labels, to verify labels match pictures.

Changes from the original: the hardcoded `D:\\PD_LAB\\...` path is now
`--data-dir`, and `--save` writes the figure to a file so the script also runs
on a machine with no display.

    python visualise_batch.py
    python visualise_batch.py --data-dir ../dataset/val/spoilage_detection --save grid.png
"""

import argparse
import math
from pathlib import Path

import matplotlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(REPO / "dataset/train/spoilage_detection"))
    parser.add_argument("--count", type=int, default=9, help="images to plot")
    parser.add_argument("--save", default=None, help="write the figure here instead of showing it")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: no such directory: {data_dir}")
        return 2

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)

    count = min(args.count, len(dataset))
    loader = DataLoader(dataset, batch_size=count, shuffle=True)
    images, labels = next(iter(loader))
    class_names = dataset.classes

    side = math.ceil(math.sqrt(count))
    plt.figure(figsize=(8, 8))
    for i in range(count):
        plt.subplot(side, side, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
