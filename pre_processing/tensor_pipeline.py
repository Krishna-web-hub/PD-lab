"""Load one batch and show what the tensor conversion produced.

Changes from the original: the hardcoded `D:\\PD_LAB\\...` path is now
`--data-dir`, and `--save` writes the figure to a file so the script also runs
on a machine with no display.

    python tensor_pipeline.py
    python tensor_pipeline.py --save batch.png
"""

import argparse
from pathlib import Path

import matplotlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(REPO / "dataset/train/spoilage_detection"))
    parser.add_argument("--batch-size", type=int, default=16)
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

    print("Classes:", dataset.classes)
    print("Total images:", len(dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    images, labels = next(iter(loader))

    print("Tensor shape:", tuple(images.shape))
    print("Value range :", f"[{images.min():.3f}, {images.max():.3f}]")
    print("Labels      :", labels.tolist())

    plt.imshow(images[0].permute(1, 2, 0))  # CHW -> HWC for display
    plt.title(f"Example tensor image: {dataset.classes[labels[0]]}")
    plt.axis("off")

    if args.save:
        plt.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"\nsaved {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
