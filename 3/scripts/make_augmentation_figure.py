"""单独生成数据增强示例图（训练时也会自动生成，这里用于调整版式后重绘）。

用法：python scripts/make_augmentation_figure.py --run runs/baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import transforms as T

from src.data import CIFAR10
from src.visualize import configure_matplotlib, plot_augmentation_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/baseline")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    configure_matplotlib()
    dataset = CIFAR10(root=args.data, train=True, download=True, transform=T.ToTensor())
    plot_augmentation_examples(dataset, Path(args.run) / "augmentation_examples.png")


if __name__ == "__main__":
    main()
