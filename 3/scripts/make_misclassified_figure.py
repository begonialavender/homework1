"""单独生成误分类样本图（可指定张数，便于贴进页数受限的报告）。

用法：
    python scripts/make_misclassified_figure.py --ckpt runs/baseline/best.pt --cases 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.evaluate import collect_misclassified
from src.train import load_checkpoint
from src.visualize import configure_matplotlib, plot_misclassified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/baseline/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    wrong = collect_misclassified(model, data_root=args.data)

    configure_matplotlib()
    out = Path(args.ckpt).parent / (args.name or f"misclassified_{args.cases}.png")
    plot_misclassified(wrong, out, n=args.cases,
                       title=f"{ckpt['model']} 误分类样本（按置信度排序）")


if __name__ == "__main__":
    main()
