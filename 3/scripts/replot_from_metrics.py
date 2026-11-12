"""从已保存的 metrics.json 重绘混淆矩阵与每类准确率，不必重新跑评估。

用法：python scripts/replot_from_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualize import configure_matplotlib, plot_class_accuracy, plot_confusion_matrix

LABELS = {
    "baseline": "baseline",
    "se": "se",
    "baseline_noaug": "baseline_noaug",
    "baseline_strong": "baseline_strong",
    "resnet18": "resnet18",
    "resnet18_scratch": "resnet18_scratch",
    "mobilenet": "mobilenet",
}


def main() -> None:
    configure_matplotlib()
    root = Path("runs")
    for name in LABELS:
        metrics_path = root / name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            root / name / "confusion_matrix.png",
            title=f"{name} 混淆矩阵（行归一化）",
        )
        plot_class_accuracy(
            metrics, root / name / "per_class_accuracy.png", title=f"{name} 各类别准确率"
        )


if __name__ == "__main__":
    main()
