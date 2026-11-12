"""汇总 runs/ 下各实验的结果，输出可直接粘贴进报告的 Markdown 表格。

用法：python scripts/summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

ORDER = [
    ("baseline", "BaselineCNN"),
    ("se", "SECNN"),
    ("baseline_noaug", "Baseline（无增强）"),
    ("baseline_strong", "Baseline（强增强）"),
    ("resnet18", "ResNet18 预训练"),
    ("resnet18_scratch", "ResNet18 从零"),
    ("mobilenet", "MobileNetV2 预训练"),
]

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main() -> None:
    runs = Path("runs")
    rows = []
    for name, label in ORDER:
        metrics_path = runs / name / "metrics.json"
        history_path = runs / name / "history.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = {}
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        train_time = sum(history.get("time", [])) / 60
        rows.append((name, label, metrics, history, train_time))

    print("## 总体结果\n")
    print("| 实验 | 测试准确率 | 宏平均准确率 | 最佳验证准确率 | 参数量 | 训练耗时 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, label, metrics, history, train_time in rows:
        params = "—"
        ckpt = runs / name / "best.pt"
        if ckpt.exists():
            params = f"{history.get('config', {}).get('notes', {}).get('params', 0) / 1e6:.2f}M" \
                if history.get("config", {}).get("notes", {}).get("params") else "—"
        print(f"| {label} | {metrics['accuracy']:.2%} | {metrics['macro_accuracy']:.2%} | "
              f"{history.get('best_val_acc', 0):.2%} | {params} | {train_time:.1f} min |")

    print("\n## 各类别准确率\n")
    print("| 实验 | " + " | ".join(CLASSES) + " |")
    print("| --- | " + " | ".join(["---"] * len(CLASSES)) + " |")
    for name, label, metrics, _, _ in rows:
        accs = [f"{metrics['per_class_accuracy'][c]:.1%}" for c in CLASSES]
        print(f"| {label} | " + " | ".join(accs) + " |")

    print("\n## 最易混淆的类别对\n")
    for name, label, metrics, _, _ in rows:
        pairs = ", ".join(
            f"{p['true']}→{p['pred']}({p['count']})" for p in metrics["top_confusions"][:3]
        )
        print(f"- **{label}**：{pairs}")


if __name__ == "__main__":
    main()
