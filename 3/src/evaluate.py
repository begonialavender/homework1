"""模型评估：总体/每类准确率、混淆矩阵与误分类样本分析。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import CLASSES, build_plain_test_loader, denormalize


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device):
    """返回 (预测概率, 真实标签)，顺序与 loader 一致。"""
    model.eval()
    all_probs, all_labels = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(targets)
    return torch.cat(all_probs), torch.cat(all_labels)


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
    """计算总体准确率、每类准确率、混淆矩阵以及最易混淆的类别对。"""
    preds = probs.argmax(dim=1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion[t.long(), p.long()] += 1

    per_class_acc = {}
    for i, name in enumerate(CLASSES[:num_classes]):
        support = confusion[i].sum().item()
        per_class_acc[name] = confusion[i, i].item() / support if support else 0.0

    # 对角线置零后找出被误判最多的（真实类, 预测类）组合
    off_diagonal = confusion.clone()
    off_diagonal.fill_diagonal_(0)
    pair_counts = []
    for i, j in zip(*torch.nonzero(off_diagonal, as_tuple=True)):
        pair_counts.append(
            (int(off_diagonal[i, j]), CLASSES[i], CLASSES[j])
        )
    pair_counts.sort(reverse=True)

    return {
        "accuracy": (preds == labels).float().mean().item(),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "top_confusions": [
            {"count": c, "true": t, "pred": p} for c, t, p in pair_counts[:8]
        ],
        "macro_accuracy": float(np.mean(list(per_class_acc.values()))),
    }


def collect_misclassified(model: nn.Module, batch_size: int = 128, data_root: str = "data"):
    """收集测试集中的误分类样本，用于报告中的人工分析与可视化。"""
    device = next(model.parameters()).device
    loader = build_plain_test_loader(data_root, batch_size=batch_size)

    model.eval()
    wrong_images, wrong_labels, wrong_preds, wrong_confs = [], [], [], []
    with torch.no_grad():
        for images, targets in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            preds = probs.argmax(dim=1)
            mask = preds != targets
            if mask.any():
                wrong_images.append(images[mask])
                wrong_labels.append(targets[mask])
                wrong_preds.append(preds[mask])
                wrong_confs.append(probs[mask].max(dim=1).values)

    if not wrong_images:
        return None
    return {
        "images": torch.cat(wrong_images),
        "labels": torch.cat(wrong_labels),
        "preds": torch.cat(wrong_preds),
        "confidence": torch.cat(wrong_confs),
    }


def summarize_misclassified(wrong: dict, top_k: int = 10) -> list[dict]:
    """按模型置信度排序，挑出"错得很自信"的样本，这类样本最能说明模型缺陷。"""
    order = torch.argsort(wrong["confidence"], descending=True)[:top_k]
    return [
        {
            "index": int(i),
            "true": CLASSES[wrong["labels"][i].item()],
            "pred": CLASSES[wrong["preds"][i].item()],
            "confidence": float(wrong["confidence"][i].item()),
        }
        for i in order
    ]


def evaluate_checkpoint(
    model: nn.Module, test_loader, out_path: str | Path, num_classes: int = 10
) -> dict:
    device = next(model.parameters()).device
    probs, labels = predict(model, test_loader, device)
    metrics = compute_metrics(probs, labels, num_classes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"测试集准确率: {metrics['accuracy']:.4f}")
    print("每类准确率:")
    for name, acc in metrics["per_class_accuracy"].items():
        print(f"  {name:<12s} {acc:.4f}")
    print("最易混淆的类别对:")
    for item in metrics["top_confusions"][:5]:
        print(f"  真实 {item['true']:<10s} -> 预测 {item['pred']:<10s} ({item['count']} 次)")

    return metrics


__all__ = [
    "predict",
    "compute_metrics",
    "evaluate_checkpoint",
    "collect_misclassified",
    "summarize_misclassified",
    "denormalize",
]
