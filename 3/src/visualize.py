"""实验图表绘制：训练曲线、混淆矩阵、每类准确率、误分类样本。"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from .data import CLASSES, CLASSES_ZH, CIFAR10_MEAN, CIFAR10_STD

# 统一的配色，保证报告里所有图表风格一致
PRIMARY = "#2F5D8A"
DEEP = "#C0504D"
LIGHT = "#8FA8C0"
GRID = "#DCDCDC"


def configure_matplotlib() -> None:
    """中文标签 + 统一风格。Windows 上优先使用雅黑，缺失时回退到黑体。"""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.edgecolor": "#BBBBBB",
            "axes.labelcolor": "#333333",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"图表已保存: {path}")
    return path


def plot_training_curves(history: dict, path: str | Path, title: str = "训练过程") -> Path:
    """绘制 loss 与 accuracy 双曲线（左：损失，右：准确率）。"""
    configure_matplotlib()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(epochs, history["train_loss"], color=PRIMARY, lw=2, label="训练损失")
    axes[0].plot(epochs, history["val_loss"], color=DEEP, lw=2, label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], color=PRIMARY, lw=2, label="训练准确率")
    axes[1].plot(epochs, history["val_acc"], color=DEEP, lw=2, label="验证准确率")
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    axes[1].scatter([best_epoch], [best_acc], color=DEEP, zorder=5, s=45)
    axes[1].annotate(
        f"最佳 {best_acc:.2%}",
        (best_epoch, best_acc),
        textcoords="offset points",
        xytext=(8, -14),
        color=DEEP,
        fontsize=10,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].legend(loc="lower right")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrix(
    cm: list[list[int]], path: str | Path, normalize: bool = True, title: str = "混淆矩阵"
) -> Path:
    """绘制混淆矩阵热力图，默认按真实类别归一化，便于比较不同类别的错误率。"""
    configure_matplotlib()
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        fmt, vmax = "{:.2f}", 1.0
    else:
        cm_display, fmt, vmax = cm, "{:.0f}", cm.max()

    cmap = LinearSegmentedColormap.from_list("custom_blue", ["#F7F9FC", "#8FA8C0", PRIMARY, "#1B3B5F"])
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=vmax)

    labels_zh = CLASSES_ZH
    ax.set_xticks(range(len(labels_zh)), labels_zh, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels_zh)), labels_zh)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(axis="x", pad=8, length=0)
    ax.tick_params(axis="y", pad=4, length=0)

    threshold = vmax * 0.55
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            color = "white" if value > threshold else "#333333"
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=color, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("行归一化比例" if normalize else "样本数")
    fig.tight_layout()
    return _save(fig, path)


def plot_class_accuracy(metrics: dict, path: str | Path, title: str = "各类别准确率") -> Path:
    """横向条形图展示每类准确率，并标注平均值参考线。"""
    configure_matplotlib()
    names = list(metrics["per_class_accuracy"])
    accs = [metrics["per_class_accuracy"][n] for n in names]
    labels_zh = [CLASSES_ZH[CLASSES.index(n)] for n in names]

    order = np.argsort(accs)
    labels_zh = [labels_zh[i] for i in order]
    accs = [accs[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [DEEP if a < metrics["accuracy"] else PRIMARY for a in accs]
    bars = ax.barh(labels_zh, accs, color=colors, height=0.62)

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.005, bar.get_y() + bar.get_height() / 2, f"{acc:.2%}",
                va="center", fontsize=9.5, color="#333333")

    ax.axvline(metrics["accuracy"], color="#555555", ls="--", lw=1.2,
               label=f"总体准确率 {metrics['accuracy']:.2%}")
    ax.set_xlim(0, min(1.08, max(accs) + 0.12))
    ax.set_xlabel("准确率")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_history_comparison(histories: dict[str, dict], path: str | Path) -> Path:
    """多个模型的验证准确率曲线叠加对比。"""
    configure_matplotlib()
    palette = [PRIMARY, DEEP, "#5B8C5A", "#B8894A", "#7B6AA8", "#4C9A9A", "#B5548C"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    for (name, history), color in zip(histories.items(), cycle(palette)):
        epochs = np.arange(1, len(history["val_acc"]) + 1)
        ax.plot(epochs, history["val_acc"], lw=2, color=color,
                label=f"{name}（最佳 {max(history['val_acc']):.2%}）")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("验证准确率")
    ax.set_title("不同模型的验证准确率对比")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def plot_misclassified(wrong: dict, path: str | Path, n: int = 12, title: str = "误分类样本") -> Path:
    """按置信度从高到低展示误分类样本，突出"错得很自信"的典型案例。

    默认取 12 张、4 列排布，保证贴进报告后每张图和标注仍然清晰可读。
    """
    configure_matplotlib()
    order = torch.argsort(wrong["confidence"], descending=True)[:n]
    n = len(order)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.55))
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, idx in zip(axes.ravel(), order):
        # 误分类样本来自只做 ToTensor 的测试集，像素已在 [0, 1] 区间
        image = wrong["images"][idx].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.axis("off")
        true_zh = CLASSES_ZH[wrong["labels"][idx].item()]
        pred_zh = CLASSES_ZH[wrong["preds"][idx].item()]
        ax.set_title(f"真:{true_zh}  预测:{pred_zh}\n置信度 {wrong['confidence'][idx]:.1%}",
                     fontsize=8.5, color=DEEP)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_augmentation_examples(
    dataset, path: str | Path, index: int = 0, rounds: int = 8
) -> Path:
    """同一张原图经过多次随机增强后的效果，用于说明数据增强的作用。"""
    configure_matplotlib()
    from PIL import Image

    from .data import build_transforms

    original = dataset[index][0].permute(1, 2, 0).numpy()  # ToTensor 输出，取值 [0,1]
    pil = Image.fromarray((original * 255).astype(np.uint8))
    aug_transform = build_transforms(True, "strong")
    mean = np.array(CIFAR10_MEAN).reshape(1, 1, 3)
    std = np.array(CIFAR10_STD).reshape(1, 1, 3)

    fig, axes = plt.subplots(2, rounds // 2, figsize=(rounds * 1.35, 6.2))
    axes = axes.ravel()
    axes[0].imshow(original)
    axes[0].set_title("原图", fontsize=10)
    axes[0].axis("off")

    for ax in axes[1:]:
        arr = aug_transform(pil).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(arr * std + mean, 0, 1))
        ax.axis("off")

    fig.suptitle("强数据增强效果示例", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
