"""Grad-CAM 可视化：展示模型在做分类决策时真正关注的图像区域。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASSES_ZH, CIFAR10_MEAN, CIFAR10_STD
from .visualize import _save, configure_matplotlib


class GradCAM:
    """对指定卷积层做梯度加权类激活映射。

    原理：取目标层特征图对目标类别分数的梯度，在空间维度上求平均得到每个通道的
    重要性权重，再对特征图做加权求和并 ReLU，最后上采样到原图尺寸。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.retain_grad()

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, image: torch.Tensor, class_idx: int | None = None):
        """image: (1, 3, H, W) 已归一化。返回 (cam, 预测类别, 归一化特征图)。"""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(image)
            pred = int(logits.argmax(dim=1).item())
            target = pred if class_idx is None else class_idx
            score = logits[0, target]
            score.backward()

        gradients = self.activations.grad
        activations = self.activations.detach()

        # 全局平均池化得到通道权重
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        self.model.zero_grad(set_to_none=True)
        return cam.detach().cpu(), pred, target


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """取各模型最后一个 4×4 卷积阶段的输出作为 CAM 目标层。

    语义越强的层，梯度越能反映"模型凭什么判成这一类"；代价是 32×32 输入在这里
    已经下采样到 4×4，热力图分辨率有限。试过退到 8×8 的中间层，定位更细但高亮的
    往往是背景纹理，反而失真，因此仍用最深的卷积层。
    """
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "mobilenet":
        return model.features[13]  # 96 通道，4×4；再往后的层只有 2×2
    return model.features[-1]


def _to_display(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_gradcam(
    model: torch.nn.Module,
    model_name: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    path: str | Path,
    n: int = 6,
) -> Path:
    """在同一行中对比 原图 / Grad-CAM 热力图 / 叠加结果。"""
    configure_matplotlib()
    cam_extractor = GradCAM(model, get_target_layer(model, model_name))

    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(5.6, n * 1.75))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    try:
        for row in range(n):
            image = images[row : row + 1].to(device)
            cam, pred_idx, _ = cam_extractor(image)
            display = _to_display(images[row])
            cam_np = cam.numpy()

            axes[row, 0].imshow(display)
            axes[row, 0].set_title(f"真实：{CLASSES_ZH[labels[row].item()]}", fontsize=9)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(cam_np, cmap="jet")
            axes[row, 1].set_title(f"Grad-CAM　预测：{CLASSES_ZH[pred_idx]}", fontsize=9)
            axes[row, 1].axis("off")

            axes[row, 2].imshow(display)
            axes[row, 2].imshow(cam_np, cmap="jet", alpha=0.45)
            axes[row, 2].set_title("叠加", fontsize=9)
            axes[row, 2].axis("off")
    finally:
        cam_extractor.remove()

    fig.suptitle(f"{model_name} 的 Grad-CAM 关注区域", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
