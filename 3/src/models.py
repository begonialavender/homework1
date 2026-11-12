"""CIFAR-10 图像分类模型定义。

包含三个模型，用于报告中的架构对比实验：

1. BaselineCNN  —— 自建 VGG 风格卷积网络，从零训练；
2. SECNN        —— 在 BaselineCNN 的每个卷积阶段后加入 SE 通道注意力，从零训练；
3. ResNet18     —— ImageNet 预训练主干，针对 32x32 小分辨率改造后微调。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """两层 3x3 卷积 + BN + ReLU 的标准卷积阶段。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。

    先做全局平均池化得到每个通道的全局响应，再由两层全连接学习通道权重，
    最后把权重乘回特征图，让网络自适应地放大重要通道、抑制无关通道。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.fc(x.mean(dim=(2, 3))).view(b, c, 1, 1)
        return x * weight


class SEConvBlock(nn.Module):
    """卷积阶段 + SE 注意力。"""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class BaselineCNN(nn.Module):
    """自建卷积网络：3 个卷积阶段（32→64→128）+ 全局平均池化 + Dropout + 全连接。"""

    def __init__(self, num_classes: int = 10, width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, width),
            ConvBlock(width, width * 2),
            ConvBlock(width * 2, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SECNN(BaselineCNN):
    """与 BaselineCNN 结构对齐、仅把卷积阶段换成 SE 版本，便于控制变量的消融对比。"""

    def __init__(self, num_classes: int = 10, width: int = 64, dropout: float = 0.3):
        super().__init__(num_classes=num_classes, width=width, dropout=dropout)
        self.features = nn.Sequential(
            SEConvBlock(3, width),
            SEConvBlock(width, width * 2),
            SEConvBlock(width * 2, width * 4),
        )


def resnet18_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """把 ImageNet 版 ResNet18 适配到 32x32 输入。

    原始主干用 7x7/stride2 卷积 + 3x3 最大池化把 224 图像下采样 4 倍，
    对 32x32 的 CIFAR 图像会直接损失过多空间信息。这里换成 3x3/stride1 卷积
    并去掉最大池化，网络总下采样倍数为 8，特征图最终为 4x4。
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def mobilenet_v2_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 迁移学习版本，用于"轻量模型 vs 大模型"的对比。"""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    # 首个卷积步长改为 1，避免 32x32 输入被过早下采样
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "se": SECNN,
    "resnet18": resnet18_cifar,
    "mobilenet": mobilenet_v2_cifar,
}

# 从零训练与迁移学习使用不同的默认超参数
PRETRAINED_MODELS = {"resnet18", "mobilenet"}


def build_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name!r}，可选：{tuple(MODEL_REGISTRY)}")
    if name in PRETRAINED_MODELS:
        return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
    return MODEL_REGISTRY[name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float):
    """对预训练模型做分层学习率：主干用小学习率微调，分类头用大学习率重学。"""
    head_names = ("fc", "classifier")
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.split(".")[0] in head_names else backbone_params).append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
