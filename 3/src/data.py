"""CIFAR-10 数据加载与数据增强策略。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CIFAR10(datasets.CIFAR10):
    """torchvision 的 CIFAR10 只认官方文件的 MD5。

    官方源在国内只有十几 KB/s，本项目用 scripts/ 下的脚本从镜像下载并转换成同样的
    目录结构。这里把校验降级为"文件是否存在"，其余加载逻辑完全沿用 torchvision。
    """

    def _check_integrity(self) -> bool:
        names = [name for name, _ in self.train_list + self.test_list]
        names.append(self.meta["filename"])
        return all((Path(self.root) / self.base_folder / name).is_file() for name in names)

    def _load_meta(self) -> None:
        path = Path(self.root) / self.base_folder / self.meta["filename"]
        with open(path, "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        self.classes = meta[self.meta["key"]]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CLASSES_ZH = (
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车",
)

AUG_STRATEGIES = ("none", "base", "strong")


def build_transforms(train: bool, aug: str = "base") -> transforms.Compose:
    """构造图像预处理流水线。

    - 验证/测试集只做 ToTensor + Normalize，保证评估可复现；
    - 训练集按 aug 选择不同强度的增强：
        none   : 不增强（用于消融对照）
        base   : 随机裁剪 + 水平翻转（基础几何增强）
        strong : 在 base 之上叠加 RandAugment 与 RandomErasing
    """
    if aug not in AUG_STRATEGIES:
        raise ValueError(f"未知的增强策略 {aug!r}，可选：{AUG_STRATEGIES}")

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    if not train or aug == "none":
        return transforms.Compose([transforms.ToTensor(), normalize])

    pipeline = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

    if aug == "strong":
        pipeline.append(transforms.RandAugment(num_ops=2, magnitude=9))
        pipeline += [
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ]
    else:
        pipeline += [transforms.ToTensor(), normalize]

    return transforms.Compose(pipeline)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """把归一化后的张量还原到 [0, 1]，用于可视化。"""
    mean = torch.tensor(CIFAR10_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_size: int
    val_size: int
    test_size: int


def _worker_init(worker_id: int) -> None:
    # 不同 worker 使用不同的随机种子，避免数据增强出图高度相关
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(
    data_root: str = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    aug: str = "base",
    seed: int = 42,
) -> DataBundle:
    """划分训练/验证/测试集并构建 DataLoader。

    CIFAR-10 官方训练集共 50000 张，这里按 val_ratio 切出验证集用于调参与早停，
    官方测试集 10000 张只在最终评估时使用，全程不参与任何训练决策。
    """
    raw_train = CIFAR10(
        root=data_root, train=True, download=True, transform=build_transforms(True, aug)
    )
    # 验证集需要"未增强"的同一批图像，因此额外读一份无增强副本
    clean_train = CIFAR10(
        root=data_root, train=True, download=True, transform=build_transforms(False)
    )
    test_set = CIFAR10(
        root=data_root, train=False, download=True, transform=build_transforms(False)
    )

    total = len(raw_train)
    n_val = int(total * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(total), [total - n_val, n_val], generator=generator
    )

    train_set = Subset(raw_train, train_subset.indices)
    val_set = Subset(clean_train, val_subset.indices)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init,
        persistent_workers=num_workers > 0,
    )

    return DataBundle(
        train=DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs),
        val=DataLoader(val_set, shuffle=False, **loader_kwargs),
        test=DataLoader(test_set, shuffle=False, **loader_kwargs),
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set),
    )


def build_plain_test_loader(
    data_root: str = "data", batch_size: int = 128, num_workers: int = 0
) -> DataLoader:
    """返回只做 ToTensor 的测试集，便于可视化时直接显示原图。"""
    test_set = CIFAR10(root=data_root, train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
