"""训练循环：AMP 混合精度、余弦学习率调度、早停与最佳模型保存。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class TrainConfig:
    model: str = "baseline"
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    aug: str = "base"
    dropout: float = 0.3
    val_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 42
    grad_clip: float = 1.0
    patience: int = 10
    amp: bool = True
    out_dir: str = "runs/exp"
    notes: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: TrainConfig, param_groups=None):
    params = param_groups if param_groups is not None else model.parameters()
    if cfg.optimizer == "sgd":
        return SGD(
            params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "adamw":
        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"未知优化器 {cfg.optimizer!r}，可选：sgd、adamw")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    device: torch.device,
    param_groups=None,
):
    """完整训练流程，返回 history 字典并把最佳权重保存到 cfg.out_dir/best.pt。"""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg, param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "time": []}
    best_acc, best_epoch, stale = 0.0, 0, 0
    checkpoint_path = out_dir / "best.pt"

    print(f"[{cfg.model}] 训练 {cfg.epochs} epochs | 设备 {device} | AMP {use_amp}")
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, cfg.grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - start
        current_lr = optimizer.param_groups[0]["lr"]
        for key, value in zip(
            ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"),
            (train_loss, train_acc, val_loss, val_acc, current_lr, elapsed),
        ):
            history[key].append(value)

        flag = ""
        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            torch.save(
                {
                    "model": cfg.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": asdict(cfg),
                },
                checkpoint_path,
            )
            flag = "  <- best"
        else:
            stale += 1

        print(
            f"epoch {epoch:3d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s{flag}"
        )

        if stale >= cfg.patience:
            print(f"验证准确率连续 {cfg.patience} 轮未提升，提前停止于 epoch {epoch}")
            break

    history["best_val_acc"] = best_acc
    history["best_epoch"] = best_epoch
    history["config"] = asdict(cfg)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"最佳验证准确率 {best_acc:.4f} (epoch {best_epoch})，权重已保存至 {checkpoint_path}")
    return history, checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device):
    """根据 checkpoint 里记录的模型名重建网络并载入权重。"""
    from .models import build_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
