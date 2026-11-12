"""CIFAR-10 图像分类 —— 命令行入口。

用法示例：
    python main.py train --model baseline --epochs 40 --out runs/baseline
    python main.py eval  --ckpt runs/baseline/best.pt --out runs/baseline
    python main.py gradcam --ckpt runs/resnet18/best.pt --out runs/resnet18
    python main.py compare --runs runs/baseline runs/se runs/resnet18
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import CLASSES, build_dataloaders
from src.evaluate import (
    collect_misclassified,
    evaluate_checkpoint,
    summarize_misclassified,
)
from src.gradcam import plot_gradcam
from src.models import (
    PRETRAINED_MODELS,
    build_model,
    count_parameters,
    parameter_groups,
)
from src.train import TrainConfig, fit, load_checkpoint, set_seed
from src.visualize import (
    configure_matplotlib,
    plot_class_accuracy,
    plot_confusion_matrix,
    plot_history_comparison,
    plot_misclassified,
    plot_training_curves,
)

# 各模型的默认超参数，命令行参数可覆盖
PRESETS = {
    "baseline": dict(epochs=30, lr=1e-3, optimizer="adamw", weight_decay=5e-4, aug="base"),
    "se": dict(epochs=30, lr=1e-3, optimizer="adamw", weight_decay=5e-4, aug="base"),
    "resnet18": dict(epochs=20, lr=0.01, optimizer="sgd", weight_decay=5e-4, aug="base"),
    "mobilenet": dict(epochs=20, lr=0.01, optimizer="sgd", weight_decay=5e-4, aug="base"),
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_train(args) -> None:
    preset = PRESETS[args.model].copy()
    for key, value in vars(args).items():
        if key in preset and value is not None:
            preset[key] = value

    cfg = TrainConfig(model=args.model, out_dir=args.out, batch_size=args.batch_size,
                      num_workers=args.num_workers, patience=args.patience,
                      amp=not args.no_amp, **preset)

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = get_device()

    bundle = build_dataloaders(
        data_root=args.data, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        val_ratio=cfg.val_ratio, aug=cfg.aug, seed=cfg.seed,
    )
    print(f"训练集 {bundle.train_size} | 验证集 {bundle.val_size} | 测试集 {bundle.test_size}")

    pretrained = args.model in PRETRAINED_MODELS and not args.scratch
    model = build_model(cfg.model, num_classes=len(CLASSES), pretrained=pretrained).to(device)
    print(f"模型 {cfg.model}（{'ImageNet 预训练' if pretrained else '随机初始化'}）"
          f"参数量 {count_parameters(model) / 1e6:.2f}M")

    groups = None
    if args.head_lr is not None:
        groups = parameter_groups(model, cfg.lr, args.head_lr)
        print(f"分层学习率：主干 {cfg.lr:g}，分类头 {args.head_lr:g}")

    cfg.notes = {"pretrained": pretrained, "params": count_parameters(model)}
    history, ckpt = fit(model, bundle.train, bundle.val, cfg, device, groups)

    configure_matplotlib()
    plot_training_curves(history, Path(args.out) / "curves.png",
                         title=f"{cfg.model} 训练过程")

    if cfg.aug != "none":
        from torchvision import transforms as T

        from src.data import CIFAR10
        from src.visualize import plot_augmentation_examples

        raw = CIFAR10(root=args.data, train=True, download=True, transform=T.ToTensor())
        plot_augmentation_examples(raw, Path(args.out) / "augmentation_examples.png")


def cmd_eval(args) -> None:
    device = get_device()
    out_dir = Path(args.out or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.ckpt, device)
    bundle = build_dataloaders(data_root=args.data, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    print(f"载入 {args.ckpt}（epoch {ckpt['epoch']}，验证准确率 {ckpt['val_acc']:.4f}）")

    metrics = evaluate_checkpoint(model, bundle.test, out_dir / "metrics.json")

    configure_matplotlib()
    plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png",
                          title=f"{ckpt['model']} 混淆矩阵（行归一化）")
    plot_class_accuracy(metrics, out_dir / "per_class_accuracy.png",
                        title=f"{ckpt['model']} 各类别准确率")

    wrong = collect_misclassified(model, batch_size=args.batch_size, data_root=args.data)
    if wrong is not None:
        plot_misclassified(wrong, out_dir / "misclassified.png",
                           title=f"{ckpt['model']} 误分类样本（按置信度排序）")
        samples = summarize_misclassified(wrong, top_k=10)
        import json

        with open(out_dir / "misclassified_cases.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"误分类样本 {len(wrong['labels'])} 张，已导出置信度最高的 10 例")


def cmd_gradcam(args) -> None:
    device = get_device()
    out_dir = Path(args.out or Path(args.ckpt).parent)
    model, ckpt = load_checkpoint(args.ckpt, device)

    from src.data import CIFAR10, build_transforms

    test_set = CIFAR10(
        root=args.data, train=False, download=True, transform=build_transforms(False)
    )
    if args.indices:
        indices = args.indices
    else:
        # 默认每个类别各取一张，避免全是同一类
        seen, indices = set(), []
        for i in range(len(test_set)):
            label = test_set[i][1]
            if label not in seen:
                seen.add(label)
                indices.append(i)
            if len(indices) == args.count:
                break
    images = torch.stack([test_set[i][0] for i in indices])
    labels = torch.tensor([test_set[i][1] for i in indices])

    plot_gradcam(model, ckpt["model"], images, labels, device,
                 out_dir / args.name, n=len(indices))


def cmd_compare(args) -> None:
    import json

    configure_matplotlib()
    histories = {}
    for run in args.runs:
        history_path = Path(run) / "history.json"
        if not history_path.exists():
            print(f"跳过 {run}：未找到 history.json")
            continue
        with open(history_path, encoding="utf-8") as f:
            histories[Path(run).name] = json.load(f)

    if not histories:
        print("没有可对比的实验记录")
        return
    plot_history_comparison(histories, Path(args.out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 图像分类实验")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="data", help="数据集目录")
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--num-workers", type=int, default=4)

    p_train = sub.add_parser("train", parents=[common], help="训练模型")
    p_train.add_argument("--model", default="baseline", choices=list(PRESETS))
    p_train.add_argument("--out", required=True, help="结果输出目录")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--optimizer", default=None, choices=["sgd", "adamw"])
    p_train.add_argument("--weight-decay", type=float, default=None)
    p_train.add_argument("--aug", default=None, choices=["none", "base", "strong"])
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--head-lr", type=float, default=None, help="分类头单独的学习率")
    p_train.add_argument("--scratch", action="store_true", help="不使用预训练权重")
    p_train.add_argument("--no-amp", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", parents=[common], help="在测试集上评估")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_cam = sub.add_parser("gradcam", parents=[common], help="生成 Grad-CAM 可视化")
    p_cam.add_argument("--ckpt", required=True)
    p_cam.add_argument("--out", default=None)
    p_cam.add_argument("--indices", type=int, nargs="+", default=None,
                       help="测试集中的样本下标")
    p_cam.add_argument("--count", type=int, default=6, help="默认每类取一张时总共取几张")
    p_cam.add_argument("--name", default="gradcam.png", help="输出文件名")
    p_cam.set_defaults(func=cmd_gradcam)

    p_cmp = sub.add_parser("compare", help="对比多个实验的验证曲线")
    p_cmp.add_argument("--runs", nargs="+", required=True)
    p_cmp.add_argument("--out", default="runs/comparison.png")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
