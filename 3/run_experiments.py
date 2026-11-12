"""一键复现全部实验。

依次训练并评估下列配置，结果写入 runs/<实验名>/：

    实验                 目的
    ------------------   --------------------------------------------
    baseline             自建 CNN 基线
    baseline_noaug       去掉数据增强（消融）
    baseline_strong      改用强增强 RandAugment + RandomErasing（消融）
    se                   加入 SE 通道注意力（改进）
    resnet18             预训练 ResNet18 微调（改进）
    resnet18_scratch     ResNet18 从零训练（对照，说明预训练的价值）
    mobilenet            预训练 MobileNetV2 微调（轻量模型对比）

用法：python run_experiments.py --only baseline resnet18
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    ("baseline", ["--model", "baseline"]),
    ("baseline_noaug", ["--model", "baseline", "--aug", "none"]),
    ("baseline_strong", ["--model", "baseline", "--aug", "strong"]),
    ("se", ["--model", "se"]),
    ("resnet18", ["--model", "resnet18"]),
    ("resnet18_scratch", ["--model", "resnet18", "--scratch"]),
    ("mobilenet", ["--model", "mobilenet"]),
]


def run(cmd: list[str]) -> bool:
    print("\n" + "=" * 72)
    print(">>> " + " ".join(cmd))
    print("=" * 72, flush=True)
    # 单个实验失败时记录并继续，避免中断后面的实验
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"!!! 命令失败（退出码 {result.returncode}），继续下一个")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None, help="只运行指定实验")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--force", action="store_true", help="已完成的实验也重新训练")
    args = parser.parse_args()

    selected = [e for e in EXPERIMENTS if args.only is None or e[0] in args.only]
    if not selected:
        print(f"未知实验名，可选：{[name for name, _ in EXPERIMENTS]}")
        sys.exit(1)

    start = time.time()
    completed = []
    for name, extra in selected:
        out_dir = Path(args.runs_dir) / name
        # 断点续跑：已有测试指标就跳过，中断后重跑不会白费之前的训练
        if not args.force and (out_dir / "metrics.json").exists():
            print(f"\n跳过 {name}：已完成（如需重训加 --force）")
            completed.append(str(out_dir))
            continue

        run([sys.executable, "main.py", "train", *extra, "--out", str(out_dir)])
        run([sys.executable, "main.py", "eval", "--ckpt", str(out_dir / "best.pt"),
             "--out", str(out_dir)])
        completed.append(str(out_dir))

    run([sys.executable, "main.py", "compare", "--runs", *completed,
         "--out", str(Path(args.runs_dir) / "comparison.png")])

    for name in ("baseline", "se", "resnet18", "mobilenet"):
        ckpt = Path(args.runs_dir) / name / "best.pt"
        if ckpt.exists():
            run([sys.executable, "main.py", "gradcam", "--ckpt", str(ckpt),
                 "--out", str(ckpt.parent)])

    print(f"\n全部实验完成，用时 {(time.time() - start) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
