# CIFAR-10 图像分类

当代人工智能作业三。基于 PyTorch 实现图像分类模型的**设计、训练、评估与可视化分析**，
包含自建 CNN 基线、SE 通道注意力改进、预训练模型微调，以及完整的消融对比实验。

## 环境

- Python 3.11
- PyTorch 2.6 + torchvision 0.21（CUDA 12.4）
- 单卡训练即可（RTX 3050 4GB 可完整跑通全部实验）

安装依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CIFAR-10 数据集会在首次运行时自动下载到 `data/`。

> 官方源 `cs.toronto.edu` 在国内常只有十几 KB/s。如果自动下载太慢，可以用项目自带的
> 镜像脚本取一份，再用项目里的 `CIFAR10` 子类直接读取（见 `src/data.py`）：
>
> ```bash
> python scripts/download_cifar10.py --out data
> python scripts/prepare_cifar10.py --data data
> ```

## 目录结构

```
cifar10-assignment/
├── main.py                 # 命令行入口：train / eval / gradcam / compare
├── run_experiments.py      # 一键复现全部实验
├── requirements.txt
├── README.md
├── src/
│   ├── data.py             # 数据划分、归一化、三级数据增强
│   ├── models.py           # BaselineCNN / SECNN / ResNet18 / MobileNetV2
│   ├── train.py            # 训练循环、AMP、余弦调度、早停、断点保存
│   ├── evaluate.py         # 总体/每类准确率、混淆矩阵、误分类样本
│   ├── visualize.py        # 训练曲线、混淆矩阵热图、误分类图、增强示例
│   └── gradcam.py          # Grad-CAM 可解释性分析
├── runs/                   # 训练产物：权重、history.json、图表
├── report/
│   ├── 实验报告.md          # 实验报告（Markdown 源文件）
│   ├── 实验报告.docx        # 实验报告（Word 版，可直接提交）
│   └── 实验报告.pdf         # 实验报告（PDF 版）
├── scripts/                # 数据集下载/转换、结果汇总、报告排版等辅助脚本
└── data/                   # CIFAR-10 数据集（自动下载）
```

## 快速开始

训练单个模型：

```bash
python main.py train --model baseline --out runs/baseline
```

在测试集上评估，并生成混淆矩阵、每类准确率、误分类样本图：

```bash
python main.py eval --ckpt runs/baseline/best.pt --out runs/baseline
```

生成 Grad-CAM 可解释性热力图：

```bash
python main.py gradcam --ckpt runs/resnet18/best.pt --out runs/resnet18
```

一键复现报告中全部 7 组实验（含对比图与 Grad-CAM）：

```bash
python run_experiments.py
```

也可只跑其中几组：

```bash
python run_experiments.py --only baseline resnet18
```

## 支持的命令与参数

| 命令 | 作用 |
| --- | --- |
| `train` | 训练模型，保存最佳权重到 `--out/best.pt`，同时输出 `history.json` 与训练曲线 |
| `eval` | 载入权重，在测试集上计算指标并绘制混淆矩阵、每类准确率、误分类样本 |
| `gradcam` | 对测试集指定样本生成 Grad-CAM 热力图 |
| `compare` | 把多个实验的验证准确率曲线画在同一张图上 |

`train` 的关键参数：

| 参数 | 说明 |
| --- | --- |
| `--model` | `baseline` / `se` / `resnet18` / `mobilenet` |
| `--epochs` | 训练轮数（默认按模型预设） |
| `--lr` | 初始学习率 |
| `--optimizer` | `sgd` 或 `adamw` |
| `--aug` | 数据增强强度：`none` / `base` / `strong` |
| `--scratch` | 不使用 ImageNet 预训练权重（用于对照实验） |
| `--head-lr` | 迁移学习时给分类头单独设置更大的学习率 |
| `--patience` | 早停耐心值 |
| `--no-amp` | 关闭混合精度训练 |

## 实验结果

在官方测试集（10000 张）上的结果：

| 实验 | 测试准确率 | 参数量 | 训练耗时 |
| --- | --- | --- | --- |
| ResNet18（ImageNet 预训练微调） | **93.46%** | 11.17M | 33.1 min |
| MobileNetV2（ImageNet 预训练微调） | 90.74% | 2.24M | 17.5 min |
| BaselineCNN（自建） | 90.46% | 1.15M | 15.1 min |
| SECNN（+ SE 通道注意力） | 90.04% | 1.17M | 18.7 min |
| BaselineCNN（强增强） | 89.92% | 1.15M | 20.0 min |
| BaselineCNN（无增强） | 87.89% | 1.15M | 16.4 min |
| ResNet18（从零训练） | 79.69% | 11.17M | 31.1 min |

详细的训练曲线、混淆矩阵、误分类分析与 Grad-CAM 见
[report/实验报告.md](report/实验报告.md)。各实验的原始指标保存在
`runs/<实验名>/metrics.json`，训练过程记录在 `runs/<实验名>/history.json`。

汇总所有结果的 Markdown 表格：

```bash
python scripts/summarize_results.py
```

## 生成 Word / PDF 版报告

`report/实验报告.docx` 与 `report/实验报告.pdf` 已生成。修改 Markdown 后重新生成：

```bash
pandoc report/实验报告.md -f gfm -t docx -o report/实验报告.docx --resource-path=.
python scripts/make_report_docx.py report/实验报告.docx
```

其中 pandoc 负责结构转换，`make_report_docx.py` 负责中文字体、页边距、首行缩进、
表格边框、页码和图片尺寸等排版。

## 复现性说明

- 固定随机种子（默认 42），划分训练/验证集使用独立的生成器；
- 所有超参数、最佳 epoch、参数量都会写入 checkpoint；
- 同一配置在同一台机器上重复运行，测试准确率波动在 ±0.3% 以内。
