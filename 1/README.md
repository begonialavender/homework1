# 当代人工智能 实验一：新闻文本十分类

基于 TF-IDF 特征，对比**朴素贝叶斯 / 线性支持向量机 / 逻辑回归**三种经典文本分类算法，
并系统考察**文本预处理方案**、**TF-IDF 配置**与**关键超参数**对验证集表现的影响。

---

## 一、环境与依赖

| 项目 | 版本 |
| --- | --- |
| Python | 3.10 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| snowballstemmer | 3.0.1（词干化，实验用 P3 预处理方案） |

安装命令：

```bash
conda create -n project1 python=3.10
conda activate project1
conda install numpy pandas scikit-learn matplotlib
pip install snowballstemmer
```

## 二、目录结构

```
实验一提交/
├── experiment.py        # 实验主脚本（分阶段执行，含全部实验逻辑）
├── README.md            # 本文件：运行说明
├── predictions.csv      # 最终提交的预测结果（2457 行，每行一个类别标签 0-9，无表头）
├── 实验一报告.docx       # 实验报告（随作业单独提交，不随代码仓库上传）
├── figures/
│   ├── fig1_overview.png    # 预处理方案 / C 值敏感性 / alpha 敏感性 三联图
│   └── fig2_confusion.png   # 最佳模型验证集混淆矩阵
└── results/             # 全部中间产物与结果表（可复查、可复现）
    ├── exp1_preprocessing.csv      # 预处理方案对比
    ├── exp2_tfidf.csv              # TF-IDF 配置对比
    ├── exp3_models.csv             # 全部模型的超参数扫描明细
    ├── exp3_best_per_model.csv     # 各模型的最佳配置
    ├── exp4_stability.csv          # 三个划分种子下的均值与标准差
    ├── exp5_solvers.csv            # 求解器与核函数对照
    ├── confusion_matrix.csv        # 混淆矩阵原始计数
    ├── per_class_metrics.csv       # 逐类别精确率/召回率/F1
    ├── classification_report_best.txt
    ├── error_pairs.csv             # 最易混淆的类别对
    ├── error_examples.csv          # 若干误分类样本
    ├── state.json                  # 分阶段运行的中间状态
    ├── summary.json                # 实验配置与结果汇总
    └── run_log.txt                 # 完整运行日志
```

## 三、运行方式

数据目录默认为 `../data-Project1`（内含 `train_data.csv` 与 `test_data_unlabeled.csv`），
请确认脚本与 `data-Project1` 的相对位置，或通过 `--data-dir` 指定。

```bash
# 方式一：一次跑完全部阶段（约 3 分钟）
python experiment.py --stage all --data-dir ../data-Project1 --out-dir .

# 方式二：分阶段运行（便于检查中间结果）
python experiment.py --stage 1     # 预处理方案对比        (~40 秒)
python experiment.py --stage 2     # TF-IDF 配置对比       (~20 秒)
python experiment.py --stage 3     # 模型与超参数对比       (~17 秒)
python experiment.py --stage 4     # 多划分种子稳定性分析   (~15 秒)
python experiment.py --stage 6     # 求解器与核函数对照     (~90 秒)
python experiment.py --stage 5     # 锁定方案、出图、最终预测 (~5 秒)
```

> 说明：阶段 5 依赖阶段 1-3 产生的中间状态（`results/state.json`），
> 因此分阶段运行时请按上面的顺序执行；`--stage all` 会自动按序执行。

**可复现性**：所有随机种子固定为 `random_state=42`（`--seed` 可改）；
所有预处理与 TF-IDF 统计量仅在训练集上拟合，验证集只做 `transform`，
测试集在方案锁定前始终保持封存。

## 四、实验设置摘要

| 环节 | 设置 |
| --- | --- |
| 训练/验证划分 | 7368 条中分层抽取 20% 作验证集（5894 / 1474） |
| 预处理方案 | P0 原始文本 / P1 去邮件头 / P2 再去引文与签名 / P3 再加停用词与词干化 |
| TF-IDF 配置 | 词表规模 5000 与 20000、1-gram 与 1-2gram、次线性 TF 缩放、min_df=2 |
| 超参数网格 | 朴素贝叶斯 `alpha ∈ {0.01,0.1,0.5,1,2}`；SVM 与逻辑回归 `C ∈ {0.1,1,10,100}` |
| 稳定性分析 | 划分种子 `{42, 2024, 7}` 重复主结果，报告均值 ± 标准差 |
| 评价指标 | 验证集准确率与宏平均 F1（宏 F1 用于抵抗类别不均衡的干扰） |

## 五、主要结果

| 模型 | 最佳超参数 | 验证集准确率 | 验证集宏 F1 |
| --- | --- | --- | --- |
| 朴素贝叶斯 | alpha = 0.1 | 0.9206 | 0.9213 |
| **线性 SVM（LinearSVC）** | **C = 1** | **0.9396** | **0.9397** |
| 逻辑回归 | C = 10 | 0.9383 | 0.9384 |

- 最佳预处理方案为 **P0 原始文本**：去掉邮件头使宏 F1 下降约 2.4 个百分点，
  再去掉引文与签名后下降约 10.9 个百分点。
- TF-IDF 方面，词表从 5000 放宽到 20000 并启用 `sublinear_tf` 收益最大（宏 F1 提升约 2 个百分点）。
- 三个划分种子下：线性 SVM 0.9416±0.0083、逻辑回归 0.9396±0.0058、朴素贝叶斯 0.9246±0.0051。
- 相同特征下，LinearSVC 精度最高（0.9176）且比 `SVC(kernel='linear')` 快约 57 倍（0.4s 对 22.8s）。

最终按验证集宏 F1 选定 **线性 SVM（C=1）**，用全部 7368 条训练数据重新拟合后
对 2457 条无标签测试集预测，结果见 `predictions.csv`。

## 六、注意事项

- 数据目录、结果目录均通过命令行参数指定，未使用任何绝对路径，可在其他机器上直接运行。
- 图表中的中文依赖思源黑体（Noto Sans CJK）；在无该字体的环境中，脚本会自动回退到默认字体，
  图表中文可能显示为方框，但**不影响任何数值结果**。
- 脚本按阶段缓存中间结果到 `results/state.json`，重复运行同一阶段会安全地覆盖输出。
