#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
当代人工智能 实验一：新闻文本 10 分类
对比 朴素贝叶斯 / 线性支持向量机 / 逻辑回归 三种经典算法，并考察预处理方案、
TF-IDF 配置与关键超参数对验证集表现的影响。

设计原则（对应课程实验规范）：
  1. 一次实验只改变一个因素：预处理实验固定模型与向量化配置，反之亦然。
  2. 先划分数据，再仅用训练集拟合所有统计量（词表、IDF、TF-IDF），杜绝数据泄露。
  3. 先建立简单基线，再判断新增方法是否带来真实改进。
  4. 固定随机种子，报告多次划分下的均值与波动。
  5. 测试集只在方案锁定后用于最终预测，不参与任何调参。

脚本按阶段执行，中间结果缓存在 results/ 与 results/state.json，
便于分次运行、检查与复现：
    python experiment.py --stage 1     # 预处理方案对比
    python experiment.py --stage 2     # TF-IDF 配置对比
    python experiment.py --stage 3     # 模型与超参数对比
    python experiment.py --stage 4     # 多划分种子稳定性分析
    python experiment.py --stage 5     # 锁定方案、最终预测、图表与汇总
    python experiment.py --stage all   # 依次执行全部阶段
"""

import argparse
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import snowballstemmer

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 全局配置：固定随机种子，保证结果可复现
# ----------------------------------------------------------------------------
SEED = 42                        # 主实验随机种子
STABILITY_SEEDS = [42, 2024, 7]  # 稳定性分析使用的多个划分种子
VAL_SIZE = 0.2                   # 验证集比例（分层抽样）
BASE_MAX_FEATURES = 5000         # 基线 TF-IDF 词表规模
CORE_MODELS = ["朴素贝叶斯", "线性SVM", "逻辑回归"]   # 参与主对比的三种算法

_stemmer = snowballstemmer.stemmer("english")
_STOP_WORDS = set(ENGLISH_STOP_WORDS)

# 20 Newsgroups 原始文本顶部的邮件头字段
HEADER_RE = re.compile(r"^(from|subject|organization|lines|reply-to|distribution|"
                       r"newsgroups|nntp-posting-host|sender|followup-to|x-newsreader|"
                       r"keywords|summary|article-i\.d\.|path|originator)\s*:", re.I)
SIGNATURE_RE = re.compile(r"^--\s*$")


# ----------------------------------------------------------------------------
# 1. 数据加载与划分
# ----------------------------------------------------------------------------
def load_data(data_dir):
    """读取助教提供的训练集与无标签测试集。"""
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data_unlabeled.csv"))
    return (train_df["text"].astype(str).tolist(),
            train_df["target"].values,
            test_df["text"].astype(str).tolist())


def split_train_val(X, y, seed=SEED):
    """分层划分训练/验证集：各类别比例一致；划分前不做任何全局统计。"""
    return train_test_split(X, y, test_size=VAL_SIZE, random_state=seed, stratify=y)


# ----------------------------------------------------------------------------
# 2. 文本预处理：四级递进的清洗方案（均为逐样本确定性变换，不含全局统计量）
# ----------------------------------------------------------------------------
def strip_headers(text):
    """去掉邮件头区块（首个空行之前的头部字段行及其续行）。"""
    kept, header_done = [], False
    for line in text.split("\n"):
        if not header_done:
            if line.strip() == "":
                header_done = True
                continue
            if HEADER_RE.match(line.strip()):
                continue
            if line[:1] in (" ", "\t"):
                continue
            header_done = True
        kept.append(line)
    return "\n".join(kept)


def strip_quotes_and_signature(text):
    """去掉引文行（> 或 | 开头）与签名块（'-- ' 之后的内容）。"""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if SIGNATURE_RE.match(s):
            break
        if s.startswith(">") or s.startswith("|"):
            continue
        out.append(line)
    return "\n".join(out)


def normalize_and_stem(text):
    """小写化、只保留字母、去停用词与单字符词、词干化。"""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return " ".join(_stemmer.stemWord(t) for t in tokens
                    if len(t) > 1 and t not in _STOP_WORDS)


PREPROCESSORS = {
    "P0_原始文本": lambda t: t,
    "P1_去邮件头": strip_headers,
    "P2_去引文签名": lambda t: strip_quotes_and_signature(strip_headers(t)),
    "P3_加词干化": lambda t: normalize_and_stem(
        strip_quotes_and_signature(strip_headers(t))),
}
PREPROCESS_LABELS = {
    "P0_原始文本": "P0 原始",
    "P1_去邮件头": "P1 去邮件头",
    "P2_去引文签名": "P2 +去引文签名",
    "P3_加词干化": "P3 +停用词/词干",
}


def apply_preprocessing(texts, name):
    fn = PREPROCESSORS[name]
    return [fn(t) for t in texts]


# ----------------------------------------------------------------------------
# 3. 特征工程与评估工具
# ----------------------------------------------------------------------------
def build_vectorizer(max_features=BASE_MAX_FEATURES, ngram_range=(1, 1),
                     sublinear_tf=False, min_df=1):
    """构造 TF-IDF 向量化器。所有统计量只在训练集上 fit。"""
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                           sublinear_tf=sublinear_tf, min_df=min_df, lowercase=True)


MODEL_ALIASES = {"线性SVM": "LinearSVC"}


def make_model(name, param=None):
    """按名称与超参数构造模型实例；全部固定随机种子。"""
    param = param or {}
    if name == "朴素贝叶斯":
        return MultinomialNB(**param)
    if name == "线性SVM":
        return LinearSVC(random_state=SEED, **param)
    if name == "SVC线性核":
        return SVC(kernel="linear", random_state=SEED, **param)
    if name == "RBF核SVM":
        return SVC(kernel="rbf", random_state=SEED, **param)
    if name == "逻辑回归":
        return LogisticRegression(random_state=SEED, max_iter=1000, **param)
    raise ValueError(name)


def parse_param(label):
    """把 'C=1' / 'alpha=0.1' / 'C=1,gamma=scale' 之类的标签还原成参数字典。"""
    d = {}
    for part in label.split(","):
        k, v = part.split("=")
        try:
            d[k.strip()] = float(v)
        except ValueError:
            d[k.strip()] = v
    return d


def evaluate(model, X_tr, y_tr, X_va, y_va):
    """在训练集上拟合、在验证集上评估；返回准确率、宏平均 F1 与训练耗时。"""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    seconds = time.time() - t0
    pred = model.predict(X_va)
    return {"acc": accuracy_score(y_va, pred),
            "macro_f1": f1_score(y_va, pred, average="macro"),
            "fit_seconds": round(seconds, 1),
            "pred": pred}


# ----------------------------------------------------------------------------
# 阶段 1：预处理方案对比（固定模型与向量化配置，只改变预处理）
# ----------------------------------------------------------------------------
def stage1_preprocessing(cfg, log):
    X_tr, y_tr, X_va, y_va, _ = cfg["split"]
    log("[阶段 1] 预处理方案对比（固定 TF-IDF(5000, 1-gram) + 逻辑回归 C=1）")
    rows = []
    for name in PREPROCESSORS:
        tr = apply_preprocessing(X_tr, name)
        va = apply_preprocessing(X_va, name)
        vec = build_vectorizer()
        A, B = vec.fit_transform(tr), vec.transform(va)
        res = evaluate(make_model("逻辑回归", {"C": 1.0}), A, y_tr, B, y_va)
        rows.append({"预处理方案": name, "有效词表规模": len(vec.vocabulary_),
                     "验证集准确率": round(res["acc"], 4),
                     "验证集宏F1": round(res["macro_f1"], 4),
                     "训练耗时(秒)": res["fit_seconds"]})
        log(f"   {name:14s} 词表={len(vec.vocabulary_):6d} "
            f"acc={res['acc']:.4f} macroF1={res['macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    best = df.loc[df["验证集宏F1"].idxmax(), "预处理方案"]
    log(f"   -> 最佳预处理方案：{best}")
    df.to_csv(os.path.join(cfg["res_dir"], "exp1_preprocessing.csv"),
              index=False, encoding="utf-8-sig")
    return {"best_prep": best}


# ----------------------------------------------------------------------------
# 阶段 2：TF-IDF 配置对比（固定最佳预处理，只改变向量化配置）
# ----------------------------------------------------------------------------
TFIDF_CONFIGS = [
    ("基线 1-gram/5000", dict(max_features=5000, ngram_range=(1, 1))),
    ("词表放宽到 20000", dict(max_features=20000, ngram_range=(1, 1))),
    ("加入二元词组 1-2gram", dict(max_features=20000, ngram_range=(1, 2))),
    ("次线性TF缩放", dict(max_features=20000, sublinear_tf=True)),
    ("忽略极低频词 min_df=2", dict(max_features=20000, min_df=2)),
]


def stage2_tfidf(cfg, log):
    X_tr, y_tr, X_va, y_va, _ = cfg["split"]
    prep = cfg["state"]["best_prep"]
    log(f"[阶段 2] TF-IDF 配置对比（固定预处理 = {prep}，模型 = 逻辑回归 C=1）")
    tr, va = apply_preprocessing(X_tr, prep), apply_preprocessing(X_va, prep)
    rows, best_f1, best_cfg, best_label = [], -1, None, None
    for label, vcfg in TFIDF_CONFIGS:
        vec = build_vectorizer(**vcfg)
        A, B = vec.fit_transform(tr), vec.transform(va)
        res = evaluate(make_model("逻辑回归", {"C": 1.0}), A, y_tr, B, y_va)
        rows.append({"TF-IDF配置": label, "特征维度": A.shape[1],
                     "验证集准确率": round(res["acc"], 4),
                     "验证集宏F1": round(res["macro_f1"], 4),
                     "训练耗时(秒)": res["fit_seconds"]})
        log(f"   {label:22s} 维度={A.shape[1]:6d} "
            f"acc={res['acc']:.4f} macroF1={res['macro_f1']:.4f}")
        if res["macro_f1"] > best_f1:
            best_f1, best_cfg, best_label = res["macro_f1"], vcfg, label
    log(f"   -> 最佳 TF-IDF 配置：{best_label}（macroF1={best_f1:.4f}）")
    pd.DataFrame(rows).to_csv(os.path.join(cfg["res_dir"], "exp2_tfidf.csv"),
                              index=False, encoding="utf-8-sig")
    return {"tfidf_cfg": best_cfg, "tfidf_label": best_label}


# ----------------------------------------------------------------------------
# 阶段 3：模型与超参数对比（固定预处理与最佳 TF-IDF 配置）
# ----------------------------------------------------------------------------
PARAM_GRIDS = {
    "朴素贝叶斯": [("alpha=0.01", {"alpha": 0.01}), ("alpha=0.1", {"alpha": 0.1}),
                   ("alpha=0.5", {"alpha": 0.5}), ("alpha=1.0", {"alpha": 1.0}),
                   ("alpha=2.0", {"alpha": 2.0})],
    "线性SVM": [("C=0.1", {"C": 0.1}), ("C=1", {"C": 1.0}),
                ("C=10", {"C": 10.0}), ("C=100", {"C": 100.0})],
    "逻辑回归": [("C=0.1", {"C": 0.1}), ("C=1", {"C": 1.0}),
                 ("C=10", {"C": 10.0}), ("C=100", {"C": 100.0})],
}
# 附加对照：固定在 5000 维基线向量化配置下，只改变 SVM 的求解器 / 核函数
EXTRA_GRIDS = {
    "线性SVM": [("C=1", {"C": 1.0})],
    "SVC线性核": [("C=1", {"C": 1.0})],
    "RBF核SVM": [("C=1,gamma=scale", {"C": 1.0})],
}
EXTRA_LABELS = {
    "线性SVM": "LinearSVC（线性核，一次求解）",
    "SVC线性核": "SVC（线性核，one-vs-one）",
    "RBF核SVM": "SVC（RBF核，非线性）",
}


def stage3_models(cfg, log):
    X_tr, y_tr, X_va, y_va, _ = cfg["split"]
    prep, vcfg = cfg["state"]["best_prep"], cfg["state"]["tfidf_cfg"]
    log(f"[阶段 3] 模型与超参数对比（预处理={prep}，TF-IDF={cfg['state']['tfidf_label']}）")
    tr, va = apply_preprocessing(X_tr, prep), apply_preprocessing(X_va, prep)
    vec = build_vectorizer(**vcfg)
    A, B = vec.fit_transform(tr), vec.transform(va)
    log(f"   特征维度={A.shape[1]}，训练样本={A.shape[0]}，验证样本={B.shape[0]}")

    rows, best = [], {}
    for name, cands in PARAM_GRIDS.items():
        for label, param in cands:
            res = evaluate(make_model(name, param), A, y_tr, B, y_va)
            rows.append({"模型": name, "超参数": label,
                         "验证集准确率": round(res["acc"], 4),
                         "验证集宏F1": round(res["macro_f1"], 4),
                         "训练耗时(秒)": res["fit_seconds"]})
            log(f"   {name:9s} {label:20s} acc={res['acc']:.4f} "
                f"macroF1={res['macro_f1']:.4f} ({res['fit_seconds']}s)")
            if name not in best or res["macro_f1"] > best[name]["验证集宏F1"]:
                best[name] = rows[-1]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["res_dir"], "exp3_models.csv"),
              index=False, encoding="utf-8-sig")

    cols = ["模型", "超参数", "验证集准确率", "验证集宏F1", "训练耗时(秒)"]
    best_df = pd.DataFrame([best[n] for n in best], columns=cols)
    best_df.to_csv(os.path.join(cfg["res_dir"], "exp3_best_per_model.csv"),
                   index=False, encoding="utf-8-sig")
    log("   -> 各模型最佳配置：")
    for _, r in best_df.iterrows():
        log(f"      {r['模型']:9s} {r['超参数']:20s} acc={r['验证集准确率']:.4f} "
            f"macroF1={r['验证集宏F1']:.4f}")
    return {"best": {n: {"param": best[n]["超参数"],
                         "acc": best[n]["验证集准确率"],
                         "macro_f1": best[n]["验证集宏F1"]} for n in best}}


# ----------------------------------------------------------------------------
# 阶段 4：稳定性分析（多个划分种子下重复主结果）
# ----------------------------------------------------------------------------
def stage4_stability(cfg, log):
    X_all, y_all = cfg["X_all"], cfg["y_all"]
    prep, vcfg = cfg["state"]["best_prep"], cfg["state"]["tfidf_cfg"]
    log(f"[阶段 4] 多划分种子稳定性分析（种子={STABILITY_SEEDS}，预处理={prep}）")
    rec = {n: {"acc": [], "macro_f1": []} for n in CORE_MODELS}
    for seed in STABILITY_SEEDS:
        X_tr, X_va, y_tr, y_va = split_train_val(X_all, y_all, seed=seed)
        tr, va = apply_preprocessing(X_tr, prep), apply_preprocessing(X_va, prep)
        vec = build_vectorizer(**vcfg)
        A, B = vec.fit_transform(tr), vec.transform(va)
        for n in CORE_MODELS:
            res = evaluate(make_model(n, parse_param(cfg["state"]["best"][n]["param"])),
                           A, y_tr, B, y_va)
            rec[n]["acc"].append(res["acc"])
            rec[n]["macro_f1"].append(res["macro_f1"])
            log(f"   seed={seed:5d} {n:8s} acc={res['acc']:.4f} "
                f"macroF1={res['macro_f1']:.4f}")
    rows = [{"模型": n, "超参数": cfg["state"]["best"][n]["param"],
             "准确率均值": round(float(np.mean(rec[n]["acc"])), 4),
             "准确率标准差": round(float(np.std(rec[n]["acc"])), 4),
             "宏F1均值": round(float(np.mean(rec[n]["macro_f1"])), 4),
             "宏F1标准差": round(float(np.std(rec[n]["macro_f1"])), 4)}
            for n in CORE_MODELS]
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        log(f"   => {r['模型']:8s} acc={r['准确率均值']:.4f}±{r['准确率标准差']:.4f} "
            f"macroF1={r['宏F1均值']:.4f}±{r['宏F1标准差']:.4f}")
    df.to_csv(os.path.join(cfg["res_dir"], "exp4_stability.csv"),
              index=False, encoding="utf-8-sig")
    return {}


# ----------------------------------------------------------------------------
# 阶段 6：求解器与核函数对照（固定 5000 维基线特征，只改变分类器）
# ----------------------------------------------------------------------------
def stage6_solvers(cfg, log):
    """在相同的基线特征上比较不同 SVM 实现与核函数，隔离"分类器"这一因素。

    说明：SVC(kernel='linear') 采用 one-vs-one 且在 20000 维稀疏特征上训练成本
    随 C 迅速增长，因此超参数扫描使用等价的 LinearSVC，此处仅作单点对照。
    """
    X_tr, y_tr, X_va, y_va, _ = cfg["split"]
    prep = cfg["state"]["best_prep"]
    log(f"[阶段 6] 求解器与核函数对照（固定 TF-IDF(5000, 1-gram)，预处理={prep}）")
    tr, va = apply_preprocessing(X_tr, prep), apply_preprocessing(X_va, prep)
    vec = build_vectorizer()
    A, B = vec.fit_transform(tr), vec.transform(va)
    log(f"   特征维度={A.shape[1]}")
    rows = []
    for name, cands in EXTRA_GRIDS.items():
        for label, param in cands:
            res = evaluate(make_model(name, param), A, y_tr, B, y_va)
            rows.append({"分类器": EXTRA_LABELS[name], "超参数": label,
                         "验证集准确率": round(res["acc"], 4),
                         "验证集宏F1": round(res["macro_f1"], 4),
                         "训练耗时(秒)": res["fit_seconds"]})
            log(f"   {name:9s} {label:18s} acc={res['acc']:.4f} "
                f"macroF1={res['macro_f1']:.4f} ({res['fit_seconds']}s)")
    pd.DataFrame(rows).to_csv(os.path.join(cfg["res_dir"], "exp5_solvers.csv"),
                              index=False, encoding="utf-8-sig")
    return {}


# ----------------------------------------------------------------------------
# 阶段 5：锁定方案、错误分析、图表与最终预测
# ----------------------------------------------------------------------------
def stage5_final(cfg, log):
    X_all, y_all, X_test = cfg["X_all"], cfg["y_all"], cfg["X_test"]
    X_tr, y_tr, X_va, y_va, _ = cfg["split"]
    prep, vcfg = cfg["state"]["best_prep"], cfg["state"]["tfidf_cfg"]
    best = cfg["state"]["best"]

    core = pd.DataFrame([{"模型": n, **best[n]} for n in CORE_MODELS])
    core = core.sort_values("macro_f1", ascending=False)
    winner = core.iloc[0]
    log(f"[阶段 5] 锁定方案：{winner['模型']}（{winner['param']}），"
        f"验证集 macroF1={winner['macro_f1']:.4f}")

    # --- 5.1 最佳模型在验证集上的预测：混淆矩阵与错误分析 ---
    tr, va = apply_preprocessing(X_tr, prep), apply_preprocessing(X_va, prep)
    vec = build_vectorizer(**vcfg)
    A, B = vec.fit_transform(tr), vec.transform(va)
    model = make_model(winner["模型"], parse_param(winner["param"]))
    model.fit(A, y_tr)
    pred = model.predict(B)

    cm = pd.DataFrame(confusion_matrix(y_va, pred),
                      index=[f"真实类{i}" for i in range(10)],
                      columns=[f"预测类{i}" for i in range(10)])
    cm.to_csv(os.path.join(cfg["res_dir"], "confusion_matrix.csv"), encoding="utf-8-sig")

    per_class = pd.DataFrame(classification_report(
        y_va, pred, output_dict=True, digits=4)).T
    per_class.to_csv(os.path.join(cfg["res_dir"], "per_class_metrics.csv"),
                     encoding="utf-8-sig")
    report_txt = classification_report(y_va, pred, digits=4)
    with open(os.path.join(cfg["res_dir"], "classification_report_best.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"模型: {winner['模型']} 超参数: {winner['param']}\n"
                f"预处理: {prep}   TF-IDF: {cfg['state']['tfidf_label']}\n\n{report_txt}")
    log("   最佳模型验证集分类报告：")
    for line in report_txt.rstrip().split("\n"):
        log("     " + line)

    y_va = np.asarray(y_va)
    err = pred != y_va
    log(f"   错误分析：验证集 {len(y_va)} 条，误分类 {int(err.sum())} 条，"
        f"错误率 {err.mean():.4f}")
    pairs = (pd.DataFrame({"真实类别": y_va[err], "预测类别": pred[err]})
             .groupby(["真实类别", "预测类别"]).size()
             .sort_values(ascending=False).head(8).reset_index(name="错误数"))
    for _, r in pairs.iterrows():
        log(f"      真实类 {int(r['真实类别'])} -> 预测类 {int(r['预测类别'])}："
            f"{int(r['错误数'])} 条")
    pairs.to_csv(os.path.join(cfg["res_dir"], "error_pairs.csv"),
                 index=False, encoding="utf-8-sig")
    examples = [{"真实类别": int(y_va[i]), "预测类别": int(pred[i]),
                 "样本文本摘要": re.sub(r"\s+", " ", X_va[i])[:200]}
                for i in np.where(err)[0][:6]]
    pd.DataFrame(examples).to_csv(os.path.join(cfg["res_dir"], "error_examples.csv"),
                                  index=False, encoding="utf-8-sig")

    # --- 5.2 图表 ---
    draw_confusion(y_va, pred, cfg["fig_dir"], log)
    draw_overview(cfg, log)

    # --- 5.3 方案锁定后的最终预测：用全部训练数据重训 ---
    tr_all = apply_preprocessing(X_all, prep)
    te_all = apply_preprocessing(X_test, prep)
    vec2 = build_vectorizer(**vcfg)
    A2 = vec2.fit_transform(tr_all)
    T2 = vec2.transform(te_all)
    final = make_model(winner["模型"], parse_param(winner["param"]))
    final.fit(A2, y_all)
    final_pred = final.predict(T2)
    out_csv = os.path.join(cfg["out_dir"], "predictions.csv")
    pd.DataFrame(final_pred).to_csv(out_csv, index=False, header=False)
    log(f"   最终预测：模型={winner['模型']} {winner['param']}，样本数={len(final_pred)}")
    log(f"   预测类别分布={dict(sorted(pd.Series(final_pred).value_counts().items()))}")
    log(f"   已保存 predictions.csv -> {out_csv}")

    summary = {
        "随机种子": SEED,
        "稳定性分析种子": STABILITY_SEEDS,
        "训练集样本数": int(len(X_all)),
        "验证集样本数": int(len(X_va)),
        "无标签测试集样本数": int(len(X_test)),
        "验证集比例": VAL_SIZE,
        "最佳预处理方案": prep,
        "最佳TFIDF配置": cfg["state"]["tfidf_label"],
        "最佳TFIDF参数": vcfg,
        "核心模型最佳结果": {n: best[n] for n in CORE_MODELS},
        "锁定模型": winner["模型"],
        "锁定超参数": winner["param"],
        "验证集准确率": float(winner["acc"]),
        "验证集宏F1": float(winner["macro_f1"]),
        "验证集误分类数": int(err.sum()),
        "验证集错误率": round(float(err.mean()), 4),
        "新增特征数": int(A.shape[1]),
        "环境": {"python": "3.10", "scikit-learn": __import__("sklearn").__version__,
                 "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(os.path.join(cfg["res_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"summary": summary}


def draw_confusion(y_va, pred, fig_dir, log):
    """图 2：最佳模型的验证集混淆矩阵（行归一化）。"""
    cm = confusion_matrix(y_va, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    n = cm.shape[0]
    ax.set_xticks(range(n), [f"类{i}" for i in range(n)], fontsize=6)
    ax.set_yticks(range(n), [f"类{i}" for i in range(n)], fontsize=6)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("最佳模型验证集混淆矩阵（行归一化）", fontsize=9)
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= 0.02:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if cm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_confusion.png"), bbox_inches="tight")
    plt.close(fig)
    log(f"   已保存图表 fig2_confusion.png")


def draw_overview(cfg, log):
    """图 1：预处理方案 / C 值敏感性 / alpha 敏感性 三联图。"""
    pre = pd.read_csv(os.path.join(cfg["res_dir"], "exp1_preprocessing.csv"))
    models = pd.read_csv(os.path.join(cfg["res_dir"], "exp3_models.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.4))

    ax = axes[0]
    labels = [PREPROCESS_LABELS[n] for n in pre["预处理方案"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pre["验证集准确率"], 0.4, label="准确率", color="#4C72B0")
    ax.bar(x + 0.2, pre["验证集宏F1"], 0.4, label="宏F1", color="#DD8452")
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=6.5)
    ax.set_ylim(max(0.0, pre["验证集宏F1"].min() - 0.1), 1.0)
    ax.set_title("(a) 预处理方案对比（宏F1/准确率）")
    ax.legend(fontsize=6, loc="lower right"); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for name, color, mk in (("逻辑回归", "#4C72B0", "o"), ("线性SVM", "#C44E52", "s")):
        sub = models[models["模型"] == name].copy()
        sub["c"] = sub["超参数"].str.replace("C=", "").astype(float)
        sub = sub.sort_values("c")
        ax.plot(sub["c"], sub["验证集宏F1"], marker=mk, color=color, label=name)
    ax.set_xscale("log"); ax.set_xlabel("正则化参数 C（对数轴）")
    ax.set_ylabel("验证集宏F1"); ax.set_title("(b) 正则化参数 C 敏感性")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[2]
    sub = models[models["模型"] == "朴素贝叶斯"].copy()
    sub["a"] = sub["超参数"].str.replace("alpha=", "").astype(float)
    sub = sub.sort_values("a")
    ax.plot(sub["a"], sub["验证集宏F1"], marker="^", color="#55A868", label="朴素贝叶斯")
    ax.set_xscale("log"); ax.set_xlabel("平滑参数 alpha（对数轴）")
    ax.set_ylabel("验证集宏F1"); ax.set_title("(c) 平滑参数 alpha 敏感性")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg["fig_dir"], "fig1_overview.png"), bbox_inches="tight")
    plt.close(fig)
    log("   已保存图表 fig1_overview.png")


# ----------------------------------------------------------------------------
# 主流程：阶段调度与中间状态缓存
# ----------------------------------------------------------------------------
def setup_cjk_font():
    """加载思源黑体，保证图表中文正常显示。"""
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=p).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "figure.dpi": 220})


STAGES = {1: stage1_preprocessing, 2: stage2_tfidf, 3: stage3_models,
          4: stage4_stability, 5: stage5_final, 6: stage6_solvers}


def main():
    ap = argparse.ArgumentParser(description="当代人工智能实验一：文本分类")
    ap.add_argument("--data-dir", default="../data-Project1")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--stage", default="all",
                    help="1-5 表示单个阶段，all 表示依次执行全部阶段")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    res_dir = os.path.join(args.out_dir, "results")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 读取已有中间状态（支持分阶段运行与断点续跑）
    state_path = os.path.join(res_dir, "state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)

    setup_cjk_font()
    X_all, y_all, X_test = load_data(args.data_dir)
    X_tr, X_va, y_tr, y_va = split_train_val(X_all, y_all, seed=args.seed)

    cfg = {"data_dir": args.data_dir, "out_dir": args.out_dir,
           "res_dir": res_dir, "fig_dir": fig_dir,
           "split": (X_tr, y_tr, X_va, y_va, None),
           "X_all": X_all, "y_all": y_all, "X_test": X_test, "state": state}

    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(str(msg))

    log("=" * 70)
    log(f"实验一：新闻文本 10 分类 | 阶段={args.stage} | 随机种子={args.seed}")
    log("=" * 70)
    log(f"训练集(有标签) {len(X_all)} 条；无标签测试集 {len(X_test)} 条；"
        f"验证集比例 {VAL_SIZE}（分层，种子={args.seed}）")
    log(f"划分结果：训练 {len(X_tr)} / 验证 {len(X_va)}")

    wanted = list(STAGES) if args.stage == "all" else [int(args.stage)]
    for s in wanted:
        need = {1: [], 2: ["best_prep"], 3: ["best_prep", "tfidf_cfg"],
                4: ["best_prep", "tfidf_cfg", "best"],
                5: ["best_prep", "tfidf_cfg", "best"],
                6: ["best_prep"]}[s]
        missing = [k for k in need if k not in state]
        if missing:
            raise SystemExit(f"阶段 {s} 缺少前置状态 {missing}，请先运行对应前置阶段。")
        out = STAGES[s](cfg, log)
        state.update(out or {})
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    log("=" * 70)
    log(f"阶段 {args.stage} 完成，耗时 {round(time.time() - t0, 1)} 秒")
    log("=" * 70)
    with open(os.path.join(res_dir, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
