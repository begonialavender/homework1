# -*- coding: utf-8 -*-
"""实验二对照实验：生成报告所需的全部数据与图表。

运行：
    python experiments/run_experiments.py

输出：
    experiments/results/*.csv    表格数据（UTF-8 BOM，Excel 可直接打开）
    experiments/results/results.json  汇总数据
    figures/*.png                报告插图
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from lab2 import (  # noqa: E402
    MAZE_GOAL,
    MAZE_GRID,
    MAZE_START,
    WEIGHTED_GOAL,
    WEIGHTED_GRID,
    WEIGHTED_START,
    alpha_beta,
    astar,
    build_sample_tree,
    distance_matrix,
    maze_cost,
    minimax,
    solve,
    terrain_cost,
)
from lab2.game_tree import GameStats, TreeNode, tree_minimax_value  # noqa: E402

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "experiments" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC"):
    if any(f.name == candidate for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def make_grid(
    rows: int,
    cols: int,
    density: float,
    seed: int,
    weighted: bool = False,
    max_tries: int = 200,
) -> List[List[str]]:
    """生成一张保证起点到终点可达的随机地图。

    weighted=False 时生成 '.'/'X' 的等代价迷宫；
    weighted=True 时生成 '1'..'9' 的地形代价（数字越大越难走）。
    """
    rng = random.Random(seed)
    for _ in range(max_tries):
        grid: List[List[str]] = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if rng.random() < density:
                    row.append("X")
                else:
                    row.append(str(rng.randint(1, 9)) if weighted else ".")
            grid.append(row)
        grid[0][0] = "S"
        grid[rows - 1][cols - 1] = "E"
        cost, _, _ = astar(grid, (0, 0), (rows - 1, cols - 1),
                           cost_fn=terrain_cost if weighted else maze_cost,
                           heuristic="zero")
        if cost is not None:
            return grid
    raise RuntimeError("无法生成可达地图，请降低障碍密度")


def timed_run(
    func: Callable,
    repeat: int = 3,
) -> Tuple[object, float]:
    """重复运行取平均耗时（秒）。"""
    best_elapsed = math.inf
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - t0
        best_elapsed = min(best_elapsed, elapsed)
    return result, best_elapsed


def write_csv(name: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path = RES_DIR / name
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# 实验一：迷宫寻路 —— h=0 与曼哈顿距离
# --------------------------------------------------------------------------
def experiment_maze() -> List[Dict]:
    configs = [
        ("题目样例 5x5", MAZE_GRID, MAZE_START, MAZE_GOAL, maze_cost),
        ("随机 10x10 密度0.20", make_grid(10, 10, 0.20, 11), (0, 0), (9, 9), maze_cost),
        ("随机 20x20 密度0.20", make_grid(20, 20, 0.20, 12), (0, 0), (19, 19), maze_cost),
        ("随机 20x20 密度0.30", make_grid(20, 20, 0.30, 13), (0, 0), (19, 19), maze_cost),
        ("随机 40x40 密度0.25", make_grid(40, 40, 0.25, 14), (0, 0), (39, 39), maze_cost),
    ]
    rows: List[Dict] = []
    for name, grid, start, goal, cost_fn in configs:
        out = {}
        for label, heuristic in (("h0", "zero"), ("h_manhattan", "manhattan")):
            result, elapsed = timed_run(
                lambda h=heuristic: astar(grid, start, goal, cost_fn=cost_fn, heuristic=h)
            )
            cost, path, stats = result
            out[label] = {
                "cost": cost,
                "expanded": stats.expanded,
                "generated": stats.generated,
                "time_ms": round(elapsed * 1000, 3),
                "steps": len(path) - 1,
            }
        rows.append(
            {
                "map": name,
                "h0_expanded": out["h0"]["expanded"],
                "h0_generated": out["h0"]["generated"],
                "h0_time_ms": out["h0"]["time_ms"],
                "manhattan_expanded": out["h_manhattan"]["expanded"],
                "manhattan_generated": out["h_manhattan"]["generated"],
                "manhattan_time_ms": out["h_manhattan"]["time_ms"],
                "cost_h0": out["h0"]["cost"],
                "cost_manhattan": out["h_manhattan"]["cost"],
                "steps": out["h_manhattan"]["steps"],
                "expanded_ratio": round(
                    out["h_manhattan"]["expanded"] / max(out["h0"]["expanded"], 1), 4
                ),
            }
        )
    write_csv(
        "exp1_maze.csv",
        rows,
        [
            "map",
            "h0_expanded",
            "h0_generated",
            "h0_time_ms",
            "manhattan_expanded",
            "manhattan_generated",
            "manhattan_time_ms",
            "cost_h0",
            "cost_manhattan",
            "steps",
            "expanded_ratio",
        ],
    )
    return rows


# --------------------------------------------------------------------------
# 实验二：加权地形 —— 启发函数可采纳性与权重
# --------------------------------------------------------------------------
def experiment_weighted() -> Dict:
    configs = [
        ("题目样例 5x5", WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_GOAL),
        ("随机 10x10 密度0.15", make_grid(10, 10, 0.15, 21, weighted=True), (0, 0), (9, 9)),
        ("随机 20x20 密度0.20", make_grid(20, 20, 0.20, 22, weighted=True), (0, 0), (19, 19)),
        ("随机 30x30 密度0.20", make_grid(30, 30, 0.20, 23, weighted=True), (0, 0), (29, 29)),
    ]
    rows: List[Dict] = []
    for name, grid, start, goal in configs:
        out = {}
        for label, heuristic, weight in (
            ("h0", "zero", 1.0),
            ("manhattan", "manhattan", 1.0),
            ("w3", "manhattan", 3.0),
        ):
            result, elapsed = timed_run(
                lambda h=heuristic, w=weight: astar(
                    grid, start, goal, cost_fn=terrain_cost, heuristic=h, h_weight=w
                )
            )
            cost, path, stats = result
            out[label] = {
                "cost": cost,
                "expanded": stats.expanded,
                "time_ms": round(elapsed * 1000, 3),
            }
        optimal = out["h0"]["cost"]
        rows.append(
            {
                "map": name,
                "optimal_cost": optimal,
                "h0_expanded": out["h0"]["expanded"],
                "manhattan_cost": out["manhattan"]["cost"],
                "manhattan_expanded": out["manhattan"]["expanded"],
                "manhattan_time_ms": out["manhattan"]["time_ms"],
                "w3_cost": out["w3"]["cost"],
                "w3_expanded": out["w3"]["expanded"],
                "w3_optimal": out["w3"]["cost"] == optimal,
                "expanded_ratio": round(
                    out["manhattan"]["expanded"] / max(out["h0"]["expanded"], 1), 4
                ),
            }
        )

    # h_weight=3 在多张随机地图上的最优性统计
    suboptimal = 0
    saved_nodes = []
    total = 30
    for seed in range(100, 100 + total):
        grid = make_grid(15, 15, 0.2, seed, weighted=True)
        c_opt, _, s_opt = astar(grid, (0, 0), (14, 14), cost_fn=terrain_cost, heuristic="zero")
        c_w3, _, s_w3 = astar(grid, (0, 0), (14, 14), cost_fn=terrain_cost,
                              heuristic="manhattan", h_weight=3.0)
        if c_w3 != c_opt:
            suboptimal += 1
        saved_nodes.append(s_opt.expanded / max(s_w3.expanded, 1))

    write_csv(
        "exp2_weighted.csv",
        rows,
        [
            "map",
            "optimal_cost",
            "h0_expanded",
            "manhattan_cost",
            "manhattan_expanded",
            "manhattan_time_ms",
            "w3_cost",
            "w3_expanded",
            "w3_optimal",
            "expanded_ratio",
        ],
    )
    return {
        "rows": rows,
        "w3_study": {
            "maps": total,
            "suboptimal_maps": suboptimal,
            "suboptimal_rate": round(suboptimal / total, 3),
            "avg_expanded_reduction": round(statistics.mean(saved_nodes) - 1, 3),
        },
    }


# --------------------------------------------------------------------------
# 实验三：博弈树 —— Minimax vs Alpha-Beta、访问顺序
# --------------------------------------------------------------------------
def random_tree(branching: int, depth: int, rng: random.Random) -> TreeNode:
    if depth == 0:
        return TreeNode(value=rng.randint(1, 100))
    return TreeNode(children=[random_tree(branching, depth - 1, rng) for _ in range(branching)])


def reorder_with_values(
    node: TreeNode, is_max: bool, best_first: bool
) -> Tuple[TreeNode, float]:
    """按子树的精确 Minimax 值重排子节点，返回 (新树, 该子树值)。"""
    if not node.children:
        return TreeNode(value=node.value), node.value
    pairs = [reorder_with_values(ch, not is_max, best_first) for ch in node.children]
    pairs.sort(key=lambda item: item[1], reverse=(is_max == best_first))
    new_node = TreeNode(children=[p[0] for p in pairs])
    values = [p[1] for p in pairs]
    return new_node, (max(values) if is_max else min(values))


def experiment_game_tree() -> Dict:
    sample = build_sample_tree()

    def run(tree: TreeNode) -> Dict:
        mm_stats, ab_stats = GameStats(), GameStats()
        mm = minimax(tree, True, mm_stats)
        ab = alpha_beta(tree, True, ab_stats)
        return {
            "value": ab,
            "mm_visits": mm_stats.visits,
            "ab_visits": ab_stats.visits,
            "cutoffs": ab_stats.cutoffs,
            "same_value": mm == ab,
        }

    sample_result = run(sample)
    best_tree, _ = reorder_with_values(sample, True, True)
    worst_tree, _ = reorder_with_values(sample, True, False)
    sample_best = run(best_tree)
    sample_worst = run(worst_tree)

    configs = [(2, 3), (3, 3), (4, 3), (2, 4), (3, 4), (2, 5)]
    trials = 20
    rows: List[Dict] = []
    for branching, depth in configs:
        mm_visits, ab_visits, best_visits, worst_visits = [], [], [], []
        cutoffs = []
        for seed in range(trials):
            rng = random.Random(1000 + seed * 97 + branching * 13 + depth)
            tree = random_tree(branching, depth, rng)
            mm_stats, ab_stats = GameStats(), GameStats()
            mm = minimax(tree, True, mm_stats)
            ab = alpha_beta(tree, True, ab_stats)
            assert mm == ab
            bt, _ = reorder_with_values(tree, True, True)
            wt, _ = reorder_with_values(tree, True, False)
            b_stats, w_stats = GameStats(), GameStats()
            alpha_beta(bt, True, b_stats)
            alpha_beta(wt, True, w_stats)

            mm_visits.append(mm_stats.visits)
            ab_visits.append(ab_stats.visits)
            best_visits.append(b_stats.visits)
            worst_visits.append(w_stats.visits)
            cutoffs.append(ab_stats.cutoffs)

        rows.append(
            {
                "branching": branching,
                "depth": depth,
                "leaves": branching ** depth,
                "mm_visits": round(statistics.mean(mm_visits), 1),
                "ab_visits": round(statistics.mean(ab_visits), 1),
                "ab_best_order": round(statistics.mean(best_visits), 1),
                "ab_worst_order": round(statistics.mean(worst_visits), 1),
                "cutoffs": round(statistics.mean(cutoffs), 1),
                "reduction": round(1 - statistics.mean(ab_visits) / statistics.mean(mm_visits), 3),
            }
        )

    write_csv(
        "exp3_game_tree.csv",
        rows,
        [
            "branching",
            "depth",
            "leaves",
            "mm_visits",
            "ab_visits",
            "ab_best_order",
            "ab_worst_order",
            "cutoffs",
            "reduction",
        ],
    )
    return {
        "sample": sample_result,
        "sample_best_order": sample_best,
        "sample_worst_order": sample_worst,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# 实验四：宝藏分配 —— 深度、算法与访问顺序
# --------------------------------------------------------------------------
def experiment_treasure() -> Dict:
    names, matrix = distance_matrix()

    depth_rows: List[Dict] = []
    for depth in (1, 2, 3, 4):
        row = {"depth": depth}
        for algorithm in ("minimax", "alpha-beta"):
            result = solve(depth, algorithm=algorithm)
            row[f"{algorithm}_target"] = result.best_target
            row[f"{algorithm}_value"] = result.best_value
            row[f"{algorithm}_nodes"] = result.stats.nodes
            row[f"{algorithm}_cutoffs"] = result.stats.cutoffs
        row["reduction"] = round(
            1 - row["alpha-beta_nodes"] / max(row["minimax_nodes"], 1), 3
        )
        depth_rows.append(row)

    order_rows: List[Dict] = []
    for depth in (2, 4):
        for order in ("natural", "value_desc", "gain_desc", "distance_asc"):
            mm = solve(depth, algorithm="minimax", order=order)
            ab = solve(depth, algorithm="alpha-beta", order=order)
            order_rows.append(
                {
                    "depth": depth,
                    "order": order,
                    "mm_target": mm.best_target,
                    "mm_value": mm.best_value,
                    "mm_nodes": mm.stats.nodes,
                    "ab_target": ab.best_target,
                    "ab_value": ab.best_value,
                    "ab_nodes": ab.stats.nodes,
                    "ab_cutoffs": ab.stats.cutoffs,
                }
            )

    write_csv(
        "exp4_treasure_depth.csv",
        depth_rows,
        [
            "depth",
            "minimax_target",
            "minimax_value",
            "minimax_nodes",
            "minimax_cutoffs",
            "alpha-beta_target",
            "alpha-beta_value",
            "alpha-beta_nodes",
            "alpha-beta_cutoffs",
            "reduction",
        ],
    )
    write_csv(
        "exp4_treasure_order.csv",
        order_rows,
        ["depth", "order", "mm_target", "mm_value", "mm_nodes", "ab_target", "ab_value", "ab_nodes", "ab_cutoffs"],
    )
    with (RES_DIR / "distance_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["节点"] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + list(row))
    return {"names": names, "matrix": matrix, "depth_rows": depth_rows, "order_rows": order_rows}


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------
def plot_maze(rows: List[Dict]) -> None:
    labels = [r["map"].replace("随机 ", "").replace("题目样例 ", "样例 ") for r in rows]
    h0 = [r["h0_expanded"] for r in rows]
    hm = [r["manhattan_expanded"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], h0, width, label="h=0（Dijkstra）", color="#9aa5b1")
    bars2 = ax.bar([i + width / 2 for i in x], hm, width, label="h=曼哈顿距离", color="#2f6db5")
    ax.set_yscale("log")
    ax.set_ylabel("展开节点数（对数轴）")
    ax.set_title("题目一：启发函数对 A* 搜索规模的影响")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}", (bar.get_x() + bar.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=7, rotation=0)
    fig.savefig(FIG_DIR / "fig1_maze_nodes.png")
    plt.close(fig)


def plot_weighted(rows: List[Dict], w3_study: Dict) -> None:
    labels = [r["map"].replace("随机 ", "").replace("题目样例 ", "样例 ") for r in rows]
    h0 = [r["h0_expanded"] for r in rows]
    hm = [r["manhattan_expanded"] for r in rows]
    w3 = [r["w3_expanded"] for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width = 0.26
    ax.bar([i - width for i in x], h0, width, label="h=0", color="#9aa5b1")
    ax.bar(list(x), hm, width, label="h=曼哈顿（可采纳）", color="#2f6db5")
    ax.bar([i + width for i in x], w3, width, label="h=3×曼哈顿（不可采纳）", color="#d08a2e")
    ax.set_yscale("log")
    ax.set_ylabel("展开节点数（对数轴）")
    ax.set_title("题目二：加权地形下不同启发函数的扩展规模")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.98,
        0.97,
        f"3×曼哈顿在 {w3_study['maps']} 张随机图中 "
        f"{w3_study['suboptimal_maps']} 张偏离最优（{w3_study['suboptimal_rate']:.0%}）",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#7a4a00",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d08a2e", alpha=0.9),
    )
    fig.savefig(FIG_DIR / "fig2_weighted_nodes.png")
    plt.close(fig)


def plot_game_tree(result: Dict) -> None:
    rows = result["rows"]
    labels = [f"b={r['branching']}, d={r['depth']}" for r in rows]
    series = [
        ("Minimax", [r["mm_visits"] for r in rows], "#9aa5b1"),
        ("Alpha-Beta（自然顺序）", [r["ab_visits"] for r in rows], "#2f6db5"),
        ("Alpha-Beta（最优顺序）", [r["ab_best_order"] for r in rows], "#2f8f5b"),
        ("Alpha-Beta（最差顺序）", [r["ab_worst_order"] for r in rows], "#c0504d"),
    ]
    x = list(range(len(rows)))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for idx, (label, values, color) in enumerate(series):
        positions = [i + (idx - 1.5) * width for i in x]
        ax.bar(positions, values, width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_ylabel("访问节点数（对数轴）")
    ax.set_title("题目三：Minimax 与 Alpha-Beta 的搜索开销（20 次随机树平均）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG_DIR / "fig3_game_tree_nodes.png")
    plt.close(fig)


def plot_treasure(result: Dict) -> None:
    depth_rows = result["depth_rows"]
    depths = [r["depth"] for r in depth_rows]
    mm = [r["minimax_nodes"] for r in depth_rows]
    ab = [r["alpha-beta_nodes"] for r in depth_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    ax = axes[0]
    ax.plot(depths, mm, marker="o", label="Minimax", color="#9aa5b1")
    ax.plot(depths, ab, marker="s", label="Alpha-Beta", color="#2f6db5")
    for d, m, a in zip(depths, mm, ab):
        ax.annotate(f"{m}", (d, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.annotate(f"{a}", (d, a), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)
    ax.set_xlabel("搜索深度 H")
    ax.set_ylabel("访问节点数")
    ax.set_title("深度对搜索开销的影响")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    order_rows = [r for r in result["order_rows"] if r["depth"] == 4]
    labels = [{"natural": "自然顺序\nT1→T2→T3→T4",
               "value_desc": "价值降序",
               "gain_desc": "即时收益降序",
               "distance_asc": "距离升序"}[r["order"]] for r in order_rows]
    values = [r["ab_nodes"] for r in order_rows]
    bars = ax.bar(range(len(values)), values, color="#2f6db5", width=0.55)
    for bar, v, row in zip(bars, values, order_rows):
        ax.annotate(f"{v}（剪枝{row['ab_cutoffs']}次）", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Alpha-Beta 访问节点数")
    ax.set_title("H=4 时访问顺序对剪枝的影响")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(FIG_DIR / "fig4_treasure.png")
    plt.close(fig)


def plot_distance_heatmap(result: Dict) -> None:
    names = result["names"]
    matrix = result["matrix"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_title("题目四：六个关键格之间的最小移动代价")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > 8 else "#1f2d3d", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="A* 最优代价")
    fig.savefig(FIG_DIR / "fig5_distance_heatmap.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    print("实验一：迷宫寻路 ...")
    maze_rows = experiment_maze()
    for row in maze_rows:
        print(f"  {row['map']}: h0={row['h0_expanded']} / 曼哈顿={row['manhattan_expanded']}"
              f"（{row['expanded_ratio']:.1%}），代价 {row['cost_h0']} vs {row['cost_manhattan']}")

    print("实验二：加权地形 ...")
    weighted = experiment_weighted()
    for row in weighted["rows"]:
        print(f"  {row['map']}: h0={row['h0_expanded']} / 曼哈顿={row['manhattan_expanded']}"
              f" / 3×曼哈顿={row['w3_expanded']}，代价 {row['manhattan_cost']}（最优 {row['optimal_cost']}）")
    print("  3×曼哈顿统计：", weighted["w3_study"])

    print("实验三：博弈树 ...")
    game_tree = experiment_game_tree()
    print("  样例树：", game_tree["sample"], "；最优顺序：", game_tree["sample_best_order"],
          "；最差顺序：", game_tree["sample_worst_order"])
    for row in game_tree["rows"]:
        print(f"  b={row['branching']}, d={row['depth']}: MM={row['mm_visits']}, "
              f"AB={row['ab_visits']}，剪枝率 {row['reduction']:.1%}")

    print("实验四：宝藏分配 ...")
    treasure = experiment_treasure()
    print("  距离矩阵：")
    for name, row in zip(treasure["names"], treasure["matrix"]):
        print(f"    {name:>3}: {row}")
    for row in treasure["depth_rows"]:
        print(f"  H={row['depth']}: MM {row['minimax_target']}/{row['minimax_value']} "
              f"({row['minimax_nodes']} 节点)，AB {row['alpha-beta_target']}/{row['alpha-beta_value']} "
              f"({row['alpha-beta_nodes']} 节点)")

    print("绘图 ...")
    plot_maze(maze_rows)
    plot_weighted(weighted["rows"], weighted["w3_study"])
    plot_game_tree(game_tree)
    plot_treasure(treasure)
    plot_distance_heatmap(treasure)

    summary = {
        "maze": maze_rows,
        "weighted": weighted,
        "game_tree": game_tree,
        "treasure": treasure,
    }
    with (RES_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n完成：表格在 {RES_DIR}，图表在 {FIG_DIR}")


if __name__ == "__main__":
    main()
