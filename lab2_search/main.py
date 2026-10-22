# -*- coding: utf-8 -*-
"""实验二总入口：依次运行四道题并核对题目给出的预期结果。

用法：
    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab2 import (  # noqa: E402
    MAZE_GOAL,
    MAZE_GRID,
    MAZE_START,
    SAMPLE_TREE_EXPECTED_VALUE,
    WEIGHTED_EXPECTED_COST,
    WEIGHTED_GOAL,
    WEIGHTED_GRID,
    WEIGHTED_START,
    alpha_beta,
    astar,
    astar_cost,
    build_sample_tree,
    distance_matrix,
    format_matrix,
    maze_cost,
    minimax,
    path_cost,
    solve,
    terrain_cost,
)


def title(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def check(name: str, actual, expected) -> bool:
    ok = actual == expected
    flag = "通过" if ok else "不通过"
    print(f"[{flag}] {name}: 实际 = {actual}，预期 = {expected}")
    return ok


def main() -> int:
    ok = True

    # ---------------- 题目一 ----------------
    title("题目一：迷宫寻路（A*，h = 曼哈顿距离）")
    cost, path, stats = astar(
        MAZE_GRID, MAZE_START, MAZE_GOAL, cost_fn=maze_cost, heuristic="manhattan"
    )
    print("路径：", path)
    print("路径长度：", len(path) - 1, "步；路径代价：", cost)
    print("搜索统计：", stats.as_dict())
    ok &= check("题一 起点", path[0], MAZE_START)
    ok &= check("题一 终点", path[-1], MAZE_GOAL)
    ok &= check("题一 最短步数", len(path) - 1, 7)
    ok &= check("题一 路径代价", cost, 7)

    # 无可达路径的情形
    blocked = [row[:] for row in MAZE_GRID]
    blocked[0][1] = "X"
    blocked[1][0] = "X"
    no_cost, no_path, _ = astar(blocked, (0, 0), (3, 4), cost_fn=maze_cost)
    ok &= check("题一 无解返回", (no_cost, no_path), (None, []))

    # ---------------- 题目二 ----------------
    title("题目二：加权地形（A*，h 计入终点 E 的 0 代价）")
    cost2, path2, stats2 = astar(
        WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_GOAL, cost_fn=terrain_cost
    )
    print("路径：", path2)
    print("路径代价：", cost2, "；逐步累计：", end="")
    running = 0
    pieces = []
    for r, c in path2[1:]:
        running += terrain_cost(WEIGHTED_GRID[r][c])
        pieces.append(f"({r},{c})={terrain_cost(WEIGHTED_GRID[r][c])}")
    print(" -> ".join(["0"] + pieces), "=", running)
    print("搜索统计：", stats2.as_dict())
    ok &= check("题二 最小代价", cost2, WEIGHTED_EXPECTED_COST)
    ok &= check("题二 代价复算", path_cost(path2, WEIGHTED_GRID, terrain_cost), WEIGHTED_EXPECTED_COST)

    blocked_weighted = [["S", "X"], ["X", "E"]]
    ok &= check(
        "题二 无解返回 -1",
        astar_cost(blocked_weighted, (0, 0), (1, 1), cost_fn=terrain_cost),
        -1,
    )

    # ---------------- 题目三 ----------------
    title("题目三：Alpha-Beta 剪枝")
    tree = build_sample_tree()
    mm_stats = None
    from lab2.game_tree import GameStats  # 局部导入，保持输出清晰

    mm_stats = GameStats()
    mm_value = minimax(tree, True, mm_stats)
    ab_stats = GameStats()
    ab_value = alpha_beta(tree, True, ab_stats)
    print("Minimax 结果：", mm_value, "统计：", mm_stats.as_dict())
    print("Alpha-Beta 结果：", ab_value, "统计：", ab_stats.as_dict())
    ok &= check("题三 Minimax 值", mm_value, SAMPLE_TREE_EXPECTED_VALUE)
    ok &= check("题三 Alpha-Beta 值", ab_value, SAMPLE_TREE_EXPECTED_VALUE)
    check("题三 剪枝后访问节点数应少于 Minimax", ab_stats.visits < mm_stats.visits, True)
    ok &= ab_stats.visits < mm_stats.visits

    # ---------------- 题目四 ----------------
    title("题目四：竞争环境下的宝藏分配")
    names, matrix = distance_matrix()
    print("6x6 距离矩阵：")
    print(format_matrix(names, matrix))

    for depth, expected_target, expected_value in ((2, "T2", -1), (4, "T3", 3)):
        print(f"\n--- 搜索深度 H = {depth} ---")
        for algorithm in ("minimax", "alpha-beta"):
            result = solve(depth, algorithm=algorithm)
            print(
                f"{algorithm:>10}: 首步 = {result.best_target}, "
                f"U = {result.best_value}, 统计 = {result.stats.as_dict()}"
            )
            print("             最优行棋序列：", " -> ".join(
                f"{p} 取 {t} ({g:+d})" for p, t, g in result.line
            ))
        result = solve(depth, algorithm="alpha-beta")
        ok &= check(f"题四 H={depth} 首步", result.best_target, expected_target)
        ok &= check(f"题四 H={depth} 收益差", result.best_value, expected_value)

    print("\n" + "=" * 72)
    print("总体结果：", "全部通过" if ok else "存在不通过项，请检查")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
