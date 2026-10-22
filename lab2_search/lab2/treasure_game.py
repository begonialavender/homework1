# -*- coding: utf-8 -*-
"""题目四：竞争环境下的宝藏分配（A* + Alpha-Beta 综合题）。

建模要点
--------
1. 先用 A* 在加权地图上求出 A、B、T1、T2、T3、T4 两两之间的最小移动代价，
   得到 6x6 距离矩阵；
2. 把玩家位置记为所在格子的名字（初始为 "A"/"B"，取宝后变成 "T1".."T4"），
   这样任意时刻的位置都能直接查距离矩阵；
3. 博弈状态 = (A 的位置, B 的位置, 剩余宝藏, A 累计收益, B 累计收益, 当前玩家)；
4. 每回合只能选一个尚未获取的宝藏，收益 Gain(p, Ti) = Vi - d(p, Ti)，
   到达深度限制或宝藏取完时，以 U = Score_A - Score_B 作为评价值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .astar import astar, terrain_cost

# 地图：X 为障碍，数字为地形代价，A/B/T* 所在格代价为 1
TREASURE_MAP: List[List[str]] = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]

TREASURE_VALUES: Dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
TREASURE_POS: Dict[str, Tuple[int, int]] = {
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
PLAYER_STARTS: Dict[str, Tuple[int, int]] = {"A": (0, 0), "B": (6, 6)}
NODE_NAMES: List[str] = ["A", "B", "T1", "T2", "T3", "T4"]
TREASURE_ORDER: Tuple[str, ...] = ("T1", "T2", "T3", "T4")


# --------------------------------------------------------------------------
# 距离矩阵
# --------------------------------------------------------------------------
def _all_positions() -> Dict[str, Tuple[int, int]]:
    pos = dict(PLAYER_STARTS)
    pos.update(TREASURE_POS)
    return pos


def distance_matrix() -> Tuple[List[str], List[List[int]]]:
    """用 A* 计算 6 个关键格两两之间的最小移动代价，返回 (名称, 6x6 矩阵)。"""
    pos = _all_positions()
    matrix: List[List[int]] = []
    for name_a in NODE_NAMES:
        row: List[int] = []
        for name_b in NODE_NAMES:
            if name_a == name_b:
                row.append(0)
                continue
            cost, _path, _stats = astar(
                TREASURE_MAP, pos[name_a], pos[name_b], cost_fn=terrain_cost
            )
            if cost is None:
                raise RuntimeError(f"{name_a} -> {name_b} 不可达，地图数据有误")
            row.append(cost)
        matrix.append(row)
    return list(NODE_NAMES), matrix


def _build_distance_lookup() -> Dict[Tuple[str, str], int]:
    names, matrix = distance_matrix()
    lookup: Dict[Tuple[str, str], int] = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            lookup[(a, b)] = matrix[i][j]
    return lookup


DIST: Dict[Tuple[str, str], int] = _build_distance_lookup()


# --------------------------------------------------------------------------
# 博弈搜索
# --------------------------------------------------------------------------
@dataclass
class TreasureStats:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0
    max_depth: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "nodes": self.nodes,
            "leaves": self.leaves,
            "cutoffs": self.cutoffs,
            "max_depth": self.max_depth,
        }


@dataclass
class SearchResult:
    best_target: Optional[str]
    best_value: float
    line: Tuple[Tuple[str, str, int], ...]  # (玩家, 宝藏, 本次收益)
    stats: TreasureStats
    depth: int
    algorithm: str


def _ordered_moves(remaining: Sequence[str], order: str, pos: str) -> List[str]:
    moves = list(remaining)
    if order in ("natural", None):
        return moves
    if order == "value_desc":
        return sorted(moves, key=lambda t: -TREASURE_VALUES[t])
    if order == "gain_desc":
        return sorted(moves, key=lambda t: -(TREASURE_VALUES[t] - DIST[(pos, t)]))
    if order == "distance_asc":
        return sorted(moves, key=lambda t: DIST[(pos, t)])
    if callable(order):
        return list(order(moves, pos, TREASURE_VALUES, DIST))
    raise ValueError(f"未知的访问顺序: {order}")


def solve(
    depth: int,
    algorithm: str = "alpha-beta",
    order: str = "natural",
    stats: Optional[TreasureStats] = None,
    root_move: Optional[str] = None,
) -> SearchResult:
    """求解 A 的最优首步。

    depth 为搜索层数（1 层 = 一个玩家选一次宝藏）；
    algorithm 取 "minimax" 或 "alpha-beta"；
    root_move 仅用于分析：指定 A 的首步，返回该首步下的博弈结果。
    """
    if stats is None:
        stats = TreasureStats()
    use_ab = algorithm in ("alpha-beta", "alphabeta", "ab")

    def rec(
        pos_a: str,
        pos_b: str,
        remaining: Tuple[str, ...],
        score_a: int,
        score_b: int,
        turn: int,
        dep: int,
        alpha: float,
        beta: float,
    ) -> Tuple[float, Tuple[Tuple[str, str, int], ...]]:
        stats.nodes += 1
        if dep > stats.max_depth:
            stats.max_depth = dep

        if dep == 0 or not remaining:
            stats.leaves += 1
            return score_a - score_b, ()

        if turn == 0:  # A (MAX)
            best = -math.inf
            best_line: Tuple[Tuple[str, str, int], ...] = ()
            for target in _ordered_moves(remaining, order, pos_a):
                if root_move is not None and dep == depth and target != root_move:
                    continue
                gain = TREASURE_VALUES[target] - DIST[(pos_a, target)]
                rest = tuple(x for x in remaining if x != target)
                value, line = rec(target, pos_b, rest, score_a + gain, score_b, 1, dep - 1, alpha, beta)
                if value > best:
                    best = value
                    best_line = (("A", target, gain),) + line
                if use_ab:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        stats.cutoffs += 1
                        break
            return best, best_line

        # B (MIN)
        best = math.inf
        best_line = ()
        for target in _ordered_moves(remaining, order, pos_b):
            gain = TREASURE_VALUES[target] - DIST[(pos_b, target)]
            rest = tuple(x for x in remaining if x != target)
            value, line = rec(pos_a, target, rest, score_a, score_b + gain, 0, dep - 1, alpha, beta)
            if value < best:
                best = value
                best_line = (("B", target, gain),) + line
            if use_ab:
                beta = min(beta, best)
                if alpha >= beta:
                    stats.cutoffs += 1
                    break
        return best, best_line

    value, line = rec("A", "B", TREASURE_ORDER, 0, 0, 0, depth, -math.inf, math.inf)
    return SearchResult(
        best_target=line[0][1] if line else None,
        best_value=value,
        line=line,
        stats=stats,
        depth=depth,
        algorithm=algorithm,
    )


def format_matrix(names: Sequence[str], matrix: Sequence[Sequence[int]]) -> str:
    """把距离矩阵格式化成等宽文本，便于打印。"""
    width = max(len(str(v)) for row in matrix for v in row)
    header = " " * 4 + "".join(f"{n:>{width + 2}}" for n in names)
    lines = [header]
    for name, row in zip(names, matrix):
        lines.append(f"{name:<4}" + "".join(f"{v:>{width + 2}}" for v in row))
    return "\n".join(lines)
