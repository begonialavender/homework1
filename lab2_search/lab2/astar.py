# -*- coding: utf-8 -*-
"""A* 搜索：题目一（等代价迷宫）与题目二（加权地形）共用同一套实现。

状态：网格坐标 (row, col)
g(n)：从起点走到 n 的实际累计代价
h(n)：曼哈顿距离下界（可切换为 0，此时退化为 Dijkstra）
f(n) = g(n) + h(n)
优先队列按 (f, h) 排序：f 相同时优先扩展 h 更小的节点，路径更直、扩展更少。
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

INF = float("inf")
Cell = Tuple[int, int]
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class SearchStats:
    """搜索过程的统计量，用于实验对照。"""

    expanded: int = 0      # 弹出并真正展开的节点数
    generated: int = 0     # 生成（入堆）的节点数
    max_frontier: int = 0  # 优先队列的最大长度
    reopened: int = 0      # 重新打开的节点数

    def as_dict(self) -> Dict[str, int]:
        return {
            "expanded": self.expanded,
            "generated": self.generated,
            "max_frontier": self.max_frontier,
            "reopened": self.reopened,
        }


def maze_cost(cell: str) -> Optional[int]:
    """题目一：'X' 不可通行，其余格子（含 'S'、'E'、'.'）进入代价均为 1。"""
    return None if cell == "X" else 1


def terrain_cost(cell: str) -> Optional[int]:
    """题目二/题目四：'X' 不可通行；数字为地形代价；'E' 进入代价为 0；其余字母格代价 1。"""
    if cell == "X":
        return None
    if cell == "E":
        return 0
    if cell.isdigit():
        return int(cell)
    return 1


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _make_heuristic(
    grid: Sequence[Sequence[str]],
    goal: Cell,
    cost_fn: Callable[[str], Optional[int]],
    heuristic: str,
    h_weight: float,
) -> Callable[[Cell], float]:
    """构造启发函数。

    h(n) = h_weight * ((曼哈顿距离 d - 1) * 最小可通行代价 + 进入终点的代价)

    说明：
    - 题目一进入终点的代价为 1，h 就是曼哈顿距离；
    - 题目二规定进入 E 的代价为 0，因此终点前一步的真实剩余代价为 0，
      必须把终点格的特殊代价计入，h 才保持可采纳；
    - 当 h_weight > 1 时启发式可能不可采纳，仅用于对照实验。
    """
    if heuristic in ("zero", "none", None):
        return lambda _n: 0.0

    goal_enter = cost_fn(grid[goal[0]][goal[1]])
    goal_enter = 0 if goal_enter is None else goal_enter

    positive_costs = [
        c
        for row in grid
        for cell in row
        for c in [cost_fn(cell)]
        if c is not None and c > 0
    ]
    min_step = min(positive_costs) if positive_costs else 1

    def h(n: Cell) -> float:
        d = manhattan(n, goal)
        return h_weight * max(0.0, (d - 1) * min_step + goal_enter)

    return h


def astar(
    grid: Sequence[Sequence[str]],
    start: Cell,
    goal: Cell,
    cost_fn: Callable[[str], Optional[int]] = terrain_cost,
    heuristic: str = "manhattan",
    h_weight: float = 1.0,
    stats: Optional[SearchStats] = None,
) -> Tuple[Optional[int], List[Cell], SearchStats]:
    """在网格上运行 A*。

    返回 (最小代价, 路径坐标列表, 统计量)。
    不可达时返回 (None, [], stats)；起点即终点时返回 (0, [start], stats)。
    """
    if stats is None:
        stats = SearchStats()

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return None, [], stats

    sr, sc = start
    gr, gc = goal
    for r, c in (start, goal):
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"坐标越界: {(r, c)}")
    if cost_fn(grid[sr][sc]) is None or cost_fn(grid[gr][gc]) is None:
        return None, [], stats
    if start == goal:
        return 0, [start], stats

    h = _make_heuristic(grid, goal, cost_fn, heuristic, h_weight)
    counter = itertools.count()
    h0 = h(start)
    open_heap: List[Tuple[float, float, int, int, Cell]] = [
        (h0, h0, 0, next(counter), start)
    ]
    best_g: Dict[Cell, int] = {start: 0}
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    closed = set()

    while open_heap:
        _f, _hv, g_cost, _cnt, node = heapq.heappop(open_heap)

        if node in closed:
            continue
        if g_cost != best_g.get(node):
            continue

        closed.add(node)
        stats.expanded += 1

        if node == goal:
            path: List[Cell] = []
            cur: Optional[Cell] = node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return g_cost, path, stats

        r, c = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            step = cost_fn(grid[nr][nc])
            if step is None:
                continue
            nb = (nr, nc)
            ng = g_cost + step
            if ng >= best_g.get(nb, INF):
                continue

            best_g[nb] = ng
            parent[nb] = node
            if nb in closed:          # 允许重新打开，便于和不可采纳启发式对照
                closed.discard(nb)
                stats.reopened += 1
            hn = h(nb)
            heapq.heappush(open_heap, (ng + hn, hn, ng, next(counter), nb))
            stats.generated += 1
            if len(open_heap) > stats.max_frontier:
                stats.max_frontier = len(open_heap)

    return None, [], stats


def astar_cost(
    grid: Sequence[Sequence[str]],
    start: Cell,
    goal: Cell,
    cost_fn: Callable[[str], Optional[int]] = terrain_cost,
    heuristic: str = "manhattan",
    h_weight: float = 1.0,
) -> int:
    """只返回最小代价；无解时按题目二的要求返回 -1。"""
    cost, _path, _stats = astar(
        grid, start, goal, cost_fn=cost_fn, heuristic=heuristic, h_weight=h_weight
    )
    return -1 if cost is None else cost


def path_cost(
    path: Sequence[Cell],
    grid: Sequence[Sequence[str]],
    cost_fn: Callable[[str], Optional[int]] = terrain_cost,
) -> Optional[int]:
    """按“进入格子的代价”计算一条给定路径的总代价，用于校验搜索结果。"""
    if not path:
        return None
    total = 0
    for r, c in path[1:]:
        step = cost_fn(grid[r][c])
        if step is None:
            return None
        total += step
    return total
