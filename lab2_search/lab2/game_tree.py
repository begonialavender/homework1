# -*- coding: utf-8 -*-
"""题目三：极大极小搜索（Minimax）与 Alpha-Beta 剪枝。

TreeNode 结构按题目要求实现：
- 非叶子节点有 children；
- 叶子节点有 value；
- 根节点是 MAX 玩家，子节点从左到右依次访问。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class TreeNode:
    def __init__(self, value=None, children: Optional[Iterable["TreeNode"]] = None):
        self.value = value
        self.children: List["TreeNode"] = list(children) if children is not None else []

    # 方便调试与报告展示
    def __repr__(self) -> str:
        if self.children:
            return f"TreeNode(children={self.children!r})"
        return f"TreeNode(value={self.value!r})"

    @classmethod
    def from_dict(cls, data: Dict) -> "TreeNode":
        """把题目给出的 JSON 结构转换为 TreeNode。"""
        if "children" in data and data["children"]:
            return cls(children=[cls.from_dict(ch) for ch in data["children"]])
        return cls(value=data.get("value"))


@dataclass
class GameStats:
    visits: int = 0        # 访问过的节点数
    leaves: int = 0        # 评估过的叶子数
    cutoffs: int = 0       # 发生剪枝的次数
    max_depth: int = 0     # 实际到达的最大深度

    def as_dict(self) -> Dict[str, int]:
        return {
            "visits": self.visits,
            "leaves": self.leaves,
            "cutoffs": self.cutoffs,
            "max_depth": self.max_depth,
        }


def minimax(
    node: TreeNode,
    is_max: bool = True,
    stats: Optional[GameStats] = None,
    depth: int = 0,
) -> float:
    """标准 Minimax：自底向上回传，不做任何剪枝。"""
    if stats is None:
        stats = GameStats()
    stats.visits += 1
    if depth > stats.max_depth:
        stats.max_depth = depth

    if not node.children:
        stats.leaves += 1
        return node.value

    if is_max:
        best = -math.inf
        for child in node.children:
            best = max(best, minimax(child, False, stats, depth + 1))
        return best
    best = math.inf
    for child in node.children:
        best = min(best, minimax(child, True, stats, depth + 1))
    return best


def alpha_beta(
    node: TreeNode,
    is_max: bool = True,
    stats: Optional[GameStats] = None,
    alpha: float = -math.inf,
    beta: float = math.inf,
    depth: int = 0,
) -> float:
    """Alpha-Beta 剪枝。

    alpha：MAX 已保证能拿到的下界；
    beta ：MIN 已保证能把结果压到的上界。
    当 alpha >= beta 时，当前节点之后的兄弟分支不可能影响祖先的选择，直接剪掉。
    """
    if stats is None:
        stats = GameStats()
    stats.visits += 1
    if depth > stats.max_depth:
        stats.max_depth = depth

    if not node.children:
        stats.leaves += 1
        return node.value

    if is_max:
        best = -math.inf
        for child in node.children:
            best = max(best, alpha_beta(child, False, stats, alpha, beta, depth + 1))
            if best > alpha:
                alpha = best
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return best

    best = math.inf
    for child in node.children:
        best = min(best, alpha_beta(child, True, stats, alpha, beta, depth + 1))
        if best < beta:
            beta = best
        if alpha >= beta:
            stats.cutoffs += 1
            break
    return best


def ordered_copy(
    node: TreeNode,
    key: Callable[[TreeNode], float],
    is_max: bool = True,
    best_first: bool = True,
) -> TreeNode:
    """按给定关键字递归排序每个节点的子节点，返回一棵新树（不修改原树）。

    用于“子节点访问顺序对剪枝效果的影响”实验：
    - best_first=True 时，MAX 层按关键字降序、MIN 层按关键字升序，
      相当于总能先访问对自己最有利的分支（理想顺序，实践中需要靠启发式近似）；
    - best_first=False 时顺序相反（最差顺序）。
    关键字一般取子树的标准 Minimax 值。
    """
    if not node.children:
        return TreeNode(value=node.value)

    children = [ordered_copy(ch, key, not is_max, best_first) for ch in node.children]
    # MAX 想先看大的，MIN 想先看小的
    reverse = (is_max == best_first)
    children.sort(key=key, reverse=reverse)
    return TreeNode(children=children)


def tree_minimax_value(node: TreeNode, is_max: bool = True) -> float:
    """不计数地求子树 Minimax 值，作为排序关键字。"""
    if not node.children:
        return node.value
    vals = [tree_minimax_value(ch, not is_max) for ch in node.children]
    return max(vals) if is_max else min(vals)
