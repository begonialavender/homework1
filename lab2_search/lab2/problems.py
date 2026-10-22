# -*- coding: utf-8 -*-
"""题目中给出的测试数据，集中放在这里，便于复现与实验脚本复用。"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .game_tree import TreeNode

# --------------------------------------------------------------------------
# 题目一：迷宫寻路
# --------------------------------------------------------------------------
MAZE_GRID: List[List[str]] = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
MAZE_START: Tuple[int, int] = (0, 0)
MAZE_GOAL: Tuple[int, int] = (3, 4)
MAZE_EXPECTED_PATH: List[Tuple[int, int]] = [
    (0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)
]

# --------------------------------------------------------------------------
# 题目二：加权地形
# --------------------------------------------------------------------------
WEIGHTED_GRID: List[List[str]] = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "8", "1", "8", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "8", "1", "1", "E"],
]
WEIGHTED_START: Tuple[int, int] = (0, 0)
WEIGHTED_GOAL: Tuple[int, int] = (4, 4)
WEIGHTED_EXPECTED_COST = 7

# --------------------------------------------------------------------------
# 题目三：极大极小博弈树（与题面 JSON 完全一致）
# --------------------------------------------------------------------------
SAMPLE_TREE_DICT: Dict = {
    "is_maximizing_player": True,
    "children": [
        {
            "is_maximizing_player": False,
            "children": [
                {
                    "is_maximizing_player": True,
                    "children": [{"value": 3}, {"value": 5}],
                },
                {
                    "is_maximizing_player": True,
                    "children": [{"value": 6}, {"value": 9}],
                },
            ],
        },
        {
            "is_maximizing_player": False,
            "children": [
                {
                    "is_maximizing_player": True,
                    "children": [{"value": 6}, {"value": 9}],
                },
                {
                    "is_maximizing_player": True,
                    "children": [{"value": 1}, {"value": 2}],
                },
            ],
        },
    ],
}
SAMPLE_TREE_EXPECTED_VALUE = 5


def build_sample_tree() -> TreeNode:
    return TreeNode.from_dict(SAMPLE_TREE_DICT)
