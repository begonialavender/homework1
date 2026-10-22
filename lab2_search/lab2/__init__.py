# -*- coding: utf-8 -*-
"""当代人工智能实验二：搜索算法（A* / Minimax / Alpha-Beta）。"""

from .astar import SearchStats, astar, astar_cost, maze_cost, path_cost, terrain_cost
from .game_tree import GameStats, TreeNode, alpha_beta, minimax
from .problems import (
    MAZE_GOAL,
    MAZE_GRID,
    MAZE_START,
    SAMPLE_TREE_DICT,
    SAMPLE_TREE_EXPECTED_VALUE,
    WEIGHTED_EXPECTED_COST,
    WEIGHTED_GOAL,
    WEIGHTED_GRID,
    WEIGHTED_START,
    build_sample_tree,
)
from .treasure_game import (
    DIST,
    NODE_NAMES,
    TREASURE_MAP,
    TREASURE_VALUES,
    SearchResult,
    TreasureStats,
    distance_matrix,
    format_matrix,
    solve,
)

__all__ = [
    "SearchStats",
    "astar",
    "astar_cost",
    "maze_cost",
    "terrain_cost",
    "path_cost",
    "GameStats",
    "TreeNode",
    "minimax",
    "alpha_beta",
    "MAZE_GRID",
    "MAZE_START",
    "MAZE_GOAL",
    "WEIGHTED_GRID",
    "WEIGHTED_START",
    "WEIGHTED_GOAL",
    "WEIGHTED_EXPECTED_COST",
    "SAMPLE_TREE_DICT",
    "SAMPLE_TREE_EXPECTED_VALUE",
    "build_sample_tree",
    "DIST",
    "NODE_NAMES",
    "TREASURE_MAP",
    "TREASURE_VALUES",
    "TreasureStats",
    "SearchResult",
    "distance_matrix",
    "format_matrix",
    "solve",
]
