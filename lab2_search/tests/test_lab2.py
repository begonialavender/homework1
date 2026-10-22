# -*- coding: utf-8 -*-
"""实验二自测用例（只依赖标准库 unittest）。

运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    maze_cost,
    minimax,
    path_cost,
    solve,
    terrain_cost,
)
from lab2.game_tree import GameStats  # noqa: E402


class TestMaze(unittest.TestCase):
    def test_shortest_path(self):
        cost, path, _ = astar(MAZE_GRID, MAZE_START, MAZE_GOAL, cost_fn=maze_cost)
        self.assertEqual(cost, 7)
        self.assertEqual(path[0], MAZE_START)
        self.assertEqual(path[-1], MAZE_GOAL)
        self.assertEqual(len(path) - 1, 7)
        # 每一步都必须是上下左右相邻格
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            self.assertEqual(abs(r1 - r2) + abs(c1 - c2), 1)

    def test_expected_path_is_optimal(self):
        self.assertEqual(path_cost(
            [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)],
            MAZE_GRID,
            maze_cost,
        ), 7)

    def test_no_path(self):
        grid = [row[:] for row in MAZE_GRID]
        grid[0][1] = "X"
        grid[1][0] = "X"
        cost, path, _ = astar(grid, (0, 0), MAZE_GOAL, cost_fn=maze_cost)
        self.assertIsNone(cost)
        self.assertEqual(path, [])

    def test_zero_heuristic_gives_same_cost(self):
        cost_h0, _, _ = astar(MAZE_GRID, MAZE_START, MAZE_GOAL, cost_fn=maze_cost, heuristic="zero")
        cost_h1, _, _ = astar(MAZE_GRID, MAZE_START, MAZE_GOAL, cost_fn=maze_cost)
        self.assertEqual(cost_h0, cost_h1)


class TestWeighted(unittest.TestCase):
    def test_expected_cost(self):
        cost, path, _ = astar(WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_GOAL, cost_fn=terrain_cost)
        self.assertEqual(cost, WEIGHTED_EXPECTED_COST)
        self.assertEqual(path_cost(path, WEIGHTED_GRID, terrain_cost), cost)

    def test_goal_entry_is_free(self):
        self.assertEqual(terrain_cost("E"), 0)
        self.assertEqual(terrain_cost("8"), 8)
        self.assertIsNone(terrain_cost("X"))

    def test_zero_heuristic_same_cost(self):
        c0, _, _ = astar(WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_GOAL, cost_fn=terrain_cost, heuristic="zero")
        c1, _, _ = astar(WEIGHTED_GRID, WEIGHTED_START, WEIGHTED_GOAL, cost_fn=terrain_cost)
        self.assertEqual(c0, c1)
        self.assertEqual(c1, 7)

    def test_no_path_returns_minus_one(self):
        grid = [["S", "X"], ["X", "E"]]
        self.assertEqual(astar_cost(grid, (0, 0), (1, 1), cost_fn=terrain_cost), -1)


class TestGameTree(unittest.TestCase):
    def setUp(self):
        self.tree = build_sample_tree()

    def test_minimax(self):
        stats = GameStats()
        self.assertEqual(minimax(self.tree, True, stats), SAMPLE_TREE_EXPECTED_VALUE)
        self.assertEqual(stats.leaves, 8)

    def test_alpha_beta_same_value_fewer_nodes(self):
        mm_stats, ab_stats = GameStats(), GameStats()
        mm = minimax(self.tree, True, mm_stats)
        ab = alpha_beta(self.tree, True, ab_stats)
        self.assertEqual(mm, ab)
        self.assertEqual(ab, SAMPLE_TREE_EXPECTED_VALUE)
        self.assertLessEqual(ab_stats.visits, mm_stats.visits)
        self.assertGreater(ab_stats.cutoffs, 0)


class TestTreasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names, cls.matrix = distance_matrix()

    def test_matrix_shape(self):
        self.assertEqual(self.names, ["A", "B", "T1", "T2", "T3", "T4"])
        self.assertEqual(len(self.matrix), 6)
        self.assertTrue(all(len(row) == 6 for row in self.matrix))

    def test_matrix_symmetry_and_diagonal(self):
        for i in range(6):
            self.assertEqual(self.matrix[i][i], 0)
            for j in range(6):
                self.assertEqual(self.matrix[i][j], self.matrix[j][i])

    def test_known_distances(self):
        d = {a: dict(zip(self.names, row)) for a, row in zip(self.names, self.matrix)}
        self.assertEqual(d["A"]["T2"], 4)
        self.assertEqual(d["A"]["T1"], 10)
        self.assertEqual(d["B"]["T3"], 4)

    def test_depth_2(self):
        result = solve(2, algorithm="alpha-beta")
        self.assertEqual(result.best_target, "T2")
        self.assertEqual(result.best_value, -1)

    def test_depth_4(self):
        result = solve(4, algorithm="alpha-beta")
        self.assertEqual(result.best_target, "T3")
        self.assertEqual(result.best_value, 3)

    def test_minimax_and_ab_agree(self):
        for depth in (1, 2, 3, 4):
            mm = solve(depth, algorithm="minimax")
            ab = solve(depth, algorithm="alpha-beta")
            self.assertEqual(mm.best_value, ab.best_value)
            self.assertEqual(mm.best_target, ab.best_target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
