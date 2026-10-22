# 实验二：搜索算法（A* / Minimax / Alpha-Beta）

当代人工智能实验课 2026 年秋季。本目录包含四道题的完整实现、单元测试与对照实验脚本。

## 目录结构

```
lab2_search/
├─ main.py                     四道题总入口：运行并自动核对题面预期
├─ lab2/
│  ├─ __init__.py              包导出
│  ├─ astar.py                 A* 通用实现（迷宫 / 加权地形）
│  ├─ game_tree.py             TreeNode、Minimax、Alpha-Beta
│  ├─ treasure_game.py         题目四：距离矩阵与博弈搜索
│  └─ problems.py              题面地图、样例树等数据
├─ tests/
│  └─ test_lab2.py             16 个单元测试
├─ experiments/
│  ├─ run_experiments.py       对照实验脚本（生成 CSV 与 PNG）
│  └─ results/                 实验数据表（运行后生成）
├─ figures/                    报告插图（运行后生成）
└─ README.md
```

## 环境要求

- Python 3.10 或更高版本；
- 四道题与单元测试只使用标准库，无需安装任何第三方包；
- 只有重新生成图表时需要 `matplotlib` 与 `numpy`：

```bash
pip install matplotlib numpy
```

## 快速开始

在本目录（`lab2_search`）下执行：

```bash
# 1. 运行四道题并核对预期结果
python main.py

# 2. 运行单元测试
python -m unittest discover -s tests -v

# 3. 重新生成对照实验数据与图表
python experiments/run_experiments.py
```

`main.py` 正常结束时最后一行会打印“总体结果：全部通过”，退出码为 0。

## 各题的调用方式

### 题目一：迷宫寻路

```python
from lab2 import astar, maze_cost

cost, path, stats = astar(
    grid, start, goal,
    cost_fn=maze_cost,      # 'X' 不可通行，其余格子代价 1
    heuristic="manhattan",  # 取 "zero" 时退化为 Dijkstra
)
# path: List[Tuple[int, int]]，无解时为 []
```

### 题目二：加权地形

```python
from lab2 import astar, terrain_cost

cost, path, stats = astar(
    grid, start, goal,
    cost_fn=terrain_cost,   # 数字=地形代价，'E' 的进入代价为 0
    heuristic="manhattan",
)
# cost: int，无解时为 None（题目要求的 -1 可在输出层自行映射）
```

`h_weight` 参数可以把启发式放大若干倍（例如 `h_weight=3.0`），用于 Weighted A* 对照实验；大于 1 时不保证最优。

### 题目三：Minimax / Alpha-Beta

```python
from lab2 import build_sample_tree, minimax, alpha_beta
from lab2.game_tree import GameStats

root = build_sample_tree()
stats = GameStats()
value = alpha_beta(root, is_max=True, stats=stats)
print(value, stats.as_dict())   # 5, 含 visits / leaves / cutoffs / max_depth
```

也可以用 `TreeNode.from_dict(题目给出的 JSON)` 直接构造任意树。

### 题目四：宝藏分配

```python
from lab2 import distance_matrix, solve

names, matrix = distance_matrix()    # 6×6 距离矩阵
result = solve(depth=4, algorithm="alpha-beta")
print(result.best_target, result.best_value)   # T3, 3
print(result.line)                             # 最优行棋序列
```

`depth` 是行动层数（H=2 表示 A、B 各走一次）；`algorithm` 可选 `"minimax"` 或 `"alpha-beta"`；`order` 可选 `"natural"`、`"value_desc"`、`"gain_desc"`、`"distance_asc"`。

## 实现要点

1. **代价函数与启发式**：A* 的启发式为 `h(n) = max(0, (d-1)·c_min + c_E)`。迷宫问题中退化为曼哈顿距离；加权地形中因为进入 E 的代价为 0，必须取 `d-1`，否则紧邻终点时会高估 1，破坏可采纳性。
2. **优先队列**：按 `(f, h)` 排序；f 相同时优先扩展 h 更小的节点，路径更直、扩展更少。允许重新打开节点，便于观察不可采纳启发式的行为。
3. **Alpha-Beta**：维护 `[alpha, beta]` 窗口，`alpha >= beta` 时剪枝；统计 `cutoffs` 次数，便于与 Minimax 比较。
4. **题目四缓存**：先用 A* 预计算 6 个关键格两两之间的最小代价，博弈过程中只查表，不再逐格寻路；状态为 `(p_A, p_B, 剩余宝藏, s_A, s_B, turn)`。

## 可复现性

- 随机实验固定种子：迷宫 11–14，加权地图 21–23，3× 启发式统计 100–129，随机博弈树按 `1000 + 种子 × 97 + …` 生成；
- 节点数、叶子数不受机器影响；耗时取 3 次运行的最小值，仅用于相对比较；
- `experiments/results/*.csv` 与报告中的表格一一对应，`figures/*.png` 对应报告插图。

## 常见问题

**`ModuleNotFoundError: No module named 'lab2'`**：请在 `lab2_search` 目录下运行命令，或使用 `python -m unittest discover -s tests -v` 从本目录启动。

**图表中文显示为方框**：`run_experiments.py` 会优先使用 Noto Sans CJK 字体；如在 Windows 上运行，可把 `plt.rcParams["font.sans-serif"]` 改为 `["Microsoft YaHei"]`。

**题目四的返回值为什么是 −1/3**：`best_value` 是 `U = Score_A − Score_B`，可以为负；题面的 H=2 结果就是 −1。
