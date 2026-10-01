# Plateau 实验实现改动

日期：2026 年 7 月 1 日

## 目标

新实验比较以下三种方法达到稳定状态后的 GN* 表现：

1. 均匀离散化 Baseline。
2. 基于 Sample 的 Adaptive Bundle 方法。
3. 基于 IPOPT 的 Adaptive Bundle 方法。

主要比较横轴是总梯度评估次数。GN* 和 Plateau 高度越低越好。

## 修改的文件

### `algorithm.py`

- 新增 `ipopt_available()`，使实验可以在开始高成本运行之前检查 IPOPT 后端。
- 扩展 IPOPT 导入异常处理，同时识别包未安装以及二进制文件或动态库损坏的情况。
- 新增 `lambda_solver`，用于明确选择 `"ipopt"` 或 `"slsqp"`。
- 新增 `require_ipopt`。启用后，如果 IPOPT 不可用，程序会立即报错，不再静默回退到 SLSQP。
- 新增可选的硬性梯度评估预算 `max_grad_evals`。
- 当剩余预算少于 `K` 次梯度评估时，也会保存最终状态的 checkpoint，包括预算不能被 `K` 整除的情况。
- 在返回结果字典中新增实际使用的选择器、求解器后端和梯度预算。
- 新增对非法选择器、求解器、严格 IPOPT 配置和预算组合的提前检查。

### `baseline.py`

- 新增可选的硬性梯度评估预算 `max_grad_evals`。
- 梯度下降循环会在下一步超过预算之前停止。
- 即使在一次单纯形网格遍历中途达到预算，也会保存最终 checkpoint。
- 在返回结果字典中新增已配置的梯度预算。

### `experiments.py`

- 新增 `detect_plateau(...)`。
- 新增以总梯度评估次数为横轴的两两 Plateau 对比图。
- 新增 `experiment_mlp_plateau_comparison(...)`，在相同的 MLP 问题、初始点、平滑性估计和 fused oracle 上运行三种方法。
- 两个 Adaptive 实验均传入 `target_cov=None`，关闭达到 Baseline 最终值后的提前停止。
- 在运行任何高成本方法之前，严格检查 IPOPT 是否可用。
- 新增三张输出图：
  - `baseline_vs_sample.png`
  - `baseline_vs_ipopt.png`
  - `sample_vs_ipopt.png`
- 新增 Plateau 高度比值和控制台汇总表。

## Plateau 定义

检测器默认参数为：

```python
plateau_window = 5
plateau_relative_improvement_tol = 0.05
plateau_consecutive_windows = 2
```

检测器首先构造历史最优 GN* 曲线：

```python
best_so_far = np.minimum.accumulate(cov_history)
```

对于每个可能的 Plateau 起点，检测器检查两个相邻且不重叠的区间，每个区间包含五个 checkpoint。每个区间的改善幅度都必须低于 5%。从候选起点到运行结束的总体历史最优改善幅度也必须低于 5%，从而避免把前期短暂的平缓区间误判为最终 Plateau。

返回的字典包含：

```python
{
    "found": bool,
    "onset_index": int | None,
    "onset_grad_evals": int | None,
    "onset_cpu_time": float | None,
    "plateau_level": float | None,
}
```

`plateau_level` 是从检测到的起点到运行结束之间，原始 GN* 数值的中位数。

## 实验输出

`experiment_mlp_plateau_comparison(...)` 返回：

```python
{
    "baseline": ...,
    "sample_adaptive": ...,
    "ipopt_adaptive": ...,
    "plateaus": ...,
    "plateau_ratios": ...,
    "plots": ...,
    "detector_options": ...,
    "max_grad_evals": ...,
}
```

报告的比值为：

```text
Baseline Plateau / Sample Adaptive Plateau
Baseline Plateau / IPOPT Adaptive Plateau
Sample Adaptive Plateau / IPOPT Adaptive Plateau
```

比值大于 1 表示分子对应方法的 Plateau 更高，表现更差。

## 默认运行配置

新实验的默认参数为：

```python
K = 3
p = 4
n = 60
h = 8
max_grad_evals = 30000
coarse_resolution = 10
steps_per_point_per_pass = 10
adaptive_eval_every_n_grads = 2000
```

Baseline 默认在每次完整 pass 后保存 checkpoint。两个 Adaptive 方法使用相同的梯度预算和 checkpoint 间隔。Sample 方法默认在每次外层权重选择时使用 512 个随机单纯形样本，IPOPT 方法使用结构化多起点优化器。

## 运行方法

显式调用新实验函数：

```python
from experiments import experiment_mlp_plateau_comparison

result = experiment_mlp_plateau_comparison(
    output_dir="plateau_results",
)
```

现有脚本的直接运行入口保持不变，避免意外启动成本较高的三方法实验。

## 当前 IPOPT 环境状态

代码已正确识别当前 IPOPT 后端不可用。原因是已安装的 `cyipopt` 扩展在加载时缺少 IPOPT 符号：

```text
symbol not found in flat namespace '_AddIpoptIntOption'
```

因此，新 Plateau 实验目前会立即给出明确错误，而不会把 SLSQP 回退结果标记为 IPOPT。修复本机 IPOPT/`cyipopt` 安装后，才能运行完整的三方法实验。

## 已完成的验证

- 所有修改过的 Python 文件均通过字节码编译检查。
- Plateau 检测器能够在已知的合成 Plateau 曲线上找到预期起点。
- 对持续改善的合成曲线，检测器会正确返回 `found=False`。
- 两两 Plateau 绘图能够生成有效图片。
- Baseline 和 Sample Adaptive 测试运行会在指定梯度预算处停止。
- 严格 IPOPT 模式会拒绝当前损坏的后端，不会回退到 SLSQP。
