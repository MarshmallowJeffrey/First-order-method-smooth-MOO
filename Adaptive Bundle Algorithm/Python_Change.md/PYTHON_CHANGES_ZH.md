# Python 代码改动记录

日期：2026 年 7 月 1 日

## 说明

- 本文只记录聊天中实际执行的代码改动，不介绍各文件原本已有的功能。
- `algorith-main.py` 后来由用户重命名为 `algorithm.py`，`experiments-main.py` 后来由用户重命名为 `experiments.py`。这两个重命名不属于代码逻辑修改，本文统一使用当前文件名。
- 已撤销的临时修改不计入最终改动。

## `algorithm.py`

### 删除

- 删除未被调用的 `_T_map_grid_batched`，只保留算法实际使用的 `_T_map_batched`。
- 删除对 `bundle.py` 中 `T_map` 和 `GN` 的导入。
- 删除 `_bundle_update_adaptive` 的 `pc_fn` 参数，以及 `pc_fn = GN` 和相关传参。
- 删除 `algorithm_adaptive` 和 `pc_star` 中无实际作用的 `mode` 参数。
- 删除 `_pc_star_metric` 别名，以及 GAP、`_maximise_GAP`、`bundle.mu` 等旧版本残留说明。
- 从本文件删除 `bundle_from_points`；该函数已移动到 `baseline.py`。
- 从 `_gn_value_and_jac_batched` 删除未使用的 `Fmat` 和 `L` 参数。

### 新增和修改

- 将 `_gn_value_and_jac_batched` 的接口改为：

  ```python
  _gn_value_and_jac_batched(Jmat, lam)
  ```

  同步更新调用点和文档，删除不存在的强凸缩放公式，并补充并列最小值处的不可微说明。
- 新增 `_gn_value_batched(Jmat, lam)`，只计算 GN 标量。内循环停止检查直接复用 `Jbuf`，不再额外计算未使用的权重梯度。
- 将算法选择器的上一权重和 checkpoint 指标的上一权重拆分为 `selector_prev_lam` 与 `metric_prev_lam`，避免 checkpoint 频率影响算法选择结果。
- 局部 IPOPT/SLSQP 求解失败时，警告信息现在包含失败的初始权重和异常原因，然后继续尝试其他起点。
- `algorithm_adaptive` 入口会自动调用 `prefer_fused_joint_oracle`，直接调用算法时也会优先使用 `joint.fused`。
- 新增 `ipopt_available()`，同时捕获 IPOPT 包缺失和动态库加载失败。
- `algorithm_adaptive` 新增 `lambda_solver` 和 `require_ipopt`：可以明确选择 IPOPT/SLSQP，并可禁止 IPOPT 静默回退。
- `algorithm_adaptive` 新增 `max_grad_evals`，达到梯度评估预算后停止，并在结果中返回实际使用的 lambda 求解器和预算；预算不能被 `K` 整除时也会保存最终 checkpoint。

## `bundle.py`

### 删除

- 删除未被调用的循环版 `T_map`。
- 删除不再使用的循环版 `GN`。

### 新增

- 新增共享函数 `prefer_fused_joint_oracle`：
  - 存在 `joint_oracle.fused` 时优先使用 fused 版本；
  - 不存在 fused 版本时使用普通 `joint_oracle`；
  - fused 版本抛出 `MemoryError` 或 `NotImplementedError` 时警告一次，并永久回退到普通版本。

## `baseline.py`

### 删除

- 删除无实际数据的 `worst_errs` 列表。
- 删除固定为 `NaN` 的 `err` 变量和返回结果中的 `"worst_errs"`。
- 删除关于旧 suboptimality、`mu_lams`、A2 和 NAG 的过时注释。
- 删除不再需要的 `Tuple` 类型导入。

### 新增和修改

- 将 `coverage_mode` 改为布尔参数 `evaluate_coverage`；只有为 `True` 时才构造临时 Bundle 并计算 GN*。
- 将 `bundle_from_points` 从 `algorithm.py` 移到本文件，因为它只用于 Baseline checkpoint。
- 保留 `_nearest_coarse_index`，并在文档中明确标注它当前未被调用。
- `uniform_discretisation` 和 `bundle_from_points` 会自动调用 `prefer_fused_joint_oracle`。
- 新增 `max_grad_evals`，Baseline 会在下一步超过预算前停止；即使在一轮网格遍历中途停止，也会保存最终 checkpoint，并把预算写入返回结果。

## `experiments.py`

### 删除

- 删除未使用的 `time` 导入。
- 删除绘图和算法调用中的 `mode="gn"`。
- 删除旧的 `coverage_mode="gn"`，改用 `evaluate_coverage=True`。
- 删除未使用的 `checkpoint_every` 参数。
- 删除实验文件内临时实现的 `_prefer_fused_joint_oracle`，改用 `bundle.py` 中的共享实现。

### 新增和修改

- Baseline 和 Adaptive 现在接收同一个经过 fused 优先处理的 `joint_oracle`。
- `_plot_coverage` 固定绘制 GN*，不再保留无效的 GN/GAP 模式分支。
- 新增 `detect_plateau`，使用连续、不重叠的 checkpoint 窗口检测 GN* 平台起点和平台高度。
- 新增 `_plot_plateau_pair`，绘制以下三张两两对比图：
  - Baseline 与 Sample Adaptive；
  - Baseline 与 IPOPT Adaptive；
  - Sample Adaptive 与 IPOPT Adaptive。
- 新增 `experiment_mlp_plateau_comparison`：三种方法共享同一问题、初始点、oracle 和梯度预算；Adaptive 的 `target_cov` 设为 `None`，避免达到 Baseline 最终值后提前停止。
- Plateau 实验要求 IPOPT 确实可用，不允许结果实际来自 SLSQP 却标记为 IPOPT。
- Plateau 实验返回每种方法的平台信息、平台高度比值和三张图的路径。

## `objectives.py`

### 删除

- 删除未使用的 PyTorch 导入及全局 PyTorch dtype/thread 设置；该文件现在只使用 NumPy。

### 新增和修改

- 在逻辑回归工厂函数旁注明：当前 `experiments.py` 未调用它，仅保留给未来强凸实验。
- `_sample_planted_data` 新增 `K > 0`、`p > 0` 和 `n >= K` 检查。
- 数据包含空类别时重新采样，最多尝试 1,000 次；仍失败则抛出 `RuntimeError`。
- 将累计类别概率的最后一列强制设为 `1.0`，避免浮点误差导致逆 CDF 采样失败。
- 类别数量使用真实样本数，不再用 1 替代空类别计数。
- `make_mlp_nonconvex` 新增 `h > 0` 检查。
- 修正 MLP 文档示例，使其解包四个返回值，并补充 `joint_oracle` 的返回说明和数组形状。

## 验证

- 五个 Python 文件均通过字节码编译检查。
- 已验证 GN 标量实现与带 Jacobian 的实现数值一致。
- 已验证 fused、普通 oracle 和 fused 失败回退路径。
- 已验证输入参数检查、空类别重采样、联合 oracle 输出形状及 Plateau 检测的基本行为。
- 当前本机 `cyipopt` 动态库加载失败，因此完整三方法 Plateau 实验尚未运行；严格 IPOPT 检查已验证会明确报错，不会回退后误标为 IPOPT。
