# 变量参考:每个旋钮在哪个文件、默认值是什么、哪些真正重要

日期:2026 年 7 月 8 日。

## 1. 各部分怎么衔接

```
objectives_torch.py     生成问题:数据、K 个目标、梯度、
 (objectives_numpy.py)  光滑常数估计 L、参数个数 d
        |
        v
baseline.py             算法 1:权重网格 + 梯度下降
algorithm.py            算法 2:自适应束方法
        |
        v
experiments.py          在同一个问题、同一预算下跑两种方法,
                        打检查点、检测平台、画图
        |
        v
run_experiments.py      两个扫描的正式参数值;写 summary.json /
                        README / 趋势图
```

`*_without_256_checkpoints.py` 是 `baseline.py`/`algorithm.py` 的
A/B 测量变体,旋钮完全相同,只有检查点记录的内容不同。

---

## 2. 逐文件清单

### 2.1 问题生成——`objectives_torch.py` 的 `make_mlp_nonconvex`

| 变量 | 含义 | 默认值 |
|---|---|---|
| `K` | 目标(类别)个数 | 3 |
| `p` | 数据的输入维度 | 4 |
| `n` | 数据样本数 | 60 |
| `hidden_sizes` | MLP 各隐藏层宽度,如 `[4]` 或 `[96, 96]` | `[8]`(可用简写 `h`:`h=4` 等于 `[4]`) |
| `seed` | 植入数据(W*、X)的随机种子 | 7 |
| `w_true_scale` | 植入 W* 的均匀分布半宽:元素 ~ U[−w, w] | 1.0(即论文的 U[−1,1]) |
| `n_probes` | 计时开始前,每个目标用来**估计**光滑常数 L 的探针对数 | 40 |
| `activation` | 隐藏层激活函数:`"relu"`、`"tanh"`、`"softplus"`、`"identity"` | `"relu"`(基准实验传 `"tanh"`——ReLU 违反论文的光滑性假设) |

派生量,不能直接设置:`d` = MLP 的参数总数(随 `p` 和 `hidden_sizes`
增长;例如 K=5、p=6 时 `[4]` 给 d=53;K=2、p=20 时 `[96,96]` 给
d=11,522)。

`objectives_numpy.py` 里有 `make_logreg_strongly_convex(K=5, p=4, n=60,
reg=0.1, seed=42, w_true_scale=1.0)`——强凸逻辑回归测试台,**只被
验证脚本使用**,实验从不用它。

`experiments.make_mlp_initial_point(K, p, h=None, seed=0, hidden_sizes=None)`
生成共享初始点 x0(基准实验传 seed 8)。

### 2.2 基线——`baseline.py` 的 `uniform_discretisation`

必填输入:`K`、目标与梯度、`L`、`x0`、`resolution`。

| 变量 | 含义 | 默认值 | 基准值 |
|---|---|---|---|
| `resolution`(r) | 权重单纯形上的网格密度;网格共 C(r+K−1, K−1) 个节点 | (必填) | 6(平台)、10(交叉) |
| `n_passes` | 最多允许扫过整个网格多少遍 | 1 | 100,000(从不触发停止) |
| `steps_per_point_per_pass` | 每遍在每个节点上走的梯度下降步数 | 20 | 5 |
| `eval_every_n_grads` | 每这么多次梯度求值打一个检查点(None = 每遍一次) | None | 预算/25(平台)、预算/13(交叉) |
| `max_grad_evals` | 梯度求值总数的硬预算 | None | 30,000 / 2,000 |
| `node_tol` | **可选**认证模式:逐节点的 ‖∇F_{λ_i}‖² 验收标准;None = 预算模式 | None | None(未使用) |
| `evaluate_coverage` | 检查点是否测量 GN* | False | True |
| `joint_oracle` | 融合求值器,一次调用返回全部 K 个值+梯度(提速手段,不改变语义) | None | torch 融合求值器 |
| `verbose` | 是否打印进度 | False | True |

值得了解的内部状态:`node_served` 与 `node_grad_sq`(认证模式的
记账;在 A/B 变体里 `node_grad_sq` 还兼作自报的检查点值)。

### 2.3 自适应方法——`algorithm.py` 的 `algorithm_adaptive`

必填输入:`K`、`d`、目标与梯度、`L`、`x0`。

| 变量 | 含义 | 默认值 | 基准值 |
|---|---|---|---|
| `max_outer` | 外层轮次(λ 搜索 + 内层步)上限 | 120 | 1,000,000(从不触发停止) |
| `max_inner` | 每个外层轮次的 T 映射步数上限;每步花 K 次梯度求值 | 25 | 25(平台)、5(交叉) |
| `epsilon` | **可选**认证模式:方法自己的搜索值 ≤ 2ε/3 时外层停止;内层目标 ε/3 | None | None(未使用) |
| `eval_every_n_grads` | 检查点节奏(None = 每个外层轮次一次) | None | 预算/25、预算/13 |
| `target_cov` | **可选**:检查点测量值 ≤ 此目标即停(遗留的到目标时间驱动使用) | None | None |
| `lambda_max_starts` | 方法**自己**每轮 λ 搜索的多起点数 | 256 | 64(平台)、8(交叉) |
| `lambda_solver` | λ 搜索用 `"ipopt"` 还是 `"slsqp"` | `"ipopt"` | `"ipopt"` |
| `require_ipopt` | 缺 IPOPT 时拒绝运行(而不是悄悄退到 SLSQP);默认 True 后,显式选 `lambda_solver="slsqp"` 的运行必须同时传 `require_ipopt=False` | True(7 月 8 日前为 False) | True |
| `max_grad_evals` | 梯度求值总数的硬预算 | None | 30,000 / 2,000 |
| `prune_inner` | 只把最好的内层候选留进束(论文第 5 节说明);被剪掉的候选仍计入预算 | True | True |
| `joint_oracle`、`verbose` | 同基线 | None / False | 融合 / True |

值得了解的内部状态:

- `L_scale`——下降引理保护机制对 L 的乘数。从 1 开始,每次"可证下降"
  不等式失败就加倍,以 `L_scale_final` 报告,超过 2^60 则中止运行
  (说明目标沿迭代路径不是 L-光滑的)。这是一个**健康标志**:2–16 是
  预期的偶尔纠正区间。
- `m`——束的大小(存储的点数)。`prune_inner=True` 时每个外层轮次
  增加 1,False 时每轮最多增加 `max_inner`。
- 固定常数,**有意不做成旋钮**:检查点度量 `pc_star` 永远用 256 个
  起点,与 `lambda_max_starts` 无关,以保证它是跨运行、跨方法的
  同一把可比尺子。

### 2.4 实验驱动——`experiments.py`

`experiment_mlp_plateau_comparison` 把上面所有东西组装成一次对照
运行。它自己的默认值(开发用):K=3、p=4、n=60、seed=10、
init_seed=None(意为 seed+1)、coarse_resolution=10、n_passes=1000、
steps_per_point_per_pass=10、baseline_eval_every_n_grads=None、
adaptive_eval_every_n_grads=2000、max_grad_evals=30000、
max_outer=10000、max_inner=25、lambda_max_starts=256、
prune_inner=True、hidden_sizes=None、activation="relu"、
w_true_scale=1.0,外加下表的平台检测旋钮。基准实验覆盖其中大多数
(见 2.5 节)。

平台检测器(`detect_plateau`)——只做测量,从不改变轨迹:

| 变量 | 含义 | 默认值 | 基准值 |
|---|---|---|---|
| `plateau_window` | 每个稳定窗口包含的检查点数 | 5 | 5(平台)、4(交叉) |
| `plateau_relative_improvement_tol` | 改进小于这个比例算"平" | 0.05 | 0.05 |
| `plateau_consecutive_windows` | 需要连续多少个平窗口 | 2 | 2 |

同文件还有 `experiment_mlp_gn_coverage`——**遗留的**到目标时间驱动
(基线跑满全部日程,其最终值作为自适应方法的 `target_cov`)。设计
不对称;不用于任何主要数字;保留用于快速单配置比较。

### 2.5 正式取值——`run_experiments.py`

常量:`DATA_SEED = 7`、`INIT_SEED = 8`(所有基准运行完全一致)。

平台扫描(`plateau_configs`):K ∈ {3,4,5,6},p=6,n=30,
hidden_sizes=[4],activation="tanh",r=6,预算 30,000,检查点间隔
1,200,steps_per_point_per_pass=5,max_inner=25,lambda_max_starts=64,
prune_inner=True,n_passes=100,000,max_outer=1,000,000。

交叉扫描(`crossover_configs`):K=2,p=20,n=50,000,
hidden_sizes ∈ {[16,16],[32,32],[64,64],[96,96],[128,128]},
activation="tanh",r=10,预算 2,000,检查点间隔 153,
steps_per_point_per_pass=5,max_inner=5,lambda_max_starts=8,
prune_inner=True,plateau_window=4。

`--smoke` 把两个扫描各跑一个缩小版,写进独立的 `*_smoke` 输出文件夹。

---

## 3. 哪些变量最重要

### 3.1 到底是什么让一次运行停下来

一次运行在下列条件中**最先触发**的那个处结束。所有正式实验里,
真正起作用的都只有第 1 条——这正是等预算设计。

基线:

1. `max_grad_evals`——梯度预算。**所有基准运行里实际的停止条件。**
2. `n_passes` 用尽——基准设为 100,000,实际永远碰不到。
3. `node_tol` 全部节点达标(认证模式)——默认关闭;开启时
   `max_grad_evals` 保留为保险丝,预算内认证失败会如实报告。

自适应方法:

1. `max_grad_evals`——梯度预算。**所有基准运行里实际的停止条件。**
2. `max_outer` 用尽——基准设为 1,000,000,永远碰不到。
3. `epsilon`(认证模式)——方法自己的 λ 搜索值 ≤ 2ε/3 即停;
   基准全部关闭。
4. `target_cov`——检查点测量值到目标即停;只有遗留驱动使用。

另有两个上限只在运行**内部**起作用、不停止运行:`max_inner` 结束
一个内层循环(回去重新做 λ 搜索),`steps_per_point_per_pass` 结束
对一个节点的一次访问(去下一个节点)。

### 3.2 什么主导成本和运行时间

先用平实语言给出成本模型:

- 一次梯度求值的代价约为 n·d(对整个数据集一次前向+反向)。
  一个标量化步花 K 次梯度求值。
- **总梯度工作量 = max_grad_evals × 单次求值的代价。**两种方法按设计
  共享这一项。
- 基线除此之外几乎不付任何东西。它的结构旋钮是网格:
  N = C(r+K−1, K−1) 个节点决定预算摊得多薄(以及质量地板在哪里)。
- 自适应方法每个外层轮次付**额外的 CPU**(不付额外梯度):λ 搜索
  最多跑 `lambda_max_starts` 次局部求解,求解器每迭代一步约花
  m·K·d 的代数运算(m = 束大小);T 映射选点每步同量级。一个预算内
  的外层轮次数约为 max_grad_evals / (K · max_inner);
  `prune_inner=True` 时 m ≈ 轮次数。
- 检查点只烧墙钟(不计入两条报告轴,但等待是真实的):每个检查点跑
  一次固定 256 起点的度量求解,同样每迭代步约 m·K·d。检查点个数 =
  max_grad_evals / eval_every_n_grads。(`experiment_mlp_gn_coverage`
  里的注释记录了真实事故:检查点加密 10 倍反而让总墙钟**变差**,
  因为每个检查点都要付那次 256 起点求解。)

按实际影响力排序:

| 名次 | 变量 | 为什么排在这里 |
|---|---|---|
| 1 | `n`、`hidden_sizes`/`p`(即 n·d) | 决定**每一次**梯度求值的价格,两种方法都受影响;交叉扫描扫的正是这个价格 |
| 2 | `max_grad_evals` | 乘在一切上面;还决定自适应优势是否可见(K=6 的预算教训) |
| 3 | `K` | 三重效应:每步花 K 次求值、基线网格按 C(r+K−1,K−1) 爆炸、λ 搜索在 K−1 维空间里进行 |
| 4 | `resolution` r | 基线节点数(关于 r 是 K−1 次多项式)及其质量地板 |
| 5 | `max_inner` | 决定给定预算下的外层轮次数,进而决定束的增长和要付多少次 λ 搜索 |
| 6 | `prune_inner` | 束大小 m 乘在自适应方法**所有**代数运算上(λ 搜索、T 映射、度量);True 使 m ≈ 轮次数,False 让它快 max_inner 倍地增长 |
| 7 | `lambda_max_starts` | 自适应方法每轮 CPU 开销的线性乘数(从不影响梯度) |
| 8 | `eval_every_n_grads` | 只影响墙钟,途径是 256 起点度量求解的次数 |

改变**语义**而非成本的开关:`epsilon`、`node_tol`、`target_cov`
(预算模式 → 认证/目标停止);`activation`(tanh 保持光滑性假设成立;
ReLU 破坏它,不能用于基准);`seed`/`init_seed`(决定你在哪个问题
实例上)。

速查——"运行太慢,该动哪个旋钮?":

- 检查点处等太久 → 调大 `eval_every_n_grads`(度量求解变少;
  图变粗糙,轨迹不变)。
- 大 d 时自适应每轮太慢 → 调小 `lambda_max_starts`、保持
  `prune_inner=True`,或调大 `max_inner`(同预算下搜索次数变少)。
- 全都太慢 → 真正的杠杆是 `n`、`hidden_sizes` 和 `max_grad_evals`;
  其他旋钮都改变不了梯度账单。
