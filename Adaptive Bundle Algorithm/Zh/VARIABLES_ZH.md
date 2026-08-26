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

---
---

# 第二部分——7月8日之后新增的旋钮(写于 2026-08-25)

上面的第一部分(第 1–3 节)覆盖原始引擎,原样保留。后来的代际
(`_fast`、`_ccp`、SVRG 基线和各 campaign runner)沿用其中仍适用的
旋钮,并新增下面这些。逐文件背景见 `CODE_MAP.md`;运行命令见
`MANUAL.md` 第二部分。

## 4. fast 引擎——`algorithm_fast_without_256_checkpoints.py` 的 `algorithm_adaptive_fast`

| 变量 | 含义 | 默认值 |
|---|---|---|
| `lambda_tier_mode` | λ-search 分档:`"strict"` = 每轮都付完整多起点搜索;`"two_tier"` = cheap 档(质心+顶点+上轮起点)加周期性 strict 校验 | `"strict"`——在 K=6 MLP 上唯一诚实的选择:two_tier 的 cheap 读数被证明低报约 2 倍且瞄错峰区(v3 诊断) |
| `lambda_max_starts` | strict 档起点数 | 64 |
| `cheap_tol` / `cheap_max_iter` | cheap 档求解容差 / 迭代上限 | 1e-4 / 30 |
| `strict_tol` / `strict_max_iter` | strict 档求解容差 / 迭代上限 | 1e-8 / 100 |
| `sticky_strict` | 一旦进入 strict 就保持 strict | True |
| `msvrg_step_const` | Momentum-SVRG 内环步长常数 | 0.1 |
| `msvrg_momentum` | 内环动量 β | 0.9(纯固定预算 runner 设 0.5) |
| `msvrg_epoch_len` | 每个 segment 的小批量步数(一个 segment = `epoch_len` 步 + 1 次全量 joint 评估) | None = 自动规则;K=6 纯预算 campaign 解析为 13 |
| `msvrg_max_segments` | 每外轮 segment 上限;撞上即健康标志 `cap_hits`(预算模式下接受,目标模式下表示内目标不可达) | 10 |
| `msvrg_trigger_rho` / `msvrg_trigger_patience` | 内环提前退出触发器(纯固定预算协议里已移除:segment 跑满) | 0.7 / 2 |
| `msvrg_rel_target` | 相对内目标(当轮搜索值的分数) | None(v4 探针 0.1;固定预算运行 0.05) |
| `prune_grid_r` | 交付时剪枝的探测 λ 网格分辨率(按位校验) | 10 |
| `epsilon`、`max_outer`、`max_grad_evals`、`eval_every_n_grads`、`require_ipopt` | 同第一部分 | 1e-3 / 150 / None / None / True |

`objectives_torch_fast.StochLamOracle`(小批量 oracle):`batch_size`
b,按类分层 ∝ n_k(campaign 标准 b = 4096;MNIST 运行用 1024),
`seed`。本代际统一的梯度当量记账:一次全量 joint 调用 = K 单位;
一步小批量 = 2·b·K/n 单位;x0 与度量/审计工作不上轴。

## 5. SVRG 认证基线——`baseline_svrg_certified_…` 经 `run_baseline_svrg_r_sweep_…` 调用

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--r-list` | 扫描的网格分辨率,逗号分隔 | `10,12,15,20` |
| `--node-tol` | 每节点认证水平(‖∇F_{λ_i}‖²) | 0.02(tol0.01 目录用 0.01) |
| `--solve-target-frac` | 内解目标 = node_tol 的这个分数 | 0.25 |
| `--share-mode` | 节点间证书共享方式 | `"gram"`(基于 Gram 的共享;节点可全部由共享签发) |
| `--ckpt-every-grads` | 检查点节奏 | 4,500 |
| `--max-wall-per-r` / `--max-grads-per-r` | 每个分辨率的保险丝 | 14,400 秒 / 2e6 |
| `--save-grams` | 存 `delivery_audit.npz`(逐节点 Gram,供节点间隙审计) | 关 |
| `--out-dirname` | 输出目录覆盖(v2 目录就是这样建的) | 旧目录 |
| `--fast-ref` | 作为参考曲线画上的自适应 `summary.json` 路径 | None |
| `--replot` | 只用已存 summary 重画图,不运行 | 关 |

## 6. 纯固定预算协议——`run_pure_budget_{K6,K2}(_ccp)_…`、`run_fixed_budget_K6_…`

协议里**没有任何容差参数**。共享旋钮:

| 参数 | 含义 | 正式取值 |
|---|---|---|
| `--run` | 跑哪条腿:`baseline` 或 `adaptive`(一次调用一条腿,串行) | — |
| `--budget` | 梯度当量总预算 B | K6:80,912(= r15@0.02 的实际成本);K2 与 MNIST 数字对:20,000 |
| `--s` | 每次分配决策连续花费的 segment 数 | 主跑 5;敏感性腿 1 |
| `--r` | 基线网格分辨率 | K6:10/12/15/20;K2:10/20/40/80;数字对:10/20/40 |
| `--targeting-starts` | 自适应 worst-λ 搜索的起点数(它的决策策略) | K6:24;K2:64(ts64 腿)与 24(ts24 腿) |
| `--eval-every` | 检查点节奏(梯度当量) | K6:2,000;K2:250 |
| `--backfill-audits` | 给已完成的基线腿补 strict 64 起点前缀审计(不低报合并) | — |
| `--figure` / `--replot` | 用已存数据重画 campaign 图 | — |
| `--force` | 允许覆盖已完成的腿 | 关 |

K=2 精确质量计附加项(`run_pure_budget_K2_…`):`--decision-mode`、
`--decision-grid`(默认 2,001)、`--audit-grid`(默认 200,001)——
K=2 时单纯形是 1 维,质量用精确密集网格测量,任何测量都不含多起点
搜索。CCP 腿(`run_pure_budget_K2_ccp_…`)只换下一个 λ 的策略,
新增 `--ccp-N0`(2,000)、`--ccp-r`(10)、`--ccp-seed`(0)。

`run_fixed_budget_K6_…`(协议 5e,前身):`--budget 80912`、
`--rel-target 0.05`、`--targeting-starts 24`、`--eval-every 2000`、
`--max-outer 5000`、`--tag`。

## 7. CCP λ 求解器——`ccp_lambda_solver.CCPConfig`

| 字段 | 含义 | 默认值 |
|---|---|---|
| `N0` | 每轮采样的随机种子数(静态模式) | 2,000(重型审计仪器:8,192) |
| `r` | 每轮做 CCP polish 的 restart 数 | 10(重型审计:20) |
| `pool_cap_factor` | 跨轮 pool 上限 = 系数 × r | 3 |
| `tau_rel` | CCP restart 的相对平稳性容差 | 1e-8 |
| `tau_eps_frac` | tau 的安全上限 0.01×epsilon(很少起作用) | 0.01 |
| `T_max` | 每个 restart 的 CCP 迭代上限 | 100 |
| `seed_sampler` | 随机种子分布:`"exp"`(Exp(1) 归一化)或 `"sobol"`(scrambled) | `"exp"`——研究 A 未发现显著差异,保留 exp |
| `adaptive_seed_schedule` | 按 rho 规则收缩 N0(消融开关) | False |
| `n_new_floor_factor` | 开启调度时的收缩下限 = 系数 × r | 10 |
| `rho_low` | 调度带边界 | 0.25 |
| `screen_sep_l1` | 留用种子间强制的 l1 间隔 | 0.05 |
| `dedup_l1_tol` / `dedup_phi_rel` | pool 去重:同点 l1 / phi 接近度 | 1e-3 / 1e-9 |
| `active_tol` | 宽容 active-set 阈值(相对) | 1e-9 |
| `collapse_frac` | pool 塌缩触发分数 | 0.5 |
| `seed` | rng / Sobol 扰动种子 | 0 |
| `use_highspy` | 强制/禁止用 HiGHS 解 game LP | None = 自动探测 |

## 8. bandit toy runner——`run_bandit_toy{,_K5,_mv}_without_256_checkpoints.py`

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--epsilon` | 精度档(已记录档位:1e-2、1e-3、1e-4;mv:1e-2、1e-3) | 1e-2 |
| `--eval-every` | 检查点节奏(梯度);**0 = 逐 segment 记录**,即精确读数模式(会话 13)——粗节奏的首次到达读数是上界伪影 | 10 |
| `--max-grad-evals` / `--max-wall` | 保险丝 | 200,000 / 3,600 秒(K5:7,200 秒) |
| `--smoke` | 小规模运行,写进 `smoke/` | 关 |
| 仅 mv:`--gamma` | 方差权重(杀死闭式解的旋钮) | None = 取 `gamma_scan.json` 记录值 |
| 仅 mv:`--gamma-scan`、`--gamma-list` | 扫描 γ 候选 | — |
| 仅 mv:`--rebuild-reference` | 重建不计时的多起点真值表(`reference_gamma1_*.npz`) | — |
| 仅 mv:`--epsilons` | 一次调用跑多个档 | None |

bandit 专属语义:equal-level stop 赋予基线一个其原生理论不承诺的
终点全局性质——**绝不**引用终点 GN\* 作为基线覆盖性的证据;覆盖
弱点由 value/PF 指标和审计体现。

## 9. MNIST runner

`run_ccp_compare_K10_mnist_…`(K=10 patch-softplus):`--budget`
55,000、`--eval-every` 1,500、`--per-class` 1,000(每类前 N 张训练
图)、`--batch` 1,024、`--s` 5、`--ts` 24(IPOPT 腿的 strict 起点
数)、`--ah16-faithful`(消融开关)、`--ccp-seed` 0。问题族:
`objectives_mnist_patch.py`(patch 局部连接 softplus MLP,d = 8,874)。

`run_pure_budget_K2_mnist_pair_…`(实验四):`--pair a b`(两个数字;
正式 campaign 为 3 5 与 7 9)、`--budget` 20,000、`--eval-every`
250、`--audit-grid` 200,001、`--s` 5、`--ccp-seed` 0。per_class 取
该对的平衡最大值(3v5 为 5,421);batch 1,024;test 值用两个数字的
全部官方 t10k 行。问题族:`objectives_mnist_pair.py`(d = 8,098)。

## 10. 审计仪器(不属于单次运行的旋钮)

- 全族通用仪器:strict 64 起点 λ-search。
- 八月 campaign:每个交付 stack 记
  `audit_v2 = max(strict-64 IPOPT, 重型 CCP N0=8,192、r=20、全新求解器)`
  (`audit_v2_K6_…py`,`--quick` = 每腿前 3 个 stack)——两者都是
  NP-hard 最大化的下界,取 max 得到更紧且方法对称的下界。
- 凡审计承担结论处,应用单调下界包络(前缀 GN\* 不增;原始值一并
  保留)。
- 时间记账:自适应的 λ-search 时间**上** CPU 轴(它驱动算法);
  检查点度量与审计工作**不上**任何轴(summary 里的
  `metric_seconds` / `audit_seconds`)。

## 11. 新旋钮里哪些最重要

| 排名 | 旋钮 | 原因 |
|---|---|---|
| 1 | `--budget` B | 整场对决在固定 B 下定义;每个结论都是"在此预算下" |
| 2 | `--s` | 覆盖塌缩杠杆:K=6 时 s=5 让基线网格只访问 r=10/12/15/20 的 39%/19%/7.5%/2.2% 节点 |
| 3 | `--r` | 基线的预算旋钮(节点数 C(r+K−1, K−1))和质量地板 |
| 4 | `--targeting-starts` / CCP 的 `N0`、`r` | 决策质量,同时是 CPU 轴上的决策成本 |
| 5 | batch b | 内解器地板:b=4,096 的证据是目标 ≲0.01 时从部分锚点撞 segment 上限;设计好的下一杠杆是 8,192/16,384 或 b=n |
| 6 | `lambda_tier_mode` | 在 MLP 族上 strict 是唯一诚实的测量模式 |
| 7 | `--eval-every` | 只影响 wall-clock(不上轴),但等待时间是真实的 |
