# SURF Offline-Bandit Toy 与 λ-Bundle 方法的结合说明

## 1. 结论先行

这个 toy example 很适合用来比较本项目的两个算法：

- Algorithm 2: λ-bundle adaptive algorithm；
- Algorithm 1: uniform discretization baseline。

原因不是它很复杂，而是它同时提供了：

1. 一个与两篇论文完全一致的、可重复的双目标问题；
2. 一个由闭式 softmax 解给出的高精度 Pareto-front oracle；
3. 一个可以把“统计误差”“优化误差”“前沿覆盖误差”分开的干净测试台。

但必须区分两个论文目标：

- SURF 解决的是“怎样选择权重，才能让有限个点沿 PF 弧长均匀分布”；
- λ-bundle 解决的是“怎样复用一个 bundle 的函数值和梯度，使所有 λ 的标量化问题都达到统一的一阶驻点精度”。

因此，SURF 的 CDF 误差和本项目的 ε-stationarity 不是同一个 ε，实验中不能混为一个指标。

## 2. SURF 论文中的理论

### 2.1 从权重到 Pareto front

SURF 首先考虑双目标线性标量化

\[
\pi_w^* \in \arg\min_\pi \bigl[wF_1(\pi)+(1-w)F_2(\pi)\bigr],
\qquad w\in[0,1].
\]

当标量化解唯一且随 \(w\) 平滑变化时，

\[
f_{\mathrm{PF}}(w)
=
\bigl(F_1(\pi_w^*),F_2(\pi_w^*)\bigr)
\]

是一条由 \(w\) 参数化的 Pareto-front 路径。均匀改变 \(w\) 并不意味着沿这条曲线匀速移动。

SURF 在第 2.1 节定义

\[
v(w)=\left\|\frac{\partial f_{\mathrm{PF}}(w)}{\partial w}\right\|_2,
\qquad
s(w)=\int_0^w v(u)\,du,
\qquad
\Phi(w)=\frac{s(w)}{s(1)}.
\]

其中：

- \(v(w)\) 是权重变化引起的 PF traversal speed；
- \(s(w)\) 是从 \(0\) 到 \(w\) 已走过的 PF 弧长；
- \(\Phi(w)\) 是归一化弧长 CDF。

若要得到 \(N+1\) 个等弧长点，应选

\[
w_n=\Phi^{-1}(n/N),\qquad n=0,\ldots,N.
\]

这就是 SURF Rule 1。它修正的是“uniform weight 不等于 uniform PF coverage”的几何错配。

原文位置：

- `SURF.pdf` 第 4-6 页：PF speed、arc length、CDF、bandit 闭式公式；
- `SURF.pdf` 第 49 页，Appendix F.1：offline-bandit toy 的数据设定。

### 2.2 Offline bandit 的具体问题

Appendix F.1 使用单状态 bandit，policy 与 normalized occupancy measure 相同：

\[
\pi\in\Delta_{|\mathcal A|}.
\]

对均匀分布在 \([0,1]\) 上的 \(x_a\)，真实奖励为

\[
R_1(a)=x_a,\qquad R_2(a)=1-x_a^4.
\]

离线数据集

\[
\mathcal D=\{(a_t,r_{1,t},r_{2,t})\}_{t=1}^{T}
\]

是 balanced 的：各 arm 出现次数大致相等；奖励观测加入标准差 \(0.5\) 的 Gaussian noise。用每个 arm 的样本均值构造 \(\widehat R_1,\widehat R_2\)。

KL-regularized minimization objective 为

\[
F_k(\pi)
=
\tau\,\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})
-\langle \pi,\widehat R_k\rangle,
\qquad k=1,2.
\]

对任意 \(w\)，精确标量化解为

\[
\pi_w^*(a)
\propto
\pi_{\mathrm{ref}}(a)
\exp\left(
\frac{w\widehat R_1(a)+(1-w)\widehat R_2(a)}{\tau}
\right).
\]

SURF 第 2.1 节的 Eq. (9) 给出

\[
v(w)
=
\tau^{-1}\sqrt{(1-w)^2+w^2}\,
(\widehat R_1-\widehat R_2)^\top
\left[\mathrm{Diag}(\pi_w^*)-\pi_w^*(\pi_w^*)^\top\right]
(\widehat R_1-\widehat R_2).
\]

对 \(v\) 积分并归一化可得 \(\widehat\Phi\)，再使用
\(\widehat\Phi^{-1}\) 选择等弧长权重。由于唯一的不确定性来自 reward-mean estimation，SURF 证明并在 Figure 5 验证

\[
\|\widehat\Phi-\Phi\|_\infty=O(T^{-1/2}).
\]

### 2.3 `uniform_PF.ipynb` 实际做了什么

导入的 notebook 使用：

- \(A=5\)；
- \(\tau=0.05\)；
- Gaussian noise standard deviation \(0.5\)；
- \(T=1000\)；
- 12 个最终 PF 点；
- 5000 个 dense weight nodes 用于数值积分和反演 CDF。

注意，代码中的 `N_target=12` 表示 12 个点，也就是 11 个 PF segments；论文 Rule 1 的记号是 \(N+1\) 个点。

核心函数的用途是：

| 函数 | 用途 |
|---|---|
| `build_offline_dataset` | 生成 balanced 固定离线数据 |
| `estimate_reward_means` | 估计 \(\widehat R_1,\widehat R_2\) |
| `softmax_policy` | 给出每个 \(w\) 的闭式最优 policy |
| `f_components_policy` | 计算两个 minimization objectives |
| `explicit_speed_from_policy` | 实现 SURF Eq. (9) |
| `one_shot_arc_length_weights` | 用 \(\widehat\Phi^{-1}\) 生成等弧长点 |
| `run_baseline_offline` | 用 uniform \(w\) 生成点 |

重要限制：

`run_baseline_offline` 不是本项目论文 Algorithm 1 的计算基线。它在每个权重直接调用闭式 softmax 解，只是 SURF 的“uniform-weight 几何基线”。本项目的 uniform-discretization baseline 必须在网格节点上运行与 adaptive method 相同的一阶 inner solver。

## 3. 本项目论文中的对应方法

### 3.1 Uniform discretization baseline

本项目论文第 3 节定义

\[
\mathcal G_r
=
\{\lambda\in\Delta_K:r\lambda\in\mathbb Z_+^K\},
\qquad
|\mathcal G_r|
=
\binom{r+K-1}{K-1}.
\]

Algorithm 1 在每个网格权重上运行 GD/SGD，并用邻近网格点的解回答任意 λ query。它的问题是：

- 每个节点基本独立求解；
- 在一个 λ 上得到的零阶和一阶信息没有系统地服务其他 λ；
- \(K\) 增大时，grid size 对 \(K\) 呈组合爆炸。

在本 toy 的 \(K=2\) 情况，

\[
|\mathcal G_r|=r+1.
\]

因此若要和 SURF notebook 的 12 个输出点对齐，应取 \(r=11\)。

### 3.2 Adaptive λ-bundle

本项目论文第 4 节维护

\[
\mathcal B_m
=
\{(\theta_i,F_1(\theta_i),\ldots,F_K(\theta_i),
\nabla F_1(\theta_i),\ldots,\nabla F_K(\theta_i))\}_{i=1}^m.
\]

对任意 λ，定义

\[
\mathrm{GN}(\lambda;\mathcal B_m)
=
\min_{\theta_i\in\mathcal B_m}
\|\nabla F_\lambda(\theta_i)\|_2^2,
\qquad
F_\lambda=\sum_k\lambda_kF_k.
\]

Algorithm 2 每个 outer iteration 选择

\[
\lambda_t
\in
\arg\max_{\lambda\in\Delta_K}
\mathrm{GN}(\lambda;\mathcal B_t).
\]

直观上，它寻找当前 bundle 中“最没有被服务好”的 preference，再在那里执行 `BundleUpdate`。因此它的自适应对象不是 PF 弧长，而是最坏 λ 的一阶驻点误差。

论文的 vanilla T-map 为

\[
i^*\in\arg\min_i
\left[
F_\lambda(\theta_i)
-\frac{1}{2L_\lambda}\|\nabla F_\lambda(\theta_i)\|_2^2
\right],
\qquad
T(\lambda;\mathcal B_m)
=
\theta_{i^*}-\frac{1}{L_\lambda}\nabla F_\lambda(\theta_{i^*}).
\]

理论上的 ε solution map 指标是

\[
\epsilon_{\mathrm{sm-stat}}(\widehat\theta)
=
\sup_{\lambda\in\Delta_K}
\|\nabla F_\lambda(\widehat\theta(\lambda))\|_2^2.
\]

Algorithm 2 使用的阈值结构是：

- outer worst case 小于 \(2\epsilon/3\) 时停止；
- active λ 的 inner update 目标是小于 \(\epsilon/3\)。

Theorem 1 的 outer-iteration complexity 是

\[
O\left((\mathrm{LipGN}/\epsilon)^{K-1}\right),
\]

再乘每个 active λ 的 \(O(1/\epsilon)\) inner-oracle cost，得到总量级 \(O(\epsilon^{-K})\)。

原文位置：

- 本项目论文第 3 节，第 4 页：uniform discretization；
- 第 4 节，第 5-6 页：bundle、GN、Algorithm 2、Theorem 1；
- 第 5.2 节，第 9 页：与 SURF 相同的 offline-bandit toy。

### 3.3 两篇论文最关键的不同

| 方面 | SURF | λ-bundle |
|---|---|---|
| 自适应变量 | 根据 PF 几何重分配 \(w\) | 选择最坏驻点误差的 \(\lambda_t\) |
| 目标 | PF 等弧长覆盖 | 所有 λ 上统一的一阶准确度 |
| 主要误差 | \(\|\widehat\Phi-\Phi\|_\infty\)、CV 等 | \(\sup_\lambda\min_i\|\nabla F_\lambda(\theta_i)\|^2\) |
| bandit 闭式解作用 | 直接构造 PF speed/CDF | 只能作为 ground truth，不应只给某个方法使用 |
| 是否保证 PF 点均匀 | 是，SURF 的核心目标 | 否，adaptive λ 可能集中在难优化区域 |

所以“adaptive λ”不自动等于“SURF 的 PF-aware weight”。前者按优化难度分配预算，后者按弧长几何分配输出权重。

## 4. 推荐的结合方式

### 4.1 只从 SURF 复用问题和 oracle

主实验只复用：

1. balanced offline dataset generator；
2. reward-mean estimator；
3. \(F_1,F_2\) 的定义；
4. closed-form \(\pi_w^*\)，作为未计时 oracle；
5. Eq. (9) 的 arc-length CDF，仅作为 front-uniformity reference。

主比较方法保持为：

- λ-bundle adaptive algorithm；
- uniform discretization Algorithm 1。

可选地把 SURF arc-uniform points 画成灰色/虚线 oracle reference，但不要把它的 closed-form runtime 与两个 first-order methods 混在同一 CPU 比较里。

### 4.2 用 reduced logits 接入当前 bundle 代码

不建议把 \(\pi\) 直接作为当前 `Original_py` 算法的无约束变量。原因是：

- \(\pi\) 必须满足 simplex constraint；
- simplex constrained optimum 一般满足 KKT 条件，但原始 \(\nabla_\pi F_\lambda\) 不会等于 0；
- 直接使用当前 raw-gradient GN 会错误地把一个最优 policy 判成“不驻点”；
- 普通 Euclidean T-map 还可能把 \(\pi\) 更新到 simplex 外。

推荐改用 \(A-1\) 维 reduced logits：

\[
z(\theta)=[\theta_1,\ldots,\theta_{A-1},0],
\qquad
\pi(\theta)=\mathrm{softmax}(z(\theta)).
\]

固定最后一个 logit 消除了 softmax 的平移不识别性。随后定义

\[
F_k(\theta)
=
\tau\mathrm{KL}(\pi(\theta)\|\pi_{\mathrm{ref}})
-\langle\pi(\theta),\widehat R_k\rangle.
\]

这会把问题变为当前 bundle/T-map 可以接受的无约束 smooth objective。若

\[
g_{\pi,k}
=
\tau(\log\pi-\log\pi_{\mathrm{ref}}+\mathbf 1)-\widehat R_k,
\]

则

\[
\nabla_\theta F_k
=
J_{\pi,\theta}^\top g_{\pi,k}.
\]

实现时可以使用 PyTorch autograd，或使用解析 Jacobian。闭式 oracle policy 也能转成 reduced logits：

\[
\theta^*_{w,a}
=
\log\frac{\pi_w^*(a)}{\pi_w^*(A)},
\qquad a=1,\ldots,A-1.
\]

注意：此时 ε 是 logit-space 的 squared-gradient stationarity，不再是错误的 raw \(\pi\)-gradient norm。

### 4.3 公平性原则

1. 两个方法使用同一 seed 下的同一固定数据集和同一 \(\widehat R\)。
2. 两个方法使用相同 objective/gradient oracle、初始点、smoothness estimate 和数值精度。
3. closed-form softmax 只生成 ground truth；不能让 uniform baseline 直接闭式求解、而 adaptive method 使用 GD。
4. adaptive λ maximization 是算法本身的成本，必须计入 runtime。
5. 公共评估，包括 dense λ scoring、oracle PF、IGD/Hausdorff/CV 和绘图，不计入两个方法的 runtime。
6. 理论版 Algorithm 2 应设 `prune_inner=False`；否则当前代码会丢弃 inner candidates，不能声称使用了完整 bundle 的 ε convergence guarantee。
7. 当前代码中的 GN 是 squared gradient norm。若想报告 \(\|\nabla F_\lambda\|\le\eta\)，代码阈值应使用 \(\epsilon=\eta^2\)。

## 5. 应该报告哪些 ε

建议明确写成三个量。

### 5.1 优化 ε

\[
\epsilon_{\mathrm{opt}}
=
\max_{w\in[0,1]}
\min_{\theta_i\in\mathcal B}
\left\|
w\nabla F_1(\theta_i)
+(1-w)\nabla F_2(\theta_i)
\right\|_2^2.
\]

这是本项目论文最核心、也是比较 adaptive 与 uniform-discretization 最公平的指标。

因为 \(K=2\)，建议用公共 dense \(w\)-grid 加局部 refinement 做严格的 post-hoc scorer，而不要用某个方法自己的 self-reported λ search 作为最终横向指标。

### 5.2 PF objective-space accuracy

对 dense query weights \(w_j\)，使用各方法的 solution map 得到
\(\widehat f(w_j)\)，再与 closed-form plug-in oracle

\[
f^*_{\widehat R}(w_j)
=
\bigl(F_1(\pi^*_{w_j}),F_2(\pi^*_{w_j})\bigr)
\]

比较。建议至少报告：

- max point-to-oracle distance；
- IGD；
- Hausdorff distance（若实现成本可接受）。

这衡量“生成的前沿是否接近精确 plug-in PF”。

### 5.3 统计 ε

算法实际优化的是 \(\widehat R\) 定义的问题。若要比较真实 PF，还应单独报告：

\[
\epsilon_R
=
\max\{
\|\widehat R_1-R_1\|_\infty,
\|\widehat R_2-R_2\|_\infty
\},
\]

以及可选的

\[
\|\widehat\Phi-\Phi\|_\infty.
\]

这样可以区分：

- 优化器没有解好 \(\widehat R\) 问题；
- 数据量不足导致 \(\widehat R\) 与 \(R\) 不同。

## 6. 推荐实验矩阵

### 6.1 固定 toy 设置

- \(A=5\)；
- \(x_a=\mathrm{linspace}(0,1,5)\)；
- \(R_1=x\)，\(R_2=1-x^4\)；
- \(\pi_{\mathrm{ref}}\) uniform；
- \(\tau=0.05\)；
- \(T=1000\)；
- reward noise standard deviation \(0.5\)；
- 30 seeds 起步，最终表格可用 100 seeds。

### 6.2 方法设置

Uniform baseline：

- 主设置 \(r=11\)，正好 12 个 grid nodes；
- 补充 sweep \(r\in\{5,11,23,47\}\)；
- 相同 GD oracle 和 warm start；
- 记录每个 checkpoint 的 common \(\epsilon_{\mathrm{opt}}\)。

Adaptive bundle：

- \(K=2\)，初始 \(\theta_0=0\)，对应 uniform policy；
- `prune_inner=False` 作为 theorem-faithful 主结果；
- 可另外报告 `prune_inner=True` 为 runtime heuristic，但必须单独标注；
- 对 \(\epsilon\in\{10^{-2},10^{-3},10^{-4}\}\) 做 time-to-ε；
- 使用与 baseline 相同的 gradient-evaluation budget。

### 6.3 两种公平比较

1. Fixed accuracy：
   - 首次达到 common \(\epsilon_{\mathrm{opt}}\le\epsilon\) 的 CPU time；
   - 首次达到该 ε 的 gradient evaluations。

2. Fixed budget：
   - 相同 CPU budget 下达到的最小 \(\epsilon_{\mathrm{opt}}\)；
   - 相同 gradient budget 下达到的最小 \(\epsilon_{\mathrm{opt}}\)；
   - 在该 checkpoint 生成的 PF 与 oracle PF 的距离。

### 6.4 CPU time 计时

论文和当前代码把横轴称为 CPU time，但现有内部实现主要使用 elapsed wall time。建议新 driver 同时记录：

- `cpu_process_s = time.process_time()`；
- `wall_s = time.perf_counter()`。

主图若明确要求 CPU time，就使用前者；wall time 放补充表。计时范围只包围 method call：

- 包含 adaptive λ search、bundle update、uniform grid optimization；
- 不包含数据生成、公共 post-hoc scoring、oracle PF 和 plotting。

## 7. 推荐输出

至少生成：

1. `epsilon_opt_vs_cpu.pdf`：common GN* 对 CPU time，log-log；
2. `epsilon_opt_vs_grad_evals.pdf`；
3. `time_to_epsilon.csv`：每个 seed 和 ε 的首次达标时间；
4. `pf_at_matched_cpu.pdf`：同一 CPU budget 下两条 approximate PF 与 exact plug-in PF；
5. `pf_at_matched_epsilon.pdf`：同一 ε 下的 PF；
6. `lambda_allocation.pdf`：uniform grid 与 adaptive selected \(\lambda_t\)；
7. `summary.json`：所有参数、seed、commit、计时和指标。

若要讨论 PF spacing，再加：

- CV of adjacent arc-length gaps；
- Gap Ratio；
- SURF arc-uniform oracle points。

但必须说明 λ-bundle 原算法没有等弧长保证。若将 SURF 的 \(\Phi^{-1}\) 后处理接在 bundle solution map 上，应把它命名为一个单独的 hybrid：

`bundle optimization + SURF geometry-aware querying`。

## 8. 本地代码与来源状态

SURF companion repository 已经放在：

`SURF_toy_example_import/`

本任务直接相关的文件是：

- `SURF_toy_example_import/uniform_PF.ipynb`
- `SURF_toy_example_import/README_IMPORT.md`
- `SURF_toy_example_import/UPSTREAM_COMMIT.txt`

本地快照 commit 为：

`867de3af2dd53570ee3a65a5c8f4446d78bf7d6e`

2026-07-25 核对上游最新 commit：

`afc0c46d219cb262bda425ba982b98860abbbacf`

上游在本地快照之后有 3 个 commits，但比较结果显示 `uniform_PF.ipynb` 没有变化；变化集中在 Fishwood、Mountaincar 和 `benchmark_moo`。因此当前 bandit toy 不需要重复导入或覆盖本地文件。

## 9. 实施时最容易踩的坑

1. 把 SURF 的 uniform-weight closed-form baseline 当成本项目的 Algorithm 1。
2. 直接在 simplex policy 上用 unconstrained raw-gradient GN。
3. 一个方法用 closed form，另一个方法用 GD，却比较 CPU time。
4. 把 \(\|\widehat\Phi-\Phi\|_\infty\) 和 squared-gradient ε 写成同一个“accuracy”。
5. 用 adaptive method 自己的 λ-search 值和 baseline 的另一种 meter 横向比较。
6. `prune_inner=True` 后仍声称使用了 theorem-faithful ε certificate。
7. 在 \(K=2\) toy 上把 bundle 的优势夸大为解决高维 simplex 爆炸；此 toy 的主要价值是正确性和可解释性，高 \(K\) 优势需要另一个实验。

