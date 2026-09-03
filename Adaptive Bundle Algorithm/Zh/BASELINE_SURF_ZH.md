# K = 2 pair campaign 的 SURF 基线 — 设计规格

日期：2026 年 9 月 2 日。状态：**设计已确认（用户 9 月 2 日拍板），代码待实现
（campaign 阶段 S3）**。
英文版：`../BASELINE_SURF.md`；两个文件必须同步修改。配套文档：
`ADAPTIVE_STEPPERS.md`（优化核；9 月 2 日修订）、`CODE_MAP.md`、`MANUAL.md`；
campaign 级计划（阶段 S0–S5、全部 leg、评判指标）记录在
`../Note/Sep_2_note.md`。

本文档规定 K = 2 MNIST pair campaign 的 SURF leg（"baseline1"）：
Jiang–Huang–Chen 的 SURF 配权策略包裹本项目自己的内层机器，外加 campaign
的评测协议。设计原则：**SURF 的"λ 放哪"数学一字不动；内层求解、oracle、
计费、safeguard 全用我们的**——各 leg 之间的唯一差异就是 λ 选择策略。

---

## 0. 理论锚点

- Bundle 论文 v2（`Reference_essay/..._MAnalytics_ (2).pdf`）：
  - 头号量（其式 4）：ε_sm−stat(θ̂) = max_{λ∈Δ_K} ‖∇F_λ(θ̂(λ))‖——
    **norm 尺度**的最坏梯度范数，即我们的 worst GN；
  - **Proposition 3**：GN(·; B) 对 λ 一致 Lipschitz（ℓ₁），覆盖论证所需的
    性质——norm 有、平方没有；这是阈值与报告全走 norm 尺度的依据；
  - **Proposition 4**：均匀离散化需要 ~eK(1 + C·LipGN/ε)^{K−1} 个格点
    （K = 2 时线性于 1/ε）——网格规模必须扫、不能猜的理由。
- SURF 论文（`Reference_essay/SURF.pdf`，arXiv:2605.20619）：Algorithm 1
  （Rule 2）、弦长估计（其式 12）、阻尼 CDF 更新（其式 13，条件 α ≤ 0.5）、
  Remark 1（不精确的暖启动内层足够）。

## 1. 设定与记号

K = 2；权重一维化：λ(w) = [w, 1−w]，w ∈ [0, 1]。

    F_w(x) = w·f₁(x) + (1−w)·f₂(x)，   ∇F_w(x) = J(x)ᵀλ(w)
    L_w = w·L₁ + (1−w)·L₂，            L̂ = L_scale·L_w（全局，只翻倍）

x_n^(t) = 第 t 轮位子 n 交付的点；y_τ = 内环迭代点；u_τ = 动量累计器；
D = 累计交付集（每点带 f 向量与 Gram G = JJᵀ）。

SURF 的几何对象（原样保留）：

    前沿路径   f_PF(w) = f(x*_w) ∈ R²
    行进速度   v(w) = ‖∂f_PF(w)/∂w‖
    弧长       s(w) = ∫₀ʷ v(p) dp，   CDF  Φ(w) = s(w)/s(1)
    配权       w_n = Φ⁻¹(n/N)  ⟹  s(w_{n+1}) − s(w_n) = s(1)/N

N 与 uniform leg 的 r 扫同一阶梯（见 §4）。

## 2. 算法（单阶段，跑到预算耗尽）

**初始化**：Φ₀(w) = w（第 0 轮 = 均匀网格）；所有位子 x_n^(−1) = x₀；在 x₀
做一次计费全量评估（每条 leg 都做——指标曲线的共同 t = 0 锚点）；
D = {(x₀, f(x₀), G(x₀))}；L_scale = 1。

**第 t 轮（t = 0, 1, 2, …）**

1. **配权**：w_n^(t) = Φ_t⁻¹(n/N)，n = 0…N（含端点 w = 0, 1——前沿两翼）。
   Φ_t 存在 1001 点 w-网格上，反函数按单调插值求。
2. **每位子解一段**，用 campaign 预实验选出的胜者优化核（见
   `ADAPTIVE_STEPPERS.md`；无 tol 停机——fixed budget）。锚点
   x̃ = x_n^(t−1)，其 Jacobian 上一轮段末已付费缓存，换权重后全梯度免费：

       g̃ = J(x̃)ᵀλ(w_n^(t))

   内环 τ = 0…m−1（分层批 S，触发器可提前停）：

       v_τ = g_S(y_τ) − g_S(x̃) + g̃           （SVRG 修正梯度）
       按胜者核走步（此处示 const）：u_τ = β·u_{τ−1} + v_τ，
       y_{τ+1} = y_τ − (0.1/L̂)·u_τ

   段末一次计费全量评估 → f_n^(t) = f(y_m)、J_n^(t)；下降检查
   F(y_m) ≤ F(x̃) + 10⁻¹⁰·(1+|F(x̃)|)；失败 ⇒ L_scale ×= 2、清动量、
   同锚重试（≤ 4）；成功 ⇒ x_n^(t) = y_m。付过费的点连同 f、Gram 一律
   进 D。这次评估一石三鸟：下降检查、前沿测量 h(u_n)、指标层的 Gram。
3. **弦长估弧长**（SURF 式 12 + 严格单调保护）：

       s̃(w₀) = 0，  s̃(w_{n+1}) = s̃(w_n) + max(‖f_{n+1}^(t) − f_n^(t)‖₂, ε_arc)

   ε_arc = 10⁻¹² 只防两点重合让 Φ 出平台、反函数失定义。
4. **单调插值 + 阻尼更新**（SURF 式 13）：PCHIP 过 {(w_n, s̃(w_n))} 插出
   s̃ : [0,1] → R₊；Φ̃_t = s̃/s̃(1)；

       Φ_{t+1} = α·Φ̃_t + (1−α)·Φ_t，   α = 0.3（论文条件 α ≤ 0.5）

5. **终止**：每个位子开工前查预算（grad_fuse 惯例）；grad_equiv ≥
   max_grad_evals 即停。轮数 T 是涌现量：
   T ≈ 预算 / [(N+1)·(平均段内行数 + 1 次全量)]。

每位子一段的粗解为何合法：SURF 的 Remark 1——不精确、暖启动、有限步
+ 小 α 即可；同一位子跨轮累积精化（纵向暖启动）。

`certify_final` flag（默认关）：冻结最终权重逐位解到目标——只作为
尾声选项；fixed-budget campaign 不用它。

## 3. 评测协议（各 leg 同一把尺）

**y 轴 — worst GN**（GN 三层约定，用户 9 月 2 日拍板）：

    单点单方向   GN(x, λ) = ‖J(x)ᵀλ‖ = √(λᵀ G(x) λ)
    单方向       gn(λ; D) = min_{x∈D} GN(x, λ)
    worst GN(D)  = max_{λ∈Δ₂} min_{x∈D} ‖J(x)ᵀλ‖

- 数值内核不动：Gram 与 val = λᵀQλ 照算（√ 与 min、max 可交换，一切
  argmax/argmin 不变）；
- 一切给定的精度以 GN（norm）尺度声明、入代码平方一次（平方尺度的
  solve_target = tol/4 恰对应 norm 尺度的 ε/2）；
- 跨 λ 聚合与全部报告用 norm；
- **精确计算**：K = 2 的 worst GN 在 **200,001 点 w-网格**上精确算
  （chunked BLAS；smoke 档 20,001）——零近似噪声；CCP λ-search 只是
  adaptive leg 的内部选点引擎，绝不用于指标；
- fixed budget ⟹ **任何图不画阈值线、front 图不做达标标记**。

**x 轴**：① total gradient evaluations = grad_equiv（λ-search 与插值
不计入）；② CPU time（进程时间；λ-search 与 CDF 开销照实计给花它的
leg）。checkpoint 按 eval_every 节奏记 (grad_equiv, cpu_time, worst GN)；
各 leg 共享 t = 0 锚点，曲线同起点。

**三张图**：① best-so-far worst GN vs total gradient evaluations（主）；
② best-so-far worst GN vs CPU time；③ Pareto front——best-per-λ 散点
（λ 四舍五入分组，组内 argmin w·f₁ + (1−w)·f₂）+ 非支配 frontier 折线
+ 颜色 = λ₁ + 灰菱形 = f(x₀)。

**副表**：frontier 弦长变异系数（覆盖均匀度）、对参考前沿的 ε-精度、
safeguard 重试数、minibatch 步数、各 leg overhead 时间占比。

## 4. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| N | 扫 {10, 20, 30, 40} | 与 uniform 的 r 同阶梯；各 leg 用各自最优（N\*、r\*） |
| α | 0.3 | CDF 阻尼，≤ 0.5 |
| 段长 / 触发器 / β / 步长常数 | = campaign 基线 | c = 0.1，β = 0.5，ρ = 0.7，patience 2，重试 ≤ 4（const 核；胜者核可换走步规则） |
| ε_arc | 10⁻¹² | 严格单调保护 |
| Φ 网格 | 1001 点 | 存 CDF、求反函数 |
| 预算 | B = 20,000 grad-equiv，eval_every = 250 | campaign 值（smoke 400/25） |
| worst-GN 网格 | 200,001（smoke 20,001） | 精确指标 |
| certify_final | 关 | 仅尾声选项 |
| Rule-1 子变体 | 仅 bandit toy | 闭式 Φ 一步配权（导入 notebook 有现成实现）；上限参照臂 |

## 5. 与原版 SURF 的差异（定稿清单）

**一字不动**：Rule-2 循环结构、分位配权 Φ_t⁻¹(n/N)、弦长式 (12)、PCHIP、
阻尼式 (13) 及 α ≤ 0.5、含端点、按位子纵向暖启动。

**改动**：① 内层 = 本项目的 Momentum-SVRG 段家族 + 下降 safeguard
（原版：任意 K 步粗解器）——与其他 leg 完全同参、把对比隔离到"λ 怎么
选"；② 前沿测量 = 计费精确全量 oracle，与下降检查共用（原版可用便宜
近似）；③ 单阶段、预算终止（原版按轮数 T）；④ 外挂 worst-GN/CPU/front
评测层（纯测量，不改动力学）；⑤ ε_arc 保护；⑥ 只上 K = 2（M > 2 是
SURF 原文自认的 future work）。

相关但不同：MODPO 分支的 SURF（同学的 LLM 实验，AdamW 内层，无精确
指标层）是另一战场的姊妹实现；画图与预算轴惯例与其对齐。

## 6. 文件计划（加新不改旧）

- `Original_py/baseline/baseline_surf_without_256_checkpoints.py` ——
  leg 本体（§2 的循环，stepper 从 `stepper_core` 注入）；
- `Original_py/experiment_plot/run_surf_compare_K2_without_256_checkpoints.py`
  —— campaign runner：三条 leg、checkpoint 记录、三张图与副表。
