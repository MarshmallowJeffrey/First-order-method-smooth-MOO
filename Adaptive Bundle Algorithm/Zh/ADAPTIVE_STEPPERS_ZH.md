# 认证 Momentum-SVRG 基线的自适应步长扩展 — 设计规格

日期：2026 年 9 月 1 日；2026 年 9 月 2 日修订（绑定 K2 campaign——见下方
修订通知）。状态：**设计已确认，代码待实现（campaign 阶段 S1）**。
英文版：`../ADAPTIVE_STEPPERS.md`；两个文件必须同步修改。
配套文档：`CODE_MAP.md`（文件地图）、`MANUAL.md`（运行方法）、
`VARIABLES.md`（命名规范）。

本文档是后续实验的依据：给认证 Momentum-SVRG 基线加入三个自适应
步长变体（stepper）。文档自成一体——未来的会话应能只凭本文件完成
实现与实验。

**范围规则。** 只改内环的"走步规则"——即拿到修正梯度 v_t 之后怎么
走。SVRG 修正本身、认证逻辑、下降 safeguard、提前停触发器、
grad_equiv 计费，对所有 stepper 一律不动。

## ⚠ 9 月 2 日修订 — 绑定 K = 2 pair campaign

下方的 stepper 数学不变。四项计划内容被 K2 campaign 计划
（`../Note/Sep_2_note.md`，用户 9 月 2 日拍板）取代：

1. **4 路开关的归属**（9 月 2 日落地版）：走步规则在共享模块
   `Original_py/Core Engine/stepper_core.py`；带开关的执行器是
   `Original_py/experiment_plot/
   run_stepper_pre_experiment_K2_without_256_checkpoints.py`——v1 pair
   campaign 的 adaptive-CCP 执行器（`_run_leg_pair`）的 stepper 参数化
   拷贝。campaign 执行器本来就在 runner 层、不在 Core Engine，因此不存在
   `algorithm_ccp_stepper_*` 引擎文件。SURF 与 uniform 两条 leg 不带
   开关，只从 `stepper_core` 引入胜者规则。（§5 的 baseline 引擎文件
   计划作废。）
2. **Gate 0** 相应改为：stepper="const" 对比 v1 pair 执行器的
   adaptive-CCP leg。**9 月 2 日已通过**：4v9 smoke 实例上 12 项检查
   全部逐位一致（gram/theta/fval/λ/计费栈、checkpoint 数组、audit
   历史）。
3. **§7 的 smoke 协议被取代**，由 campaign 预实验（阶段 S2）执行：
   stepper 执行器 × 锁定对子 4v9，B = 2,500（1/8 预算），eval_every =
   50，11 配置（const 1 + bb 1 + adagrad 3 + adam 6）× 3 个 sampler
   种子 {41, 141, 241}；裁判 = best-so-far worst GN（norm 尺度）vs
   grad_equiv（CPU time 为副，平局取更简单者）。胜者冻结，campaign
   全部 leg 通用。**Gate 1**（安全门）改在同一 4v9 smoke 实例上跑而非
   bandit toy——意图相同（廉价抓 NaN/发散），且直接检验 S2 要用的
   机器。**9 月 2 日已通过**：bb / adagrad(×3) / adam 全部有限、
   L_scale 有界（bb 经 safeguard 到 8，属设计内行为）、audit 单调、
   各臂均有下降。
4. **GN 尺度**：一切给定精度以梯度 **norm** 尺度声明、入代码平方一次；
   内部 Gram/val 计算与一切 argmax/argmin 不变；全部报告用 norm
   （bundle 论文 v2 的 Proposition 3；见 `BASELINE_SURF_ZH.md` §3）。

---

## 0. 基线回顾（stepper = "const"）

参考实现：
`Original_py/baseline/baseline_svrg_certified_without_256_checkpoints.py`
（活跃的 `_without_256_checkpoints` 轨道，见 `CODE_MAP.md`）。

每个节点上 λ ∈ Δ_K 固定。记号：

- F_λ(x) = Σ_k λ_k f_k(x)；光滑常数 L_λ = Σ_k λ_k L_k；L̂ = L_scale·L_λ
  （L_scale 是 safeguard 乘子，从 1 起步、只翻倍）。
- 锚点 x̃ 带完整 Jacobian J(x̃)（管线已算好）；全梯度 g̃ = J(x̃)ᵀλ，
  零额外 oracle 开销。
- 内环 t = 0 … m−1（m = 段长），分层批 S：

      v_t = g_S(y_t) − g_S(x̃) + g̃          （SVRG 修正梯度）
      u_t = β·u_{t−1} + v_t                  （重球动量）
      y_{t+1} = y_t − η·u_t，  η = c/L̂      （常数标量步长）

- 提前停：‖v_t‖² ≤ ρ·solve_target 连续 patience 步。
- 段末：在 y_m 做一次计费全量评估；下降检查
  F_λ(y_m) ≤ F_λ(x̃) + 1e-10·(1+|F_λ(x̃)|)。失败：L_scale ×= 2、清动量、
  同锚重试（≤ max_segment_retries）。成功：锚点 ← y_m（Option I），
  认证量 val = ‖J(y_m)ᵀλ‖² 对 node_tol 检查。

当前默认值：c = msvrg_step_const = 0.1（Johnson & Zhang 2013 的
η = 0.1/L 规则）、β = msvrg_momentum = 0.5、msvrg_max_segments = 10、
msvrg_trigger_rho = 0.7、msvrg_trigger_patience = 2、
max_segment_retries = 4。

三个新 stepper 都原样保留下降 safeguard；每个 stepper 每小步消耗的
`grad_pair` 调用完全相同，因此 grad_equiv 计费在各 stepper 之间天然
一致。

---

## 1. stepper = "bb" — SVRG-BB：正则化 + 夹紧 + 保留动量

标量步长，每段重算一次；段内走步规则与 const 完全相同，只是 η 换成 η_k。

第 k 段开始时，用当前节点最近两个**已接受**锚点（全梯度都已在手——
零额外 oracle 开销）：

    s = x̃_k − x̃_{k−1}，     r = g̃_k − g̃_{k−1}
    D = max(sᵀr, δ‖s‖²)                     其中 δ = bb_delta_rel · L_λ
    η_k = clip( (1−β)·‖s‖² / (m·D)，  c_min/L̂，  c_max/L̂ )

设计说明：

1. **(1−β) 是动量修正。** Tan et al. 的公式 η = ‖s‖²/(m·sᵀr) 假设无
   动量的朴素 SVRG 步。重球把每个梯度放大约 1/(1−β) 倍（β = 0.5 即
   2 倍），BB 校准的是**整段总位移** ≈ 1/曲率：m·η/(1−β) = ‖s‖²/D，
   解出上式。
2. **δ 只负责分母为正**（非凸目标下 sᵀr 可能 ≤ 0）。防 η_k 爆炸是
   clip 的职责，不是 δ 的。取 max() 形式时，曲率估计失效就退化为
   常数兜底步 (1−β)/(m·δ)，再被 clip 收进窗口。
3. **夹紧窗口以现行规则为锚**：c_min = 0.01、c_max = 1.0，即 BB 最多
   在 0.1/L̂ 之上放大 10 倍、之下缩小 10 倍。safeguard 把 L_scale 翻倍
   时窗口自动下移。
4. **三种情形退回 const 规则** η = 0.1/L̂：节点第一段（还没有锚点对）；
   ‖s‖² ≈ 0（触发器立刻停过）；任何 safeguard 重试段。重试时丢弃该段
   的 BB 提议——失灵时行为 = 现在的算法。
5. **按节点存记忆。** (x̃_prev, g̃_prev) 是节点内状态；换节点清空
   （λ 变了，割线对无意义）。只有被接受的锚点推进这对记忆；重试不
   推进。公式里的 m 取计划段长（曲率估计与上一段实际走了几步无关）。

新参数：`bb_delta_rel = 1e-3`、`bb_clip = (0.01, 1.0)`。
调参税：**0 个配置**（全部是结构性常数）。

出处：SVRG-BB，Tan–Ma–Dai–Qian，NeurIPS 2016
（`Reference_essay/SVRG-BB_Barzilai-Borwein_step_size_for_SGD.pdf`，
arXiv:1605.04131）——线性收敛只在强凸下证明。正则化分母的思路来自
Li & Giannakis 2019（arXiv:1910.06532）：用二次正则把非凸接到强凸。
我们的目标非凸，所以这里的 BB 是带保险的加速器而非有证书的收敛率：
理论锚点仍是固定步长非凸 SVRG 分析（Reddi et al. 2016），兜底路径
恰好复现它。

## 2. stepper = "adagrad" — AdaGrad-on-SVRG：逐坐标 + 温启动

逐坐标步长；动量在分子、AdaGrad 缩放在分母。G 累计的是 v 不是 u，
避免把动量放大算两遍：

    G_t = G_{t−1} + v_t ⊙ v_t               （逐坐标累加，只增不减）
    u_t = β·u_{t−1} + v_t
    y_{t+1} = y_t − α_mult · u_t ⊘ (√G_t + ε)

**G₀ 温启动（核心设计）。** 每个节点初始化

    G₀ = (L̂/c)² · 𝟙                         （c = 0.1，每坐标同值）

于是第一步在每个坐标上都恰好等于可信的 const 规则 0.1/L̂。之后
AdaGrad 只会**选择性收缩**：梯度历史大的坐标步子变小，平坦坐标保持
≈ 0.1/L̂。稳定性来自"从可信起点单调不增"。这消掉了裸 AdaGrad 会引入
的自由步长旋钮。

`α_mult`（唯一的粗调旋钮，默认 1.0）把整体起始水平抬到
α_mult·0.1/L̂。α_mult = 1 是零调参保底；加速红利用 α_mult ∈ {1, 3, 10}
探测（冒进由下降 safeguard 兜住）。

重置规则：**节点内跨段保留 G**（v_t 流是连续的；方差在锚点更新处收缩
而非跳变）；换节点按 G₀ 规则重初始化；safeguard 重试时清 u 和 G——
重初始化的 G₀ 用翻倍后的 L̂，起步自动减半，与现行重试语义对齐。
ε = 1e-12（G₀ 很大，ε 实际不起作用）。

新参数：`adagrad_alpha_mult = 1.0`、`adagrad_eps = 1e-12`。
调参税：**3 个配置**（α_mult 网格）。

出处：Allen-Zhu & Hazan 2016 的 SVRG-3
（`Reference_essay/Variance Reduction for-Faster-Non-Convex-Optimization.pdf`）
——实验性推荐，无定理。组合的凸理论：AdaSVRG，Dubois-Taine et al.，
Machine Learning 2022（arXiv:2102.09645）。非凸：无现成定理；由
safeguard 兜底。

## 3. stepper = "adam" — VR + Adam（方差缩减喂给 Adam）

EMA 一阶矩本身就是动量（不再叠 u）：

    m_t = β₁·m_{t−1} + (1−β₁)·v_t
    G_t = β₂·G_{t−1} + (1−β₂)·v_t ⊙ v_t
    m̂ = m_t/(1−β₁ᵗ)，  Ĝ = G_t/(1−β₂ᵗ)     （t = 节点内累计步数）
    y_{t+1} = y_t − α · m̂ ⊘ (√Ĝ + ε)

规则：

- β₁ = 0.9 固定。**β₂ 只允许 {0.9, 0.99}**——记忆长度 1/(1−β₂) 为
  10~100 步，与段长匹配；深度学习默认的 0.999（记忆 1000 步）禁用：
  锚点一动，陈旧统计会拖过远超一段的时间。
- ε = 1e-8。v_t → 0 时更新被 ε 主导的认证点附近区域根本不会进入：
  现有触发器 ‖v_t‖² ≤ ρ·solve_target 在那之前就停了内环（触发器读的
  是 v_t 而非 Adam 方向，与优化器无关）。
- 状态 (m, G, t) 节点内跨段保留，换节点清零。safeguard 重试：清矩且
  **α ← α/2**（α 与 L 无关，重试减步直接作用在 α 上），保持"重试即
  减步"的语义。
- **只喂 v_t 给 Adam。** 全梯度只以 v_t 内修正项的身份进入；段末全量
  评估留在优化器外面（认证/下降检查/checkpoint），绝不作为一步。
  这保证 Adam 的输入流统计均质。

为什么认证轨道必须有 VR：裸 minibatch 梯度下
E‖g_S‖² = ‖∇F_λ‖² + Var，解附近 Adam 的 Ĝ 量到的是噪声，迭代停在
高于 node_tol 的噪声地板上。换成 v_t 后 y → x̃ 时 Var → 0，认证仍然
可达。

新参数：`adam_alpha = 3e-4`、`adam_beta1 = 0.9`、`adam_beta2 = 0.99`、
`adam_eps = 1e-8`。
调参税：**6 个配置**（α ∈ {1e-4, 3e-4, 1e-3} × β₂ ∈ {0.9, 0.99}），
按下面第 7 节的 smoke 协议选出后再进主对比。

## 4. 已定的设计决策

| 问题 | 决定 | 理由 |
|---|---|---|
| 是否重置累计器（G / 矩）？ | 节点内跨段保留；换节点重置；safeguard 重试时重置（AdaGrad 按翻倍后的 L̂ 重初始化 G₀；Adam 把 α 减半） | v_t 流跨段连续（方差在锚点更新处收缩而非跳变）；换节点连目标函数都换了 |
| 标量还是逐坐标累计？ | 默认逐坐标；标量 AdaGrad-Norm（b_t² += ‖v_t‖²）只留 ablation flag | 逐坐标才是新能力（各向异性）；标量版与 BB 职能重叠且信息更弱 |
| 下一锚点取最后一步还是平均？ | 最后一步（Option I）；不做平均版 | 认证点、交付点、锚点保持同一个点；平均要每段多付一次计费全量评估；平均的理论只有凸情形 |

## 5. 实现计划

按项目惯例：不改旧文件，每阶段加新文件。

1. **新引擎**
   `Original_py/baseline/baseline_svrg_adaptive_certified_without_256_checkpoints.py`：
   certified 基线的复制品 + `stepper` 开关
   ∈ {"const", "bb", "adagrad", "adam"} 和上述新参数（先
   `import _layout` 再做同级导入）。`"const"` 复现现行算法，专为等价门
   存在。
2. **新 runner**
   `Original_py/experiment_plot/run_adaptive_stepper_smoke_without_256_checkpoints.py`，
   执行第 6–7 节的门与 smoke 协议。
3. 各 stepper 的节点内状态：BB —— (x̃_prev, g̃_prev, η_k)；
   AdaGrad —— (G, u)；Adam —— (m, G, t)。结果字典新增：stepper 名、
   每段步长轨迹（BB：η_k 列表；AdaGrad/Adam：每段有效逐坐标步长的
   min/median/max），其余计数器沿用。
4. **Gate 0 的实现警示**：const 路径上，stepper 开关不得多消耗任何
   RNG、不得改变浮点运算顺序，否则与参考基线的逐位一致会被破坏。

## 6. 验证门（任何实验之前先过）

- **Gate 0（等价门）。** stepper="const" 对比现有 certified 基线，
  同种子：段末 F 值、grad_equiv、safeguard 计数、交付点必须完全一致。
- **Gate 1（安全门）。** 三个自适应 stepper 各在 bandit-toy 目标上跑
  2 个节点：无 NaN/Inf、重试次数有限、认证达成。

## 7. Smoke 与主对比协议

- **场景**：K = 3 MNIST triple 节点网格，Exp 5（μ = 0）配置，全预算的
  1/8，3 个种子。全预算只在用户看过 smoke 结果拍板后再跑（家规）。
- **公平性**：同种子同分层批流；grad_equiv 计费一致（自动成立——
  `grad_pair` 调用相同）；段长与触发器设置各 stepper 一致。
- **对比臂**：const / bb（1 配置）/ adagrad（α_mult ∈ {1,3,10}）/
  adam（6 配置，选最优者进主对比）。**调参税**（0 / 0 / 3 / 6 个配置）
  作为结果的一部分报告，不隐藏。
- **指标。** 主指标：served 节点数 vs grad_equiv 曲线（家规标准）与
  每节点认证成本。诊断：步长轨迹（BB 的 η_k；AdaGrad/Adam 的有效步长
  分位数）、safeguard_retries、minibatch_steps_total。合规：对所有
  served 节点复核 val ≤ node_tol。
- **预期信号**（各设计成立的证据）：BB —— 多数段 η_k > 0.1/L̂
  （放大即加速证据）；AdaGrad —— 收益集中在 α_mult > 1 档；
  Adam —— 若同税负下不超过 BB/AdaGrad，即坐实"认证轨道不值得"。

## 8. 环境注意

- 用 venv 的 Python 3.13 解释器显式启动；`run.sh` 指向内层 3.11 venv
  （已知坑）。
- 长（全预算）运行必须先获用户同意。

## 9. 参考文献

`Reference_essay/` 内：

- `accelerating-stochastic-gradient-descent-using-predictive-variance-reduction-Paper.pdf` —— Johnson & Zhang 2013（SVRG；η = 0.1/L 规则；Option I/II）。
- `Stochastic_Variance_Reduction_for_Nonconvex_Optimization.pdf` —— Reddi et al. 2016（非凸固定步长理论锚点；η = μ₁/(L n^{2/3})）。
- `Variance Reduction for-Faster-Non-Convex-Optimization.pdf` —— Allen-Zhu & Hazan 2016（SVRG-3 = AdaGrad-on-SVRG，实验性）。
- `SVRG-BB_Barzilai-Borwein_step_size_for_SGD.pdf` —— Tan et al. 2016（BB 步长；强凸理论；首个 Option I 证明）。

外部：

- Li & Giannakis 2019，*Adaptive Step Sizes in Variance Reduction via Regularization*，arXiv:1910.06532（非凸下的正则化 BB 分母）。
- Dubois-Taine et al. 2022，*SVRG Meets AdaGrad: Painless Variance Reduction*，Machine Learning（arXiv:2102.09645）—— AdaGrad-on-SVRG 的凸理论。
- Kingma & Ba 2015，*Adam*（arXiv:1412.6980）。
