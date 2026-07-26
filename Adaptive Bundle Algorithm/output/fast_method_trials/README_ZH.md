# fast_method_trials —— 加速方法（Gram + Momentum-SVRG）系列实验

本文件夹下四次 run 共享同一问题实例与设置（除表中注明的旋钮外）：
K=6, p=20, n=50,000, hidden [96,96], tanh, 种子 7/8, d=11,910,
epsilon=0.001，without-256-checkpoints 轨道。对比用的参照曲线
（baseline r=10 与原版 adaptive）来自上一层的 7 月 11 日 run：
`../trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/`
——它不属于本系列，**不要移动**（本系列所有 summary 都指向它）。
`../calibration_speed_test_B2000/`（7 月 16 日由
`trial_K6_..._B2000_without_256_checkpoints` 改名）是 7 月 11 日为
B180180 正式实验做的**测速校准**试跑，同样与本系列无关。

| 文件夹 | 日期 | 是什么 | 关键旋钮 | 结果 | 角色 |
|---|---|---|---|---|---|
| `v1_plan_defaults` | 7/15 | 第一次 fast run，方案默认参数 | b=1024, β=0.9, step_const=0.1, 全程严格档, 150 轮 | best GN 0.774——尾部饱和（方差地板） | 调参记录；被 v2 取代 |
| `v2_tuned_b4096_beta0.5` | 7/15 | 按调参表对 v1 的修正 | b=4096, β=0.5（其余同 v1） | best 0.1526 ≈ 原版终值 0.1473，用时 1,225 s 对 10,375 s（**约 8.5×**） | 方案默认设计的主结果 |
| `v3_rel_target_two_tier` | 7/16 | 用户批准的再设计 | + 两档 λ-search（粗档值直接进图）、+ rel_target=0.25、500 轮 | best 0.0581，仅 **293 s** / 9,226 梯度当量；cap_hits 0/500；每轮恰好 1 段 | 自适应目标设计的主结果 |
| `cert_attempt_partial` | 7/16 | 认证尝试，max_outer=3000——**用户于约 1441 轮停止** | v3 配置 + 更长保险丝 | best 0.0291（7,173 s；由检查点日志重建，未重算任何量） | 证据：斜率衰减至 plateau 级（~2.9e-2）→ 单靠加轮数无法认证 ε=1e-3；同时暴露堆叠拷贝浪费（当日已修） |

自报告 best-so-far GN 的演进（原版方法：0.1473 @ 10,375 s）：
v1 0.774 → v2 0.1526 → v3 0.0581 → cert 0.0291。

## 版本之间到底改了什么

**共同基座（四次 run 都一样）**：Gram 化 λ-search（精确恒等改写）、
Momentum-SVRG 分段内层（full 梯度验收）、梯度当量记账、交付时修剪、
同一问题实例与种子、epsilon=0.001。

* **v1 → v2——只改参数，算法没动。**
  `msvrg_batch` 1024 → 4096（梯度方差 ÷4）、`msvrg_momentum` 0.9 → 0.5
  （heavy-ball 噪声放大 1/(1−β)：10 倍 → 2 倍）；`p_seg = ⌈n/b⌉` 自动
  从 49 变 13，所以每段成本不变。动机：v1 尾部撞方差地板饱和
  （safeguard 静默、148/150 轮触段帽、尾部震荡——正是方案调参表列的症状）。
* **v2 → v3——两处算法改动 + 一处保险丝。**
  （1）λ-search：全程严格档（每轮 64 起点）→ 两档——平时粗档
  （约 K+2 起点、tol 1e-4）且其值直接进图，严格档（64 起点、tol 1e-8）
  只负责签发停机证书（停机复核）。
  （2）内层目标：绝对 eps/3 → 相对 max(eps/3, 0.25×pc_val)——
  "把本轮最坏方向砍到四分之一"，尾部由 eps/3 兜底还原论文
  （Algorithm-2 变体；停机证书不受影响）。这一条消灭了 v2 的
  cap_hits（150/150 → 0/500）。
  （3）`max_outer` 150 → 500。MSVRG 参数沿用 v2。
  涌现行为：每轮恰好 1 段（自适应目标自动实现了最快换 λ 的节奏）。
* **v3 → cert——只改一个数字：`max_outer` 500 → 3000。**其余完全相同；
  用户于约 1441 轮停止（每轮成本随 bundle 增大而上涨——在此发现的
  堆叠拷贝浪费当日已修——且斜率衰减至 plateau 级 ~2.9e-2）。

## 旧文件夹名（重组前的 Note/LEDGER 引用按此表对照，重组于 7 月 16 日）

| 旧路径（output/ 下） | 现在 |
|---|---|
| `trial_K6_d11910_h96x96_tanh_n50000_eps0.001_fast_msvrg_without_256_checkpoints` | `fast_method_trials/v1_plan_defaults` |
| `..._fast_msvrg_without_256_checkpoints_v2` | `fast_method_trials/v2_tuned_b4096_beta0.5` |
| `..._fast_msvrg_without_256_checkpoints_v3` | `fast_method_trials/v3_rel_target_two_tier` |
| `..._fast_msvrg_without_256_checkpoints_cert` | `fast_method_trials/cert_attempt_partial` |

代码：`Original_py/` 下的 `_fast` 五件套。设计文档：桌面
`MOO_bundle_acceleration_plan_{ZH,EN}.md`。叙事记录：
`Note/Jul_15_note.md`（v1/v2）、`Note/Jul_16_note.md`（改名、v3、认证尝试、修复）。
