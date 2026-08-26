# K=6 FAST 试验（Gram + Momentum-SVRG，eps=0.001，without-256 轨道）

由 `Original_py/run_trial_K6_fast_without_256_checkpoints.py` 生成
（2026-07-15）。模块 docstring 是完整规格；设计参考为 7 月 15 日的双语
方案文档（桌面 ZH/EN 两版）。所有 `_fast` 代码均为新文件，原文件未动。

## 跑了什么

加速版 adaptive 方法（`algorithm_adaptive_fast`），方案四项全部生效：

1. **Gram 化 λ-search**——精确恒等改写 GN(λ)=min_i λ^T M_i λ；本次实测
   λ-search 耗时 **756.4 s / 总 926.1 s
   （81.7%）**，对比 7 月 11 日原版 run 的 ~95%。
2. **两档 λ-search + 停机复核**——
   `lambda_tier_mode="strict"`。全程严格档（64 起点）：自报告口径与旧曲线完全一致。
   档位统计：{'strict': 300}。
3. **Momentum-SVRG 内层**——分层 minibatch 方差缩减 + heavy-ball 动量的
   分段结构（b=4096，p_seg=ceil(n/b)，
   step_const=0.1，beta=0.5，
   rho=0.7，patience=2，
   max_segments=10，
   rel_target=0.1）。启用 rel_target 时每轮内层目标 =
   max(eps/3, rel_target×pc_val)——针对 v2 cap_hits 的 Algorithm-2 变体
   （证书不受影响，停机线不动）。入 bundle 与验收全部基于 FULL 梯度
   （随机性无法伪造证书）。
4. **交付时修剪**——bundle 663 -> 499 点
   （r=10 simplex 网格 + 末轮搜索赢家的 λ-激活检测）；
   探针 λ 上的 GN 值逐位不变。

结果：stop_reason=`round_fuse`，最终自报告 best-so-far GN =
**9.1288e-02**，消耗 grad-equivalents = 12346
（joint 调用 662 次，minibatch IFO 69779456），
墙钟 926.1 s。L_scale_final=1.0，
inner_cap_hits=28。

## 轴与记账口径

梯度轴 = GRAD-EQUIVALENTS：一次 joint oracle = K（其 n 个 per-sample
梯度恰好把 K 个互不相交的 per-class loss 各覆盖一遍）；一步 minibatch
Momentum-SVRG = 2b·K/n。复用曲线的轴是同一单位（其所有步都是 full joint
调用），两轴直接可比。CPU 轴照本轨道惯例剔除 checkpoint 开销。

## 复用的参照曲线（披露）

Baseline（r=10）与原版 adaptive 曲线取自
`output/trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/summary.json`（7 月 11/12），未重跑：同一问题实例与
x0（种子 7/8）、同一 fuse
（max_outer=150）、同一 64 起点自报告口径。旧 adaptive run 为 budget 模式
（epsilon=None）；在 epsilon=0.001 下其已记录区间的轨迹逐位相同（其 GN
最低 0.147，远高于 2eps/3=6.7e-4，两处 epsilon 检查均不可能触发）——且
budget 模式的 CPU 轴对原版方法只会更有利（epsilon 模式还要多付每步一次
GN 检查）。CPU 对比为同机跨 run；7 月 11 日文件夹记录了机器负载与
oracle 速率校准（偏差 6.5% 以内）。

## 关键对比（自报告口径）

对 baseline：共同目标 GN 0.6161，baseline 于
2406.8119235038757 s / 103500.0
grads 达到，fast 于 82.91079306602478 s /
608.3347200000001 grad-equivalents 达到
（比值：CPU 29.028933803434334，
梯度 170.13659848315083）。

对原版 adaptive：共同目标 GN 0.1473，原版于
10375.001036643982 s / 22500.0
grads 达到，fast 于 459.8561384677887 s /
3613.05792 grad-equivalents 达到。
（`time_to_target_vs_original_adaptive` 中的 "baseline_*" 字段指原版
adaptive——沿用工具函数的第一槽位命名。）

## 图

- `gn_vs_grad_evals_baseline_vs_fast.png`、`gn_vs_cpu_time_baseline_vs_fast.png`
- `gn_vs_grad_evals_adaptive_orig_vs_fast.png`、`gn_vs_cpu_time_adaptive_orig_vs_fast.png`

CPU 图为对数时间轴；竖线 = 等预算点。

## 注意事项（一次说清）

自报告口径（baseline 从不看网格节点之间；adaptive 值是 NP-hard max 的
启发式下界——λ-search 是求最大化器，搜索不足只会让报告值偏低、即偏
保守）。单实例（种子 7/8）、单机器；
复用曲线为跨 run CPU 对比。Momentum-SVRG 的内层保证是期望型；所有验收
检查均为精确（full 梯度）。触到段数帽的轮次不适用 Algorithm 2 的终止
论证（inner_cap_hits=28；warning 已记录）。
