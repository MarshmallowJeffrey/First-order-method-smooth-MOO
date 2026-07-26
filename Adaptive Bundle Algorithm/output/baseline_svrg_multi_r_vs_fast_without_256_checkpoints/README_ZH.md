# Baseline r-扫描（v3 Momentum-SVRG 内层，without-256 轨道）

由 `Original_py/run_baseline_svrg_r_sweep_without_256_checkpoints.py`
生成（2026-07-20）。设计记录：`Note/Jul_20_note.md`。引擎：
`Original_py/baseline_svrg_certified_without_256_checkpoints.py`
（新文件，原文件未动）。

每个 r 是 Algorithm 1 在分辨率 r 均匀网格上的一次完整运行，内层求解器
与 v3 fast 试验完全一致（同实例、种子 7/8、
b=4096、同动量/步长/下降 safeguard/梯度当量记账）。
每节点服务证书：存在交付点 x 使 lambda^T G(x) lambda <= node_tol =
0.02（精确、full-gradient Gram——随机性无法伪造证书）。被
求解的节点推进到 solve_target = 0.005
（= 0.25 x node_tol，对应 v3 的 rel_target=0.25）。
share_mode=`gram`：交付点的 Gram 以缓存 lambda 重加权服务
所有未服务节点（零 oracle 调用——与 fast 方法 lambda-search 的缓存
纪律同一原则）。

散点 y 值 = 交付点集在交付时刻一次严格档 64 起点族内 lambda-search 的
GN*（测量成本不进两条成本轴；不是外部 256 起点尺子——轨道规矩不变）。
x 值 = 该 run 的总 grad-equivalents / 墙钟秒。

Fast adaptive 曲线原样复用自 `/Users/shirch/vscode101/.venv/First-order-method-smooth-MOO/Adaptive Bundle Algorithm/output/fast_method_trials/v3_rel_target_two_tier`（已披露；其画的是粗档
搜索值，欠搜索只会低报——见其 README；用户 7 月 20 日决定保持该口径）。

## 当前结果

| r | N nodes | delivered GN* (strict, full simplex) | grid cert end | grad-equivalents | wall s | solved | served-by-share | censored | stop |
|---|---------|--------------------------------------|---------------|------------------|--------|--------|-----------------|----------|------|
| 10 | 3,003 | 1.6352e-01 | 1.9911e-02 | 41327 | 599 | 2879 | 3003 | 0 | completed |
| 12 | 6,188 | 9.5415e-02 | 2.0000e-02 | 64428 | 986 | 4441 | 6188 | 0 | completed |
| 15 | 15,504 | 5.9456e-02 | 1.9997e-02 | 80912 | 1173 | 4758 | 15504 | 0 | completed |
| 20 | 53,130 | 6.3415e-02 | 1.9999e-02 | 241721 | 3689 | 11820 | 53130 | 0 | completed |

图中竖虚线：不复用缓存的 Algorithm-1 下限（每节点至少一次 full joint
调用，N(r) x 6 grad-equivalents / N(r) x 0.1384 s，
按 7 月 11 日实测速率）——忠实按论文逐节点 oracle 验收的 baseline 在
任何求解之前就要付的量。

## 图

- `gn_vs_grad_evals_baseline_r_sweep_vs_fast.png`
- `gn_vs_cpu_time_baseline_r_sweep_vs_fast.png`

每个 r 画成一条轨迹线 + 一个终点方块，两者都在可比口径上
（7 月 25 日修正，见 Note/Jul_25_note.md §6）：线是每个 checkpoint
时刻"已交付点集前缀"的严格档 64 起点全 simplex GN*（事后在缓存 Gram
上计算，成本计入 `metric_seconds`，不进两条成本轴）；方块是其最终值。
这与 fast 曲线的 y 轴同属一个口径族，两方法的线可以共享坐标轴而不产生
跨口径误读。run 自己的网格口径（所有节点最好已知值的最大值，7 月 8 日
滞后语义）不再入主图，保留在各 summary 的 `cov_history`，其终点在上表
"grid cert end" 列。"grid cert end" 与 "delivered GN*" 之差就是该网格
被实测出的节点间误差。空心方块（如有）= 触保险丝/censored 的 run：该
点是此容差下该 r 成本的下界，不是收敛测量。

## 注意

单实例（种子 7/8）、单机器；交付点坐标未存盘
（内存原因），但种子确定、可精确重导出。SVRG 内层保证为期望型；所有
证书值均精确（full-gradient Gram）。每个 r 用新采样器（种子
41），各 r 可独立复现。
