# Certification attempt — PARTIAL (user-stopped, round 1441/3000)

v3 configuration with max_outer=3000, stopped by the user on July 16 at
round 1441 (~7173 s CPU, best-so-far GN
2.9069e-02, 80794 grad-equivalents, bundle 4313
points).  Curves in `summary_partial.json` and the two figures are
rebuilt from the run's own checkpoint log lines — nothing recomputed.

Why stopped: per-round cost had grown 0.6 s -> ~24 s (stacked-copy waste
in the inner loop — fix planned — plus segments/round rising to ~7.6 as
the relative target hardens at low GN), and the best-so-far slope had
decayed from ~1.2 to ~0.32 dex per 1,000 rounds, making certification
within the 3,000-round fuse unlikely.  Full read-out:
`Note/Jul_16_note.md`.

The partial result still extends v3: best-so-far 2.907e-02 vs v3's
5.81e-2 and the original method's 1.47e-1 (10,375 s).

---

# 认证尝试——部分结果（用户于第 1441/3000 轮停止）

v3 配置 + max_outer=3000，7 月 16 日由用户在第 1441 轮停止
（CPU 约 7173 s，best-so-far GN 2.9069e-02，
80794 梯度当量，bundle 4313 点）。曲线与两张图由
run 自身的检查点日志逐行重建，未重算任何量。

停止原因：每轮成本从 0.6 s 涨到约 24 s（内层堆叠拷贝浪费——已列入修复
——叠加低 GN 水位下相对目标变难、每轮段数回升到约 7.6），且 best-so-far
斜率从每千轮约 1.2 个数量级衰减到约 0.32，3000 轮内认证已不太可能。
完整记录见 `Note/Jul_16_note.md`。

部分结果依然推进了 v3：best-so-far 2.907e-02，对比 v3 的 5.81e-2
与原版方法的 1.47e-1（10,375 s）。
