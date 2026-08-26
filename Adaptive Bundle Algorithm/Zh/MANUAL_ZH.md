# 使用手册:每个文件是什么,以及如何复现实验

日期:2026 年 7 月 7 日。配套文档:`EXPERIMENTS.md`(实验是什么、结果、
怎么读图)及其中文版 `EXPERIMENTS_ZH.md`。本手册的英文版保存在
`MANUAL.md`;修改其中一份时必须同步修改另一份。

本手册回答两个问题:

1. 这个项目里每个文件和目录是做什么的?
2. 要复现 `output/` 里的结果,具体运行什么?

---

## 1. 项目布局

仓库根目录:`First-order-method-smooth-MOO/`,分支
`mlp-comparison-results`。除特别说明外,下文路径都相对于仓库根目录。
路径中含有空格——在 shell 里永远加引号。

### 1.1 仓库根目录

| 路径 | 是什么 |
|---|---|
| `Adaptive Bundle Algorithm/` | 整个项目:代码、实验输出、文档、参考论文。 |
| `Note/Jul_5_note.md` | 7 月 4 日正确性修复的详细记录(下降引理保护机制、单纯形投影一致性、ε 模式诚实化、蛇形网格顺序、修正的逻辑回归光滑常数),含每一处的数学理由。 |
| `Note/Jul_6_note.md` | 7 月 6 日论文一致性审查与修复的详细记录(W* 分布、平台水平定义、激活函数参数、文档字符串修正),以及 ReLU 光滑性问题的调查。 |
| `.venv/` | 项目的 Python 虚拟环境(Python 3.11.5,PyTorch,cyipopt/IPOPT,NumPy,SciPy,Matplotlib)。本手册所有命令都用它的解释器。 |

### 1.2 `Adaptive Bundle Algorithm/`——文档与资料

| 路径 | 是什么 |
|---|---|
| `EXPERIMENTS.md` / `EXPERIMENTS_ZH.md` | 实验分析报告(英文/中文):定义、由理论推出的预期、精确协议、结果、参数参考、读图指南、曲线行为解释、诚实报告说明。 |
| `MANUAL.md` / `MANUAL_ZH.md` | 本手册(英文/中文)。 |
| `Python_Change.md/PYTHON_CHANGES.md` | 把原始笔记本代码改造成当前模块布局过程中所做代码变更的时间顺序记录,含 7 月 4 日正确性修复。 |
| `Python_Change.md/PYTHON_CHANGES_ZH.md` | 上一文件的中文翻译。 |
| `Python_Change.md/PLATEAU_EXPERIMENT_CHANGES.md` | 搭建平台实验机制的变更记录(`detect_plateau`、`experiment_mlp_plateau_comparison`、预算记账、成对比较图)。 |
| `Python_Change.md/PLATEAU_EXPERIMENT_CHANGES_ZH.md` | 上一文件的中文翻译。 |
| `Reference_essay/A_first_order_bundle_method_for_smooth_multi_objective_optimization.pdf` | **主论文**。这里实现的算法 1(均匀网格基线)和算法 2(自适应束方法)出自它。 |
| `Reference_essay/Smooth Tchebycheff Scalarization for Multi-Objective Optimization.pdf` | 相关工作(Lin 等,2024),草稿引用。 |
| `Reference_essay/Beyond One-Preference-Fits-All Alignment- Multi-Objective Direct Preference Optimization.pdf` | 相关工作(MODPO),草稿引用。 |
| `Reference_essay/reference essay.pdf` | 其他参考材料。 |
| `output/` | 当前全部实验结果。结构见 1.4 节。 |

### 1.3 `Adaptive Bundle Algorithm/Original_py/`——代码

每个模块负责一件事;通常你只需要 `run_experiments.py` 这一个入口。

| 文件 | 做什么 |
|---|---|
| `algorithm.py` | 论文的**算法 2**(自适应束方法):外层循环、多起点最大—最小 λ 搜索(`_maximise_GN`,IPOPT,失败时退到 SLSQP)、T 映射内层步(式 10,批量化,最小下标破平局)、下降引理保护机制(自适应 `L_scale` 加倍并发 RuntimeWarning)、ε 模式,以及 GN\* 质量度量(`pc_star`,固定 256 起点的尺子)。 |
| `bundle.py` | `Bundle` 容器:每个算过的点连同全部 K 个分目标梯度;可在任意 λ 处由存储组装 ∇F_λ。 |
| `baseline.py` | 论文的**算法 1**(均匀离散化基线):分辨率 r 的单纯形网格、带热启动的蛇形顺序、每节点固定步长梯度下降、检查点,以及用同一 GN\* 度量给基线打分所用的快照束。另有可选的认证模式(`node_tol`,默认关闭,当前所有实验均未使用):逐节点的 ‖∇F_{λ_i}‖² 验收检查,全部节点达标即停(见 `Note/Jul_8_note.md`)。 |
| `objectives_torch.py` | MLP 测试台(PyTorch):植入式线性 softmax 数据(论文 §5.1.1,W* ~ U[−1,1])、K 个按类交叉熵目标、可选激活函数(`relu`/`tanh`/`softplus`/`identity`——基准用 `tanh`)、基于探针的光滑常数估计、融合式联合梯度调用。 |
| `objectives_numpy.py` | NumPy 问题生成器,含只被验证脚本使用的强凸逻辑回归测试台(没有实验驱动用它)。 |
| `experiments.py` | 实验驱动与分析:`experiment_mlp_plateau_comparison`(两个扫描都用的等预算一对一比较,**核心驱动**)、`experiment_mlp_gn_coverage`(遗留的到目标时间设计——不用于主要数字)、`detect_plateau`,以及全部绘图(`_plot_plateau_pair` 画出 `EXPERIMENTS.md` §8 描述的每个图元素)。 |
| `run_experiments.py` | 统一实验运行器,复现入口:定义两个扫描的全部配置,运行它们,在 `output/` 下写出每个 `summary.json`、`README.md`、趋势图和 `sweep_index.json`。 |
| `run.sh` | 便捷包装:用项目 venv 并设好 `KMP_DUPLICATE_LIB_OK=TRUE` 来运行本目录下任意脚本(见第 2 节)。 |

### 1.4 `Adaptive Bundle Algorithm/output/`——结果

```
output/
  README.md                  # 目录树怎么组织,读图指引
  plateau/                   # 固定 30k 预算的 K 扫描 + K=6 预算研究
    README.md                # 跨配置分析、K=6 调查
    sweep_index.json         # 4 个主扫描配置的机器可读索引
    plateau_ratio_vs_K.png   # 趋势图
    K6_budget_study.png      # K=6 三个预算的历史最优曲线
    K{3..6}_p6_n30_h4_tanh_r6_B30000/       # 主扫描
    K6_p6_n30_h4_tanh_r6_B{90000,240000}/   # K=6 预算研究
  crossover/                 # K=2、n=50k、2k 预算的宽度扫描
    README.md, sweep_index.json, crossover_ratio_vs_d.png
    d{642,1794,5634,11522,19458}_h*_tanh_n50000_B2000/
```

每个配置目录恰好包含四个文件:

- `gn_vs_grad_evals.png`——原始 GN\* 对累计梯度求值次数。
- `gn_vs_cpu_time.png`——原始 GN\* 对 CPU 时间(x 轴对数,带等预算标记)。
  每个元素怎么读:`EXPERIMENTS.md` §8。
- `summary.json`——完整记录:`config`(全部参数)、`baseline` 与
  `adaptive` 块(检查点历史 `cov_history`、`best_so_far`、
  `grad_evals_history`、`cpu_times`,加健康标志 `L_scale_final`、
  `inner_cap_hits`)、`plateaus`(每种方法的检测器输出)、
  `time_to_target`(最终值与对称的到共同目标统计量)、
  `runtime_warnings`。
- `README.md`——参数、选择理由、结果、健康标志、单配置分析。

### 1.5 仓库之外的归档

`/Users/shirch/vscode101/.venv/ledger-artifacts/`(有意放在 git 仓库
之外)存有:`verify_fixes.py` 与 `prefix_repro.py`(验证脚本,见第 5 节)、
`orig_backup/`(7 月 4 日之前的代码)、`pre_fix_outputs_archive/`
(全部修复前的实验输出——与当前结果不可比较)、`relu_sweep_archive/`
(促使改用 tanh 的 ReLU 诊断扫描)。

---

## 2. 环境

- 解释器:`<仓库根目录>/.venv/bin/python`(Python 3.11.5)。
- 所需包已装在该 venv 里:PyTorch、cyipopt(IPOPT)、NumPy、SciPy、
  Matplotlib。
- **凡是同时导入 PyTorch 和 cyipopt 的程序,在 macOS 上必须设
  `KMP_DUPLICATE_LIB_OK=TRUE`**:PyTorch 自带一份 OpenMP 运行时,
  IPOPT/OpenBLAS 又引入 Homebrew 的那份;不设此标志进程会以
  "OMP: Error #15" 中止。`experiments.py` 和 `run_experiments.py`
  自己会设;独立脚本必须显式设置(或用 `run.sh`)。
- 实验必须有 IPOPT(`experiment_mlp_plateau_comparison` 没有它会拒绝
  运行),以保证 λ 搜索和度量用的是预期的求解器,而不是悄悄退到 SLSQP。

快速检查环境是否可用:

```bash
cd "<仓库根目录>/Adaptive Bundle Algorithm/Original_py"
KMP_DUPLICATE_LIB_OK=TRUE ../../.venv/bin/python -c \
  "import torch, cyipopt; print('environment OK')"
```

---

## 3. 复现实验

以下命令都在
`"<仓库根目录>/Adaptive Bundle Algorithm/Original_py"` 下运行。
`$PY` 指 `../../.venv/bin/python`。

### 3.0 运行之前

- **在空闲机器上串行运行。**CPU 时间轴是墙钟时间;并发负载会扭曲它
  (梯度轴不受影响)。
- **覆盖行为:**重复运行时 `summary.json` 和 `README.md` 原地覆盖;
  **PNG 文件永不覆盖**——重跑会在旧图旁边生成带 `_001`、`_002`…
  后缀的新图。想要干净目录,先删除旧的配置目录。
- **可复现性:**扫描定义中数据与初始化种子固定(7/8),所以梯度轴
  曲线和所有质量数字应当复现(误差在浮点/BLAS 线程级别)。
  CPU 时间以及由此得出的等时间比值依赖机器:应期望复现定性图景
  (单调增长、交叉点在小 d 处),而不是 1.3–101.4 这些精确数字。

### 3.1 冒烟测试(几分钟——先跑这个)

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py plateau   --smoke
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py crossover --smoke
```

两个扫描的微型版(平台:K=3,4,预算 4k)。它们写到
`output/plateau_smoke/` 和 `output/crossover_smoke/`,不会碰真实结果。
在把几个小时投给完整扫描之前,用它们确认环境无误。冒烟目录之后可删。

### 3.2 平台扫描(K = 3,4,5,6,预算 30k)

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py plateau
```

在 `output/plateau/` 下写出四个 `K*_B30000` 目录、
`plateau_ratio_vs_K.png`、`sweep_index.json` 和扫描 `README.md`。
参考机器上的迭代工作时间:总计约 36 分钟(各基线只需数秒;自适应
运行在 K=3,4,5,6 分别为 442 秒、494 秒、531 秒、639 秒)。真实墙钟
时间更长,因为检查点度量求值(不计入报告轴)也要时间。

### 3.3 K=6 预算研究(预算 90k 与 240k)

`run_experiments.py` 有意把主扫描固定在 30k,所以两个预算研究目录
是直接调用它的 `run_one_config` 辅助函数产生的——该函数运行一个配置
**并**写出标准的四文件布局(`summary.json`、两张改名后的图、
`README.md`)。在 `Original_py/` 下:

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY - <<'EOF'
from pathlib import Path
from run_experiments import run_one_config, PLATEAU_RATIONALE

# Checkpoint cadences as recorded in the committed summary.json files.
for budget, cadence in ((90_000, 3_600), (240_000, 8_000)):
    config = dict(
        K=6, p=6, n=30, hidden_sizes=[4], activation="tanh",
        coarse_resolution=6, n_passes=100_000,
        steps_per_point_per_pass=5,
        max_grad_evals=budget,
        baseline_eval_every_n_grads=cadence,
        adaptive_eval_every_n_grads=cadence,
        max_outer=1_000_000, max_inner=25,
        lambda_max_starts=64, prune_inner=True,
    )
    out = Path(f"../output/plateau/K6_p6_n30_h4_tanh_r6_B{budget}")
    run_one_config(config, out, PLATEAU_RATIONALE)
EOF
```

(种子 7/8 由 `run_one_config` 自己通过 `DATA_SEED`/`INIT_SEED`
常量提供。)实测迭代工作时间:90k 约 35 分钟,240k 约 2.2 小时
(自适应 2,096 秒与 7,880 秒;基线 21 秒与 59 秒)。合并图
`K6_budget_study.png` 由三份 summary.json 的 `best_so_far` 曲线拼成
(30k/90k/240k 的自适应 + 240k 的基线);任何读出这四条曲线、
在对数 y 轴上作图的小脚本都能重画它。

### 3.4 交叉扫描(宽度 16x16 … 128x128)

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py crossover
```

在 `output/crossover/` 下写出五个 `d*` 目录、`crossover_ratio_vs_d.png`、
`sweep_index.json` 和扫描 `README.md`。**这是最贵的一个**:实测迭代
工作时间共约 4.9 小时,大头在大宽度的自适应运行(64x64:2,467 秒;
96x96:4,632 秒;128x128:9,040 秒)。加上检查点度量求值(d 大时
占比可观),请在空闲机器上按大半天规划。

### 3.5 单个自定义配置

想跑一个你自己的配置(不同的 K、宽度、预算、激活函数),照 3.3 节
那样复用 `run_one_config`,换成你自己的 `config` 字典和输出目录,
就能得到标准四文件布局。`EXPERIMENTS.md` 第 6 节的每个参数都是
`config` 的键。如果还想换种子(常量 `DATA_SEED`/`INIT_SEED` 固定为
7/8),就改为调用底层驱动:

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY - <<'EOF'
from experiments import experiment_mlp_plateau_comparison

result = experiment_mlp_plateau_comparison(
    K=4, p=6, n=30, hidden_sizes=[4], activation="tanh",
    seed=7, init_seed=8,
    coarse_resolution=6,
    steps_per_point_per_pass=5, n_passes=100_000,
    max_grad_evals=30_000,
    baseline_eval_every_n_grads=1_200,
    adaptive_eval_every_n_grads=1_200,
    max_outer=1_000_000, max_inner=25,
    lambda_max_starts=64, prune_inner=True,
    output_dir="../output/my_custom_run",
)
print(result["plateaus"])
EOF
```

底层驱动写出两张图(用原始的 `baseline_vs_ipopt_*.png` 文件名)并
返回结果字典;它**不**写 `summary.json` 和单配置 `README.md`——
那两样是 `run_one_config` 附加的。

### 3.6 把复现结果与已记录结果对照

把你新生成的 `summary.json` 与同目录里已提交的那份比较:`config`
必须完全一致;`time_to_target.baseline_final_best_gn` 与
`time_to_target.adaptive_final_best_gn` 在同一台机器上应当吻合到
若干位有效数字(梯度轴);`cpu_times` 跨机器会不同。主要数字表在
`EXPERIMENTS.md` §5 和各扫描的 `README.md` 里。

---

## 4. 读结果

- 从 `EXPERIMENTS.md` 开始:§5 结果与趋势,§8 每个图元素的含义,
  §9 曲线的不直观行为(原始曲线上升、大 d 处的波动、128x128 的虚线)。
- 然后读扫描 `README.md`(跨配置分析)和单配置 `README.md`
  (该次运行的参数与理由)。
- `summary.json` 是任何地方引用的每个数字的机器可读事实来源。

---

## 5. 验证代码本身(可选)

两个独立脚本放在仓库**之外**的
`/Users/shirch/vscode101/.venv/ledger-artifacts/`:

```bash
PY="<仓库根目录>/.venv/bin/python"
cd /Users/shirch/vscode101/.venv/ledger-artifacts
KMP_DUPLICATE_LIB_OK=TRUE "$PY" verify_fixes.py   # 预期:10 passed, 0 failed
KMP_DUPLICATE_LIB_OK=TRUE "$PY" prefix_repro.py   # 预期:duplicates: 39,pc_history 全为 1.0
```

`verify_fixes.py` 对**现行**代码跑 10 项检查(保护机制收敛、无点
重复、修正后的逻辑回归 L、ε 模式停止、λ/值一致性、蛇形顺序界)。
`prefix_repro.py` 运行**归档的** 7 月 4 日之前代码,复现其冻结循环
缺陷(39 个重复束点,质量钉死在 1.0)——证明缺陷确实存在过、
现在确实没了。

---

## 6. 有意不在这里的东西

- 旧实验笔记本(`run_plateau*.ipynb`、`mlp_crossover_h*.ipynb`、
  `Mlp_Compare.ipynb`、`mlp_complexity_crossover_experiment.ipynb`)
  和独立模块 `gn_sample_ipopt.py` 已删除;它们做的一切都被
  `run_experiments.py` + `experiments.py` 覆盖。
- 修复前的实验输出(含旧的 `output/plateau result/` 目录树)已从
  仓库移除;它们产自带有已证实缺陷的代码,归档于仓库之外的
  `ledger-artifacts/pre_fix_outputs_archive/`。
- 演示用 DOCX 文件已删除;将来的报告应根据 `EXPERIMENTS.md` §5
  重新制作。

---
---

# 第二部分——7月9日之后新增的全部内容(写于 2026-08-25)

上面的第一部分(第 1–6 节,写于 7月7日)原样保留,它对"原始轨道"
(`run_experiments.py` 跑的 plateau 与 crossover 两个扫描)的描述仍然
准确。之后新增的每条工作线都记录在本部分,仍然回答同样的两个问题:
每样东西是什么,以及跑什么命令来复现。

## ⚠ 8月25日重组通告(先读这个)

8月25日晚些时候,用户对整个仓库做了大范围重组。本手册(第一、二
部分)里的所有路径都指**重组前**的布局,请用下面两张地图换算:

- **代码**:`Original_py/` 不再是扁平结构。文件被移入
  `Core Engine/`(引擎 + bundle + CCP 求解器)、`baseline/`、
  `objective/`、`experiment_plot/`(全部 runner、画图、审计、
  `experiments.py`)、`sanity_check/`;只有 `run.sh` 留在顶层。
  当天已完成**修补**(用户批准):每个子文件夹放置 `_layout.py`
  (sys.path 引导),所有有同级 import 的模块先加载它;输出/数据
  锚点加深一级;全部输出目录常量改指**新**位置。修补后五个
  sanity 门禁与两个遗留验证器全部通过。`./run.sh <脚本>.py`
  恢复可用,既接受裸文件名(自动在子文件夹中查找)也接受
  `子文件夹/脚本.py` 路径——下文第一、二部分的命令照写即可运行。
- **结果**:`output/` 被分组改名;新旧对照表和被删除目录清单
  (Pareto_front、旧 r-sweep 目录、标定测试、若干日志/备份)见
  `output/README.md`。`*_ZH` 文档移入 `Zh/`;`EXPERIMENTS(_ZH).md`
  和 `Python_Change.md/` 已删除(git 历史里可找回)。

第 9 节的命令照写可运行(runner 现在读写**新**目录);第 10 节的
表保留重组前的名字作历史标识——实际位置一律通过
`output/README.md` 换算。

本部分的两个常备参照:

- `CODE_MAP.md`(本目录;8月25日之前它是 `Original_py/README.md`)——
  全部代码代际的逐文件地图、import 分层图,以及 `Original_py/`
  保持扁平结构的原因。
- 每个实验的权威记录是其 `output/` 结果目录内的 `README.md`,加上
  运行该实验那次会话的 `Note/` 日期文件。本手册只给命令并指向它们,
  不重复结果数字。

如果工作区里找不到第一部分引用的 `EXPERIMENTS.md`,从 git 历史恢复
——它是七月 plateau/crossover 结果唯一的完整报告。

## 7. 轨道、文件代际与测量规则

文件名后缀标记代码代际(完整表在 `CODE_MAP.md`):

| 后缀 | 代际 |
|---|---|
| *(无)* | 原始参考实现;检查点用外部固定 256 起点量尺打分 |
| `_without_256_checkpoints` | **现行**测量轨道(7月8日起):轨迹完全相同,但检查点记录方法自己最近一次 λ-search 的值 |
| `_fast` | 7月15日加速集:Gram 缓存、带停止校验的两档 λ-search、Momentum-SVRG 内环、交付时剪枝 |
| `_ccp` | 8月9日 λ 求解器替换:多起点凸-凹程序(CCP)取代 IPOPT |

常备测量规则(7月8日起):所有实验都在 without-256 轨道上。任何地方
都没有外部 256 起点量尺;本族自己的 strict 64 起点 λ-search 就是
证书、停止、审计和共享坐标轴曲线的仪器。凡审计承担结论的地方,一律
报告单调下界包络(每次审计都是 NP-hard 最大化的下界,可能低报;
前缀 GN\* 不增)。八月的 campaign 另外使用方法对称的双仪器审计
`audit_v2 = max(strict-64 IPOPT, 重型 CCP)`——见 9.7 节。

## 8. 环境新增项

- venv 与 `KMP_DUPLICATE_LIB_OK=TRUE` 规则同第一部分 §2。便捷入口是
  `Original_py/run.sh`:`./run.sh <脚本.py> [参数]` 用 venv 解释器
  并带上该环境变量运行本目录任意脚本。
- `ccp_lambda_solver.py` 会自动探测 `highspy`(HiGHS)来解它的
  game LP,强烈建议装上(热启动 LP 是 CCP 的内步);没有时退回
  `scipy.optimize.linprog`。CCP 腿**不需要** IPOPT;`ts*`
  (IPOPT strict 档)腿和 `audit_v2` 的 IPOPT 半边仍需要 IPOPT。
- 凡 wall-clock 会落到 CPU 轴上的运行,必须在空闲机器上**串行**跑
  (检查点/审计时间由 runner 记在轴外,但决策时间是真实上轴的)。
- 可复现性注意(会话 12 发现,7月27日):本环境下 MLP torch 运行
  **不可比特级复现**——把每条已存 MLP 轨迹当作一次实现,核对已存的
  `summary.json`,不要靠重跑对帐。bandit numpy 运行可以比特级复现。

## 9. 复现 7月9日之后的实验

约定:每个 runner 都有 `--smoke`(小规模运行,写进独立的
`*_SMOKE`/`smoke` 目录——先跑它);campaign runner 不加 `--force`
不会覆盖已完成的腿。以下命令都在 `Original_py/` 里用
`./run.sh <脚本> [参数]` 执行。

### 9.1 七月在原始引擎上的新增实验

| 实验 | 命令 | 输出位置 |
|---|---|---|
| K=2 认证 Pareto front(7月8日) | `./run.sh run_pareto_certified_without_256_checkpoints.py` | `output/Pareto_front/pareto_certified_without_256_checkpoints{,_r20}/` |
| K=3 λ 路径图(7月9日) | `./run.sh run_lambda_path_without_256_checkpoints.py` | `output/lambda_path_K3/` |
| 测量变体 A/B 复跑(K5 plateau + 96×96 crossover) | `./run.sh run_experiments_without_256_checkpoints.py` | `output/without_256_checkpoints/` |
| K=6 参考试跑,B=180,180(7月11日) | `./run.sh run_trial_K6_without_256_checkpoints.py` | `output/trial_K6_…_B180180_without_256_checkpoints/`——**不要移动**;fast 试跑与这些已存曲线对比 |

### 9.2 fast 引擎试跑(7月15–16日;会话 12 的探针 7月27日)

先跑门禁:`./run.sh sanity_checks_fast.py`——必须 8/8 PASS。

```bash
./run.sh run_trial_K6_fast_without_256_checkpoints.py \
  --tier-mode strict --rel-target 0.1 --max-outer 300 --variant-tag v4_strict_rel0.1
```

输出位置:`output/fast_method_trials/`。v1/v2/v3 三个文件夹是同一
runner 更早的参数组(各自 README 记录精确身份);v4(上面的命令)是
诚实 strict 仪器的探针。`v3` 画出的 cheap 档读数后来被证明低报约
2 倍——文件夹保留存档,不要引用它的曲线。

### 9.3 固定 node_tol 的基线 r-sweep + 节点间隙(7月20–26日)

```bash
./run.sh run_baseline_svrg_r_sweep_without_256_checkpoints.py \
  --r-list 10,12,15,20 --node-tol 0.02 \
  --out-dirname baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/tol0.02 \
  --fast-ref "../output/fast_method_trials/trial_K6_…_v4_strict_rel0.1/summary.json" \
  --save-grams
```

(tol0.01 目录用 `--node-tol 0.01`。)旧的 7月25日风格目录
`output/baseline_svrg_multi_r_vs_fast_without_256_checkpoints/` 已冻结;
v2 目录 `…_v2_without_256_checkpoints/` 是对比的正式归档
(其中 `original/` 是旧目录的逐字副本)。间隙图:
`./run.sh plot_between_node_gap_without_256_checkpoints.py`。
促成换用 strict 仪器的 v3 停滞诊断:
`./run.sh diag_v3_plateau_without_256_checkpoints.py` →
`…_v2_…/diag_v3_plateau/diag.json`。

### 9.4 K=6 固定预算实验(7月26日;协议 5e)

```bash
./run.sh run_fixed_budget_K6_without_256_checkpoints.py \
  --budget 80912 --rel-target 0.05 --targeting-starts 24 --eval-every 2000
```

输出位置:`…_v2_…/fixed_budget_B80912/`(`--replot` 用已存数据重画;
`--smoke` → `fixed_budget_B600_SMOKE/`)。图上的基线点来自 9.3 节
已完成的 r-sweep 运行。

### 9.5 纯固定预算协议(7月27日 K=6;7月30日 K=2)——headline 协议

协议里任何地方都没有容差参数:共享 segment 工作单元、共享 `s`、
链式热启动、花完预算即停;各腿唯一的差别是"下一个 λ 选在哪"。
一次调用跑一条腿,串行执行:

```bash
./run.sh run_pure_budget_K6_without_256_checkpoints.py --run adaptive --s 5 --budget 80912 --targeting-starts 24
```

```bash
./run.sh run_pure_budget_K6_without_256_checkpoints.py --run baseline --r 10 --s 1
```

已记录的腿:adaptive s5;baseline r ∈ {10,12,15,20} 的 s=5,外加
r10/r15 的 s=1。`--backfill-audits` 给已完成的基线腿补 strict 64
起点前缀审计;`--figure` 重画 campaign 图。输出位置:
`…_v2_…/pure_budget_B80912/`。

K=2 版本用**精确 1-D 质量计**(所有测量都不含多起点搜索;
`--decision-grid 2001`,`--audit-grid 200001`):

```bash
./run.sh run_pure_budget_K2_without_256_checkpoints.py --run adaptive --s 5 --budget 20000 --targeting-starts 24
```

基线腿 `--run baseline --r 10|20|40|80 --s 5`;CCP 选点腿是独立
runner(同一执行循环,只换下一个 λ 的策略):

```bash
./run.sh run_pure_budget_K2_ccp_without_256_checkpoints.py --s 5 --budget 20000
```

输出位置:`output/pure_budget_K2_without_256_checkpoints/B20000/`。
K=6 campaign 的事后 ε-Pareto front 指标:
`./run.sh front_metrics_K6_pure_budget_without_256_checkpoints.py`。

### 9.6 SURF bandit toy(7月26日 K=2/K=5;7月31日 mean-variance)

门禁:`./run.sh sanity_checks_bandit_toy.py`(9/9)、
`sanity_checks_bandit_toy_K5.py`(9/9)、`sanity_checks_bandit_toy_mv.py`。

```bash
./run.sh run_bandit_toy_without_256_checkpoints.py --epsilon 1e-2 --eval-every 0
```

`--eval-every 0` = 逐 segment 记录,即精确读数模式(会话 13)。
eps1e-3/1e-4 文件夹仍是 7月26日的粗检查点节奏;引用它们的首次到达
读数前先用 `--eval-every 0` 重跑。K=5:
`run_bandit_toy_K5_without_256_checkpoints.py --epsilon …`。
mean-variance(非凸):
`run_bandit_toy_mv_without_256_checkpoints.py --epsilon …`
(`--gamma-scan` 选 γ;`--rebuild-reference` 重建不计时的多起点
真值参考表)。输出位置:
`output/bandit_toy_{surf,K5,mv}_without_256_checkpoints/eps*/`。

### 9.7 CCP campaign(8月8–10日):实验一、MNIST K=10 试跑、研究 A/B

先跑门禁:`./run.sh sanity_checks_ccp.py`——全部必须 PASS。

- **研究 A——种子采样器消融**(决定 `CCPConfig.seed_sampler`):
  `./run.sh run_ccp_smoke_sampler_without_256_checkpoints.py --reps 50`
  → `output/ccp_smoke_sampler/`。
- **研究 B——受控 λ 求解器基准**(2a 配对起点 polish 对比 + 2b 60 秒
  定时赛,跑在 `output/ccp_compare_…/lambda_solver_bench/snapshots/`
  的冻结 Gram 快照上):
  `./run.sh run_lambda_solver_bench_without_256_checkpoints.py --T 60 --batches 20`。
- **实验一(K=2 与 K=6)**——每条命令把自己的全部腿串行跑进
  `output/ccp_compare_without_256_checkpoints/{K2_B20000,K6_B80912}/`:

```bash
./run.sh run_ccp_compare_K2_without_256_checkpoints.py --smoke
```

```bash
./run.sh run_ccp_compare_K2_without_256_checkpoints.py
```

  (K=6 同样用 `run_ccp_compare_K6_without_256_checkpoints.py`。)
- **审计:**`./run.sh audit_v2_K6_without_256_checkpoints.py`
  (`--quick` 只查每腿前 3 个 stack)给每条 K6 腿写 `audit_v2.json`
  ——即第 7 节的双仪器质量计。
- **画图:**`./run.sh plot_ccp_compare_without_256_checkpoints.py --which K2`
  (以及 `--which K6`)。
- **MNIST K=10 试跑**(报告里的"实验二";仓库内更早的文档编号为
  三——同一个实验)。从 `data/mnist/` 读 idx 文件;只有两条
  adaptive 腿(CCP vs IPOPT ts24):

```bash
./run.sh run_ccp_compare_K10_mnist_without_256_checkpoints.py --budget 55000 --per-class 1000 --batch 1024 --s 5 --ts 24
```

  然后 `./run.sh plot_ccp_compare_K10_mnist_without_256_checkpoints.py`。

### 9.8 实验四——K=2 MNIST 数字对 campaign(8月13日)

先选数字对(给 5 个候选对按冲突程度排名):

```bash
./run.sh run_conflict_smoke_K2_mnist_pairs_without_256_checkpoints.py
```

然后每个选定的对各跑一个 campaign(baseline r ∈ {10,20,40} +
adaptive CCP,B = 20,000,精确 1-D 质量计,train+test 两个 front):

```bash
./run.sh run_pure_budget_K2_mnist_pair_without_256_checkpoints.py --pair 3 5
```

(第二个对用 `--pair 7 9`;`--smoke` → `SMOKE/pair_3v5_B800/`。)
画图:`./run.sh plot_K2_mnist_pair_without_256_checkpoints.py`。
输出位置:`output/K2_mnist_pair_without_256_checkpoints/`。

## 10. 结果地图——7月之后的 output 目录一览

| output 目录 | 实验(记录所在) |
|---|---|
| `Pareto_front/` | K=2 认证 Pareto front,主跑 r=10 + r=20 复跑(各自 README + `FINDINGS_ZH.md`) |
| `lambda_path_K3/` | K=3 λ 路径图(README + 象限分析) |
| `without_256_checkpoints/` | 测量变体 A/B 复跑(`FINDINGS.md`) |
| `trial_K6_…_B180180_…/` | K=6 参考试跑(冻结的参考曲线) |
| `calibration_speed_test_B2000/` | 试跑前的速度标定——不是实验结果 |
| `fast_method_trials/` | fast 引擎系列 v1/v2/v3/v4 + 被中停的认证尝试 |
| `baseline_svrg_multi_r_vs_fast_without_256_checkpoints/` | 7月25日 r-sweep 目录(图风格冻结) |
| `baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/` | 对比正式归档:`original/`、`tol0.02/`、`tol0.01/`、`adaptive_extended/`、`diag_v3_plateau/`、`fixed_budget_B80912/`、`pure_budget_B80912/`(+各 SMOKE) |
| `bandit_toy_surf_…/`、`bandit_toy_K5_…/`、`bandit_toy_mv_…/` | SURF bandit toy K=2 / K=5 / mean-variance,各 eps 档 + smoke |
| `pure_budget_K2_…/B20000/` | K=2 纯固定预算,精确 1-D 质量计(`REPORT_ZH.md`) |
| `ccp_compare_…/K2_B20000/`、`…/K6_B80912/` | 实验一(CCP vs IPOPT vs 网格) |
| `ccp_compare_…/K10_mnist10k_B55000/` | MNIST K=10 试跑(报告"实验二",更早编号"三") |
| `ccp_compare_…/lambda_solver_bench/` | 研究 B:受控 λ 求解器基准 |
| `ccp_smoke_sampler/` | 研究 A:Exp(1) vs Sobol 种子采样器消融 |
| `K2_mnist_pair_…/` | 实验四:数字对冲突 smoke + pair_3v5 / pair_7v9 两个 campaign |

每条腿的文件约定:每腿都有 `summary.json`(完整曲线+参数+健康标志)
和 `grams.npz`(交付点的 Gram 矩阵);`thetas.npz`(参数向量)只在
MNIST 数字对的腿里;bandit 运行存 `raw_histories.npz`;被审计的
K6/K10 腿有 `audit_v2.json`;campaign 层面有 `campaign_manifest.json`
和 `*.log`。

## 11. 验证门禁——当前完整清单

| 门禁 | 预期 |
|---|---|
| `./run.sh sanity_checks_fast.py` | 8/8 PASS(Gram 路径 ≡ einsum、MSVRG 退化比特级一致、剪枝按位安全等) |
| `./run.sh sanity_checks_ccp.py` | 全部 PASS(LP 热启动 ≡ 冷启动、单调上升、K=2 ≡ 精确包络等) |
| `./run.sh sanity_checks_bandit_toy.py` | 9/9 |
| `./run.sh sanity_checks_bandit_toy_K5.py` | 9/9 |
| `./run.sh sanity_checks_bandit_toy_mv.py` | 全部 PASS |
| 旧版 `verify_fixes.py` / `prefix_repro.py`(仓库外,第一部分 §5) | 10 passed / duplicates: 39 |

改任何引擎文件之前先跑对应门禁;每项检查都必须打印 PASS。
已存结果的抽查命令(精确到 JSON 数值的预期)在仓库外的会话台账里。
