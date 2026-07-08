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
