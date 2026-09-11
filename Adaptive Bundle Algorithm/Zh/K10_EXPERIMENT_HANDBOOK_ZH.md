# K = 10 MNIST 全部数字实验手册

Experiment handbook for the K = 10 fixed-budget campaign (adaptive λ-bundle with CCP decisions versus uniform simplex grids, all ten MNIST digits, B = 500,000, seed 41). 版本 v1，2026 年 9 月 10 日。
英文版：`../K10_EXPERIMENT_HANDBOOK.md`；两个文件必须同步修改。

这份手册写给帮忙在 GPU 机器上运行实验的人。第一部分说明这个实验是什么，第二部分说明怎么运行、输出放在哪里、要交回什么。所有需要的代码都已经写好并在 CPU 上做过完整的小规模检验，运行者不需要改任何代码，只需要按第二部分的步骤执行命令。

---

## 第一部分：这个实验是什么

### 1.1 实验概括

问题是 MNIST 全部十个数字的十目标训练。第 k 个目标 F_k(θ) 是数字 k 的训练图上的平均交叉熵（cross-entropy），再加一个 ridge 项 (μ/2)‖θ‖²，μ = 1e-4。网络是一个小的三层网络（patch(5×5)-64 → dense-96 → 10 logits，8,874 个参数），激活函数用 softplus，所以每个目标都是光滑的。十个目标同时训练时，用权重 λ = (λ₁, …, λ₁₀) 把它们合成一个标量目标 F_λ = Σ_k λ_k F_k；λ 在九维单纯形上（各分量非负、和为 1）。每一个 λ 对应一个训练问题，"多目标训练"就是要把整个单纯形上的问题都解好。

实验比较两种"把权重 λ 放在哪里训练"的方法，两者用同一个网络、同一份数据、同一个内层求解器、同样的计算预算、同样的随机种子，唯一的差别是每一轮选哪个 λ 去训练：

* **uniform grid**（基线）：把单纯形等分成分辨率为 r 的格点，共 C(r+9, 9) 个，按蛇形顺序轮流在每个格点上训练，直到预算用完。
* **自适应 λ-bundle，决策用 CCP**（下文简称 CCP）：每一轮先在当前所有已交付模型上找出"最没被解决"的权重 λ（用 CCP 子问题求解一个最大-最小问题），就在这个 λ 上训练。

评价指标有两个：

* **主指标 worst GN**（worst-case gradient norm，最坏方向梯度范数）。对单纯形上每一个 λ，在已交付的全部模型里取梯度 ‖Σ_k λ_k ∇F_k(θ)‖ 最小的那个；再在所有 λ 上取最大值。它衡量"最难的那个权重被解到什么程度"，越小越好。
* **次指标 训练前沿的超体积**（hypervolume，HV）。把每个 run 交付的全部模型放到十维损失空间里，取非支配集，计算它相对参考点 (ln 10, …, ln 10) 支配的体积，用 Monte Carlo 估计。越大说明前沿覆盖得越好。

预算 B = 500,000 个"梯度评估当量"（见 1.3 的预算计量）。uniform grid 取 r ∈ {3, 4, 5, 6, 7}，格点数分别为 220、715、2,002、5,005、11,440；每档一个 run，CCP 一个 run，共 6 个 run，全部用种子 41。不做测试集评价，但保存每个 run 交付的全部参数 θ，供以后做测试集实验。

### 1.2 整体思路和步骤

实验分四个阶段，由同一个脚本的四个 `--stage` 完成：

1. **train**：6 个 run 串行训练。每个 run 在预算内不断交付模型（每一"段"训练结束交付一个 θ），并每隔 12,500 单位预算记一个 checkpoint（记录此刻的预算消耗、墙钟时间和已交付模型数）。训练在 GPU 上，CCP 的决策在 CPU 上。输出 summary.json、grams.npz（每个交付点的 10×10 梯度 Gram 矩阵和十个损失值）、thetas.npz（全部 θ）。
2. **audit**：事后在每个 checkpoint 的模型集合上计算 worst GN。K = 10 没有精确的计算方法，用 CCP 仪器搜索（它给出的是下界）；终点再用更重的仪器搜一次并做一次不含局部搜索的随机核验，取最大值作为表里的终值。这一阶段只用 CPU，放在全部训练之后做，避免它的 CPU 负载干扰其他 run 的墙钟计时。
3. **tables**：表 1（终值 worst GN 与倍数）和表 2（超体积）。
4. **figures**：图 1（全部曲线）和图 2（基线每个分辨率一个点）。

思路上与之前 K = 2、K = 3 的固定预算实验相同：固定预算下比较，基线每个分辨率概括成一个点，看 CCP 的曲线是否在所有点的下方。K = 10 特有的几点如下。

* **网格的结构事实。** 格点的坐标是 j/r，十个坐标同时非零需要 r ≥ 10，所以 r < 10 时没有任何格点同时训练十个类：r = 3 的格点最多同时训练 3 个类，r = 7 最多 7 个类。等权 λ（单纯形中心）到最近格点的 ℓ₁ 距离随 r 从 1.4 降到 0.6，而单纯形的直径是 2。第一个覆盖中心的分辨率是 r = 10，有 92,378 个格点，一遍轮转要约 950 万单位，是预算的 19 倍。这就是 K = 10 要说明的事：可负担的分辨率都盖不住单纯形内部，而 CCP 会自己找到最坏的权重去训练。
* **预算为什么是 500,000。** K = 10 每段的满支持成本是 30 单位，K = 3 是 9 单位，同样的预算买到的段数少三倍；500,000 下 CCP 约 2.2 万段。各档能完成的轮转数：r = 3 约 30 遍，r = 4 约 8.7 遍，r = 5 约 2.9 遍，r = 6 约 1.1 遍，r = 7 只访问 47% 的格点。r = 7 是"再加密也没用"的展示档。
* **预算计量与支持效应。** 一段 = 1 次锚点全梯度（十个目标各算一次，计 10 单位）+ 53 步小批修正（每步 1,024 行，小批只含 λ 中有权重的类，成本按实际用到的行数计）。因此每段的成本是 10 + 2.0 × 有权重的类数：格点段只含 3 到 7 个类，每段 16 到 19 单位；CCP 常在满支持附近训练，每段接近 30 单位。所有 run 用同一把尺子，格点因此买到更多段数；表里放格点数与遍数，不放段数。
* **点的定义（图 2）。** 每条基线的 best-so-far 曲线是阶梯状，点取最后一级台阶的起点，也就是曲线最后一次下降发生的 checkpoint；纵坐标是终值。

### 1.3 参数选择

| 项目 | 取值 |
|---|---|
| 数据 | MNIST 训练集十个数字各 5,421 张（受最少的数字 5 限制，平衡取法），共 n = 54,210；不用测试集 |
| 模型 | patch(5×5)-64 → dense-96 → 10 logits，softplus 激活，参数量 d = 8,874 |
| 目标 | 每个数字的平均交叉熵 + (μ/2)‖θ‖²，μ = 1e-4，全部参数含偏置；λ ∈ Δ₉ |
| 内层求解器 | SVRG 段：1 次锚点全梯度 + m = 53 步小批修正步（一遍数据），批 1,024（每类 102 到 103 张）；光滑常数 L 由 40 对随机探针估计，再加 μ |
| 步长规则 | 所有方法统一用 Adam，α = 1e-3，β₁ = 0.9，β₂ = 0.9；λ 变化时清零动量，某段不下降则清零并把 α 减半 |
| uniform grid | r ∈ {3, 4, 5, 6, 7}，格点数 220 / 715 / 2,002 / 5,005 / 11,440，蛇形顺序轮转，每次访问训练 s = 5 段，上一个格点的解作热启动 |
| CCP | 每次决策采 N₀ = 2,000 个 λ，筛出 10 个重启点做 CCP 上升，pool 上限 30，停止阈值 τ = 1e-8·max(1, φ)；选中的权重训练 s = 5 段 |
| 预算 | 每个 run B = 500,000 梯度评估当量；每 12,500 单位记一个 checkpoint，共 40 个 |
| worst GN 度量 | 每个 checkpoint：CCP 仪器 N₀ = 8,192、r = 20，全新启动。终点：CCP N₀ = 32,768、r = 20 两个采样种子，加十万个均匀随机 λ 逐点精确算 GN 作核验，取最大。表里报告范数尺度（平方根） |
| 超体积 | 训练损失空间（含 ridge 项）；参考点 (ln 10, …, ln 10)；每个 run 交付的全部点取非支配集；Monte Carlo 一百万个样本 |
| 时间轴 | 各 run 的墙钟时间，训练在 GPU、决策在 CPU，审计不计入；6 个 run 串行、机器空闲 |
| 种子 | 采样 41、初始化 8、探针 7，全部 run 相同；同一初始点 |
| 硬件 | 训练：NVIDIA GPU，float64；决策与审计：CPU |

### 1.4 期待结果

* CCP 的 worst GN 终值明显低于所有 uniform grid，曲线在所有点的下方。预期倍数比 K = 3 的 6 倍更大，原因是 1.2 里的结构事实：r ≤ 7 的格点从不训练一个同时看十个类的模型。
* 倍数随 r 不一定单调：粗网格卡在格点间隙的下限，细网格在预算内走不完一遍。
* 超体积上 uniform 远低于 CCP，见 1.6 的说明。
* 时间轴上 CCP 的曲线约为 uniform 的 2 到 2.5 倍长（λ 搜索的决策开销），但仍应在所有点的下方。

### 1.5 冒烟实验说明

`--smoke` 是一个一分钟的小规模检验，只检查整条链路能不能跑通，不产生任何科学结论：每类 300 张图、预算 800、每次访问 2 段、uniform 只跑 r = 2、审计和超体积的样本数都缩小，全部在 CPU 上。它会依次跑 train → audit → tables → figures，最后打印 `SMOKE OK`。输出在单独的 SMOKE 目录里，不会碰正式实验的目录。运行者在 GPU 机器上装好环境后先跑一遍它（第二部分 Step 0），通过了再做后面的事。

### 1.6 需要提前说明的事项

1. **表 2 的结构性结果。** 超体积的参考点是 (ln 10, …, ln 10)，即随机猜测的损失。uniform 的格点最多训练 r 个类，没训练的类的交叉熵会超过 ln 10 = 2.30，这样的交付点落在参考区域之外，对超体积贡献为零。8 月的 K = 10 试验里，训练 2 个类以内的交付点 0% 落在区域内，3 个类 25%，5 个类 57%，8 个类以上 99%。所以 uniform 的超体积会远低于 CCP；这张表记录的是"网格从不交付十类都会的模型"这个事实，不是前沿形状的细微差别。表 2 因此多给两列诊断：交付点里落在参考区域内的比例，和"最均衡的模型"的最大类损失 min_θ max_k F_k(θ)。
2. **时间轴的含义。** 图上的时间是同一台机器上的墙钟：训练在 GPU 上，λ 搜索的决策在 CPU 上，审计不计入。CCP 的曲线预计约为 uniform 的 2 到 2.5 倍长，来自决策开销；图注要写明机器型号和这一点。
3. **worst GN 是下界。** 没有 K = 10 的精确量表；仪器返回的是它搜到的最坏 λ 的值，真值只可能更大。各 run 用同一套仪器，比较是公平的；终点用更重的仪器加随机核验，是为了把进表的数字搜紧。
4. **CCP 决策步的实现。** 每次 CCP 迭代解一个线性规划（LP）：max t，约束 2(M_i λ_c)ᵀλ − λ_cᵀM_iλ_c ≥ t 对每个已交付点 i 成立，λ 在单纯形上。原实现把 LP 的系数逐个传给 HiGHS，m 个点就要 m×10 次 Python 到 C++ 的调用，在 m 达两万时一次决策要 40 秒，整条 CCP run 的决策要 20 小时。本实验用新文件 `ccp_lambda_solver_bulk.py`，把 LP 整块传入并保留上一次的基（basis）做热启动。LP 本身、CCP 的迭代规则、τ、重启数、pool 都没有变。门检（`sanity_checks_ccp_bulk.py`）在记录的 1,371 个 LP 上验证两种写法给出相同的最优值（相对差 8.7e-13）和相同的 λ。但要说明：整条决策序列不是逐位相同的。CCP 多起点里有离散选择（筛选排序、pool 去重、停止判断），LP 解在最后几位的差别在第 170 次决策处翻转了一个离散选择，之后两条路径各自访问不同但同样合法的局部极大点，与换一台 CPU 或换一个 HiGHS 版本运行原求解器是同一种效应。在 K = 3 的 600 次决策上两条序列逐位一致。K = 10 的 475 次决策上，两条序列的 φ 相对差中位数为 0、均值 +0.8%、最大 20%，新写法给出的值在 93% 的决策上不低于原写法，终点 φ 分别为 7.90e-2 与 7.78e-2；也就是说新写法没有系统性地搜得更差。门检报告存在 `output/CCP/ccp_compare_without_256_checkpoints/K10_mnist10k_B55000/adaptive_s5_ccp/gate_ccp_bulk.json`。
5. **单种子。** 全部 run 用种子 41，与 K = 2、K = 3 相同。结论是单种子下的比较。

---

## 第二部分：这个实验怎么运行

### 2.1 环境准备

需要一台有 NVIDIA GPU 的 Linux 机器（任何 CUDA 卡都可以，float64 不改：这个网络很小，每段的时间由 kernel 启动次数决定，不由 float64 算力决定；显存 1 GB 足够）。Mac 的 GPU（MPS）不支持 float64，不能用。

```bash
git clone https://github.com/MarshmallowJeffrey/First-order-method-smooth-MOO.git
cd First-order-method-smooth-MOO
git checkout mlp-comparison-results   # 实验代码在这个分支上，不在 main
python3 -m venv .venv            # Python 3.10 或更新
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib highspy
# torch 的 CUDA 版按 https://pytorch.org/get-started/locally/ 选与机器 CUDA 版本匹配的命令，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch, highspy; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一行应打印 `True` 和显卡型号。不需要 cyipopt（K = 10 不用 IPOPT），不需要 torchvision。MNIST 数据（10 MB）第一次运行时自动下载到 `Adaptive Bundle Algorithm/data/mnist/`；机器不能联网时，把这个目录从别处拷过来即可。

所有命令都在这个目录下执行：

```bash
cd "First-order-method-smooth-MOO/Adaptive Bundle Algorithm/Original_py/experiment_plot"
```

### 2.2 代码文件清单

本实验新写的文件（都在 `Adaptive Bundle Algorithm/Original_py/` 下），运行者不需要修改：

| 文件 | 作用 |
|---|---|
| `objective/objectives_mnist_patch_k10.py` | K = 10 的目标函数：每类 5,421 张的数据、ridge 项、按类前向的联合梯度、device 选择（cpu / cuda） |
| `Core Engine/ccp_lambda_solver_bulk.py` | CCP 决策求解器，LP 整块写入 + basis 热启动（见 1.6 第 4 条） |
| `experiment_plot/run_dots_K10_without_256_checkpoints.py` | 主脚本：train / audit / tables / figures 四个阶段，`--smoke` 冒烟 |
| `experiment_plot/calibrate_gpu_K10_without_256_checkpoints.py` | GPU 校准脚本：数值与 CPU 一致性、确定性、每段耗时、决策耗时；写 `calibration_K10.json` |
| `sanity_check/sanity_checks_k10_objective.py` | 目标函数门检（已在开发机上通过） |
| `sanity_check/sanity_checks_ccp_bulk.py` | 求解器等价性门检（已在开发机上通过，见 1.6 第 4 条） |

复用的已有文件（不需要关心细节）：`plot_curves_family_without_256_checkpoints.py` 与 `plot_dots_figure_without_256_checkpoints.py`（画图 1、图 2），`baseline/baseline_without_256_checkpoints.py`（单纯形格点与蛇形排序），`Core Engine/stepper_core.py`（Adam 步长），`Core Engine/ccp_lambda_solver.py`（CCP 基类），`objective/objectives_mnist_patch.py`（网络与数据读取）。

### 2.3 运行步骤

每一步的命令都在 `experiment_plot/` 目录下执行，前面加上 venv 里的 `python`。长时间的步骤用 `nohup … &` 或 `tmux`，让它不受 SSH 断开影响；日志用 `tee` 同时写到文件。

**Step 0：冒烟（1 分钟，CPU）**

```bash
python run_dots_K10_without_256_checkpoints.py --smoke 2>&1 | tee smoke_K10.log
```

最后一行必须是 `SMOKE OK`。输出在 `output/CCP/K10_mnist_without_256_checkpoints/SMOKE/`，可以删掉也可以留着。

**Step 1：GPU 校准（2 到 3 分钟）**

```bash
python calibrate_gpu_K10_without_256_checkpoints.py --device cuda 2>&1 | tee calibrate_K10.log
```

最后一行必须是 `[calib] PASS -> …/calibration_K10.json`。它检查四件事：C1 GPU 与 CPU 在同一 θ 上的十个损失值和梯度相对差 ≤ 1e-10；C2 同一段在 GPU 上跑两遍逐位相同；C3 每段耗时并按 B = 500,000 推算训练总时间（看日志里 `projected training for all legs` 的分钟数）；C4 一次 CCP 决策使用的是 `highspy-bulk` 后端。**若 C2 不通过**，加 `--deterministic` 再跑一次，并在 RUN_NOTES.md 里写明是用这个开关通过的；若仍不通过，停下来把 `calibrate_K10.log` 发回来。正式训练脚本在 cuda 上要求存在通过的 `calibration_K10.json`，否则会拒绝启动。

**Step 2：训练（约 4 到 5 小时，GPU；机器必须空闲）**

```bash
nvidia-smi        # 确认 GPU 上没有别的进程
top -bn1 | head   # 确认 CPU 空闲
nohup python run_dots_K10_without_256_checkpoints.py --stage train --device cuda \
      > train_K10.log 2>&1 &
tail -f train_K10.log
```

顺序是 CCP 一个 run，然后 uniform r = 3、4、5、6、7。每个 run 每完成 10% 预算打印一行进度（段数、墙钟、决策时间）；结束时打印预算消耗、段数、墙钟、决策占比、支持直方图，并写出 summary.json、grams.npz、thetas.npz。断了可以直接重跑同一条命令：已完成的 run（目录里有 summary.json）自动跳过，未完成的从头跑。**训练期间机器上不能跑别的任务**，因为墙钟时间要进图 2 的时间轴。

**Step 3：审计（约 1 小时，CPU）**

```bash
nohup python run_dots_K10_without_256_checkpoints.py --stage audit > audit_K10.log 2>&1 &
```

对每个 run 的 40 个 checkpoint 和终点计算 worst GN，写 audit.json 并把结果并入 summary.json；日志每个 run 一行，含终值和终点三项仪器里哪一项胜出。已审计的 run 自动跳过。这一步也可以不在 GPU 机器上做：把每个 run 的 summary.json 和 grams.npz 拷到任何一台有 Python 环境的机器上运行同一条命令即可。

**Step 4：表格（约 30 分钟，CPU）**

```bash
python run_dots_K10_without_256_checkpoints.py --stage tables 2>&1 | tee tables_K10.log
```

写 `table1_K10.md/json` 和 `table2_K10.md/json`。表 2 的 Monte Carlo 一百万个样本，每个 run 几分钟。

**Step 5：图（1 分钟）**

```bash
python run_dots_K10_without_256_checkpoints.py --stage figures 2>&1 | tee figures_K10.log
```

写 `worst_gn_curves_uniform_all_laststep.png`（图 1）和 `worst_gn_dots_paper.png` 加同名 .json/.md（图 2）。

**Step 6：打包交回**（见 2.9）

以上 Step 2 到 5 也可以一条命令连续执行：`--stage all`。分开执行的好处是训练结束后就能先把小文件交回来。

### 2.4 输出目录与文件

实验的输出全部在仓库内的这个目录（"home"）下，与 K = 2、K = 3 的输出并列放在 `output/CCP/` 里：

```
Adaptive Bundle Algorithm/output/CCP/K10_mnist_without_256_checkpoints/
├── calibration_K10.json                 Step 1 的校准结果（机器、耗时、PASS）
├── SMOKE/                               Step 0 的冒烟输出（可删）
└── mu0.0001/dots_B500000/adam_1e-3_b0.9/          <- home
    ├── campaign_manifest.json           每个 run 的墙钟、决策时间、段数、机器信息、git 提交号
    ├── RUN_NOTES.md                     运行者手写的记录（见 2.5）
    ├── adaptive_ccp_seed41/
    │   ├── summary.json                 参数、预算消耗、checkpoint、墙钟、决策时间、CCP 遥测、审计结果
    │   ├── grams.npz                    每个交付点的 Gram 矩阵 (m,10,10)、损失值 (m,10)、λ 历史
    │   ├── thetas.npz                   每个交付点的参数 θ (m, 8874)，float64，每个 run 1.5 到 2.4 GB
    │   └── audit.json                   Step 3 的审计明细
    ├── uniform_r3_seed41/  … uniform_r7_seed41/      同上，每档一个目录
    ├── table1_K10.md / table1_K10.json  表 1
    ├── table2_K10.md / table2_K10.json  表 2
    ├── worst_gn_curves_uniform_all_laststep.png      图 1
    └── worst_gn_dots_paper.png / .json / .md         图 2 及其数字
```

目录名里的 `mu0.0001`、`dots_B500000`、`adam_1e-3_b0.9` 分别记录 ridge 系数、预算和步长核，与 K = 3 的目录 `K3_mnist_triple_without_256_checkpoints/v2_stepper_mu0.0001/dots_B100000/adam_1e-3_b0.9/` 同一套命名。`thetas.npz` 被仓库的 .gitignore 排除（超过 GitHub 的单文件上限），其余文件都可以提交。

### 2.5 需要记录的 log、json 和 md

脚本自动产生的：每个 run 的 `summary.json`（含 `machine` 字段：平台、torch 版本、显卡型号、CPU 核数、线程数、git 提交号、开始与结束时间）、`audit.json`、`campaign_manifest.json`、`calibration_K10.json`，以及各阶段的控制台日志（`smoke_K10.log`、`calibrate_K10.log`、`train_K10.log`、`audit_K10.log`、`tables_K10.log`、`figures_K10.log`，由 `tee`/`nohup` 写出，请一并交回）。

运行者手写的：在 home 目录下建一个 `RUN_NOTES.md`，内容按下面的模板填：

```markdown
# K10 run notes
- Machine: (hostname, GPU model, CUDA version, CPU model, cores, RAM)
- Python / torch / highspy versions: (from `pip list`)
- Repo commit: (git rev-parse --short HEAD)
- Step 0 smoke: date, SMOKE OK? 
- Step 1 calibration: date, PASS? deterministic flag used? projected minutes printed:
- Step 2 train: start/end time of each run; anything else running on the machine? any restart?
- Step 3 audit: date, machine used (same GPU box or another), wall time
- Step 4/5 tables and figures: date
- Anomalies: (crashes, warnings in the logs, re-runs, machine sleep, other users)
```

### 2.6 CPU 与 GPU 的分工和注意事项

训练段的前向和反向在 GPU 上；θ、梯度、Adam 更新、预算计量、CCP 决策（HiGHS 的线性规划）、审计（CCP 仪器与随机核验）、表格和图都在 CPU 上。每步训练把 θ 拷到 GPU、把梯度拷回 CPU（各 71 KB），开销约 0.3 毫秒。决策时 GPU 空转、训练时 CPU 空转是算法的顺序依赖，属正常，无法重叠。

要注意的：

1. **数值与可复现。** GPU 的 float64 矩阵乘与 CPU 在最后几位不同（求和顺序、乘加融合），所以 GPU 上的训练轨迹不可能与 CPU 逐位相同，也不需要。校准脚本检查的是：同一个 θ 上 GPU 与 CPU 的损失值和梯度相对差 ≤ 1e-10；同一段在 GPU 上跑两遍逐位相同。后者若不通过，用 `--deterministic` 重跑校准（它打开 torch 的确定性算法开关），并记录。这个网络的反向没有原子加操作，预期是确定的。
2. **计时。** CUDA 是异步的；脚本在每个 checkpoint 和 run 结束时调用 `torch.cuda.synchronize()` 再读时钟，每步把梯度拷回 CPU 的操作也自带同步。运行者不需要做什么，只需要保证机器空闲。
3. **时间轴的含义。** 图 2 的时间是同一台机器上的墙钟：训练在 GPU、决策在 CPU，图注要写明机器型号和这一点。
4. **机器要空。** 6 个 run 串行，训练期间 GPU 和 CPU 上不能有别的任务（`nvidia-smi` 和 `top` 各看一眼）。不要在训练时同时跑审计。若机器是共享的，请选一个没人用的时段。笔记本请插电并关掉休眠。
5. **审计放在训练之后**，或在另一台机器上做（只需要每个 run 的 summary.json 和 grams.npz，共约 200 MB）。
6. **线程。** 默认用 torch 的线程数；若想固定，加 `--threads 8`。HiGHS 单线程。
7. **显存与内存。** 数据 340 MB 加每类 5,421 行的前向激活约 70 MB，任何卡都够；内存里最大的对象是 θ 栈（每个 run 1.5 到 2.4 GB，训练结束时压缩写盘），16 GB 内存足够。
8. **磁盘。** 6 个 run 的 thetas.npz 共约 10 到 12 GB，其余文件不到 300 MB。磁盘不够时可以加 `--thetas-float32`（θ 存成 float32，体积减半；只影响以后测试集实验的精度，不影响本实验的任何数字）。
9. **中断与重跑。** 训练脚本按 run 断点续跑：目录里有 summary.json 的 run 跳过，没有的从头跑。一个 run 中途被杀会丢掉这个 run 的进度（约 40 分钟），不会影响其他 run。

### 2.7 成果的标准

**图 1**（`worst_gn_curves_uniform_all_laststep.png`）：1 × 2 面板。左图横轴是梯度评估量（线性），右图横轴是墙钟秒；纵轴都是对数刻度的 best-so-far worst GN（范数尺度）。CCP 一条粗曲线，uniform r = 3 到 7 五条细曲线，每条基线曲线上一个圆点标它最后一次下降的位置。所有曲线从同一个起点出发（x = 0 处只有初始点 θ₀）。

**图 2**（`worst_gn_dots_paper.png`）：1 × 2 面板，同样的两个横轴（默认对数刻度）。CCP 画曲线；uniform 每个 r 一个方点，点在各自曲线最后一次下降的 checkpoint，纵坐标为终值，按横坐标顺序虚线相连，标签为 r 的数值。与它同名的 .md 给出每个点的数字（终值、点的横坐标、CCP 在同一预算处的值和倍数）。

**表 1**（`table1_K10.md`）：列为 方法、r、格点数、遍数、终值 worst GN（范数尺度）、方法 / CCP。遍数 = 该 run 实际消耗的预算 / 该网格一遍的预算（按支持效应计的每段成本）。不放段数。

**表 2**（`table2_K10.md`）：列为 run、HV（train，Monte Carlo，附标准误）、与 CCP 的差距（(HV_CCP − HV_run)/HV_CCP）、落在参考区域内的交付点比例、参考区域内的非支配点数、最均衡模型的最大类损失。论文只用前三列，后三列是解释差距来源的诊断。

### 2.8 出了问题怎么办

* `SMOKE OK` 没出现：看 `smoke_K10.log` 最后的报错。常见原因是包没装全（`ModuleNotFoundError`：按 2.1 补装）、MNIST 下载失败（机器不能联网：把 `Adaptive Bundle Algorithm/data/mnist/` 目录拷过来）。
* 校准 C1 不通过（GPU 与 CPU 数值差 > 1e-10）：把 `calibrate_K10.log` 发回来，不要继续。
* 校准 C4 提示 `highspy missing`：`pip install highspy`；没有它决策会退到 scipy 的冷启动求解，慢几十倍。
* 训练脚本报 `GPU training requires a passed calibration`：先跑 Step 1。
* 训练中途机器重启或进程被杀：重跑 Step 2 的同一条命令，完成的 run 会跳过。
* `nvidia-smi` 显示别人在用 GPU：等空闲再跑，已经跑完的 run 若受了干扰（墙钟明显偏长），删掉那个 run 的目录重跑。
* 只有在排查代码本身的问题时才需要参考之前的实验：K = 3 的同类脚本是 `run_dots_K3_without_256_checkpoints.py`（它的执行器与本脚本几乎相同，只是问题是三个数字、审计用 IPOPT），K = 2 的是 `run_dots_K2_without_256_checkpoints.py`；它们的输出在 `output/CCP/K3_mnist_triple_without_256_checkpoints/` 和 `output/CCP/K2_mnist_pair_without_256_checkpoints/` 下，可以对照 summary.json 的字段。除此之外不需要了解 K = 2、K = 3 的实验。

### 2.9 交付清单

训练结束后（Step 2 完成即可先交一次，审计和表图之后再交一次）：

1. home 目录下除 `thetas.npz` 以外的全部文件（6 个 run 的 summary.json、grams.npz、audit.json，campaign_manifest.json，表和图），约 300 MB；
2. `calibration_K10.json` 与全部 `*.log`；
3. `RUN_NOTES.md`；
4. 6 个 `thetas.npz`（约 10 到 12 GB），可以晚一些用网盘或硬盘交，或先留在机器上。

打包命令示例（在仓库根目录）：

```bash
cd "Adaptive Bundle Algorithm/output/CCP/K10_mnist_without_256_checkpoints"
tar --exclude='thetas.npz' -czf K10_results_$(date +%Y%m%d).tgz mu0.0001 calibration_K10.json
```

---

## 附录：命令速查

```bash
cd First-order-method-smooth-MOO && git checkout mlp-comparison-results
cd "Adaptive Bundle Algorithm/Original_py/experiment_plot"
source ../../../.venv/bin/activate
python run_dots_K10_without_256_checkpoints.py --smoke 2>&1 | tee smoke_K10.log                 # Step 0, 1 min
python calibrate_gpu_K10_without_256_checkpoints.py --device cuda 2>&1 | tee calibrate_K10.log  # Step 1, 3 min
nohup python run_dots_K10_without_256_checkpoints.py --stage train --device cuda > train_K10.log 2>&1 &   # Step 2, ~4-5 h
nohup python run_dots_K10_without_256_checkpoints.py --stage audit > audit_K10.log 2>&1 &                 # Step 3, ~1 h
python run_dots_K10_without_256_checkpoints.py --stage tables 2>&1 | tee tables_K10.log         # Step 4, ~30 min
python run_dots_K10_without_256_checkpoints.py --stage figures 2>&1 | tee figures_K10.log       # Step 5, 1 min
```

预计耗时以校准脚本推算的分钟数为准；上表的小时数是按每段 60 到 90 毫秒估计的。
