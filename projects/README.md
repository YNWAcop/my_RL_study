# 两份导入项目的使用指南

这两个项目在 2026-11-09 从 Code_1_minimal.zip 和 Code_2.zip 导入。本目录沿用 RL 根目录的 Git 仓库。根目录已有的 lessons/、rlstudy/、docs/ 继续作为主学习项目，导入内容集中放在 projects/。

```text
RL/
  README.md
  .venv/                         已有学习项目的 Python 环境
  lessons/                       已有的算法练习
  rlstudy/                       已有的公共模块
  projects/
    README.md                    本指南
    IMPORTS.json                 来源压缩包 SHA-256、文件数量及文件名修复记录
    Code_1_minimal/
      README.md                  原项目说明
      执行清单.md                后续调度项目计划
      docs/                      PPO 实验结果、汇报和精读批注
      ref/cleanrl/cleanrl/ppo.py  PPO 训练入口
      scripts/                   图表生成脚本
      data/iris.csv              kNN 演示数据
    Code_2/
      README.md                  原整理者的环境记录
      check_env.py               环境自检
      smoke_lessons.py           六组课程代码快速检查
      docs/第三周周报.md         从 .idea 中复制出的学习周报
      lesson1/                   格子环境
      lesson2/                   SARSA 和 Q-learning
      lesson3/dqn/               DQN
      lesson4/policy_gradient/   策略梯度
      lesson5/ddpg/              连续动作 DDPG
```

**本次整理与验证**

压缩包内每个文件均已解压，并与原条目逐字节比较。Code_2 中四个中文文件名由 ZIP 的旧编码恢复。所有 Python 源文件都做了语法解析检查。原有 .idea 和 __pycache__ 文件保存在本地，由 Git 忽略；周报额外复制到 docs/，便于阅读和版本管理。原项目代码和依赖清单保留原样。

导入时检查了现有 .venv，它使用 Python 3.9.12，已有 NumPy 1.26.4、PyTorch 2.5.1、Gymnasium 1.0.0 和 Matplotlib 3.9.4。tyro、tensorboard、gym、paddlepaddle 和 parl 尚未安装。

下方命令根据压缩包中的说明整理。本次完成了解压、字节核对、语法检查和 Git 忽略规则检查，依赖安装及训练命令尚未在本机执行。两套旧项目分别创建环境，便于按各自记录使用对应依赖。环境目录 .venv-paper/ 和 .venv-parl/ 已加入忽略规则。

**先阅读什么**

1. 刚开始学习时，先看 Code_2/README.md，再读 lesson2 的两个 agent.py，对照 sample、predict、learn。
2. 然后读 lesson3/dqn/train.py，从入口跟进 agent.py、algorithm.py、model.py 和 replay_memory.py。
3. 理解策略梯度后，读 Code_1_minimal/docs/S234_汇报.md，再看 S4_ppo精读批注.md 和 ppo.py。
4. Code_1_minimal 的调度环境、手写 PPO、GIN 编码器仍是执行清单中的后续工作。algo/、envs/、models/ 等目录目前只有空骨架。

**当前已有学习项目的使用方式**

以下命令在 RL 根目录的 PowerShell 终端执行，使用现有 .venv。

```powershell
Set-Location 'E:\UserData\Documents\ChatGPT\RL'
.\.venv\Scripts\python.exe -m lessons.03_td_control --algo q_learning --episodes 1000 --seed 0
```

已有学习路线和 VS Code 运行方法见 [主项目 README](../README.md) 和 [VS Code 指南](../docs/08_vscode.md)。

**Code_1_minimal 的环境和 PPO 训练**

先在支持 conda 的 PowerShell 终端中回到 RL 根目录，创建 Python 3.12 环境。下面的 Torch 与 Gymnasium 版本来自原包的实测记录，其余依赖沿用原包列出的包名。

```powershell
Set-Location 'E:\UserData\Documents\ChatGPT\RL'
conda create --prefix .\.venv-paper python=3.12 -y
conda activate 'E:\UserData\Documents\ChatGPT\RL\.venv-paper'
python -m pip install torch==2.9.1 gymnasium==1.4.0 tyro tensorboard numpy pandas matplotlib scipy seaborn
Set-Location .\projects\Code_1_minimal

# 短跑检查，用于确认开始训练并能输出回报
python .\ref\cleanrl\cleanrl\ppo.py --env-id CartPole-v1 --total-timesteps 10000 --seed 1 --no-cuda

# 训练较完整的基线，运行时间随机器变化
python .\ref\cleanrl\cleanrl\ppo.py --env-id CartPole-v1 --total-timesteps 500000 --seed 1 --exp-name baseline --no-cuda
```

原包 README 的 5 万步达到 400 分门槛与 logs/S2_baseline.md 的实测记录有出入。实测记录中，5 万步末 10 局均值为 211.4，50 万步为 500。后续比较时要同时记录训练步数、随机种子、回报和训练过程中的波动。

在另一个终端激活同一个环境，并进入 Code_1_minimal 目录，可以启动 TensorBoard。

```powershell
conda activate 'E:\UserData\Documents\ChatGPT\RL\.venv-paper'
Set-Location 'E:\UserData\Documents\ChatGPT\RL\projects\Code_1_minimal'
tensorboard --logdir .\runs
```

浏览器打开终端给出的地址，查看 charts/episodic_return。S3 操作手册列出了参数对照命令。现有图表来自原包中的汇总值，原始 S3 训练日志和 TensorBoard 事件文件没有随包附上。重新运行训练会生成新的 runs/。

```powershell
# 在 Code_1_minimal 目录和对应环境内重画已有汇总图
python .\scripts\make_s234_figures.py
python .\scripts\make_knn_figures.py
```

结果文档中的部分机理解释需要结合源码复核，例如 clip_coef=0 时“策略梯度恒为零”的表述过强。现有消融结果只使用一个随机种子，后续实验应保留多个种子的原始结果。

**Code_2 的课程环境和运行**

Code_2 使用旧版 Gym、PaddlePaddle 和 PARL。下面的版本组合来自原整理者 README 中的本机记录。原包 requirements_动态图.txt 和两份课程原始 README 记的是更早的依赖组合。

从 RL 根目录创建课程环境。安装旧版 Gym 的工具链版本也按原记录固定。

```powershell
Set-Location 'E:\UserData\Documents\ChatGPT\RL'
conda create --prefix .\.venv-parl python=3.9 pip=23.3.1 setuptools=65.5.0 numpy=1.23.5 scipy=1.13.1 matplotlib=3.9.2 tqdm=4.67.1 -y
conda activate 'E:\UserData\Documents\ChatGPT\RL\.venv-parl'
python -m pip install paddlepaddle==2.6.2 parl==2.2.1 gym==0.21.0 pyglet==1.5.27 cloudpickle==1.6.0 pillow==11.1.0

# 原包记录的 Windows 后端适配，在启动 Python 之前设置
$env:PARL_BACKEND = 'paddle'
Set-Location .\projects\Code_2
python .\check_env.py
python .\smoke_lessons.py
```

每次重新开终端使用课程时，都要激活 .venv-parl 并设置 PARL_BACKEND。check_env.py 应逐项输出 [ OK ]，smoke_lessons.py 应最终输出“全部通过”。快速检查通过后，再进入具体课程目录训练。

```powershell
# 在 Code_2 根目录和对应环境内执行
Push-Location .\lesson2\q_learning
python .\train.py
Pop-Location

Push-Location .\lesson2\sarsa
python .\train.py
Pop-Location

Push-Location .\lesson3\dqn
python .\train.py
Pop-Location

Push-Location .\lesson4\policy_gradient
python .\train.py
Pop-Location

Push-Location .\lesson5\ddpg
python .\train.py
Pop-Location
```

每条训练命令会占用终端，前一个结束后再执行下一条。格子世界的 turtle 图形界面需要桌面会话。lesson2 的原始训练循环没有每局步数上限，如果策略原地打转，可以 Ctrl+C 停止。周报提到的独立实验脚本和 results/ 未包含在原包里。根目录 main.py 是 PyCharm 示例文件，课程训练入口是各目录中的 train.py。

在 VS Code 中，可用“打开文件夹”分别打开 projects/Code_1_minimal 或 projects/Code_2，然后通过“Python: 选择解释器”选择 RL/.venv-paper/python.exe 或 RL/.venv-parl/python.exe。终端运行前进入上面的相应工作目录。

**日常文件与 Git 管理**

修改原课程代码可直接在对应 projects 子目录里进行，学习笔记可放在根目录 notes/。先保留基线，再一次改一个参数，记录种子、命令和结果。要提交的精简结果表、解释和图可以挑选放入 notes/。

两份导入项目共用根目录 Git 仓库。现有分支继续使用；.venv、额外环境、IDE 设置、缓存、训练日志和模型产物均按忽略规则保留在本地。CleanRL 的 LICENSE 和课程源码中的版权说明随导入保留。

```powershell
Set-Location 'E:\UserData\Documents\ChatGPT\RL'
git status --short
git diff
# 按实际改动选择具体文件加入暂存区
git add -- projects/Code_2/lesson3/dqn/agent.py
git diff --cached
git commit -m "Study DQN action selection"
```

需要同步到已配置的远端时，再执行 git push。现有远端说明见 [GitHub 同步](../docs/06_sync.md)。
