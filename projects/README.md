# 两份导入项目的使用指南

两份压缩包已整理到本目录，并纳入 RL 根目录的 Git 仓库。三套代码统一使用现有 RL/.venv，完整安装记录见 [统一环境指南](../docs/09_unified_environment.md)。

```text
RL/
  .venv/                         三套代码共用的现有环境
  lessons/                       主学习项目的算法练习
  rlstudy/                       主学习项目的公共模块
  projects/
    README.md                    本指南
    IMPORTS.json                 来源 SHA-256 与中文文件名修复记录
    Code_1_minimal/
      docs/                      PPO 汇报、实验结果与精读批注
      ref/cleanrl/cleanrl/ppo.py  PPO 训练入口
      scripts/                   两个绘图脚本
      data/iris.csv              kNN 演示数据
    Code_2/
      check_env.py               环境自检
      smoke_lessons.py           课程快速检查
      docs/第三周周报.md         从 .idea 中复制出的学习周报
      lesson1/                   格子环境
      lesson2/                   SARSA 和 Q-learning
      lesson3/dqn/               DQN
      lesson4/policy_gradient/   策略梯度
      lesson5/ddpg/              连续动作 DDPG
```

**先运行检查**

以下命令从 RL 根目录的 PowerShell 终端执行。原包 README 和依赖文件记录了历史版本，当前本机的用法以统一环境指南为准。

```powershell
Set-Location 'E:\UserData\Documents\ChatGPT\RL'
.\.venv\Scripts\python.exe .\projects\Code_2\check_env.py
.\.venv\Scripts\python.exe .\projects\Code_2\smoke_lessons.py
.\.venv\Scripts\python.exe -m pytest -q
```

**学习顺序与训练入口**

1. 看 [Code_2 项目说明](Code_2/README.md)，再读 lesson2 两个 agent.py，对照 sample、predict 和 learn。
2. 读 lesson3/dqn/train.py，跟进 agent.py、algorithm.py、model.py 和 replay_memory.py。
3. 理解策略梯度后，读 [PPO 汇报](Code_1_minimal/docs/S234_汇报.md) 和 [PPO 精读批注](Code_1_minimal/docs/S4_ppo精读批注.md)。
4. 装配调度是 Code_1 的后续目标，手写 PPO、调度环境和 GIN 编码器仍在 [执行清单](Code_1_minimal/执行清单.md) 中。algo/、envs/、models/ 等目前只有空骨架。

运行课程训练时，保存解释器绝对路径，再进入具体课程目录。

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
Push-Location .\projects\Code_2\lesson3\dqn
& $python .\train.py
Pop-Location
```

将目录换成 lesson2/q_learning、lesson2/sarsa、lesson4/policy_gradient 或 lesson5/ddpg 即可运行其他课程。Code_2 根目录 main.py 是 PyCharm 示例。lesson2 原训练循环没有每局步数上限，策略打转时可 Ctrl+C 停止；turtle 地图显示需要桌面会话。

从根目录短跑 PPO，检查终端的回报输出。

```powershell
.\.venv\Scripts\python.exe .\projects\Code_1_minimal\ref\cleanrl\cleanrl\ppo.py --env-id CartPole-v1 --total-timesteps 10000 --seed 1 --no-cuda
```

完整基线可将步数改为 500000。训练事件写入根目录 runs/，另开终端启动 TensorBoard。

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir .\runs
```

Code_1 的 README 中，5 万步达到 400 分的门槛与 [实测日志](Code_1_minimal/logs/S2_baseline.md) 有出入。实测记录中，5 万步末 10 局均值为 211.4，50 万步为 500。消融结果只有一个随机种子，后续要多种子验证；“clip_coef=0 时策略梯度恒为零”的表述需结合源码复核。

重画已有图表可以执行以下命令，输出到 Code_1_minimal/docs/img/。

```powershell
.\.venv\Scripts\python.exe .\projects\Code_1_minimal\scripts\make_s234_figures.py
.\.venv\Scripts\python.exe .\projects\Code_1_minimal\scripts\make_knn_figures.py
```

**文件与 Git 管理**

导入时已逐字节核对 72 个原始文件，恢复 Code_2 的四个中文文件名。来源校验与命名修复记录在 IMPORTS.json。原 .idea 和 __pycache__ 保存在本地并由 Git 忽略，周报额外复制到 docs/。CleanRL 的 LICENSE 和课程源码版权说明随导入保留。

Code_1 的 S3 原始日志与 TensorBoard 文件、Code_2 周报提及的独立实验脚本和 results/ 均未包含在原包中。绘图脚本使用随包汇总值或 iris.csv。

代码修改保存在对应 projects 子目录，个人解释和精简结果可放到根目录 notes/。提交前检查改动并选择具体文件。

```powershell
git status --short
git diff
git add -- projects/Code_2/lesson3/dqn/agent.py
git diff --cached
git commit -m "Study DQN action selection"
```

需要同步时执行 git push，现有远端说明见 [GitHub 同步](../docs/06_sync.md)。
