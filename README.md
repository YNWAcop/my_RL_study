# my_RL_study

用 Python、NumPy 和 PyTorch 学习强化学习。按“理解问题 → 推导目标 → 阅读实现 → 跑对照实验 → 解释结果”的顺序推进。

## 实战项目

- [高速公路驾驶决策](projects/highway_dqn/README.md)使用 HighwayEnv 和 DQN，在不同车流密度下与规则策略对比。
- [作业车间排程](projects/job_shop/README.md)使用派工规则、带动作掩码的 PPO 和 CP-SAT，对比完工时间并输出甘特图。

## 从这里开始

现在可按 [22 步学习手册](docs/10_step_by_step.md) 逐步运行代码，并在 [学习进度](notes/study_progress.md) 中记录。

本机 .venv 已配齐整条路线的依赖。重建完整环境时使用 requirements-all.txt，并按 [统一环境指南](docs/09_unified_environment.md) 恢复 PARL 配置；上游阅读源码可通过 scripts.fetch_sources 恢复。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study list
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 00 --quick
```

1. 阅读 [环境配置](docs/00_setup.md)，运行第一个实验。
2. 先读 [基础概念与张量形状](docs/07_foundations.md)，再跟随 [学习路线](docs/01_roadmap.md)，每一阶段都有阅读材料、代码入口、实验和验收问题。
3. 用 [算法对照表](docs/02_algorithms.md) 对照更新目标和数据流。
4. 用 [源码阅读地图](docs/03_sources.md) 阅读 Hands-on-RL、dennybritz、CleanRL 和 Spinning Up。
5. 按 [实验规范](docs/04_experiments.md) 保存数据，用 [学习记录模板](notes/template.md) 写自己的解释。

## 目录

```text
docs/                     中文路线、原理、源码导读与实验规范
lessons/                  22 步的顺序与预算见 study_plan.json
  00_foundations.py       数组、张量、环境交互与 GAE 手算
  01_bandit.py            探索与利用
  02_value_iteration.py   MDP、贝尔曼方程与动态规划
  02_mc_prediction.py     回报、首次访问 MC 与 TD(0)
  03_td_control.py        SARSA / Q-learning
  04_dqn.py               DQN / Double DQN
  05_policy_gradient.py   REINFORCE / 一步 Actor–Critic
  06_ppo.py               PPO-Clip / GAE
  07_continuous.py        DDPG / TD3 / SAC
  08_sb3.py               成熟 DQN/PPO 基线
  09_scheduling.py        固定机器调度、动作掩码与 GIN 前向
external/                 固定版本的上游阅读源码，Git 忽略
rlstudy/                  日志、网络和回放池等通用部分
scripts/                  结果汇总和曲线绘制
tests/                    终止语义和数学更新的检查
notes/                    个人学习记录
runs/                     运行时生成，Git 忽略
```

## 第一个实验

本机的 VS Code 点击运行方式见 [VS Code 操作指南](docs/08_vscode.md)。打开章节文件后可点击右上角“运行 Python 文件”，或在“运行和调试”中选择具体算法后点击绿色三角形。项目通过 editable 安装提供公共模块，修改代码后即可重新运行。

所有命令均在仓库根目录执行。以下使用环境中的 `python`。

```powershell
python -m pip install -r requirements.txt
python -m lessons.01_bandit --steps 2000 --seed 0
python -m lessons.02_value_iteration
python -m lessons.03_td_control --algo q_learning --episodes 1000 --seed 0
python -m lessons.03_td_control --algo sarsa --episodes 1000 --seed 0
python -m scripts.plot_runs --pattern "runs/td_control/*/train.csv"
```

学习顺序为 **Python/张量 → 多臂老虎机 → MDP/动态规划 → MC/TD → SARSA/Q-learning → DQN → REINFORCE → Actor–Critic → PPO → DDPG/TD3 → SAC → 多随机种子对比**。其中 MC 提供随机游走预测代码，调度和 GIN 入门位于路线末尾。

本仓库是面向初学者的新写教学实现，强调显式训练循环。深度算法使用 CPU、低维状态和小网络。算法参数与实现简化写在各阶段说明中，实验结果由实际运行生成。

## 资料与验证

参考项目的角色、阅读范围、授权边界见 [资料说明](docs/03_sources.md)。本地验证状态见 [验证记录](docs/05_validation.md)，远端连接情况与同步方法见 [GitHub 同步](docs/06_sync.md)。当前路线按有基本 Python 能力、首次系统学习 RL、每周约 6–8 小时安排，可根据每阶段验收结果调整进度。

两份压缩包已整理到 [projects/](projects/README.md)，包含 PPO 实验材料和 PARL 课程代码。三套代码统一使用现有 .venv，安装记录与运行命令见 [统一环境指南](docs/09_unified_environment.md)。
