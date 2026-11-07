# my_RL_study

用 Python、NumPy 和 PyTorch 学习强化学习。按“理解问题 → 推导目标 → 阅读实现 → 跑对照实验 → 解释结果”的顺序推进。

## 从这里开始

1. 阅读 [环境配置](docs/00_setup.md)，运行第一个实验。
2. 先读 [基础概念与张量形状](docs/07_foundations.md)，再跟随 [学习路线](docs/01_roadmap.md)，每一阶段都有阅读材料、代码入口、实验和验收问题。
3. 用 [算法对照表](docs/02_algorithms.md) 对照更新目标和数据流。
4. 用 [源码阅读地图](docs/03_sources.md) 阅读 Hands-on-RL、dennybritz、CleanRL 和 Spinning Up。
5. 按 [实验规范](docs/04_experiments.md) 保存数据，用 [学习记录模板](notes/template.md) 写自己的解释。

## 目录

```text
docs/                     中文路线、原理、源码导读与实验规范
lessons/
  01_bandit.py            探索与利用
  02_value_iteration.py   MDP、贝尔曼方程与动态规划
  03_td_control.py        SARSA / Q-learning
  04_dqn.py               DQN / Double DQN
  05_policy_gradient.py   REINFORCE / 一步 Actor–Critic
  06_ppo.py               PPO-Clip / GAE
  07_continuous.py        DDPG / TD3 / SAC
rlstudy/                  日志、网络和回放池等通用部分
scripts/                  结果汇总和曲线绘制
tests/                    终止语义和数学更新的检查
notes/                    个人学习记录
runs/                     运行时生成，Git 忽略
```

## 第一个实验

所有命令均在仓库根目录执行。以下使用环境中的 `python`。

```powershell
python -m pip install -r requirements.txt
python -m lessons.01_bandit --steps 2000 --seed 0
python -m lessons.02_value_iteration
python -m lessons.03_td_control --algo q_learning --episodes 1000 --seed 0
python -m lessons.03_td_control --algo sarsa --episodes 1000 --seed 0
python -m scripts.plot_runs --pattern "runs/td_control/*/train.csv"
```

学习顺序为 **Python/张量 → 多臂老虎机 → MDP/动态规划 → MC/TD → SARSA/Q-learning → DQN → REINFORCE → Actor–Critic → PPO → DDPG/TD3 → SAC → 多随机种子对比**。其中 MC 是推导与手算练习，代码入口集中实现后续主线。

本仓库是面向初学者的新写教学实现，强调显式训练循环。深度算法使用 CPU、低维状态和小网络。算法参数与实现简化写在各阶段说明中，实验结果由实际运行生成。

## 资料与验证

参考项目的角色、阅读范围、授权边界见 [资料说明](docs/03_sources.md)。本地验证状态见 [验证记录](docs/05_validation.md)，远端连接情况与同步方法见 [GitHub 同步](docs/06_sync.md)。当前路线按有基本 Python 能力、首次系统学习 RL、每周约 6–8 小时安排，可根据每阶段验收结果调整进度。
