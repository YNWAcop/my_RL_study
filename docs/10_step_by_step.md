# 从零到 GIN 调度入门的 22 步

每一步按“读指定函数 → 快速运行 → 做一次修改 → 用自己的话回答验收问题”完成。快速预算用于检查数据流和更新过程，正式实验使用章节里的正常预算。当前环境为本仓库 .venv。

先在 PowerShell 中切到 RL 根目录，显示步骤清单并开始第 00 步。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study list
.\.venv\Scripts\python.exe -X utf8 -m scripts.study show 00
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 00 --quick
```

后面将编号换为 01、02，依次推进到 21。VS Code 的“运行和调试”菜单中也有“学习步骤 · 快速运行”和“学习步骤 · 正式实验”，下拉选择包含编号和名称。

章节源文件保存在 lessons/，公共机械性逻辑位于 rlstudy/，两份原项目位于 projects/。external/ 保存固定版本的上游阅读源码，版本与恢复方法见 [源码地图](03_sources.md)。本地执行入口已适配当前环境，上游快照用于逐段对照。

完成一项后，在 [学习进度](../notes/study_progress.md) 勾选，并按 [笔记模板](../notes/template.md) 保存自己的推导、命令和观察。

| 编号 | 这一步学习什么 | 阅读入口 |
|---|---|---|
| 00 | 数组、索引与增量均值 | [lessons/00_foundations.py](../lessons/00_foundations.py) |
| 01 | 张量、网络与反向传播 | [lessons/00_foundations.py](../lessons/00_foundations.py) |
| 02 | 环境交互与回合边界 | [lessons/00_foundations.py](../lessons/00_foundations.py) |
| 03 | 探索与利用 | [lessons/01_bandit.py](../lessons/01_bandit.py) |
| 04 | MDP 与价值迭代 | [lessons/02_value_iteration.py](../lessons/02_value_iteration.py) |
| 05 | 回报、MC 与 TD(0) | [lessons/02_mc_prediction.py](../lessons/02_mc_prediction.py) |
| 06 | SARSA 的实际下一动作 | [lessons/03_td_control.py](../lessons/03_td_control.py) |
| 07 | Q-learning 的贪心目标 | [lessons/03_td_control.py](../lessons/03_td_control.py) |
| 08 | DQN 的完整训练循环 | [lessons/04_dqn.py](../lessons/04_dqn.py) |
| 09 | Double DQN 的两张网络 | [lessons/04_dqn.py](../lessons/04_dqn.py) |
| 10 | REINFORCE 与轨迹回报 | [lessons/05_policy_gradient.py](../lessons/05_policy_gradient.py) |
| 11 | 一步 Actor–Critic | [lessons/05_policy_gradient.py](../lessons/05_policy_gradient.py) |
| 12 | GAE 和裁剪目标手算 | [lessons/00_foundations.py](../lessons/00_foundations.py) |
| 13 | 手写 PPO 与单因素实验 | [lessons/06_ppo.py](../lessons/06_ppo.py) |
| 14 | 阅读与运行 CleanRL PPO | [projects/Code_1_minimal/ref/cleanrl/cleanrl/ppo.py](../projects/Code_1_minimal/ref/cleanrl/cleanrl/ppo.py) |
| 15 | 对照 PARL 的分层实现 | [projects/Code_2/lesson3/dqn/train.py](../projects/Code_2/lesson3/dqn/train.py) |
| 16 | DDPG 的连续动作 | [lessons/07_continuous.py](../lessons/07_continuous.py) |
| 17 | TD3 的三个改动 | [lessons/07_continuous.py](../lessons/07_continuous.py) |
| 18 | SAC 的随机策略与熵 | [lessons/07_continuous.py](../lessons/07_continuous.py) |
| 19 | 成熟基线与保存评估 | [lessons/08_sb3.py](../lessons/08_sb3.py) |
| 20 | 把调度写成决策环境 | [lessons/09_scheduling.py](../lessons/09_scheduling.py) |
| 21 | GIN 与合法动作概率 | [lessons/09_scheduling.py](../lessons/09_scheduling.py) |

正式实验可增加种子并覆盖参数。例如先查看第 13 步，再通过 --args 把裁剪系数交给原章节。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 13 --quick --seed 1 --args --clip-coef 0.1
```

检查整个学习目录可执行以下命令。每一步的终端输出保存在 runs/study_checks/ 下，summary.json 记录返回码和耗时。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study check --all
```

**00 数组、索引与增量均值**

阅读 [lessons/00_foundations.py](../lessons/00_foundations.py)。

定位 `numpy_demo`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 00 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 00
```

观察 输出选定动作的 Q 值 [4,3,0.8]。

动手 改动奖励序列 [1,0,1]，手算每一次均值更新。

验收 为什么一批动作要同时配合行编号索引？

源码对照 `external/handson/第2章-多臂老虎机问题.ipynb`。

**01 张量、网络与反向传播**

阅读 [lessons/00_foundations.py](../lessons/00_foundations.py), [rlstudy/deep.py](../rlstudy/deep.py)。

定位 `tensor_demo → mlp → gather → backward`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 01 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 01
```

观察 网络输出 [8,2]，选定动作输出 [8]，拟合后的 MSE 小于初始值。

动手 把 batch 从 8 改为 16，并同步改动 actions 长度。

验收 detach、no_grad 和 optimizer.zero_grad 分别负责什么？

**02 环境交互与回合边界**

阅读 [lessons/00_foundations.py](../lessons/00_foundations.py), [docs/07_foundations.md](../docs/07_foundations.md)。

定位 `environment_demo → reset → step`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 02 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 02
```

观察 显示五返回值和真正终止、时间截断对应的 TD 目标。

动手 把 max_episode_steps 从 3 改为 5，观察 truncated。

验收 为什么时间截断需要重置环境，却仍然可以自举？

**03 探索与利用**

阅读 [lessons/01_bandit.py](../lessons/01_bandit.py)。

定位 `main 中的 counts、estimates、bonus、regret`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 03 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 03
```

观察 生成 bandit/train.csv，打印 epsilon-greedy 与 UCB 的伪遗憾。

动手 通过 --epsilon 0、0.01、0.1 分别运行，保持同一个 seed。

验收 奖励噪声和累计伪遗憾有什么区别？

源码对照 `external/handson/第2章-多臂老虎机问题.ipynb`。

**04 MDP 与价值迭代**

阅读 [lessons/02_value_iteration.py](../lessons/02_value_iteration.py)。

定位 `solve → transitions → Bellman 更新`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 04 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 04
```

观察 打印 4×4 价值表、动作表和收敛残差。

动手 把 FrozenLake 的 is_slippery 改为 True，再运行。

验收 已知转移模型时，怎样计算一个动作的期望价值？

源码对照 `external/handson/第3章-马尔可夫决策过程.ipynb`, `external/handson/第4章-动态规划算法.ipynb`。

**05 回报、MC 与 TD(0)**

阅读 [lessons/02_mc_prediction.py](../lessons/02_mc_prediction.py)。

定位 `discounted_returns → first_visit_update → TD 更新`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 05 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 05
```

观察 手算回报 [5.23,4.7,3]，生成两种价值估计与 rmse 日志。

动手 对照 seed 0、1、2 的价值误差，再将 TD 学习率 0.05 改为 0.1。

验收 首次访问 MC 为什么要记录本回合已经更新过的状态？

源码对照 `external/handson/第5章-时序差分算法.ipynb`。

**06 SARSA 的实际下一动作**

阅读 [lessons/03_td_control.py](../lessons/03_td_control.py)。

定位 `epsilon_action → td_target 的 sarsa 分支`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 06 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 06
```

观察 保存 Q 表，输出独立的贪心路径。

动手 用 --epsilon 0 与 --epsilon 0.1 对照，记录训练回报和贪心路径。

验收 SARSA 的更新目标为什么需要实际采样出的 next_action？

源码对照 `projects/Code_2/lesson2/sarsa/agent.py`, `external/handson/第5章-时序差分算法.ipynb`。

**07 Q-learning 的贪心目标**

阅读 [lessons/03_td_control.py](../lessons/03_td_control.py)。

定位 `td_target 的 max 分支 → Q 更新`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 07 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 07
```

观察 保存 Q 表，对照 SARSA 的训练回报和贪心路径。

动手 保持第 06 步的种子与 epsilon，再用 --learning-rate 0.1 对照。

验收 探索时可能掉崖，为何 Q-learning 的贪心路径仍可能贴近悬崖？

源码对照 `projects/Code_2/lesson2/q_learning/agent.py`, `external/handson/第5章-时序差分算法.ipynb`。

**08 DQN 的完整训练循环**

阅读 [lessons/04_dqn.py](../lessons/04_dqn.py), [rlstudy/deep.py](../rlstudy/deep.py)。

定位 `ReplayBuffer → targets → gather → loss → target 同步`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 08 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 08
```

观察 完成采样和梯度更新，保存模型与 evaluation.json。

动手 将目标网络的同步间隔由 500 改为 1000，保持种子和步数。

验收 经验回放与目标网络各自解决什么问题？

源码对照 `projects/Code_2/lesson3/dqn`, `external/handson/第7章-DQN算法.ipynb`, `external/cleanrl/cleanrl/dqn.py`。

**09 Double DQN 的两张网络**

阅读 [lessons/04_dqn.py](../lessons/04_dqn.py)。

定位 `online 选动作 → target 评估动作`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 09 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 09
```

观察 数据流中能找到两个网络承担不同工作的位置。

动手 与第 08 步使用相同预算和至少 3 个训练 seed 比较。

验收 max 同时选动作和估值时，估计噪声如何导致偏差？

源码对照 `external/handson/第8章-DQN改进算法.ipynb`。

**10 REINFORCE 与轨迹回报**

阅读 [lessons/05_policy_gradient.py](../lessons/05_policy_gradient.py)。

定位 `Categorical → sample → log_prob → returns → loss`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 10 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 10
```

观察 一局结束后更新策略，保存训练日志和模型。

动手 把回报折扣改为 0.95，并记录外层时间权重的对应改动。

验收 为什么最大化奖励会对应带负号的策略损失？

源码对照 `projects/Code_2/lesson4/policy_gradient`, `external/handson/第9章-策略梯度算法.ipynb`。

**11 一步 Actor–Critic**

阅读 [lessons/05_policy_gradient.py](../lessons/05_policy_gradient.py)。

定位 `value → target → advantage.detach → 两个优化器`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 11 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 11
```

观察 每步更新 Actor 与 Critic，最终独立评估策略。

动手 固定预算，观察 Critic 目标和 Actor 更新发生的频率。

验收 Actor 的 advantage 为什么要 detach？

源码对照 `external/handson/第10章-Actor-Critic算法.ipynb`。

**12 GAE 和裁剪目标手算**

阅读 [lessons/00_foundations.py](../lessons/00_foundations.py), [lessons/06_ppo.py](../lessons/06_ppo.py)。

定位 `gae_demo → gae → bootstrap mask / trace mask`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 12 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 12
```

观察 GAE 为 [1.7,100]，四组裁剪例子均输出未裁剪和 PPO 目标。

动手 将 ratio 改为 0.9、1.1、1.5，分别配正负优势手算。

验收 为什么概率比相同而优势符号不同，会改变裁剪方向？

源码对照 `external/handson/第12章-PPO算法.ipynb`。

**13 手写 PPO 与单因素实验**

阅读 [lessons/06_ppo.py](../lessons/06_ppo.py)。

定位 `rollout → gae → 固定旧概率 → minibatch → clip`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 13 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 13
```

观察 完成 PPO 更新，保存模型和独立评估结果。

动手 分别尝试 --clip-coef 0.1、--update-epochs 1、--gae-lambda 0，逐次只改一个参数。

验收 为什么同一批旧策略数据可以用于多轮更新？

源码对照 `external/handson/第12章-PPO算法.ipynb`。

**14 阅读与运行 CleanRL PPO**

阅读 [projects/Code_1_minimal/ref/cleanrl/cleanrl/ppo.py](../projects/Code_1_minimal/ref/cleanrl/cleanrl/ppo.py), [projects/Code_1_minimal/docs/S4_ppo精读批注.md](../projects/Code_1_minimal/docs/S4_ppo精读批注.md)。

定位 `Agent → rollout → lastgaelam → pg_loss1/2`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 14 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 14
```

观察 终端打印 episodic_return，runs/ 中出现 TensorBoard 事件。

动手 按 S3 操作手册选一项实验，先记录基线与随机种子。

验收 CleanRL 的多环境采样怎样摊平成一个训练批次？

源码对照 `external/cleanrl/cleanrl/ppo.py`。

**15 对照 PARL 的分层实现**

阅读 [projects/Code_2/lesson3/dqn/train.py](../projects/Code_2/lesson3/dqn/train.py), [projects/Code_2/lesson3/dqn/agent.py](../projects/Code_2/lesson3/dqn/agent.py), [projects/Code_2/lesson3/dqn/algorithm.py](../projects/Code_2/lesson3/dqn/algorithm.py), [projects/Code_2/lesson3/dqn/model.py](../projects/Code_2/lesson3/dqn/model.py)。

定位 `train / agent / algorithm / model / replay_memory`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 15 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 15
```

观察 六组课程自检全部通过，DQN、PG、DDPG 的 learn 能执行。

动手 逐项定位本地手写 DQN 的相同职责，写一张文件对应表。

验收 拆成不同文件后，状态、动作、目标和梯度分别在哪里流动？

源码对照 `projects/Code_2/docs/第三周周报.md`。

**16 DDPG 的连续动作**

阅读 [lessons/07_continuous.py](../lessons/07_continuous.py)。

定位 `Actor → q_value → actor_loss → soft_update`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 16 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 16
```

观察 在 Pendulum 上采样连续动作，完成 Actor/Critic 更新。

动手 把探索噪声标准差由 0.2 改为 0.1，观察动作范围。

验收 Critic 对动作的梯度怎样传给 Actor？

源码对照 `projects/Code_2/lesson5/ddpg`, `external/handson/第13章-DDPG算法.ipynb`, `external/cleanrl/cleanrl/ddpg_continuous_action.py`。

**17 TD3 的三个改动**

阅读 [lessons/07_continuous.py](../lessons/07_continuous.py)。

定位 `双 Critic → target smoothing → delayed policy`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 17 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 17
```

观察 对照 DDPG 找到三个差异，并保存评估结果。

动手 将 policy 的更新频率从每 2 次改为每次，单独记录配置差异。

验收 探索噪声和目标动作平滑噪声分别加在什么位置？

源码对照 `external/cleanrl/cleanrl/td3_continuous_action.py`。

**18 SAC 的随机策略与熵**

阅读 [lessons/07_continuous.py](../lessons/07_continuous.py)。

定位 `rsample → tanh → log_prob 修正 → entropy target`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 18 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 18
```

观察 动作保持在 [-2,2]，log_prob 有限，完成参数更新。

动手 将固定 alpha 0.2 改为 0.1，再保持其他参数对照。

验收 tanh 与动作缩放为什么都要修正概率密度？

源码对照 `external/handson/第14章-SAC算法.ipynb`, `external/cleanrl/cleanrl/sac_continuous_action.py`。

**19 成熟基线与保存评估**

阅读 [lessons/08_sb3.py](../lessons/08_sb3.py)。

定位 `Monitor → PPO/DQN → learn → save → load → evaluate`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 19 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 19
```

观察 保存 monitor.csv、训练日志、SB3 模型和独立评估结果。

动手 改为 --algo dqn，比较库的默认参数和手写实现的参数。

验收 调用同名算法时，哪些超参数差异会影响比较？

源码对照 `external/sb3/stable_baselines3/ppo/ppo.py`, `external/sb3/stable_baselines3/common/on_policy_algorithm.py`。

**20 把调度写成决策环境**

阅读 [lessons/09_scheduling.py](../lessons/09_scheduling.py)。

定位 `Operation → DispatchEnv → action_mask → make_schedule`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 20 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 20
```

观察 初始实例 FIFO=12、SPT=17、LPT=12，输出甘特图。

动手 更改一项 duration 或机器编号，再比较 FIFO/SPT/LPT 与穷举最优。

验收 前置依赖和机器占用分别怎样限制开始时间？

源码对照 `external/l2d/JSSP_Env.py`, `external/l2d/permissibleLS.py`。

**21 GIN 与合法动作概率**

阅读 [lessons/09_scheduling.py](../lessons/09_scheduling.py)。

定位 `graph_adjacency → GINPolicy → 邻居求和 → pooling → mask`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 21 --quick
.\.venv\Scripts\python.exe -X utf8 -m scripts.study run 21
```

观察 随机初始化 GIN 输出合法分布，非法动作概率为零，完成一份计划。

动手 调整网络随机种子，观察合法概率；将终止状态传入网络，应得到明确错误。

验收 怎样将每个工序的节点表示变为可用于 PPO 的动作概率？

源码对照 `external/l2d/models`, `external/l2d/PPO_jssp_multiInstances.py`。

**多种子实验与图表**

完成 DQN/PPO 或 TD3/SAC 后，分别保留三次独立训练。下面以 DQN 为例，把结果放在独立目录，以便汇总时识别预算和训练种子。

```powershell
foreach ($seed in 0,1,2) {
    .\.venv\Scripts\python.exe -X utf8 -m scripts.study run 08 --seed $seed --args --output runs/study_experiments/dqn
}
.\.venv\Scripts\python.exe -m scripts.plot_runs --pattern 'runs/study_experiments/dqn/dqn/*/train.csv' --output runs/study_experiments/dqn.png
.\.venv\Scripts\python.exe -X utf8 -m scripts.summarize --pattern 'runs/study_experiments/dqn/dqn/*/evaluation.json'
```

比较算法时记录环境、实际采样步数、学习率、更新频率和种子。SB3 的 on-policy 训练按完整 rollout 采样，实际步数会在第 19 步终端输出。实验规范见 [04_experiments.md](04_experiments.md)。

**调度例子的定义**

第 20–21 步使用六道工序、两台固定机器。动作逐项构造计划，前置工序已经排入计划后，其后继才可以被选择。开始时间由前置工序完工时间和机器可用时间共同决定。奖励是 makespan 的负增量，一局奖励和等于负的最终 makespan。

第 21 步执行随机初始化 GIN 的前向、合法动作分布和计划构造。下一阶段可以将 PPO 的 Actor 改为图编码器，并按装配业务重新定义工序、资源和奖励。GIN 调度论文的完整训练需要在这一基础上继续实现和验证。

MC 价值误差曲线可使用已有绘图工具。

```powershell
.\.venv\Scripts\python.exe -m scripts.plot_runs --pattern 'runs/mc_prediction/*/train.csv' --metric rmse --output runs/mc_prediction.png
```
