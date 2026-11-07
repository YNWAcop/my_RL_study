# 学习路线

这条路线按每周 6–8 小时、约 10–12 周安排。适用假设是已有基本 Python 能力，强化学习从零开始。具体时间以每阶段验收为准。顺序依据概念依赖安排，“最优”取决于基础、可用时间和应用方向。

中文主线用 [Hands-on-RL](https://hrl.boyuai.com/chapter/)，基础习题用 [dennybritz](https://github.com/dennybritz/reinforcement-learning)，深度实现用 [CleanRL](https://github.com/vwxyzjn/cleanrl)，英文推导用 [Spinning Up](https://spinningup.openai.com/en/latest/)。每阶段先读相应材料，再运行本仓库代码，最后回到上游核对实现。更详细的文件入口见 [源码地图](03_sources.md)。

## 阶段 0 · Python 与张量 · 3–6 小时

阅读 [环境配置](00_setup.md)。练习 NumPy `argmax`、高级索引、均值和随机数生成器，再用 PyTorch 写一个两层网络。画出 `[64,4] → [64,64] → [64,2]` 的形状变化。

验收时能解释 `detach()` 与 `no_grad()` 的作用，以及为什么 TD 目标不参与反向传播。能区分环境步数、回合数、梯度更新次数和批大小。

## 阶段 1 · 探索与利用 · 第 1 周

读 Hands-on-RL 第 1–2 章。运行 `python -m lessons.01_bandit`，读增量样本均值更新、随机探索与 UCB 奖励上界。输出中的伪遗憾使用已知真实成功率计算，表示所选动作相对最佳动作的累计期望损失。

实验比较 0、1、2 三个 seed，观察奖励噪声与伪遗憾的区别。再修改 epsilon 为 0、0.01、0.1 做独立运行，将代码差异记入学习笔记。

验收时能解释贪心为什么可能停在次优动作，以及增量更新中的 `1/N(a)` 从何而来。

## 阶段 2 · MDP、动态规划和 MC · 第 2 周

读 Hands-on-RL 第 3–4 章，再做 dennybritz 的 MDP、DP 和 MC 练习。定义状态、动作、转移概率、奖励、折扣和策略。手算三步奖励 `[1,2,3]`、gamma=0.9 的每个时刻回报，并写出一次访问 MC 的均值更新。

运行 `python -m lessons.02_value_iteration`。在已知的确定性 FrozenLake 模型上检查价值迭代、贪心策略和贝尔曼残差。把 `is_slippery=False` 改为 `True`，重新观察转移随机性如何改变策略。

验收时能从 `r + gamma * V(s')` 推导贝尔曼更新，解释策略评估与价值迭代的区别，并说明为什么 MC 必须等到观察到后续回报。

## 阶段 3 · SARSA 与 Q-learning · 第 3 周

读 Hands-on-RL 第 5 章，做 dennybritz 的 SARSA、Q-learning 练习。本地统一在 CliffWalking 上运行两种方法。

```powershell
python -m lessons.03_td_control --algo q_learning --episodes 1000 --seed 0
python -m lessons.03_td_control --algo sarsa --episodes 1000 --seed 0
```

逐行跟踪 `epsilon_action → env.step → td_target → Q 更新`。画训练回报，比较程序打印的最终贪心路线。保持 epsilon=0.1，再试 epsilon=0，观察训练行为与最终贪心策略之间的关系。

验收时能说明目标中的 `max` 和实际 `next_action` 各自代表哪个策略，解释为什么落入悬崖后返回起点不等于整个任务终止。能用数字手算一次 TD 更新。

## 阶段 4 · DQN 与 Double DQN · 第 4–5 周

读 Hands-on-RL 第 7–8 章，然后阅读 CleanRL `dqn.py` 的 `QNetwork`、探索日程、回放池、目标网络和 `gather`。本地用相同网络比较两种目标。

```powershell
python -m lessons.04_dqn --algo dqn --steps 60000 --seed 0
python -m lessons.04_dqn --algo double_dqn --steps 60000 --seed 0
```

分别写出 batch 内 `states/actions/targets/predictions` 的形状。对 Double DQN 标出“在线网络选动作、目标网络评动作”。做一次目标网络更新频率的对照，记录收益波动和价值估计。

验收时能说明回放池与目标网络解决的问题不同。能解释神经网络输出维数为什么等于离散动作数量，以及 TD loss 下降和策略回报上升之间为什么没有简单对应关系。

## 阶段 5 · REINFORCE 与 Actor–Critic · 第 6–7 周

读 Hands-on-RL 第 9–10 章，配合 Spinning Up 的 VPG 页面。先读 REINFORCE 分支，再读一步 AC 分支。

```powershell
python -m lessons.05_policy_gradient --algo reinforce --episodes 1000 --seed 0
python -m lessons.05_policy_gradient --algo actor_critic --episodes 1000 --seed 0
```

标出 `Categorical → sample → log_prob → actor_loss`。REINFORCE 使用完整回合回报，一步 AC 使用 TD 误差。两分支的更新频率和熵项也不同，对比时逐项记录，不能把曲线差异全部归因于 Critic。

验收时能解释策略梯度中 `log_prob` 的作用、baseline 为什么不能依赖当前采样动作，以及为什么 Actor loss 中的 advantage 要 `detach()`。能说明 REINFORCE 当前优化有限时间回合，而 AC 对时间截断自举。

## 阶段 6 · PPO 与 GAE · 第 8–9 周

读 Hands-on-RL 第 12 章，TRPO 第 11 章先掌握限制更新幅度的动机。读 Spinning Up PPO 方程，再读 CleanRL `ppo.py` 的 rollout、GAE、概率比与裁剪部分。

```powershell
python -m lessons.06_ppo --steps 100000 --seed 0
```

本地 PPO 使用单环境、1024 步 rollout、64 的 minibatch、10 轮更新、0.2 裁剪、0.95 GAE lambda。先画一次 rollout 如何进入多个 minibatch。手算优势为正与负、概率比为 0.6 和 1.4 的四种裁剪情况。

验收时能解释旧 log probability 为什么固定、为什么同一批数据能更新多轮，以及终止、自举和 GAE 递推边界为何需要分别处理。

## 阶段 7 · DDPG → TD3 → SAC · 第 10–11 周

读 Hands-on-RL 第 13–14 章与 Spinning Up 的 DDPG、TD3、SAC 页面，最后对照 CleanRL 的三个 continuous_action 文件。本仓库固定 Pendulum，状态为 3 维、动作为 1 维且范围为 [-2,2]。

```powershell
python -m lessons.07_continuous --algo ddpg --steps 60000 --seed 0
python -m lessons.07_continuous --algo td3 --steps 60000 --seed 0
python -m lessons.07_continuous --algo sac --steps 60000 --seed 0
```

先比较 Critic 目标，再比较 Actor 目标。TD3 检查两个 Critic、延迟更新、目标动作噪声。SAC 检查 `rsample()`、tanh 变换与 log probability 修正。本地 SAC 固定 alpha=0.2，学完后再读 CleanRL 的自动温度学习分支。

验收时能解释探索噪声与目标平滑噪声的用途不同，说明 SAC 训练时为何使用随机动作，最终确定性评估又如何得到动作。

## 阶段 8 · 复现报告 · 第 12 周

遵循 [实验规范](04_experiments.md)，分别在 CartPole 比较 DQN/PPO，在 Pendulum 比较 TD3/SAC。每组使用相同环境步数预算、至少 3 个训练 seed、每个模型 10 个独立评估回合。记录失败种子、环境版本和代码修改。

交付自己的学习报告，包含目标函数、数据流、配置、训练曲线、独立评估均值与种子间标准差、一个消融实验及结论限制。之后再用 SB3 核对成熟实现，或选择离线 RL、多智能体等方向。
