# 第一课 · 从交互到更新

## 一个完整样本

机器人当前位于格子 s，选择向右动作 a，环境返回奖励 r 和下一格 s'。训练程序收集 `(s,a,r,s',terminated,truncated)`，根据这些数据调整价值或策略。环境动力学决定动作会带来什么结果，策略决定当前会选择什么动作。

MDP 假设给定当前状态和动作，下一状态及奖励的条件分布不再需要完整历史。若仅观察到遮挡图像，观测可能不足以构成这样的状态，之后可用历史堆叠或记忆模型处理。当前课程使用 Gymnasium 的低维状态与完整观察任务。

## 回报和价值

有限轨迹的折扣回报为

$$G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.$$

这里 `r_t` 指执行 `a_t` 后得到的奖励。轨迹奖励 `[1,2,3]`、gamma=0.9 时，`G_2=3`、`G_1=4.7`、`G_0=5.23`。递推式 `G_t=r_t+gamma*G_{t+1}` 让程序可以反向扫描一次得到所有回报。

$$V^\pi(s)=E_\pi[G_t\mid s_t=s],\qquad Q^\pi(s,a)=E_\pi[G_t\mid s_t=s,a_t=a].$$

单条轨迹的回报带有采样噪声，价值函数估计多次可能结果的平均水平。优势函数 `A=Q-V` 比较当前动作与策略在该状态的平均表现。

## 贝尔曼关系

将回报的第一项拆开，得到

$$Q^\pi(s,a)=E[r+\gamma(1-d)E_{a'\sim\pi}Q^\pi(s',a')].$$

动态规划使用已知转移概率计算期望。MC 使用完整样本回报估计期望。TD 使用即时奖励与下一状态价值估计构造目标。TD 中“用估计更新估计”的过程称为自举。

当学习最优策略时，Q-learning 将下一步策略期望替换为最佳动作值。SARSA 使用当前策略实际采样的下一动作。二者更新形式均为 `旧估计 + 学习率*(目标-旧估计)`。

## 采样策略与目标策略

采样策略生成交互数据，目标策略是算法希望学习或评价的策略。Q-learning 可用 epsilon-greedy 采样，并学习贪心目标策略。SARSA 评价包含探索行为的当前策略。PPO 在当前策略采集的数据上计算更新，DQN、TD3、SAC 则从回放池重用不同时间采集的转移。

epsilon-greedy 以 epsilon 概率随机选动作，其余时间选择最高评分动作。本地表格实现随机打破相同 Q 值的平局，以免初始化时固定偏向编号最小的动作。

## 从策略分布到梯度

离散策略的网络输出 logits。`Categorical(logits=...)` 将其变成动作分布，`sample()` 选择动作，`log_prob(action)` 给出当前动作的对数概率。

对于目标 `J=E[G_0]`，REINFORCE 的一次轨迹梯度估计包含

$$\sum_t \gamma^t G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).$$

PyTorch 优化器执行梯度下降，因此实现用负号把“提高回报”的目标写成损失。本地 REINFORCE 保留外层 `gamma^t`，并用固定 1/100 缩放控制梯度大小。一步 AC、PPO 使用常见的按采样状态加权的近似策略更新，不能把每种实现的状态权重直接视为完全相同的严格目标。理解差异后再对照不同教程的公式。

加入只依赖状态的 baseline 可以降低梯度方差，因为在固定状态下 `E_a[grad log pi(a|s)]=0`。使用 Critic 估计 baseline 与优势时，价值误差也会影响策略更新。

## 一批数据的形状

| 算法/变量 | 当前课程形状 | 含义 |
|---|---|---|
| CartPole observations | `[B,4]` | 每个状态 4 个数 |
| DQN Q 输出 | `[B,2]` | 每个离散动作一个评分 |
| DQN actions | `[B]`，gather 时转为 `[B,1]` | 实际执行的动作编号 |
| DQN targets/predictions | `[B]` | 一个转移对应一个回归目标 |
| PPO logits | `[B,2]` | 两个动作的未归一化分数 |
| PPO log_prob/advantages | `[B]` | 一动作一概率、一转移一优势 |
| Pendulum observations/actions | `[B,3]` / `[B,1]` | 三维观察、一维力矩 |
| 连续动作 Q 输入/输出 | `[B,4]` / `[B]` | 拼接状态动作，输出单个价值 |

完成本课后，把自己的“采集数据 → 构造目标 → 计算损失 → 更新参数”流程写进学习笔记，再进入 [学习路线](01_roadmap.md) 的第一阶段。概念定义可对照 [Hands-on-RL](https://hrl.boyuai.com/chapter/) 与 [Spinning Up 关键概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)。
