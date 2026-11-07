# 实验方法

## 每次只改变一个因素

先运行默认配置，确认日志、模型和评估结果可生成，再改变学习率、探索率或目标网络更新频率。记录 Git 提交号、Python 版本、依赖快照、命令、固定超参数和修改差异。`config.json` 仅保存命令行参数，硬编码设置以代码版本为准。

训练回报含探索和参数变化，最终评估使用固定模型与确定性动作。DQN 取最大 Q，策略梯度和 PPO 取概率最大动作，SAC 使用高斯均值经 tanh 变换的动作。它们对应一种明确的评估约定，随机策略评估可作为后续补充。

## 固定预算与多个种子

```powershell
foreach ($seed in 0, 1, 2) {
    python -m lessons.04_dqn --algo dqn --steps 60000 --seed $seed
    python -m lessons.06_ppo --steps 60000 --seed $seed
    python -m lessons.07_continuous --algo td3 --steps 60000 --seed $seed
    python -m lessons.07_continuous --algo sac --steps 60000 --seed $seed
}
python -m scripts.plot_runs --pattern "runs/continuous/*/train.csv" --output runs/continuous.png
python -m scripts.summarize --pattern "runs/continuous/*/evaluation.json"
```

上述训练相互独立，按顺序运行。每个模型结束后在独立环境上评估 10 个回合，评估种子从训练 seed 加 100000 开始。先得到每个训练 seed 的评估均值，再计算训练种子间均值和样本标准差。单个模型的 10 个评估回合不能替代 10 次独立训练。

同算法重复运行同一 seed 时，汇总器会要求缩小 pattern，避免重复计数。`steps/episodes` 预算不同的运行分组输出。其他固定参数发生变化时也应选择独立输出目录，不把它们汇总在一起。

`plot_runs` 展示每条运行的原始曲线及窗口均值，不自动绘制跨种子置信区间。窗口均值仅用于观察趋势，正式报告保留独立评估统计。横轴使用环境步数，不以回合数代替采样成本。训练提前在回合中结束时，CSV 只记录完整回合，模型仍在预算末尾保存。

## 推荐的对照问题

| 阶段 | 固定条件 | 只改变 | 观察 |
|---|---|---|---|
| TD | 地图、epsilon、学习率 | SARSA / Q-learning | 训练风险与贪心路线 |
| DQN | 网络、步数、seed | DQN / Double DQN | 回报与 Q 高估倾向 |
| PPO | 环境与预算 | clip 范围 | 稳定性及更新幅度 |
| 连续控制 | Pendulum、步数、网络规模 | DDPG / TD3 | 双 Q 与延迟更新的联合影响 |
| SAC | 其余配置不变 | 固定 alpha | 探索和最终回报 |

算法整体比较与单因素消融分别记录。DDPG 和 TD3 之间同时有三项变化，要归因于某一项需单独做消融。单个简单环境上的结果不能直接推出复杂机器人任务中的优劣。

## 排错

先检查 observation、action、target 和 prediction 的形状，避免 `[batch,1]` 与 `[batch]` 广播成错误矩阵。确认 TD target 使用 `no_grad()`，Actor 权重使用 `detach()`，经验池存储真实下一状态。若曲线不改善，先排查奖励与终止语义，再调整超参数。PyTorch 导入失败时确认解释器路径和 Windows 运行库，依赖安装失败时确认包源及系统代理。

默认训练长度用于学习实验，不承诺每个随机种子均达到某个回报。短运行的用途是检查训练路径、梯度和文件输出，学习效果需要完整预算和多 seed 验证。
