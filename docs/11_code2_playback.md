# 加载 Code_2 训练结果并显示画面

在仓库根目录 `E:\UserData\Documents\ChatGPT\RL` 打开 PowerShell。
`scripts/play_code2.py` 直接复用原课程的智能体、网络和画面封装，只调用 `predict()`，不会继续训练或覆盖模型。

## DQN

如果训练结果保存在仓库根目录的 `dqn_model.ckpt`，可以指定它运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.play_code2 --algo dqn --model .\dqn_model.ckpt --episodes 3
```

会显示 CartPole 小车，运行三回合，打印各回合回报和平均值。需要慢放可增加 `--delay 0.03`。

仅评估、不打开窗口：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.play_code2 --algo dqn --model .\dqn_model.ckpt --episodes 10 --no-render
```

该文件必须是原 Code_2 的 Paddle/PARL 模型，网络结构要与 `lesson3/dqn/model.py` 相符。
其他教程生成的 PyTorch `model.pt` 或 SB3 模型不能用这个入口加载。

## SARSA 和 Q-learning

两份 `train.py` 都会在训练循环结束后、最终动画之前调用 `agent.save()`。已经退出且没有保存的内存 Q 表不能恢复。

`agent.save()` 写入的是**当前终端目录**的 `q_table.npy`。分别在对应目录训练，避免两种算法互相覆盖：

```powershell
$rlPython = "E:\UserData\Documents\ChatGPT\RL\.venv\Scripts\python.exe"
cd E:\UserData\Documents\ChatGPT\RL\projects\Code_2\lesson2\sarsa
& $rlPython -X utf8 train.py
```

Q-learning 则进入 `projects\Code_2\lesson2\q_learning`，运行同样的训练命令。

保存好后，回到仓库根目录演示：

```powershell
cd E:\UserData\Documents\ChatGPT\RL
.\.venv\Scripts\python.exe -X utf8 -m scripts.play_code2 --algo sarsa --episodes 3
.\.venv\Scripts\python.exe -X utf8 -m scripts.play_code2 --algo q_learning --episodes 3
```

默认读取各自课程目录里的 `q_table.npy`。文件在其他位置时，用 `--model "完整路径"` 指定。
同一形状的 Q 表不记录训练算法和地图，使用者需要确保选对了文件，且状态、动作编号与训练一致。

## 调用原理

```python
agent.restore(模型路径)
obs = env.reset()
action = agent.predict(obs)
obs, reward, done, info = env.step(action)
```

重复预测和执行，直到回合结束。推理时不调用 `learn()`。
演示入口默认每回合最多 1000 步，避免未学好的格子策略无限循环；可用 `--max-steps` 调整。
格子画面默认每步额外暂停 0.2 秒，小车不额外暂停。按 `Ctrl+C` 停止。

模型加载只恢复 Q 表或网络参数。DQN 的优化器、经验池、探索进度不会由原课程的 `agent.save()` 一并恢复，因此这些步骤是使用已学策略，不是完整续训。
