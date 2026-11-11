# 高速公路驾驶决策

使用 HighwayEnv 的 `highway-fast-v0` 模拟多车道高速公路。DQN 从车辆运动状态估计离散动作的价值，动作包含保持、加减速和左右换道。规则基线保持当前车道，根据前车距离减速，路况允许时加速。

在仓库根目录运行以下命令。Windows 上可将 `python` 换成 `.\.venv\Scripts\python.exe`。

```powershell
python -m pip install -r projects/highway_dqn/requirements.txt
python -m projects.highway_dqn.run train --steps 60000 --seed 0 --output runs/highway_dqn/seed0
python -m projects.highway_dqn.run evaluate --model runs/highway_dqn/seed0/model.zip --densities 0.7 1.0 1.5 --episodes 20 --seed 10000 --csv runs/highway_dqn/seed0/evaluation.csv
```

只跑规则基线时省略 `--model`。训练约 640 步可用于检查流程，正式比较应使用相同训练步数和多个训练种子。训练目录和评估 CSV 使用新路径，避免覆盖已有实验。

加载任意一份训练结果并打开动画：

```powershell
python -m projects.highway_dqn.run evaluate --model runs/highway_dqn/seed0/model.zip --densities 1.0 --episodes 3 --render
```

评估时两种策略使用相同的场景配置与回合种子。CSV 记录每回合的回报、步数、是否碰撞、平均速度（m/s）及车道变化次数。对照碰撞率与速度，可以观察策略在拥挤程度变化时怎样取舍。这里的密度参数控制车辆生成间距，固定环境类型和其余配置。

参考 [HighwayEnv 源码和配置](https://github.com/Farama-Foundation/HighwayEnv) 与 [Stable Baselines3 的 DQN 文档](https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/dqn.html)。本目录提供训练和评估脚本，模拟器与 DQN 由依赖包提供。
