# 高速公路驾驶决策

使用 HighwayEnv 的 `highway-fast-v0` 模拟多车道高速公路。DQN 从车辆运动状态估计离散动作的价值，动作包含保持、加减速和左右换道。规则基线保持当前车道，根据前车距离减速，路况允许时加速。

用 VS Code 打开仓库根目录，在「运行和调试」中选择「高速公路 · 规则驾驶演示」。也可直接用仓库的 `.venv` 运行本目录的 `run.py`。不传参数时会显示一回合规则驾驶，无需训练或下载模型。想在终端启动同样的演示，可以运行：

```powershell
.\.venv\Scripts\python.exe -m projects.highway_dqn.run
```

动画按正常时间速度播放，每局最长为 30 秒仿真时间；发生碰撞会提前结束。默认演示只运行一局，结束后窗口自动关闭，终端会输出结束原因：`time limit` 表示时间到，`collision` 表示碰撞。默认演示使用规则驾驶，没有加载 DQN 模型。

想连续观看 5 局规则驾驶，可以运行：

```powershell
.\.venv\Scripts\python.exe -m projects.highway_dqn.run evaluate --densities 1.0 --episodes 5 --render
```

之前窗口约一秒就关闭，是因为未启用实时渲染，30 秒仿真被快速计算完了。现在 `--render` 会限制画面播放速度；不显示画面的训练和评估仍按计算速度运行。

从仓库根目录运行以下训练和评估命令。只有环境缺少依赖时才需要执行安装命令。训练完成后才会生成评估要用的 `model.zip`。

```powershell
.\.venv\Scripts\python.exe -m pip install -r projects/highway_dqn/requirements.txt
.\.venv\Scripts\python.exe -m projects.highway_dqn.run train --steps 60000 --seed 0 --output runs/highway_dqn/seed0
.\.venv\Scripts\python.exe -m projects.highway_dqn.run evaluate --model runs/highway_dqn/seed0/model.zip --densities 0.7 1.0 1.5 --episodes 20 --seed 10000 --csv runs/highway_dqn/seed0/evaluation.csv
```

只跑规则基线时省略 `--model`。训练约 640 步可用于检查流程，正式比较应使用相同训练步数和多个训练种子。训练目录和评估 CSV 使用新路径，避免覆盖已有实验。

加载任意一份训练结果并打开动画：

```powershell
.\.venv\Scripts\python.exe -m projects.highway_dqn.run evaluate --model runs/highway_dqn/seed0/model.zip --densities 1.0 --episodes 3 --render
```

把 `--model` 的值换成你在本项目训练得到的模型路径。CartPole 的 `dqn_model.ckpt` 使用不同的模型格式和输入维度。

评估时两种策略使用相同的场景配置与回合种子。CSV 记录每回合的回报、步数、是否碰撞、平均速度（m/s）及车道变化次数。对照碰撞率与速度，可以观察策略在拥挤程度变化时怎样取舍。这里的密度参数控制车辆生成间距，固定环境类型和其余配置。

参考 [HighwayEnv 源码和配置](https://github.com/Farama-Foundation/HighwayEnv) 与 [Stable Baselines3 的 DQN 文档](https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/dqn.html)。本目录提供训练和评估脚本，模拟器与 DQN 由依赖包提供。
