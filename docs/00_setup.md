# 环境与第一个实验

## Python 基础要求

能读懂函数、类、列表、字典、NumPy 数组索引和广播。深度算法还需要理解张量形状、标量损失、`backward()`、优化器和 `torch.no_grad()`。用一个输入 `[batch, 4]`、输出 `[batch, 2]` 的网络做一次前向和反向传播，再开始 DQN。

## 独立环境

当前完整学习环境使用 Windows / Python 3.9.12，包含两份导入项目的配置见 [统一环境指南](09_unified_environment.md)。以下是主学习代码的基本安装方式，创建 .venv 时使用 Python 3.9。所有命令均在仓库根目录执行，PowerShell 可直接调用解释器。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m lessons.01_bandit --steps 2000
.\.venv\Scripts\python.exe -m lessons.02_value_iteration
.\.venv\Scripts\python.exe -m pytest -q
```

之后的文档用 `python` 简写该环境解释器。其他平台运行主学习代码时使用其虚拟环境解释器并重新验证依赖。当前课程采用 CPU、低维状态与小网络，两个导入项目和 SB3 的完整安装清单在 requirements-all.txt。

依赖安装同时以 editable 模式安装本项目的 `rlstudy` 公共模块。完成后支持直接运行 `lessons` 中的文件，VS Code 操作见 [点击运行指南](08_vscode.md)。本机现有 `.venv` 已完成该安装，换电脑或重新创建环境时需要重新安装 requirements。

依赖固定顶层版本。完整环境快照见 [本地验证版本](environment_windows_py39.txt)。改变 Python 或平台后应重新运行检查，不把版本固定理解为跨平台结果完全一致。

## Gymnasium 接口

```python
observation, info = env.reset(seed=0)
next_observation, reward, terminated, truncated, info = env.step(action)
episode_finished = terminated or truncated
```

`terminated` 表示任务的终止状态，TD 目标在此屏蔽未来价值。`truncated` 在课程环境中通常来自外部时间上限，回合需要重置，但 TD 估计保留最后状态的自举项。先存储 `next_observation`，再调用 `reset()`。GAE 还要在两种回合边界切断递推。参见 [Gymnasium Env API](https://gymnasium.farama.org/api/env/)。

Hands-on-RL README 推荐的是旧 Gym，Spinning Up 的 setup.py 固定了旧 Gym、TensorFlow 1 和 PyTorch。课程将它们作为阅读材料，本地实验统一使用 requirements 中的 Gymnasium 和 PyTorch。参见 [Hands-on-RL README](https://github.com/boyu-ai/Hands-on-RL)、[Spinning Up 依赖](https://github.com/openai/spinningup/blob/master/setup.py)。

projects/Code_2 保留 Gym 与 Paddle/PARL 的课程接口，用于第 15 步对照分层实现；两套接口已在当前 .venv 中共存验证。

## 输出

训练目录包含 `config.json` 和 `train.csv`。表格算法额外保存 `q_table.npy`，深度算法额外保存 `model.pt` 和 `evaluation.json`。配置 JSON 保存命令行选项，固定的网络、学习率等设置保留在代码中，对比时还应记录 Git 提交号和修改差异。

```powershell
python -m scripts.plot_runs --pattern "runs/bandit/*/train.csv" --metric pseudo_regret
```

无显示器时也能输出 PNG。所有生成数据位于被 Git 忽略的 `runs/`。
