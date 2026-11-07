# 验证记录

验证日期为 2026-11-07，系统为 Windows，解释器为 Python 3.9.12。依赖使用仓库独立 `.venv`。

## 已完成

- Python 源文件通过语法检查。
- 多臂老虎机 2000 步、seed=0 正常生成 CSV。epsilon-greedy 累计伪遗憾为 197.1，UCB 为 121.8。
- 确定性 FrozenLake 价值迭代在 7 次更新后达到残差 0，起点价值约 0.951。
- CliffWalking 上 Q-learning 与 SARSA 各训练 1000 回合、seed=0，生成 CSV 和 Q 表。最终贪心评估回报分别为 -13、-17。
- 训练曲线工具成功生成 `runs/td_comparison.png`。
- 官方 PyTorch wheel 通过 PyPI 元数据提供的 SHA256 校验。

- `pytest -q` 通过 5 项检查，覆盖 SARSA/Q-learning 目标差异、真实终止屏蔽、GAE 截断边界、价值迭代与 SAC 动作密度梯度。
- DQN、Double DQN、PPO、DDPG、TD3、SAC 各跑 1200 环境步，REINFORCE 与一步 Actor–Critic 各跑 3 回合。8 个深度入口均完成采样、梯度更新、模型保存和 10 回合独立评估。
- 汇总工具能读取 DQN/Double DQN 的独立评估 JSON，并按算法及预算分组。
- `pip check` 无依赖冲突。实际依赖版本见 [Windows Python 3.9 环境快照](environment_windows_py39.txt)，其中本地安装 wheel 的路径已规范化为版本号。
- Markdown 本地文件链接通过存在性检查，训练曲线 PNG 已目视检查。

上述数值来自实际本地运行，单个 seed 的结果用于确认实现行为。短训练尚未达到完整学习预算，没有据此排序算法性能。

## 验证边界

短训练检查覆盖入口、更新、模型保存和评估文件格式。完整预算的学习效果需要按实验规范做多 seed 复现。当前代码面向 CPU、CartPole 与 Pendulum，尚未验证 CUDA、图像输入、多环境并行或其他连续动作边界。

SAC 使用固定温度，PPO 采用单环境简化实现。全部运行产物放在被 Git 忽略的 `runs/` 中。本次未将大模型权重推送到远端。

## 同步状态

Git 通过现有本地代理成功连接 origin，确认初始远端没有提交。课程首建提交 `333a680` 已成功推送至 `codex/rl-learning-path`，可从 [远端课程入口](https://github.com/YNWAcop/my_RL_study/tree/codex/rl-learning-path) 阅读。由于本地没有配置提交者，首建提交使用 `Codex <codex@local.invalid>` 标识自动整理，后续学习提交可使用仓库所有者自己的 Git 署名。

## VS Code 运行方式更新

2026-11-07 增加项目打包配置，在当前 `.venv` 中完成 editable 安装。已从 `lessons` 目录直接启动 `01_bandit.py` 跑通实验，验证直接文件运行能够导入 `rlstudy`。VS Code 配置包含当前文件入口和 12 个算法入口，已检查配置 JSON 及各入口参数。实际 VS Code 按钮交互由使用者按操作指南启动。原有 5 项数学检查再次通过，`pip check` 无依赖冲突。

随后修正 VS Code 将 `.venv` 目录当成可执行程序的问题，将默认解释器及全部调试配置显式指向 `.venv/Scripts/python.exe`，并在当前 VS Code 窗口中重新选择此解释器。已实际点击编辑器右上角运行按钮，确认终端输出老虎机两种策略结果，epsilon-greedy 与 UCB 累计伪遗憾分别为 197.1 和 121.8，程序正常结束并返回提示符。
