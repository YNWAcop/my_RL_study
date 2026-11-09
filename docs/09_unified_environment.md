# 当前目录的统一 Python 环境

2026-11-09 检查了 7 个已有环境，选择本仓库的 `.venv`。解释器为 `E:\UserData\Documents\ChatGPT\RL\.venv\Scripts\python.exe`，Python 3.9.12。它同时用于根目录 `lessons/`、`projects/Code_1_minimal/` 和 `projects/Code_2/`。

**选择依据**

下表记录安装前的情况。缺失数量按 NumPy、PyTorch、Gymnasium、Matplotlib、pandas、tyro、TensorBoard、Gym、PaddlePaddle、PARL、pyglet 和 pytest 共 12 项计算，传递依赖与额外渲染支持另计。

| 已有环境 | Python | 缺少的核心依赖数 | 判断 |
|---|---|---|---|
| `RL/.venv` | 3.9.12 | 7 | 满足项目 Python 要求，缺失最少 |
| `E:/anaconda` | 3.9.12 | 8 | 还需要安装 PyTorch 和 Gymnasium |
| `EMT-DLFR/.conda` | 3.10.21 | 9 | 缺少课程框架和多数学习工具 |
| `API-main/.conda` | 3.10.21 | 10 | 缺少课程框架、绘图与学习工具 |
| `LNLN/.conda` | 3.10.21 | 10 | 缺少课程框架、绘图与学习工具 |
| `mult-repro-mulT` | 3.10.21 | 10 | 缺少课程框架、绘图与学习工具 |
| `deeplearning` | 3.8.20 | 7 | 低于本项目声明的 Python 3.9 下限 |

**直接使用现有环境**

在 PowerShell 中打开仓库根目录，通过解释器路径运行即可。

```powershell
Set-Location 'E:\UserData\Documents\ChatGPT\RL'
.\.venv\Scripts\python.exe -m lessons.01_bandit --steps 2000 --seed 0
.\.venv\Scripts\python.exe .\projects\Code_2\check_env.py
.\.venv\Scripts\python.exe .\projects\Code_2\smoke_lessons.py
.\.venv\Scripts\python.exe .\projects\Code_1_minimal\ref\cleanrl\cleanrl\ppo.py --total-timesteps 10000 --seed 1 --no-cuda
```

最后一条从仓库根目录运行，TensorBoard 事件写入根目录 `runs/`。查看时在另一个终端执行以下命令，再打开终端给出的网页地址。

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir .\runs
```

各课训练会运行较久。运行原 PARL 课程时，先保存解释器绝对路径，再进入课程目录，使模型文件保存在该课程内。

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
Push-Location .\projects\Code_2\lesson3\dqn
& $python .\train.py
Pop-Location
```

将 `lesson3\dqn` 换成 `lesson2\q_learning`、`lesson2\sarsa`、`lesson4\policy_gradient` 或 `lesson5\ddpg`，即可运行对应课程。VS Code 的默认解释器与调试配置已指向这一个 `.venv`。如果窗口此前手动选过其他解释器，用“Python: 选择解释器”选择上面的路径。

**兼容性配置与复现**

Gym 0.21.0 提供课程使用的旧式四返回值接口，Gymnasium 1.0.0 提供主学习项目和 PPO 使用的接口。两者分别通过 `import gym` 与 `import gymnasium` 使用。

NumPy 从原来的 1.26.4 调整为 1.23.5。旧 Gym 的 CliffWalking 使用了已在较新 NumPy 中移除的 np.bool 别名，1.23.5 同时满足这里的 PyTorch、Gymnasium、Matplotlib、pandas 和 SciPy 版本要求。

PARL 2.2.1 的依赖约束包括 cloudpickle 1.6.0、TensorBoard 不高于 2.11.0，以及 protobuf 不高于 3.20.0。本环境相应固定为 1.6.0、2.11.0 和 3.20.0。约束可在 [PARL 官方发行元数据](https://pypi.org/pypi/parl/2.2.1/json) 中核对。

本环境安装了 `.venv/Lib/site-packages/rl_parl_backend.pth` 和 `_rl_framework_compat.py`。它们设置默认 `PARL_BACKEND=paddle`，并在首次导入 Paddle 时先加载 PyTorch。实测中，Paddle 先加载时，PARL 随后探测 PyTorch 会触发 shm.dll 加载错误；调整顺序后两个框架可以共存。这项配置由 [环境配置脚本](../scripts/configure_environment.py) 生成。

主依赖清单在 [requirements-all.txt](../requirements-all.txt)。在 Windows / Python 3.9 下恢复依赖，可从根目录执行以下命令。先固定安装工具，便于处理 Gym 0.21 的旧式包元数据。

```powershell
.\.venv\Scripts\python.exe -m pip install pip==23.3.1 setuptools==65.5.0 wheel==0.38.4
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --cache-dir .\.wheelhouse\pip-cache -r .\requirements-all.txt
.\.venv\Scripts\python.exe .\scripts\configure_environment.py
```

最后一条命令会恢复后端选择和 DLL 加载顺序配置，适用于本仓库根目录的 .venv。当前环境已完成该配置。首次导入 Paddle 时还会创建用户目录下的 .cache/paddle/dataset 缓存。

清单也包含 PPO 可选分支使用的 MoviePy 1.0.3 和 wandb 0.16.6。录像依赖本地 FFmpeg，实验追踪连到在线服务时需要用户自己的账号设置。

安装前的软件包记录保存在本地 `.wheelhouse/environment-before-20261109.json`。下载缓存也位于 `.wheelhouse/`，大体积安装包不进入 Git。原压缩包中的依赖说明保留为历史记录，当前本机的用法以本页为准。

**当前版本与验证结果**

| 包 | 已安装版本 |
|---|---|
| NumPy / PyTorch | 1.23.5 / 2.5.1 CPU |
| Gymnasium / Gym | 1.0.0 / 0.21.0 |
| PaddlePaddle / PARL | 2.6.2 / 2.2.1 |
| Matplotlib / pandas / SciPy | 3.9.4 / 2.2.3 / 1.13.1 |
| tyro / TensorBoard | 0.9.35 / 2.11.0 |
| cloudpickle / protobuf | 1.6.0 / 3.20.0 |
| MoviePy / wandb | 1.0.3 / 0.16.6 |

完整的 108 项已安装软件包记录见 [环境快照](environment_windows_py39.txt)。2026-11-09 的验证结果如下。

- pip check 通过，现有数学与终止边界测试 5 项通过。
- Code_2/check_env.py 全部通过，smoke_lessons.py 六组课程检查全部通过。
- 根目录的老虎机、动态规划、SARSA、Q-learning，以及 DQN、Double DQN、REINFORCE、Actor–Critic、PPO、DDPG、TD3、SAC 入口均成功执行。深度值函数与连续控制入口使用 1200 环境步，策略梯度使用 3 回合。
- Code_2 的 SARSA、Q-learning、DQN、策略梯度和 DDPG 都执行了真实环境交互与学习，检查了 Q 表或网络参数确实变化。
- Code_1 的 PPO 执行了 2048 环境步，事件文件中读取到 81 条回报记录。两份原绘图脚本在临时项目副本中生成了 6 张图片。
- 上述共 21 组入口与运行检查全部通过，明细在本地 runs/environment_check/summary.json，日志和结果也在该目录。
- PPO 录像入口额外执行了 128 步，成功生成 2 个 MP4。TensorBoard 本地 HTTP 服务及事件发现成功，wandb 导入成功，曲线绘制与结果汇总工具成功执行。Tcl 8.6.12 可初始化。

真实环境检查还发现课程 DDPG 的探索动作在加入 NumPy 噪声后变成 float64，被环境的 float32 动作空间拒绝。已在 [DDPG Agent](../projects/Code_2/lesson5/ddpg/agent.py) 中将探索动作转换为 float32，并在课程冒烟检查中加入真实动作执行。该修复使原训练入口能够继续采样和更新。

本次验证使用 CPU 和短训练预算，回报数值用于确认代码执行及学习更新。完整实验的收敛效果应按课程或研究项目要求增加预算和随机种子。
