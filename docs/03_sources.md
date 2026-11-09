# 参考仓库与源码阅读地图

**本机已准备的阅读源码**

2026-11-09 已将四份源码放入 external/，与本地 22 步课程分开管理。锁定版本与内容摘要见 [source_manifest.json](source_manifest.json)。

| 来源 | 本地目录 | 固定版本 |
|---|---|---|
| boyu-ai/Hands-on-RL | `external/handson` | `4a151a4bfeafd3061c0e0ef59bece25ca8fb2bf8` |
| vwxyzjn/cleanrl | `external/cleanrl` | `1.2.0` |
| DLR-RM/stable-baselines3 | `external/sb3` | `9555dd9859025561db12f331fc33be279b023a0a` |
| zcaicaros/L2D | `external/l2d` | `7b2efbb1ffc960260b16952f2bed68e500765bf0` |

CleanRL 使用官方 PyPI 1.2.0 源码发行包，包含 DQN/PPO/DDPG/TD3/SAC 等算法文件和 LICENSE。该源码的校验值来自 PyPI 发行记录，阅读时可与 projects/Code_1_minimal 中已经适配当前环境的 PPO 对照。其他三份来自 GitHub，使用提交号与文件树摘要核对版本。SB3 选用与现有 Python 3.9 环境兼容的 2.4.1，包和对应源码都已准备。

上游源码保留原目录与授权文件，供公式和实现对照。执行课程时使用 lessons/ 和 projects/ 的入口。L2D 的旧版依赖及业务建模需要在进入论文复现阶段时专门适配。SB3 的 .dockerignore 符号链接在 Windows 阅读副本中保存为 .gitignore 内容的常规文件。

external/ 由 Git 忽略，仓库同步后可以按固定清单恢复。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.fetch_sources
```


## 阅读范围与课程取舍

2026-11-07 查阅项目 README、目录、相关章节、依赖以及主线算法的目标计算与更新循环。这里记录与课程有关的重点阅读范围，属于定向源码导读，不声称逐文件审计整个上游项目。网页内容来自当次可读取版本，链接指向上游默认分支，后续可能变化。

| 来源 | 本次重点 | 在课程中的角色 |
|---|---|---|
| [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL) | README、章节目录、DQN 章节、各算法 Notebook 入口 | 中文教材，按章节读公式与解释 |
| [dennybritz](https://github.com/dennybritz/reinforcement-learning) | README、TD 学习目标、SARSA/Q-learning 解答 | MDP/DP/MC/TD 的练习与手算对照 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | README、DQN/PPO/DDPG/TD3/SAC 脚本与文档 | 理解完整训练循环和实现细节 |
| [Spinning Up](https://github.com/openai/spinningup) | 算法目录、PPO/TD3/SAC 方程、PyTorch PPOBuffer/损失/更新、setup.py | 英文推导和伪代码 |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | README 和适用定位 | 学完教学代码后的成熟基线 |
| [joyrl-book](https://github.com/datawhalechina/joyrl-book) | README 与目录 | 补充中文材料，不加入主线以减少重复 |

路线选择依据是概念先后关系、中文可读性、单文件代码可追踪性和环境成本。星标只反映关注度，不作为学习顺序依据。

## Hands-on-RL

[在线教材目录](https://hrl.boyuai.com/chapter/) 提供完整文字讲解；GitHub README 说明仓库主要提供纯代码 Notebook。阅读顺序为第 2–5、7–10、12–14 章，第 6 章 Dyna-Q 和第 11 章 TRPO 作为拓展。

定位时先看环境与算法类，再看 `update()`，最后看训练调用。网页讲解与代码结合阅读。旧版示例中的 `reset()`/`step()` 返回格式与当前 Gymnasium 不同。对照 [Env API](https://gymnasium.farama.org/api/env/)，理解差异后再迁移。

## dennybritz

先做练习 Notebook，再打开同目录的 Solution。

- [DP](https://github.com/dennybritz/reinforcement-learning/tree/master/DP) 的 Policy Evaluation、Policy Iteration、Value Iteration，区分求期望与取最大值。
- [MC](https://github.com/dennybritz/reinforcement-learning/tree/master/MC) 的 Prediction 与 Control，观察样本回报均值。
- [Q-learning 解答](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb) 的 `make_epsilon_greedy_policy` 与 `q_learning`。
- [SARSA 解答](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb) 的 `sarsa`，核对下一步实际动作进入目标的位置。

原始两个 TD 示例分别使用 CliffWalking 和 WindyGridworld，所以不能直接拿原始曲线做同环境算法比较。本地 `03_td_control` 将二者统一到 CliffWalking。深度部分包含早期 TensorFlow 写法及 WIP 内容，主线深度代码转向 CleanRL。

## CleanRL

| 代码 | 阅读锚点 | 对照问题 |
|---|---|---|
| [dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) | `QNetwork`、`linear_schedule`、`rb.add`、`target_max`、`gather` | 哪个张量参与梯度？截断后的真实 observation 如何保存？ |
| [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) | `get_action_and_value`、`lastgaelam`、`logratio`、`pg_loss1/2` | 旧概率如何固定？优势如何归一化？ |
| [ddpg_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py) | Actor、`qf1_target`、`actor_loss` | Q 对动作的导数如何到达 Actor？ |
| [td3_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) | `clipped_noise`、`min_qf_next_target`、`policy_frequency` | 三项改进各改了哪里？ |
| [sac_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py) | `get_action`、`log_alpha`、`next_state_log_pi` | log 概率如何修正？温度参数如何学习？ |

先略读追踪、上传和视频等实验管理代码，再追踪一批数据经过哪些计算。先读 PyTorch 低维状态版本，之后再看 Atari/JAX/并行实现。CleanRL 的完整脚本还依赖工具模块与第三方库，单文件结构不等于可以省略其依赖。参见 [项目 README](https://github.com/vwxyzjn/cleanrl)。

## Spinning Up

先阅读 [算法目录](https://spinningup.openai.com/en/latest/user/algorithms.html)，再读 [VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)、[PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)、[DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)、[TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)、[SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)。

代码看 [PyTorch PPO](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py) 的 `PPOBuffer.finish_path`、`compute_loss_pi` 与 `update`。比较本地 GAE 和上游路径结束时的 last value 处理。上游还包含 MPI、KL 提前停止和日志框架。其 [setup.py](https://github.com/openai/spinningup/blob/master/setup.py) 依赖年代较早，原环境应另建环境复现。

## 来源与授权

lessons/ 中的课程代码为本地教学实现。external/ 保存固定版本的上游源码与原有授权文件，projects/ 保存用户提供的项目材料。改编上游具体文件时，应保留版权声明、对应 LICENSE，并标注来源和修改。Hands-on-RL 网页文字与仓库代码的授权分别核对，本仓库的整体发布许可证由仓库所有者决定。
