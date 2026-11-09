# 李科浇《世界冠军带你从零实践强化学习》— 代码工作区

课程地址：B站 BV1yv411i7xd（8 节，约 5h48m，配套 PARL + gym）

## 环境

| 项 | 值 |
|---|---|
| conda 环境名 | `rl` |
| 环境路径 | `D:\envs\conda-envs\rl` |
| 解释器 | `D:\envs\conda-envs\rl\python.exe` |
| Python | 3.9.25 |
| 包管理器 | conda（默认通道）+ pip 仅补 conda 没有的包 |

### 已装版本（不要随意升级）

| 包 | 版本 | 来源 | 说明 |
|---|---|---|---|
| numpy | 1.23.5 | conda | 必须 <1.24，gym 0.21 用了 `np.bool8` |
| scipy | 1.13.1 | conda | |
| matplotlib | 3.9.2 | conda | 画训练曲线 |
| tqdm | 4.67.1 | conda | |
| pillow / cloudpickle | 11.1.0 / 1.6.0 | conda / pip | cloudpickle 被 parl 锁到 1.6.0 |
| **gym** | **0.21.0** | pip | 旧版 4 返回值 API：`reset()->obs`、`step()->obs,r,done,info` |
| **paddlepaddle** | **2.6.2** | pip | PARL 的后端框架，PyPI 上最后的 2.x |
| **parl** | **2.2.1** | pip | 课程用的算法框架 |
| pyglet | 1.5.27 | pip | gym 渲染窗口依赖 |
| pip / setuptools | 23.3.1 / 65.5.0 | conda | pip 必须 <24.1，否则拒装 gym 0.21 |

## ⚠️ 关键坑：PARL 在 Windows 上识别不到 paddle

PARL 2.2.1 的 Windows 后端探测写死了 `from paddle import fluid`，
而 paddle 2.6 已删除 `paddle.fluid`，于是静默降级成
「No deep learning framework was found」，`parl.Model` / `parl.algorithms` 全部消失。

已在环境里用两个位置同时兜住：

1. `D:\envs\conda-envs\rl\Lib\site-packages\sitecustomize.py`
   → 每个进程启动时自动 `setdefault("PARL_BACKEND", "paddle")`
2. `conda env config vars set PARL_BACKEND=paddle -n rl`

**不要删这两处。** 如果换机器重建环境，`sitecustomize.py` 要重新拷一份。

## 自检

```bash
D:\envs\conda-envs\rl\python.exe check_env.py
```

全部 `[ OK ]` 才算环境可用。该脚本会真的跑一遍 CartPole 和 parl DQN 前向。

## 注意

- **不要 `pip install -U gym`**。0.26+ 把 `reset()` 改成返回 `(obs, info)`、`step()` 返回 5 元组，
  课程代码全线报废。
- **不要 `pip install -U parl`**。2.2.1 之后没有再适配 paddle 2.6 的版本。
- 课程代码里的 `Pendulum-v0` 在 gym 0.21 中已更名为 `Pendulum-v1`，遇到报错直接改名。
- 课程用 PARL 写，工业界基本是 PyTorch + gymnasium。学算法思路，别把这套 API 当标准。

---

## 课程代码（已下载）

来源 `PaddlePaddle/PARL` 仓库 `examples/tutorials/`，只取能跑的算法实现。

| 目录 | 内容 | 对应视频 |
|---|---|---|
| `lesson1/gridworld.py` | 格子世界渲染封装（turtle） | Lesson1-3 |
| `lesson2/sarsa` | 表格型 Sarsa，跑 CliffWalking | Lesson2-2 ~ 2-3 |
| `lesson2/q_learning` | 表格型 Q-learning，跑 CliffWalking | Lesson2-4 |
| `lesson3/dqn` | DQN 解 CartPole | Lesson3-2 ~ 3-4 |
| `lesson4/policy_gradient` | PG 解 CartPole | Lesson4-2 ~ 4-3 |
| `lesson5/ddpg` | DDPG 解连续动作 CartPole | Lesson5-1 ~ 5-2 |

**刻意没下的**：
- `homework/` —— 作业答案，用不上
- 静态图版 lesson3/4/5 —— 依赖 `paddle.fluid`，paddle 2.6 已删除该模块，必挂。上表用的是仓库里的 `parl2_dygraph` 动态图版。

## 为跑通做的 2 处本地适配

| 位置 | 改动 | 原因 |
|---|---|---|
| 5 个 `train.py` + `lesson1/gridworld.py` | 删掉 `assert xx.__version__ == "x.y.z"` | 原版硬锁 paddle 2.2.0 / parl 2.0.3 / gym 0.18.0，本机不匹配会在 import 阶段直接崩 |
| `lesson1/gridworld.py`、`lesson2/*/gridworld.py` | `FrozenLake-v0` → `FrozenLake-v1` | gym 0.21 中 v0 已被 `DeprecatedEnv` 拦掉 |

算法逻辑一行未改，改动处都留了 `[本地适配]` 注释。

## 运行

```bash
cd lesson3/dqn
D:\envs\conda-envs\rl\python.exe train.py
```

`smoke_lessons.py` 一次性验证 6 个 lesson 能否构造 / 前向 / learn。改代码后先跑它，比逐个开训练快得多：

```bash
D:\envs\conda-envs\rl\python.exe smoke_lessons.py
```

