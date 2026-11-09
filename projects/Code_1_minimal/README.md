# Code — GIN + PPO 解单工位装配调度

论文复现项目。目标：交付「复现一种 RL 算法 + 掌握原理 + 展示调参过程」。

**当前进度**：S0–S4 已完成（环境、跑通 CleanRL PPO、七个消融实验、`ppo.py` 精读）。
S5 之后（手写 PPO、调度环境、GIN 编码器）代码尚未开始，`algo/` `envs/` `models/` 等目录目前只有空 `__init__.py`，是预留骨架。

---

## 快速开始

```bash
conda create -n paper python=3.12 -y
conda activate paper
pip install torch gymnasium tyro tensorboard numpy pandas matplotlib scipy seaborn
```

不需要 `.venv`，也不需要 GPU —— CartPole 是纯 CPU 任务。

### 复现 S2：跑通 CleanRL PPO

```bash
python ref/cleanrl/cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 50000 --seed 1
```

验收标准：末尾 episodic return ≥ 400。当时的记录在 `logs/S2_baseline.md`。

### 复现 S3：七个破坏实验

对照组是 `S3_00` baseline（不改任何参数），其余七组各破坏一个 PPO 组件，看曲线怎么塌。
完整命令与预期现象见 `docs/S3_操作手册.md`，实测数据与结论见 `docs/S3_results.md`。

```bash
# 以「关掉 clip」为例
python ref/cleanrl/cleanrl/ppo.py \
  --total-timesteps 500000 --seed 1 --exp-name S3_01_no_clip --clip-coef 0.0
```

七组分别是：`--clip-coef 0.0` / `--clip-coef 5.0` / `--ent-coef 0.0` /
`--num-steps 16` / `--update-epochs 1` / `--no-norm-adv` / `--max-grad-norm 1e9`。

跑完看曲线：

```bash
tensorboard --logdir runs
```

### 重新生成文档配图

```bash
python scripts/make_s234_figures.py     # S2 学习曲线 / S3 曲线与柱状 / PPO 数据流图
python scripts/make_knn_figures.py      # kNN 演示图（读 data/iris.csv）
```

---

## 目录

| 目录 / 文件 | 放什么 | 状态 |
|---|---|---|
| `执行清单.md` | 13 步执行计划 + 每步验收标准 | 主线索引，建议先看 |
| `docs/S234_汇报.md` | S2–S4 汇报正文（含全部图） | 已完成 |
| `docs/S3_results.md` | 七组消融实验数据表与结论 | 已完成 |
| `docs/S3_操作手册.md` | 消融实验的完整命令行与超参 | 已完成 |
| `docs/S4_ppo精读批注.md` | `ppo.py` 逐块拆解，含行号与数据流图 | 已完成 |
| `docs/img/` | 文档引用的 6 张图 | 已完成 |
| `ref/cleanrl/cleanrl/ppo.py` | 唯一保留的参考代码（S2/S3 的全部依赖） | 上游原文件 |
| `scripts/` | 画图脚本（matplotlib） | 可用 |
| `data/iris.csv` | kNN 演示数据 | 已附 |
| `logs/S2_baseline.md` | S2 训练记录 | 已附 |
| `algo/` `envs/` `models/` `baselines/` `tests/` | 后续 S5–S13 的代码位置 | 空骨架 |

---

## 参考仓库

`ref/` 原本克隆了三个上游仓库，**已移除**，只留下上面那一个 `ppo.py`（S2/S3 只需要它）。
要看原文自己拉：

```bash
git clone --depth 1 https://github.com/vwxyzjn/cleanrl.git ref/cleanrl
git clone --depth 1 https://github.com/zcaicaros/L2D.git ref/L2D
git clone --depth 1 https://github.com/doubleton/Dispatching-rules-for-FJSP.git ref/Dispatching-rules-for-FJSP
```

- **cleanrl** — PPO 单文件实现，骨架来源，入口 `cleanrl/ppo.py`
- **L2D** — Learning to Dispatch，图的建法与「允许工序集」写法参考，入口 `permissibleLS.py`
- **Dispatching-rules-for-FJSP** — 4 条 dispatch rule baseline 的严格定义来源

`ref/` 已在 `.gitignore` 里，clone 下来不会污染仓库。

---

## 这份包被砍掉了什么

从完整工作目录挑出的最小可读集合，**218 MB → 不到 1 MB**：

| 排除项 | 大小 | 原因 |
|---|---|---|
| `ref/L2D`、`ref/Dispatching-rules-for-FJSP` | 25 MB | 纯上游代码，按上面命令自行拉取 |
| `ref/cleanrl` 其余部分 | 179 MB | 只需要 `ppo.py` 一个文件 |
| `logs/S3_*.log` | 1.5 MB | 原始训练日志，结论已汇总进 `docs/S3_results.md` |
| `runs/` | 11 MB | TensorBoard 事件文件，重跑就能再生 |
| `docs/chart.umd.min.js` | 200 KB | 未被任何文档引用 |
| `.git/` | — | 原始提交历史，未随包发送 |
| `.idea/` | — | 本机 IDE 配置 |

文档里出现的 `D:\envs\conda-envs\paper\python.exe` 是我的本机解释器路径，
换机器时按自己的环境路径替换即可，不影响内容。
