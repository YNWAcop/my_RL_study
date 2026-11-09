# S2 基线记录：CleanRL PPO on CartPole-v1

> 实测日期：2026-10-27
> 解释器：`D:\envs\conda-envs\paper\python.exe`（Python 3.12.15 / torch 2.9.1+cu128 / gymnasium 1.4.0）
> 脚本：`ref/cleanrl/cleanrl/ppo.py`

---

## 一、命令（Git Bash，实测可跑）

```bash
cd /d/Papers/Code
"D:/envs/conda-envs/paper/python.exe" ref/cleanrl/cleanrl/ppo.py \
  --env-id CartPole-v1 --total-timesteps 500000 --seed 1
```

不要加 `--track`（wandb 需登录）。
`D:\envs\...` 写法在 Git Bash 下也能执行（MSYS 会自动转换），但正斜杠更保险。

---

## 二、必须先打的补丁：episodic_return 打印不出来

### 症状
按原代码跑到底，终端**一条 `episodic_return` 都没有**，TensorBoard 里
`charts/episodic_return` / `charts/episodic_length` **完全没有数据**。

### 根因
`ppo.py` 原第 210 行的判断是 gymnasium 0.2x 时代的 API：

```python
if "final_info" in infos:            # ← 在 gymnasium 1.4.0 下恒为 False
```

gymnasium ≥ 1.0 的 `SyncVectorEnv` 改了 info 聚合方式：
- 不再有 `final_info` / `final_observation` 键
- 改为 `infos["episode"]` 批量字典（`{'r': array([...]), 'l': array([...]), ...}`）
- 外加 `infos["_episode"]` 布尔掩码，标记本步哪些子环境结束

实测证据（`SyncVectorEnv(CartPole) × 2`，某步两个 env 同时终止）：

```
infos = {'episode': {'r': array([10., 10.]), '_r': array([True, True]),
                     'l': array([10, 10]),  '_l': array([True, True]),
                     't': array([...]),     '_t': array([True, True])},
         '_episode': array([True, True])}
terminations = array([True, True])
```

### 补丁（已应用到 `ref/cleanrl/cleanrl/ppo.py` 第 210 行）

```python
            # PATCH(2026-10-27, gymnasium 1.4.0): 新版 vector env 不再返回 "final_info"，
            # 改为 infos["episode"] 批量字典 + infos["_episode"] 布尔掩码（标记本步结束的 env）。
            if "episode" in infos:
                for _i in np.flatnonzero(infos["_episode"]):
                    ep_r = float(infos["episode"]["r"][_i])
                    ep_l = int(infos["episode"]["l"][_i])
                    print(f"global_step={global_step}, episodic_return={ep_r}")
                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episodic_length", ep_l, global_step)
```

验证：补丁后 20k 步跑出 454 条 episode 记录，TB 的 `charts/episodic_return` 有对应标量。
**`ref/` 在 `.gitignore` 里，重新 clone cleanrl 后补丁会丢失，需重打。**

---

## 三、50k 步达不到 400（清单原验收标准有误）

清单写的 `--total-timesteps 50000` + `return ≥ 400` 不可能达成。实测 seed=1：

| 训练步数 | episode 数 | 最后 10 个均值 | 分段均值 | 墙钟耗时 |
|---|---|---|---|---|
| 50,000 | 492 | **211.4** | 45k–50k 段仅 199.0 | ~30 s |
| 200,000 | 969 | 236.1 | 峰值段 125k–150k 为 353.8 | 2 min 07 s |
| **500,000** | **1,523** | **500.0** | 40万步后稳定 485–500 | **4 min 51 s** |

500k 运行的关键节点：
- `global_step=106,540`：首次出现「连续 10 个 episode 均值 ≥ 400」
- `global_step≈370,000` 起：基本钉在 500

50k 曲线的上限只有 ~200，离 400 差一半。**不是环境装坏，是步数不够。**

---

## 四、修正后的 S2 验收标准

```bash
"D:/envs/conda-envs/paper/python.exe" ref/cleanrl/cleanrl/ppo.py \
  --env-id CartPole-v1 --total-timesteps 500000 --seed 1
```

- [ ] 最后 10 个 `episodic_return` 均值 ≥ 400（实测 500.0）
- [ ] 墙钟耗时约 290–300 秒（SPS 稳态 ~1670，期间偶有 1 秒级卡顿）
- [ ] `tensorboard --logdir runs` 能看到 `charts/episodic_return` 上升曲线

若坚持只跑 50k 省时间，验收线须降到 **≥ 150**，且只能证明「代码跑通」，证明不了「算法收敛」。

---

## 五、其他无害现象

两条 tyro UserWarning（`wandb-entity` / `target-kl`）来自 `Args` 里 `str = None` 这类标注，
不影响运行。要消掉就把类型改成 `Optional[str]` / `Optional[float]`。
