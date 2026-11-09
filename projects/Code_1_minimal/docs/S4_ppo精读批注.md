# S4 · CleanRL `ppo.py` 精读批注

> 精读对象：`ref/cleanrl/cleanrl/ppo.py`，原版 **312 行**
> clone 版本：commit `fe8d8a0`（2026-04-20）
> 验收目标：合上文件能口述下面这 6 步，且在纸上画出张量怎么流
> 用法：左边开 PyCharm 看 `ppo.py`，右边看这份，按行号对

> ⚠️ **行号偏移提示（2026-10-27）**：为解决 gymnasium 1.4.0 不兼容，
> 在第 210 行打了 patch（`final_info` → `infos["episode"]` + `_episode` 掩码），
> 该位置从 6 行变成 10 行，**210 行之后的所有行号 +4**：
> ④ 192–**219**　⑤ 221–243　⑥ 245–313（原 192–215 / 217–239 / 241–309）。
> ① ② ③ 三块在 patch 之上，行号不变。补丁说明见 `logs/S2_baseline.md`。

---

## 0. 全局：六块，行号一目了然

| 块 | 行号 | 一句话 |
|---|---|---|
| ① 超参账本 | 17–78 | 所有可调项在一个 dataclass 里，命令行直接覆盖 |
| ② 装配 | 81–126 | 环境工厂 / 权重初始化 / actor-critic 双头网络 |
| ③ 初始化 | 129–183 | 派生批量大小、定种子、建 4 个并行环境、开 6 个存储张量 |
| ④ Rollout 采样 | 192–215 | 跑 128 步，边跑边存 `(s, a, logπ, r, d, V)` |
| ⑤ GAE + 摊平 | 217–239 | 反算优势，把 `(128, 4)` 摊成 `(512,)` |
| ⑥ K 轮小批更新 | 241–309 | 4 个 epoch × 4 个 minibatch，重算 logπ 做 clip |

**注意 129 行之后全在 `if __name__ == "__main__":` 里。** 也就是说这个文件没有函数封装，训练逻辑是平铺的脚本。读的时候别找 `def train()`，不存在。

---

## ① 超参账本（17–78）

```python
@dataclass
class Args:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
```

用 `@dataclass` + `tyro.cli(Args)`（130 行），效果是**每个字段自动变成命令行参数**，`--total-timesteps 50000` 就是这个来的。这是 CleanRL 敢说"单文件"的底气——不需要 argparse 那一坨。

### 三个占位字段（73–78 行）

```python
batch_size: int = 0        # 运行时算
minibatch_size: int = 0    # 运行时算
num_iterations: int = 0    # 运行时算
```

**为什么留 0**：它们不是超参，是 `num_envs × num_steps` 的派生量。放在 dataclass 里只是为了让 131–133 行有个地方赋值，也顺便让 tensorboard 的超参表能记全。

### 关键参数表

| 参数 | 默认 | 实际含义 | S3 破坏实验 |
|---|---|---|---|
| `num_envs` | 4 | 并行环境数 | — |
| `num_steps` | 128 | 每个环境每次采样多少步 | 第 4 条 `--num-steps 16` |
| `num_minibatches` | 4 | 一批数据切几份 | — |
| `update_epochs` | 4 | 同一批数据重复训几轮 | 第 5 条 `--update-epochs 1` |
| `clip_coef` | 0.2 | 信任域半径 | 第 1、2 条 `0.0` / `5.0` |
| `ent_coef` | 0.01 | 熵奖励权重 | 第 3 条 `0.0` |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 | 第 7 条注释掉 |
| `norm_adv` | True | 优势是否标准化 | 第 6 条注释掉 |
| `gae_lambda` | 0.95 | GAE 的 λ | — |
| `gamma` | 0.99 | 折扣 | — |
| `anneal_lr` | True | 学习率线性衰减 | — |
| `clip_vloss` | True | 价值函数也做 clip | — |
| `target_kl` | None | KL 超阈值就提前停 epoch | — |

### 派生量算出来是多少

```
batch_size      = num_envs × num_steps = 4 × 128 = 512
minibatch_size  = batch_size ÷ num_minibatches = 512 ÷ 4 = 128
num_iterations  = total_timesteps ÷ batch_size = 500000 ÷ 512 = 976
```

**这三个数字要记住 512 / 128 / 976。** 后面所有张量形状都在围着它们转。

> ⚠️ `//` 是整除，最后不足 512 步的那点数据会被**丢掉**。所以 `--total-timesteps 50000` 实际只跑 49,664 步，不是 50,000。S2 看 return 时别纠结这个差额。

---

## ② 装配（81–126）

### `make_env`（81–91）为什么套两层

```python
def make_env(env_id, idx, capture_video, run_name):
    def thunk():          # ← 内层，无参
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk          # ← 返回函数本身，不返回环境
```

`SyncVectorEnv` 要一个「无参可调用对象」的列表，因为并行环境必须**延迟创建**（子进程里各自建）。所以这里返回的是工厂，不是环境。这是 gym 的惯例，写错成直接返回 env 会报 `TypeError: 'CartPoleEnv' object is not callable`。

`RecordEpisodeStatistics` 是日志的关键：它把每局的 return 和长度塞进 `infos["episode"]`，210–215 行读的就是这个。

### `layer_init`（94–97）

```python
torch.nn.init.orthogonal_(layer.weight, std)
torch.nn.init.constant_(layer.bias, bias_const)
return layer
```

正交初始化。用 `return layer` 的写法是为了能内联写进 `nn.Sequential`。

**PPO 里这不是可选项。** 均匀初始化 + PPO 经常直接训崩，原因是初始策略太极端，第一批数据全是坏样本，优势估计全是同一个符号，梯度方向一开始就歪了。S5 手写时如果忘了这步，你会以为是 clip 写错了。

### `Agent`（100–126）三个设计决定

**决定一：critic 和 actor 是两套完全独立的 MLP，不共享 trunk。**

```python
self.critic = nn.Sequential(...)   # 独立
self.actor  = nn.Sequential(...)   # 独立
```

很多教程共享特征提取层，CleanRL 不共享。理由：价值和策略的梯度尺度不同，共享会让 value loss（数值大）压过 policy loss。

**决定二：最后一层的 `std` 故意不一样。**

```python
layer_init(nn.Linear(64, 1), std=1.0)              # critic 最后一层
layer_init(nn.Linear(64, n_actions), std=0.01)     # actor 最后一层
```

- actor 用 `0.01` → 初始 logits 几乎全 0 → softmax 接近均匀分布 → **初始策略最大程度探索**
- critic 用 `1.0` → 初始 V 有正常方差 → 不会因为 V≡0 让优势全等于 return

**这两个数写反，或者都用默认的 √2，就是典型的「训练前期完全不动」故障。**

**决定三：`get_action_and_value` 一个函数干三件事。**

```python
def get_action_and_value(self, x, action=None):
    logits = self.actor(x)
    probs = Categorical(logits=logits)      # 注意是 logits 不是 probs
    if action is None:
        action = probs.sample()             # 采样时用
    return action, probs.log_prob(action), probs.entropy(), self.critic(x)
```

`action is None` 这个分支是核心设计：

- **采样时**（199 行）：不传 action → 现采一个
- **更新时**（250 行）：传入旧 action → 只要 `log_prob`，不采样

同一个函数两种用法，保证采样和更新用的是**同一份 logits 计算路径**。这点很重要，S5 手写时如果采样和更新走了两条代码路径，ratio 会因为数值差异而不等于 1。

`Categorical(logits=...)` 而不是 `Categorical(probs=softmax(...))`：内部会自己做 log_softmax，数值更稳。

---

## ③ 初始化（129–183）

### 定种子（153–157）三行都要

```python
random.seed(args.seed)        # python 原生
np.random.seed(args.seed)     # numpy（环境内部采样用）
torch.manual_seed(args.seed)  # torch
torch.backends.cudnn.deterministic = args.torch_deterministic
```

只设 torch 的种子不够——CartPole 的初始状态是 numpy 随机生成的。**S3 七个实验要可比，靠的就是这三行。**

### 六个存储张量（171–176）

```python
obs      = torch.zeros((num_steps, num_envs) + obs_shape)   # (128, 4, 4)
actions  = torch.zeros((num_steps, num_envs) + act_shape)   # (128, 4)
logprobs = torch.zeros((num_steps, num_envs))               # (128, 4)
rewards  = torch.zeros((num_steps, num_envs))               # (128, 4)
dones    = torch.zeros((num_steps, num_envs))               # (128, 4)
values   = torch.zeros((num_steps, num_envs))               # (128, 4)
```

| 张量 | 形状 | 存的是什么时刻的量 |
|---|---|---|
| `obs[s]` | (128, 4, 4) | **动作之前**的观测 = s_t |
| `actions[s]` | (128, 4) | 在 s_t 上采的 a_t |
| `logprobs[s]` | (128, 4) | **旧策略**给出的 logπ_old(a_t\|s_t) |
| `rewards[s]` | (128, 4) | 执行 a_t 拿到的 r_t |
| `dones[s]` | (128, 4) | 执行 a_t 后是否终止 d_t |
| `values[s]` | (128, 4) | V(s_t)，**采样时**算的 |

**第一维是"时间"，第二维是"环境"。** 这个顺序跟直觉相反（一般是 batch 在前），踩过的坑：`obs.reshape(-1, ...)` 摊平时，展开顺序是 `[t0e0, t0e1, t0e2, t0e3, t1e0, ...]`，也就是**相邻索引是同一步的不同环境**，不是同一环境的连续步。

这不影响正确性（只要所有张量用同一种摊平方式），但影响你调试时的直觉。

---

## ④ Rollout 采样（192–215）

```python
for step in range(0, args.num_steps):
    global_step += args.num_envs              # ← 一次加 4
    obs[step] = next_obs                      # ← 先存 s_t
    dones[step] = next_done

    with torch.no_grad():                     # ← 采样不算梯度
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
    actions[step] = action
    logprobs[step] = logprob

    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
    next_done = np.logical_or(terminations, truncations)
```

### 四个容易看错的细节

**1. `obs[step] = next_obs` 在动作之前。** 存的是**执行动作前**的状态。这就是为什么第五块能直接用 `values[t]` 和 `obs[t]` 配对——两边都是 s_t。

**2. `global_step += args.num_envs` 不是 `+= 1`。** 4 个环境同时走一步 = 4 个时间步。这个数只用于日志横轴。

**3. `next_done` 是 terminations 和 truncations 的并集**（206 行）。gymnasium 把"任务完成"和"时间超限"拆成两个返回值，PPO 只关心"这条轨迹断没断"。用 numpy 的 `logical_or` 而不是 `or`——后者对数组会报错。

**4. `torch.no_grad()` 只包住网络前向，不包 `envs.step`。** 环境本来就不在计算图里，但 `values` 必须 detach 后再存，否则整个 rollout 的计算图会挂在内存里，跑 128 步就 OOM。

### 日志（210–215）

```python
if "final_info" in infos:
    for info in infos["final_info"]:
        if info and "episode" in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
```

`final_info` 只在**某个环境刚好结束那一局**时才出现，`info` 可能是 `None`（其他环境没结束，占位），所以那个 `if info and` 不能省。

**S2 的验收就是盯这一行输出。** 最后 10 次 `episodic_return` 平均 ≥ 400 才算过。

---

## ⑤ GAE + 摊平（217–239）

这是全文数学密度最高的 23 行。公式先摆出来：

```
δ_t   = r_t + γ·V(s_{t+1})·(1-d_t) − V(s_t)
Â_t   = δ_t + γ·λ·(1-d_t)·Â_{t+1}
```

对应代码：

```python
for t in reversed(range(args.num_steps)):          # ← 必须倒着走
    if t == args.num_steps - 1:
        nextnonterminal = 1.0 - next_done          # 最后一步后面没有数据，用真实 next_done
        nextvalues = next_value                    # 需要额外算一次 V(s_T)
    else:
        nextnonterminal = 1.0 - dones[t + 1]       # ← 注意是 t+1
        nextvalues = values[t + 1]
    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
returns = advantages + values
```

### 为什么必须 `reversed`

`Â_t` 依赖 `Â_{t+1}`。正着走拿不到。这个递推形式是 GAE 的定义，等价于对 TD 残差做指数加权和。

### `nextnonterminal` 的索引陷阱

| 计算 `δ_t` 时 | 用哪个 done |
|---|---|
| t = 127（最后一步） | `next_done`（环境返回的） |
| t ≤ 126 | `dones[t+1]` |

**不是 `dones[t]`。** 因为 `δ_t` 里要判断的是「s_{t+1} 是不是终止状态」，所以用 t+1 时刻的 done。写成 `dones[t]` 不会报错，只会让优势估计系统性偏小，表现为学得慢但"看起来在学"——最难查的一类 bug。

### `returns = advantages + values`（231 行）

这一步很容易漏看。`returns` 是**价值函数的回归目标**：

```
return_t = Â_t + V(s_t) ≈ 蒙特卡洛回报
```

因为 `Â_t = Q(s_t,a_t) − V(s_t)`，所以 `Â_t + V(s_t) = Q(s_t,a_t)`，而 GAE 的 Â 本身就是 Q 的 λ-加权估计。**value loss 优化的就是让 V(s_t) 逼近这个数。**

### 摊平（233–239）

```python
b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)   # (128,4,4) → (512,4)
b_advantages = advantages.reshape(-1)                              # (128,4) → (512,)
```

**5 个张量必须用同一种 reshape 方式**，否则 `b_obs[i]` 和 `b_advantages[i]` 对不上。这里全靠 `.reshape(-1)` 的默认行优先顺序，天然一致。S5 手写时如果用 `permute` 再 reshape，就是典型的错位 bug：不报错，奖励也涨，但 ratio 永远是乱的。

---

## ⑥ K 轮小批更新（241–309）

### 双层循环结构

```python
b_inds = np.arange(args.batch_size)          # 512 个下标
for epoch in range(args.update_epochs):      # 4 轮
    np.random.shuffle(b_inds)                # ← 每轮重新打乱
    for start in range(0, args.batch_size, args.minibatch_size):   # 4 个 minibatch
        mb_inds = b_inds[start:end]          # 128 个下标
```

**同一批 512 条数据，被重复利用 4 次。** 这就是 PPO「样本效率高」的来源——SAC/DQN 一条数据只用一次，PPO 用 4 次（K=4）。

`shuffle` 在 epoch 外层，所以 4 轮的分组不同。

### 核心三行：重算 logπ

```python
_, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
logratio = newlogprob - b_logprobs[mb_inds]
ratio = logratio.exp()
```

**这里是整个 PPO 的心脏。**

- `b_logprobs` 是**采样时**（旧策略 π_old）算的，冻在那里
- `newlogprob` 是**现在**（当前策略 π_θ，已经训过几个 minibatch 了）重算的
- `ratio = π_θ(a|s) / π_old(a|s)`，实现上用 `exp(logπ_θ − logπ_old)` 而不是直接除

**为什么用 exp 做差**：log 概率可能是 −20 这种量级，直接 `exp(a)/exp(b)` 会溢出。做差再 exp 数值稳定。

**为什么 `b_actions.long()`**：`actions` 张量建的时候是 float（因为 `envs.single_action_space.shape` 是空 tuple，`torch.zeros` 默认 float32），`log_prob` 要求整数索引。`.long()` 是必需的，忘了会报 `Expected Long`。

### pg_loss 为什么用 `max` 而不是 `min`

```python
pg_loss1 = -mb_advantages * ratio
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

**论文里写的是 `min`，代码里是 `max`，两者等价** —— 因为代码里先把符号取负了：

```
论文：L = E[min(r·A, clip(r)·A)]        （要最大化）
代码：loss = -L = E[max(-r·A, -clip(r)·A)]   （要最小化）
```

`max(-x, -y) = -min(x, y)`。**看到 max 不要以为 CleanRL 写错了**，这是被问最多的一个问题。

### 三个 loss 怎么拼起来（284–285）

```python
entropy_loss = entropy.mean()
loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
```

| 项 | 符号 | 意味着 |
|---|---|---|
| `pg_loss` | + | 越小越好 |
| `- ent_coef × entropy` | **−** | 熵越大越好（鼓励探索） |
| `+ vf_coef × v_loss` | + | 越小越好 |

**注意熵前面的减号。** 抄代码时把减号写成加号，策略会迅速坍缩成确定性策略，CartPole 上表现为前 50k 步还行、之后突然掉下去。

### value loss 的 clip 分支（271–282）

```python
if args.clip_vloss:
    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()
```

这个是 **PPO2（2017）的实现细节，原版 PPO 论文没有**。作用是防止 value 网络一次更新跨太大。默认开着，S3 可以顺手关掉试试。

`0.5 *` 是 MSE 求导后 2 被约掉的传统遗留。

### 梯度裁剪（289 行）

```python
nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)   # 0.5
```

**这是 S3 第 7 条要注释掉的那行。** 注意它和 PPO 的 `clip_coef` 是两件完全不同的事：

| | 对象 | 层级 |
|---|---|---|
| `clip_coef` | `ratio`（概率比） | **loss 层**，改的是目标函数 |
| `max_grad_norm` | 梯度范数 | **优化器层**，改的是更新步长 |

混淆这两个是新手常见误解。

### `target_kl` 的一个隐蔽问题（292–293）

```python
for epoch in range(args.update_epochs):     # ← epoch 循环里
    ...
    for start in range(...):                # ← minibatch 循环里
        approx_kl = ((ratio - 1) - logratio).mean()
    if args.target_kl is not None and approx_kl > args.target_kl:
        break
```

`approx_kl` 在 minibatch 循环里每次被覆盖，`break` 判断时用的是**最后一个 minibatch** 的值，不是整批的。默认 `target_kl=None` 所以不触发，但你哪天想启用要自己改成累积平均。

### `approx_kl` 为什么不用 `-logratio.mean()`

```python
old_approx_kl = (-logratio).mean()                 # k1 估计，有偏
approx_kl    = ((ratio - 1) - logratio).mean()     # k3 估计，恒非负、方差更小
```

k3 估计 `KL ≈ (r−1) − log r` 对任意 r > 0 都非负，而 k1 在小批样本上方差很大。两个都记进 tensorboard 是为了对照。

---

## ⑦ 数据流图

```
                    ┌──────────────────────────────┐
                    │  next_obs (4, 4)  next_done   │
                    └───────────────┬──────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │  ④ ROLLOUT   × 128 步                                  │
        │                                                        │
        │   obs[s]=next_obs ──► agent(next_obs) ──► a, logπ, V   │
        │        │                    │                          │
        │        │                    ▼                          │
        │        │            envs.step(a) ──► r, term, trunc    │
        │        │                    │                          │
        │        └──── 存 6 个 (128,4) 张量 ◄── next_obs 回到顶部 │
        └───────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  ⑤ GAE   从 t=127 倒着递推                             │
        │     δ_t = r_t + γV_{t+1}(1-d) − V_t                    │
        │     Â_t = δ_t + γλ(1-d)Â_{t+1}                         │
        │     returns = Â + V        ← value 网络的回归目标       │
        └───────────────────────────┬───────────────────────────┘
                                    │  reshape(-1)
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  ⑥ UPDATE   epoch×4 ─► minibatch×4                    │
        │                                                        │
        │     b_obs ─► agent ─► newlogπ, entropy, newV           │
        │                         │                              │
        │     b_logprobs ─────────┴─► ratio = exp(new−old)        │
        │                              │                         │
        │              ┌───────────────┼───────────────┐         │
        │              ▼               ▼               ▼         │
        │        pg_loss(max)     v_loss(clip)   −ent_coef·H     │
        │              └───────────────┴───────────────┘         │
        │                        loss.backward()                 │
        │                        clip_grad_norm_                 │
        │                        optimizer.step()                │
        └───────────────────────────┬───────────────────────────┘
                                    │  回到 ④，下一轮 iteration
                                    ▼
```

---

## ⑧ 四个最容易看错的地方（自查清单）

| # | 现象 | 正确理解 |
|---|---|---|
| 1 | `pg_loss = torch.max(...)` 但论文写 `min` | 代码先取了负号，`max(-x,-y) = -min(x,y)`，等价 |
| 2 | GAE 里 `dones[t+1]` | `δ_t` 判断的是 s_{t+1} 是否终止，所以取 t+1 的 done |
| 3 | `loss = pg_loss - ent_coef*entropy` | 熵要**最大化**，所以是减号 |
| 4 | `clip_coef` 和 `max_grad_norm` 都叫"clip" | 前者是 loss 层的**概率比**裁剪（0.2），后者是优化器层的**梯度范数**裁剪（0.5），作用对象完全不同 |

---

## ⑨ 口述自测（S4 验收）

合上代码，对着空白纸回答这 6 问。答不出来的回去重看对应行号。

- [ ] **① 一批数据多大？** 512 = 4 环境 × 128 步；切 4 份每份 128；整个训练 976 轮。
- [ ] **② rollout 阶段存了哪 6 个量？** `obs / actions / logprobs / rewards / dones / values`，都是 `(128, 4)`，第一维时间第二维环境。
- [ ] **③ GAE 为什么要倒着算？** `Â_t` 依赖 `Â_{t+1}`。
- [ ] **④ `returns` 是什么，拿来干嘛？** `Â + V`，作为 value 网络的回归目标。
- [ ] **⑤ `ratio` 怎么算，为什么要重算 logπ？** `exp(logπ_θ − logπ_old)`；π_old 是采样时冻住的，π_θ 是当前更新中的，两者比值反映策略漂移了多少。
- [ ] **⑥ 三部分 loss 的符号各是什么？** `+pg_loss`、`−ent_coef·H`、`+vf_coef·v_loss`。

如果你能一口气答完，S4 就过了，可以进 S5 手写。

---

## ⑩ 改成你的调度任务时，要动哪几处

现在心里有数就行，S12 再动手：

| 原代码 | 你的版本 | 为什么 |
|---|---|---|
| `envs.single_action_space.n`（115 行） | 节点数（不固定） | 动作数 = 工序数，随算例变 |
| `Categorical(logits=logits)`（123 行） | `Categorical(logits=logits.masked_fill(mask, -1e9))` | 非法动作必须屏蔽，**这是 `validator.py` 的出口** |
| `nn.Sequential(Linear(4,64),...)`（110 行） | GIN 编码器 + 打分头 | 观测从 4 维向量变成图 |
| `obs` 张量定长 `(128,4,4)` | 需要 padding 或变长处理 | 不同算例节点数不同 |

**`masked_fill(-1e9)` 用 −1e9 不用 −inf**：`softmax` 遇到 `inf - inf` 会出 `nan`，而 −1e9 在 float32 下 exp 后是 0，安全。
