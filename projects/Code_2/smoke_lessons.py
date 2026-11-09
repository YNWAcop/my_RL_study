# -*- coding: utf-8 -*-
"""课程代码冒烟测试：验证各 lesson 在当前 rl 环境能构造、能前向、能 learn。"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
MASK = ["model", "agent", "algorithm", "replay_memory", "env", "gridworld", "train"]
OK = True


def load(rel):
    for m in MASK:
        sys.modules.pop(m, None)
    d = os.path.join(ROOT, rel)
    if d not in sys.path:
        sys.path.insert(0, d)
    os.chdir(d)


def chk(name, fn):
    global OK
    try:
        detail = fn()
        print("[ OK ] %-22s %s" % (name, detail or ""))
    except Exception as e:  # noqa: BLE001
        OK = False
        print("[FAIL] %-22s %s: %s" % (name, type(e).__name__, str(e)[:300]))


def l1():
    load("lesson1")
    import gym
    import gridworld  # noqa: F401
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset()
    env.step(0)
    return "gridworld 导入 + FrozenLake 可交互"


def _l2(rel, cls_name):
    load(rel)
    import importlib
    import gym

    tr = importlib.import_module("train")
    ag = importlib.import_module("agent")
    env = tr.CliffWalkingWapper(gym.make("CliffWalking-v0"))
    agent = getattr(ag, cls_name)(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1,
    )
    obs = env.reset()
    try:
        act = agent.sample(obs)
    except TypeError:
        act = agent.sample(obs, True)
    import inspect

    n = len(inspect.signature(agent.learn).parameters)
    agent.learn(*([obs, act, 0, obs, act, False][:n]))
    return "env+agent 构造成功, sample=%s" % act


def l3():
    load(r"lesson3\dqn")
    import numpy as np
    from model import Model
    from algorithm import DQN
    from agent import Agent
    from replay_memory import ReplayMemory

    obs_dim, act_dim = 4, 2
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=0.99, lr=0.0005)
    agent = Agent(alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)
    rpm = ReplayMemory(2000)
    for _ in range(300):
        o = np.random.randn(obs_dim).astype("float32")
        rpm.append((o, np.random.randint(act_dim), 1.0, o, False))
    batch = rpm.sample(64)
    loss = agent.learn(*batch)
    return "DQN sample=%s  learn_loss=%.4f" % (agent.sample(batch[0][0]), loss)


def l4():
    load(r"lesson4\policy_gradient")
    import numpy as np
    from model import Model
    from algorithm import PolicyGradient
    from agent import Agent

    obs_dim, act_dim = 4, 2
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = PolicyGradient(model, lr=0.001)
    agent = Agent(alg)
    obs = np.random.randn(obs_dim).astype("float32")
    act = agent.sample(obs)
    batch_obs = np.random.randn(8, obs_dim).astype("float32")
    batch_act = np.random.randint(act_dim, size=8)
    batch_rew = np.random.rand(8).astype("float32")
    agent.learn(batch_obs, batch_act, batch_rew)
    return "PG sample=%s  learn 成功" % act


def l5():
    load(r"lesson5\ddpg")
    import numpy as np
    from model import Model
    from algorithm import DDPG
    from agent import Agent
    from env import ContinuousCartPoleEnv

    env = ContinuousCartPoleEnv()
    obs = env.reset()
    env.step(np.float32(0.1))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = Model(act_dim=act_dim, obs_dim=obs_dim)
    alg = DDPG(model, gamma=0.99, tau=0.001, actor_lr=0.0001, critic_lr=0.001)
    agent = Agent(alg, act_dim, expl_noise=0.1)
    act = agent.sample(obs.astype("float32"))
    n = 8
    c_loss, a_loss = agent.learn(
        np.random.randn(n, obs_dim).astype("float32"),
        np.random.randn(n, act_dim).astype("float32"),
        np.random.rand(n).astype("float32"),
        np.random.randn(n, obs_dim).astype("float32"),
        np.zeros(n, dtype="float32"),
    )
    return "DDPG sample=%s  critic=%.3f actor=%.3f" % (np.round(act, 3), c_loss, a_loss)


chk("lesson1 gridworld", l1)
chk("lesson2 sarsa", lambda: _l2(r"lesson2\sarsa", "SarsaAgent"))
chk("lesson2 q_learning", lambda: _l2(r"lesson2\q_learning", "QLearningAgent"))
chk("lesson3 dqn", l3)
chk("lesson4 policy_gradient", l4)
chk("lesson5 ddpg", l5)

print("=" * 60)
print("全部通过" if OK else "有失败项，见上")
sys.exit(0 if OK else 1)
