# -*- coding: utf-8 -*-
"""环境自检：确认 RL 课程所需依赖可用。运行: python check_env.py"""
import sys
import warnings

warnings.filterwarnings("ignore")

print("=" * 56)
print("Python :", sys.version.split()[0])
print("路径   :", sys.executable)

ok = True


def check(name, fn):
    global ok
    try:
        info = fn()
        print(f"[ OK ] {name:<14} {info}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name:<14} {type(e).__name__}: {e}")


def _np():
    import numpy
    return numpy.__version__


def _gym():
    import gym
    return gym.__version__


def _paddle():
    import paddle
    return paddle.__version__


def _parl():
    import parl
    return getattr(parl, "__version__", "?")


check("numpy", _np)
check("gym", _gym)
check("paddle", _paddle)
check("parl", _parl)


# --- 1. gym 环境能否创建并 step（旧版 4 返回值 API） ---
def _gym_rollout():
    import gym
    env = gym.make("CartPole-v0")
    obs = env.reset()
    assert len(obs) == 4, "reset 返回值异常"
    total = 0.0
    for _ in range(10):
        obs, r, done, info = env.step(env.action_space.sample())
        total += r
    env.close()
    return f"CartPole-v0 跑通 10 步, 累计奖励={total}"


check("gym 环境", _gym_rollout)


# --- 2. parl Model + DQN 算法能否构建并前向 ---
def _parl_dqn():
    import paddle
    import paddle.nn as nn
    import parl

    class Model(parl.Model):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, act_dim)
            )

        def forward(self, obs):
            return self.fc(obs)

        def value(self, obs):
            return self.forward(obs)

    model = Model(4, 2)
    alg = parl.algorithms.DQN(model, gamma=0.99, lr=1e-3)
    act = alg.predict(paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32"))
    return f"DQN 前向输出={act.numpy().ravel()}"


check("parl DQN", _parl_dqn)

print("=" * 56)
print("结论：全部通过 ✅" if ok else "结论：存在失败项 ❌")
sys.exit(0 if ok else 1)
