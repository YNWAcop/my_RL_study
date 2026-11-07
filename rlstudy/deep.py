"""深度算法共用的网络、回放池和评估，不包含算法更新。"""
import json
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn


def mlp(inputs, outputs, hidden=64):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


def tensor(array):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated):
        self.data.append((np.array(state, copy=True), np.array(action, copy=True),
                          reward, np.array(next_state, copy=True), terminated))

    def sample(self, batch_size):
        return tuple(tensor(column) for column in zip(*random.sample(self.data, batch_size)))

    def __len__(self):
        return len(self.data)


def soft_update(target, source, tau=0.005):
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(source_parameter, tau)


def evaluate(env_id, action_fn, seed, output, episodes=10):
    env = gym.make(env_id)
    returns = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + 100000 + episode)
            score = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    action = action_fn(tensor(state).unsqueeze(0))
                state, reward, terminated, truncated, _ = env.step(action)
                score += reward
                done = terminated or truncated
            returns.append(score)
    finally:
        env.close()
    result = {"returns": returns, "mean": float(np.mean(returns)),
              "std": float(np.std(returns)), "episodes": episodes,
              "first_seed": seed + 100000, "policy": "deterministic"}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("独立评估", result["mean"], "+/-", result["std"])
