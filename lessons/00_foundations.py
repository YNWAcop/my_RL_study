"""从数组、张量和环境交互，走到回报与 PPO 的几个手算例子。"""
import importlib

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from rlstudy.common import parser, seed_everything
from rlstudy.deep import mlp


def numpy_demo(seed):
    rng = np.random.default_rng(seed)
    q_values = np.array([[1.0, 4.0], [3.0, 2.0], [0.5, 0.8]])
    actions = np.array([1, 0, 1])
    selected = q_values[np.arange(3), actions]
    print("Q 表的形状", q_values.shape)
    print("每行最大值所在的动作", q_values.argmax(axis=1))
    print("按实际动作取值", selected)
    print("三次奖励样本", rng.binomial(1, 0.7, size=3))
    estimate = 0.0
    for count, reward in enumerate([1.0, 0.0, 1.0], start=1):
        estimate += (reward - estimate) / count
        print("第", count, "次观察后的均值", estimate)
    np.testing.assert_allclose(selected, [4.0, 3.0, 0.8])


def tensor_demo(seed):
    seed_everything(seed)
    observations = torch.randn(8, 4)
    network = mlp(4, 2, hidden=16)
    actions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    targets = observations[:, 0] * 2 - observations[:, 1]
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    logits = network(observations)
    predictions = logits.gather(1, actions[:, None]).squeeze(1)
    initial_loss = F.mse_loss(predictions, targets).item()
    print("状态、网络输出、选定动作的形状", observations.shape, logits.shape, predictions.shape)
    print("采样动作的 log_prob 形状", Categorical(logits=logits).log_prob(actions).shape)
    for _ in range(100):
        predictions = network(observations).gather(1, actions[:, None]).squeeze(1)
        loss = F.mse_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = F.mse_loss(network(observations).gather(1, actions[:, None]).squeeze(1), targets).item()
    print("拟合前后的 MSE", initial_loss, final_loss)
    print("detach 后需要梯度吗", logits.detach().requires_grad)
    assert final_loss < initial_loss


def environment_demo(seed):
    env = gym.make("CartPole-v1", max_episode_steps=3)
    env.action_space.seed(seed)
    try:
        observation, info = env.reset(seed=seed)
        print("reset 返回状态形状", observation.shape, "info", info)
        for step in range(3):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            print("step", step + 1, "action", action, "reward", reward,
                  "terminated", terminated, "truncated", truncated)
            if terminated or truncated:
                break
        print("TD 目标中，真正终止", 1.0, "时间截断", 1.0 + 0.9 * 3.0)
    finally:
        env.close()


def gae_demo(seed):
    gae = importlib.import_module("lessons.06_ppo").gae
    advantages = gae([1, 100], [2, 0], [3, 0], [False, True], [True, True], gamma=0.9)
    print("时间截断与下一回合之间的 GAE", advantages)
    np.testing.assert_allclose(advantages, [1.7, 100])
    for advantage in [1.0, -1.0]:
        for ratio in [0.6, 1.4]:
            raw = ratio * advantage
            clipped = np.clip(ratio, 0.8, 1.2) * advantage
            print("advantage", advantage, "ratio", ratio,
                  "未裁剪目标", raw, "PPO 目标", min(raw, clipped))


def main():
    cli = parser(__doc__)
    cli.add_argument("--topic", choices=["numpy", "tensor", "environment", "gae"], default="numpy")
    args = cli.parse_args()
    demonstrations = {"numpy": numpy_demo, "tensor": tensor_demo,
                      "environment": environment_demo, "gae": gae_demo}
    demonstrations[args.topic](args.seed)


if __name__ == "__main__":
    main()
