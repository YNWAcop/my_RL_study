"""针对容易静默出错的数学目标与回合边界做检查。"""
import importlib

import gymnasium as gym
import numpy as np
import torch

td = importlib.import_module("lessons.03_td_control")
ppo = importlib.import_module("lessons.06_ppo")
continuous = importlib.import_module("lessons.07_continuous")
dp = importlib.import_module("lessons.02_value_iteration")


def test_td_on_policy_and_off_policy_targets():
    next_values = np.array([1.0, 4.0])
    assert td.td_target(1, next_values, 0, False, 0.9, "sarsa") == 1.9
    assert td.td_target(1, next_values, 0, False, 0.9, "q_learning") == 4.6
    assert td.td_target(1, next_values, 0, True, 0.9, "q_learning") == 1


def test_gae_bootstraps_timeout_without_crossing_reset():
    advantages = ppo.gae([1, 100], [2, 0], [3, 0], [False, True], [True, True], gamma=0.9)
    np.testing.assert_allclose(advantages, [1.7, 100])


def test_gae_true_terminal_masks_future_value():
    np.testing.assert_allclose(ppo.gae([1], [2], [300], [True], [True]), [-1])


def test_value_iteration_reaches_deterministic_goal():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        values, policy, _, residual = dp.solve(env)
        assert residual < 1e-10
        assert np.isclose(values[0], 0.99 ** 5)
        state, _ = env.reset(seed=0)
        reward = 0
        for _ in range(20):
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            if terminated or truncated:
                break
        assert reward == 1
    finally:
        env.close()


def test_sac_action_density_and_gradient():
    torch.manual_seed(0)
    actor = continuous.Actor(stochastic=True)
    action, log_probability = actor(torch.zeros(16, 3))
    assert action.shape == (16, 1)
    assert log_probability.shape == (16,)
    assert action.abs().max() <= 2
    assert torch.isfinite(log_probability).all()
    loss = (log_probability - action.squeeze(1)).mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in actor.parameters())
