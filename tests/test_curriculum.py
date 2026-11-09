"""学习路线补充代码的数值边界与调度合法性。"""
import importlib
import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch


mc = importlib.import_module("lessons.02_mc_prediction")
scheduling = importlib.import_module("lessons.09_scheduling")


def test_first_visit_uses_first_occurrence_return():
    np.testing.assert_allclose(mc.discounted_returns([1, 2, 3], 0.9), [5.23, 4.7, 3])
    values = np.zeros(7)
    counts = np.zeros(7, dtype=int)
    mc.first_visit_update(values, counts, [(1, 1, 2), (2, 2, 1), (1, 3, 6)])
    assert counts[1] == counts[2] == 1
    assert values[1] == 6
    assert values[2] == 5


def test_schedule_obeys_precedence_resources_and_reward():
    env = scheduling.DispatchEnv()
    observation, info = env.reset()
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(info["action_mask"], [True, True, True, False, False, False])
    with pytest.raises(ValueError):
        env.step(3)
    reward_sum = 0
    for action in [0, 1, 2, 3, 4, 5]:
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        assert env.observation_space.contains(observation)
    assert terminated and not truncated
    assert not info["action_mask"].any()
    assert reward_sum == -info["makespan"] == -12
    for index, operation in enumerate(scheduling.OPERATIONS):
        for predecessor in operation.predecessors:
            assert env.start_times[index] >= env.finish_times[predecessor]
    for machine in (0, 1):
        indices = [index for index, operation in enumerate(scheduling.OPERATIONS) if operation.machine == machine]
        indices.sort(key=lambda index: env.start_times[index])
        assert all(env.finish_times[left] <= env.start_times[right] for left, right in zip(indices, indices[1:]))


def test_gin_mask_and_gradient():
    env = scheduling.DispatchEnv()
    observation, info = env.reset()
    model = scheduling.GINPolicy()
    mask = torch.from_numpy(info["action_mask"])
    distribution = model(torch.from_numpy(observation), mask)
    assert torch.isclose(distribution.probs.sum(), torch.tensor(1.0))
    assert torch.all(distribution.probs[~mask] == 0)
    (-distribution.log_prob(torch.tensor(0))).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with pytest.raises(ValueError):
        model(torch.from_numpy(observation), torch.zeros_like(mask))


def test_curriculum_entries_are_local_and_unique():
    root = Path(__file__).resolve().parents[1]
    steps = json.loads((root / "lessons/study_plan.json").read_text(encoding="utf-8"))["steps"]
    assert [step["id"] for step in steps] == [f"{number:02d}" for number in range(22)]
    for step in steps:
        entry = step["entry"]
        path = root.joinpath(*entry[1].split(".")).with_suffix(".py") if entry[0] == "-m" else root / entry[0]
        assert path.is_file()
        assert all((root / reading).exists() for reading in step["read"])
    from scripts.study import get_step
    assert get_step(steps, "01 张量、网络与反向传播", argparse.ArgumentParser())["id"] == "01"
