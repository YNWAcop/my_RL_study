"""六道工序、两台固定机器，学习优先级调度、动作掩码和 GIN 前向。"""
from dataclasses import dataclass
from itertools import permutations

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rlstudy.common import Run, parser, seed_everything


@dataclass(frozen=True)
class Operation:
    name: str
    machine: int
    duration: float
    predecessors: tuple


OPERATIONS = (
    Operation("A1", 0, 3, ()), Operation("B1", 0, 2, ()), Operation("C1", 1, 4, ()),
    Operation("A2", 1, 3, (0,)), Operation("B2", 1, 5, (1,)), Operation("C2", 0, 2, (2,)),
)


class DispatchEnv(gym.Env):
    """动作逐项构造计划，前置工序已排入计划时，其后继才可被选择。"""
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(OPERATIONS))
        self.observation_space = gym.spaces.Box(0, np.inf, shape=(len(OPERATIONS), 5), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scheduled = np.zeros(len(OPERATIONS), dtype=bool)
        self.start_times = np.zeros(len(OPERATIONS))
        self.finish_times = np.zeros(len(OPERATIONS))
        self.machine_free = np.zeros(2)
        return self.observation(), self.info()

    def action_mask(self):
        return np.array([not self.scheduled[index] and all(self.scheduled[parent] for parent in operation.predecessors)
                         for index, operation in enumerate(OPERATIONS)], dtype=bool)

    def observation(self):
        return np.array([[operation.duration / 5, operation.machine, float(self.scheduled[index]),
                          self.start_times[index] / 20, self.finish_times[index] / 20]
                         for index, operation in enumerate(OPERATIONS)], dtype=np.float32)

    def info(self):
        return {"action_mask": self.action_mask(), "makespan": float(self.finish_times.max())}

    def step(self, action):
        if not self.action_space.contains(action) or not self.action_mask()[action]:
            raise ValueError("动作必须是尚未安排且所有前置工序均已安排的工序")
        operation = OPERATIONS[action]
        previous_makespan = self.finish_times.max()
        ready_at = max((self.finish_times[parent] for parent in operation.predecessors), default=0)
        start = max(ready_at, self.machine_free[operation.machine])
        finish = start + operation.duration
        self.start_times[action], self.finish_times[action] = start, finish
        self.machine_free[operation.machine] = finish
        self.scheduled[action] = True
        reward = previous_makespan - self.finish_times.max()
        terminated = bool(self.scheduled.all())
        return self.observation(), float(reward), terminated, False, self.info()


def graph_adjacency():
    # 编码器双向聚合依赖边；合法动作仍由有向前置关系判断。
    adjacency = torch.zeros(len(OPERATIONS), len(OPERATIONS))
    for index, operation in enumerate(OPERATIONS):
        for parent in operation.predecessors:
            adjacency[index, parent] = adjacency[parent, index] = 1
    return adjacency


class GINPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(5, 32)
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32)) for _ in range(2)])
        self.epsilon = nn.Parameter(torch.zeros(2))
        self.score = nn.Linear(64, 1)
        self.register_buffer("adjacency", graph_adjacency())

    def forward(self, observations, mask):
        if not mask.any():
            raise ValueError("所有动作都被屏蔽；终止状态应结束回合")
        nodes = torch.tanh(self.input_layer(observations))
        for index, update in enumerate(self.updates):
            nodes = torch.tanh(update((1 + self.epsilon[index]) * nodes + self.adjacency @ nodes))
        graph = nodes.mean(dim=0).expand_as(nodes)
        logits = self.score(torch.cat([nodes, graph], dim=1)).squeeze(1)
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def make_schedule(rule, network=None):
    env = DispatchEnv()
    observation, info = env.reset()
    order = []
    done = False
    while not done:
        available = np.flatnonzero(info["action_mask"]).tolist()
        if rule == "spt":
            action = min(available, key=lambda index: (OPERATIONS[index].duration, index))
        elif rule == "lpt":
            action = min(available, key=lambda index: (-OPERATIONS[index].duration, index))
        elif rule == "gin":
            with torch.no_grad():
                distribution = network(torch.from_numpy(observation), torch.from_numpy(info["action_mask"]))
                action = int(distribution.probs.argmax())
        else:
            action = available[0]
        observation, reward, done, truncated, info = env.step(action)
        order.append(action)
    return env, order


def exhaustive_best():
    best = np.inf
    best_order = None
    for order in permutations(range(len(OPERATIONS))):
        positions = {operation: index for index, operation in enumerate(order)}
        valid = all(positions[parent] < positions[index] for index, operation in enumerate(OPERATIONS)
                    for parent in operation.predecessors)
        if valid:
            env = DispatchEnv()
            env.reset()
            for action in order:
                env.step(action)
            makespan = env.info()["makespan"]
            if makespan < best:
                best, best_order = makespan, order
    return best, best_order


def main():
    cli = parser(__doc__)
    cli.add_argument("--stage", choices=["rules", "gin"], default="rules")
    args = cli.parse_args()
    seed_everything(args.seed)
    run = Run("scheduling", args, ["rule", "makespan", "order"])
    schedules = []
    try:
        for rule in ["fifo", "spt", "lpt"]:
            env, order = make_schedule(rule)
            schedules.append((rule, env))
            run.log(rule=rule, makespan=env.info()["makespan"], order=" ".join(map(str, order)))
            print(rule, "makespan", env.info()["makespan"], "顺序", order)
        best, order = exhaustive_best()
        print("穷举最优 makespan", best, "顺序", order)
        run.log(rule="exhaustive", makespan=best, order=" ".join(map(str, order)))
        if args.stage == "gin":
            network = GINPolicy()
            env = DispatchEnv()
            observation, info = env.reset()
            distribution = network(torch.from_numpy(observation), torch.from_numpy(info["action_mask"]))
            print("初始合法动作", info["action_mask"], "GIN 概率", distribution.probs.detach().numpy())
            env, order = make_schedule("gin", network)
            print("随机初始化 GIN 的计划", order, "makespan", env.info()["makespan"])
            run.log(rule="untrained_gin", makespan=env.info()["makespan"], order=" ".join(map(str, order)))
        figure, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
        for axis, (rule, env) in zip(axes, schedules):
            for index, operation in enumerate(OPERATIONS):
                axis.barh(operation.machine, operation.duration, left=env.start_times[index], height=0.5)
                axis.text(env.start_times[index] + operation.duration / 2, operation.machine, operation.name, ha="center", va="center")
            axis.set(title=rule.upper() + " / Cmax=" + str(env.info()["makespan"]), xlabel="Time", yticks=[0, 1], yticklabels=["M0", "M1"])
        figure.tight_layout()
        figure.savefig(run.path / "gantt.png", dpi=150)
        plt.close(figure)
    finally:
        run.close()


if __name__ == "__main__":
    main()
