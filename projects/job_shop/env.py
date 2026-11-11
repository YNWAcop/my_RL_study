"""Job-shop scheduling with fixed routes and one machine per operation."""
import gymnasium as gym
import numpy as np


EXAMPLE = (
    ((0, 3), (1, 3)),
    ((0, 2), (1, 5)),
    ((1, 4), (0, 2)),
)


def random_instance(rng, jobs, machines):
    return tuple(
        tuple((int(machine), int(rng.integers(1, 20))) for machine in rng.permutation(machines))
        for _ in range(jobs)
    )


class JobShop(gym.Env):
    """Each action appends one job's next operation to its machine queue."""

    def __init__(self, jobs, machines, instance=None):
        super().__init__()
        self.jobs, self.machines = jobs, machines
        self.fixed_instance = instance
        self.action_space = gym.spaces.Discrete(jobs)
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(jobs * machines * 5 + machines,), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.instance = self.fixed_instance
        if self.instance is None:
            self.instance = random_instance(self.np_random, self.jobs, self.machines)
        self.horizon = sum(duration for job in self.instance for _, duration in job)
        self.next_op = np.zeros(self.jobs, dtype=int)
        self.machine_free = np.zeros(self.machines, dtype=int)
        self.job_free = np.zeros(self.jobs, dtype=int)
        self.schedule = {}
        return self.observation(), {}

    def action_masks(self):
        return self.next_op < self.machines

    def observation(self):
        features = []
        for job, route in enumerate(self.instance):
            for op, (machine, duration) in enumerate(route):
                start, end = self.schedule.get((job, op), (0, 0))
                features.extend((machine / max(1, self.machines - 1), duration / self.horizon,
                                 float(op < self.next_op[job]), start / self.horizon, end / self.horizon))
        features.extend(self.machine_free / self.horizon)
        return np.asarray(features, dtype=np.float32)

    def step(self, action):
        if not self.action_space.contains(action) or not self.action_masks()[action]:
            raise ValueError("Action must select an unfinished job.")
        job, op = int(action), int(self.next_op[action])
        machine, duration = self.instance[job][op]
        previous = int(self.job_free.max())
        start = int(max(self.job_free[job], self.machine_free[machine]))
        end = start + duration
        self.schedule[job, op] = (start, end)
        self.next_op[job] += 1
        self.job_free[job] = self.machine_free[machine] = end
        makespan = int(self.job_free.max())
        done = not self.action_masks().any()
        return self.observation(), float(previous - makespan), bool(done), False, {"makespan": makespan}


def validate_schedule(instance, schedule):
    """Check completeness, duration, precedence, and machine exclusivity."""
    expected = {(job, op) for job, route in enumerate(instance) for op in range(len(route))}
    if set(schedule) != expected:
        raise ValueError("Schedule has missing or unknown operations.")
    machines = {}
    for job, route in enumerate(instance):
        previous_end = 0
        for op, (machine, duration) in enumerate(route):
            start, end = schedule[job, op]
            if start < previous_end or end - start != duration:
                raise ValueError(f"Invalid duration or precedence at {(job, op)}.")
            previous_end = end
            machines.setdefault(machine, []).append((start, end))
    for machine, intervals in machines.items():
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"Overlapping operations on machine {machine}.")
    return max(end for _, end in schedule.values())
