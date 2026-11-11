"""Train MaskablePPO or benchmark dispatching rules on job-shop instances."""
import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import EXAMPLE, JobShop, random_instance, validate_schedule


def dispatch(instance, rule, model=None):
    env = JobShop(len(instance), len(instance[0]), instance=instance)
    observation, _ = env.reset()
    done = False
    while not done:
        mask = env.action_masks()
        if model is not None:
            action = int(model.predict(observation, action_masks=mask, deterministic=True)[0])
        else:
            eligible = np.flatnonzero(mask)
            sign = 1 if rule == "spt" else -1
            action = min(eligible, key=lambda job: (sign * instance[job][env.next_op[job]][1], job))
        observation, _, done, _, _ = env.step(action)
    return env.schedule


def cp_sat(instance, seconds, cp_model):
    model = cp_model.CpModel()
    horizon = sum(duration for route in instance for _, duration in route)
    starts, ends, by_machine = {}, {}, {}
    for job, route in enumerate(instance):
        for op, (machine, duration) in enumerate(route):
            key = (job, op)
            starts[key] = model.NewIntVar(0, horizon, f"start_{job}_{op}")
            ends[key] = model.NewIntVar(0, horizon, f"end_{job}_{op}")
            interval = model.NewIntervalVar(starts[key], duration, ends[key], f"op_{job}_{op}")
            by_machine.setdefault(machine, []).append(interval)
            if op:
                model.Add(starts[key] >= ends[job, op - 1])
    for intervals in by_machine.values():
        model.AddNoOverlap(intervals)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(ends.values()))
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    schedule = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {key: (solver.Value(starts[key]), solver.Value(ends[key])) for key in starts}
    return schedule, solver.StatusName(status)


def plot_schedule(instance, schedule, output):
    figure, axis = plt.subplots(figsize=(8, 3))
    for (job, op), (start, end) in schedule.items():
        machine, _ = instance[job][op]
        axis.barh(machine, end - start, left=start, height=0.6, color=f"C{job % 10}")
        axis.text((start + end) / 2, machine, f"J{job + 1}.{op + 1}", ha="center", va="center")
    axis.set(xlabel="Time", ylabel="Machine", yticks=range(len(instance[0])),
             title=f"Makespan = {validate_schedule(instance, schedule)}")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def train(args):
    from sb3_contrib import MaskablePPO

    args.output.mkdir(parents=True, exist_ok=False)
    env = JobShop(args.jobs, args.machines)
    try:
        model = MaskablePPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=4,
                            gamma=1.0, seed=args.seed, device="cpu", verbose=1)
        model.learn(total_timesteps=args.steps)
        model.save(args.output / "model.zip")
        config = {"jobs": args.jobs, "machines": args.machines, "seed": args.seed,
                  "requested_steps": args.steps, "actual_steps": model.num_timesteps}
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    finally:
        env.close()
    print(args.output / "model.zip")


def evaluate(args):
    if args.instance_file:
        instance = json.loads(args.instance_file.read_text(encoding="utf-8"))
        if not isinstance(instance, list) or not instance or not isinstance(instance[0], list) or not instance[0]:
            raise ValueError("Expected a nonempty list of job routes.")
        jobs, machines = len(instance), len(instance[0])
        for route in instance:
            if not isinstance(route, list) or len(route) != machines:
                raise ValueError("All job routes must have the same number of operations.")
            for operation in route:
                if (not isinstance(operation, list) or len(operation) != 2
                        or any(type(value) is not int for value in operation)
                        or not 0 <= operation[0] < machines or operation[1] <= 0):
                    raise ValueError("Each operation must be [machine ID, positive integer duration].")
            if {machine for machine, _ in route} != set(range(machines)):
                raise ValueError("Each route must visit each machine once.")
        instances = [instance]
    elif args.example:
        jobs, machines = 3, 2
        instances = [EXAMPLE]
    else:
        jobs, machines = args.jobs, args.machines
        instances = [random_instance(np.random.default_rng(args.seed + index), jobs, machines)
                     for index in range(args.instances)]
    model = None
    if args.model:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(args.model, device="cpu")
        env = JobShop(jobs, machines)
        if env.observation_space != model.observation_space or env.action_space != model.action_space:
            raise ValueError("Evaluation jobs and machines must match the model's training dimensions.")
    if args.cp_sat:
        from ortools.sat.python import cp_model

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "instances.json").write_text(json.dumps(instances, indent=2), encoding="utf-8")
    rows = []
    methods = ["spt", "lpt"] + (["ppo"] if model is not None else []) + (["cp_sat"] if args.cp_sat else [])
    for index, instance in enumerate(instances):
        for method in methods:
            started = time.perf_counter()
            if method == "cp_sat":
                schedule, status = cp_sat(instance, args.time_limit, cp_model)
            else:
                schedule = dispatch(instance, method, model if method == "ppo" else None)
                status = "DISPATCHED"
            elapsed_ms = (time.perf_counter() - started) * 1000
            makespan = validate_schedule(instance, schedule) if schedule is not None else None
            rows.append({"instance": index, "method": method, "status": status,
                         "makespan": makespan, "solve_ms": round(elapsed_ms, 3)})
            if index == 0 and schedule is not None:
                (args.output / f"{method}_schedule.json").write_text(
                    json.dumps([{"job": job, "op": op, "start": start, "end": end}
                                for (job, op), (start, end) in sorted(schedule.items())], indent=2),
                    encoding="utf-8",
                )
                plot_schedule(instance, schedule, args.output / f"{method}_gantt.png")
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in methods:
        values = [r["makespan"] for r in rows if r["method"] == method and r["makespan"] is not None]
        mean = f"{np.mean(values):.2f}" if values else "n/a"
        print(f"{method}: schedules={len(values)}/{len(instances)}, mean makespan={mean}")
    print(args.output / "results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int, default=50_000)
    training.add_argument("--seed", type=int, default=0)
    testing = commands.add_parser("evaluate")
    testing.add_argument("--seed", type=int, default=10_000)
    testing.add_argument("--instances", type=int, default=20)
    instance_source = testing.add_mutually_exclusive_group()
    instance_source.add_argument("--example", action="store_true")
    instance_source.add_argument("--instance-file", type=Path)
    testing.add_argument("--model", type=Path)
    testing.add_argument("--cp-sat", action="store_true")
    testing.add_argument("--time-limit", type=float, default=2.0)
    for command in (training, testing):
        command.add_argument("--jobs", type=int, default=3)
        command.add_argument("--machines", type=int, default=2)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.jobs, args.machines) < 1 or args.seed < 0:
        parser.error("jobs and machines must be positive, seed must be nonnegative")
    if args.command == "train":
        if args.steps < 1:
            parser.error("steps must be positive")
        train(args)
    else:
        if args.instances < 1 or not np.isfinite(args.time_limit) or args.time_limit <= 0:
            parser.error("instances and time limit must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
