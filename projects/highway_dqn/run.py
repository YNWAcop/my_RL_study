"""Train or evaluate a DQN policy in HighwayEnv's simulated highway."""
import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # Registers the environments.
import numpy as np
from stable_baselines3 import DQN


def make_env(density, render=False):
    return gym.make(
        "highway-fast-v0",
        render_mode="human" if render else None,
        config={
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "vehicles_density": density,
        },
    )


def keep_lane(env):
    """A simple speed controller that brakes for a close lead vehicle."""
    highway = env.unwrapped
    ego = highway.vehicle
    front, _ = highway.road.neighbour_vehicles(ego)
    distance = ego.lane_distance_to(front) if front is not None else float("inf")
    actions = highway.action_type.actions_indexes
    if distance < max(12.0, ego.speed * 1.2):
        return actions["SLOWER"]
    return actions["FASTER"] if ego.speed < 25.0 else actions["IDLE"]


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    env = make_env(args.density)
    try:
        model = DQN(
            "MlpPolicy", env, seed=args.seed, device="cpu",
            learning_rate=5e-4, buffer_size=50_000, learning_starts=512,
            batch_size=64, train_freq=4, target_update_interval=500,
            exploration_fraction=0.35, verbose=1,
        )
        model.learn(total_timesteps=args.steps)
        model.save(args.output / "model.zip")
        config = {"environment": env.spec.id, "density": args.density,
                  "seed": args.seed, "requested_steps": args.steps,
                  "actual_steps": model.num_timesteps}
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    finally:
        env.close()
    print(args.output / "model.zip")


def rollout(env, model, seed):
    observation, _ = env.reset(seed=seed)
    score, steps, speed_sum, lane_changes = 0.0, 0, 0.0, 0
    previous_lane = env.unwrapped.vehicle.lane_index
    while True:
        action = keep_lane(env) if model is None else int(model.predict(observation, deterministic=True)[0])
        observation, reward, terminated, truncated, info = env.step(action)
        lane = env.unwrapped.vehicle.lane_index
        lane_changes += lane != previous_lane
        previous_lane = lane
        score += reward
        steps += 1
        speed_sum += info["speed"]
        if terminated or truncated:
            return {"seed": seed, "return": score, "steps": steps, "crashed": int(info["crashed"]),
                    "mean_speed_mps": speed_sum / steps, "lane_changes": lane_changes}


def evaluate(args):
    model = DQN.load(args.model, device="cpu") if args.model is not None else None
    policies = {"keep_lane": None} if model is None else {"keep_lane": None, "dqn": model}
    if args.render and model is not None:
        policies = {"dqn": model}
    rows = []
    for density in args.densities:
        for name, policy in policies.items():
            env = make_env(density, args.render)
            try:
                group = [rollout(env, policy, args.seed + i) for i in range(args.episodes)]
            finally:
                env.close()
            rows.extend({"policy": name, "density": density, **row} for row in group)
            crash_rate = np.mean([row["crashed"] for row in group])
            speed = np.mean([row["mean_speed_mps"] for row in group])
            print(f"{name} density={density:g} collision_rate={crash_rate:.1%} mean_speed={speed:.2f} m/s")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(args.csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--density", type=float, default=1.0)
    training.add_argument("--steps", type=int, default=60_000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--output", type=Path, required=True)
    testing = commands.add_parser("evaluate")
    testing.add_argument("--model", type=Path)
    testing.add_argument("--densities", type=float, nargs="+", default=[0.7, 1.0, 1.5])
    testing.add_argument("--episodes", type=int, default=20)
    testing.add_argument("--seed", type=int, default=10_000)
    testing.add_argument("--render", action="store_true")
    testing.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        if args.steps < 1 or not np.isfinite(args.density) or args.density <= 0:
            parser.error("steps and density must be positive")
        train(args)
    else:
        if args.episodes < 1 or any(not np.isfinite(d) or d <= 0 for d in args.densities):
            parser.error("episodes and densities must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
