"""Bernoulli 多臂老虎机，比较 epsilon-greedy 和 UCB。"""
import numpy as np

from rlstudy.common import Run, parser, positive_int


def main():
    cli = parser(__doc__)
    cli.add_argument("--steps", type=positive_int, default=2000)
    args = cli.parse_args()
    probabilities = np.array([0.1, 0.3, 0.5, 0.7, 0.8])
    # 两种策略使用同一潜在奖励表，减少比较时的奖励噪声差异。
    rewards = np.random.default_rng(args.seed).random((args.steps, 5)) < probabilities
    run = Run("bandit", args, ["algo", "step", "reward", "pseudo_regret"])
    try:
        for algorithm in ("epsilon_greedy", "ucb"):
            rng = np.random.default_rng(args.seed)
            counts = np.zeros(5)
            estimates = np.zeros(5)
            regret = 0.0
            for step in range(args.steps):
                if algorithm == "ucb":
                    if step < 5:
                        action = step
                    else:
                        bonus = np.sqrt(2 * np.log(step + 1) / counts)
                        action = int(np.argmax(estimates + bonus))
                elif rng.random() < 0.1:
                    action = int(rng.integers(5))
                else:
                    best = np.flatnonzero(estimates == estimates.max())
                    action = int(rng.choice(best))
                reward = float(rewards[step, action])
                counts[action] += 1
                estimates[action] += (reward - estimates[action]) / counts[action]
                regret += probabilities.max() - probabilities[action]
                run.log(algo=algorithm, step=step + 1, reward=reward, pseudo_regret=regret)
            print(algorithm, "累计伪遗憾", round(regret, 2), "估计", estimates.round(2))
    finally:
        run.close()


if __name__ == "__main__":
    main()
