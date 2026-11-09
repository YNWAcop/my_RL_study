"""五状态随机游走，对比首次访问 MC 与 TD(0) 的价值预测。"""
import numpy as np

from rlstudy.common import Run, parser, positive_int


def discounted_returns(rewards, gamma=1.0):
    returns = np.zeros(len(rewards))
    total = 0.0
    for index in reversed(range(len(rewards))):
        total = rewards[index] + gamma * total
        returns[index] = total
    return returns


def first_visit_update(values, counts, trajectory):
    returns = discounted_returns([reward for state, reward, next_state in trajectory])
    visited = set()
    for (state, reward, next_state), target in zip(trajectory, returns):
        if state not in visited:
            visited.add(state)
            counts[state] += 1
            values[state] += (target - values[state]) / counts[state]


def sample_episode(rng):
    state = 3
    trajectory = []
    for _ in range(10000):
        next_state = state + rng.choice([-1, 1])
        reward = float(next_state == 6)
        trajectory.append((state, reward, next_state))
        state = next_state
        if state in (0, 6):
            return trajectory
    raise RuntimeError("随机游走超过安全步数上限")


def main():
    cli = parser(__doc__)
    cli.add_argument("--episodes", type=positive_int, default=2000)
    args = cli.parse_args()
    rng = np.random.default_rng(args.seed)
    mc_values, td_values = np.zeros(7), np.zeros(7)
    counts = np.zeros(7, dtype=int)
    true_values = np.arange(1, 6) / 6
    run = Run("mc_prediction", args, ["algo", "episode", "step", "rmse"])
    total_steps = 0
    print("奖励 [1,2,3]，gamma=0.9 的回报", discounted_returns([1, 2, 3], 0.9))
    try:
        for episode in range(args.episodes):
            trajectory = sample_episode(rng)
            total_steps += len(trajectory)
            first_visit_update(mc_values, counts, trajectory)
            for state, reward, next_state in trajectory:
                # 0 和 6 是终止状态，保留零价值；进入 6 时获得奖励 1。
                target = reward + td_values[next_state]
                td_values[state] += 0.05 * (target - td_values[state])
            for method, values in [("first_visit_mc", mc_values), ("td_zero", td_values)]:
                rmse = np.sqrt(np.mean((values[1:6] - true_values) ** 2))
                run.log(algo=method, episode=episode + 1, step=total_steps, rmse=rmse)
        np.savez(run.path / "values.npz", mc=mc_values, td=td_values, reference=true_values)
        print("解析价值", true_values.round(3))
        print("首次访问 MC", mc_values[1:6].round(3))
        print("TD(0)", td_values[1:6].round(3))
    finally:
        run.close()


if __name__ == "__main__":
    main()
