"""在相同的 CliffWalking 上比较 SARSA 和 Q-learning。"""
import gymnasium as gym
import numpy as np

from rlstudy.common import Run, parser, positive_int


def epsilon_action(q, state, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    best = np.flatnonzero(q[state] == q[state].max())
    return int(rng.choice(best))


def td_target(reward, next_values, next_action, terminated, gamma, algorithm):
    if terminated:
        return reward
    if algorithm == "sarsa":
        return reward + gamma * next_values[next_action]
    return reward + gamma * next_values.max()


def main():
    cli = parser(__doc__)
    cli.add_argument("--algo", choices=["q_learning", "sarsa"], default="q_learning")
    cli.add_argument("--episodes", type=positive_int, default=1000)
    args = cli.parse_args()
    env = gym.make("CliffWalking-v0", max_episode_steps=1000)
    rng = np.random.default_rng(args.seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    run = Run("td_control", args, ["episode", "step", "return", "length"])
    total_steps = 0
    try:
        for episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed + episode)
            action = epsilon_action(q, state, rng, 0.1)
            episode_return = 0.0
            done = False
            length = 0
            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = epsilon_action(q, next_state, rng, 0.1)
                target = td_target(reward, q[next_state], next_action, terminated, 1.0, args.algo)
                q[state, action] += 0.5 * (target - q[state, action])
                state, action = next_state, next_action
                episode_return += reward
                length += 1
                total_steps += 1
                done = terminated or truncated
            run.log(episode=episode + 1, step=total_steps, **{"return": episode_return}, length=length)
        np.save(run.path / "q_table.npy", q)
        # 固定贪心策略单独评估，与含探索噪声的训练回报区分。
        state, _ = env.reset(seed=args.seed + 100000)
        greedy_return = 0.0
        path = [int(state)]
        for _ in range(1000):
            state, reward, terminated, truncated, _ = env.step(int(q[state].argmax()))
            greedy_return += reward
            path.append(int(state))
            if terminated or truncated:
                break
        print("贪心评估回报", greedy_return, "路径", path[:60])
    finally:
        env.close()
        run.close()


if __name__ == "__main__":
    main()
