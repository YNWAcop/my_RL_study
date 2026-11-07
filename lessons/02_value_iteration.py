"""已知环境转移模型时，对确定性 FrozenLake 做价值迭代。"""
import gymnasium as gym
import numpy as np


def solve(env, gamma=0.99, tolerance=1e-10):
    values = np.zeros(env.observation_space.n)
    transitions = env.unwrapped.P
    for iteration in range(10000):
        action_values = np.zeros((len(values), env.action_space.n))
        for state in range(len(values)):
            for action in range(env.action_space.n):
                for probability, next_state, reward, terminated in transitions[state][action]:
                    action_values[state, action] += probability * (
                        reward + gamma * (1 - terminated) * values[next_state])
        updated = action_values.max(axis=1)
        residual = np.max(np.abs(updated - values))
        values = updated
        if residual < tolerance:
            return values, action_values.argmax(axis=1), iteration + 1, residual
    raise RuntimeError("价值迭代未收敛")


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        values, policy, iterations, residual = solve(env)
        print("V(s)\n", values.reshape(4, 4).round(3))
        print("策略 0左 1下 2右 3上\n", policy.reshape(4, 4))
        print("迭代次数", iterations, "贝尔曼残差", residual)
    finally:
        env.close()


if __name__ == "__main__":
    main()
