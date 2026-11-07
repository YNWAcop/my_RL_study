"""DQN / Double DQN，CartPole 低维状态。"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from rlstudy.common import Run, parser, positive_int, seed_everything
from rlstudy.deep import ReplayBuffer, evaluate, mlp, tensor


def main():
    cli = parser(__doc__)
    cli.add_argument("--algo", choices=["dqn", "double_dqn"], default="dqn")
    cli.add_argument("--steps", type=positive_int, default=60000)
    args = cli.parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    online = mlp(4, 2)
    target = deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    buffer = ReplayBuffer(20000)
    run = Run("dqn", args, ["episode", "step", "return", "length", "loss"])
    state, _ = env.reset(seed=args.seed)
    episode = length = 0
    score = loss_value = 0.0
    try:
        for step in range(1, args.steps + 1):
            epsilon = max(0.05, 1 - 0.95 * step / 20000)
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(online(tensor(state)).argmax())
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.add(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
            length += 1
            if len(buffer) >= 1000:
                states, actions, rewards, next_states, terminals = buffer.sample(64)
                with torch.no_grad():
                    if args.algo == "double_dqn":
                        next_actions = online(next_states).argmax(dim=1, keepdim=True)
                        next_values = target(next_states).gather(1, next_actions).squeeze(1)
                    else:
                        next_values = target(next_states).max(dim=1).values
                    # 时间上限允许自举，真正终止时屏蔽下一状态价值。
                    targets = rewards + 0.99 * (1 - terminals) * next_values
                predictions = online(states).gather(1, actions.long().view(-1, 1)).squeeze(1)
                loss = F.smooth_l1_loss(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                loss_value = loss.item()
            if step % 500 == 0:
                target.load_state_dict(online.state_dict())
            if terminated or truncated:
                episode += 1
                run.log(episode=episode, step=step, **{"return": score}, length=length, loss=loss_value)
                state, _ = env.reset()
                score = 0.0
                length = 0
        torch.save(online.state_dict(), run.path / "model.pt")
        evaluate("CartPole-v1", lambda state: int(online(state).argmax()), args.seed, run.path)
    finally:
        env.close()
        run.close()


if __name__ == "__main__":
    main()
