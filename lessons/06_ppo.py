"""单环境 PPO-Clip，显式区分 GAE 的 bootstrap mask 和 trace mask。"""
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from rlstudy.common import Run, parser, positive_int, seed_everything
from rlstudy.deep import evaluate, mlp, tensor


def gae(rewards, values, next_values, terminated, boundaries, gamma=0.99, lam=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * (1 - terminated[index]) * next_values[index] - values[index]
        # 真正终止与时间截断均切断轨迹递推，防止串到下一回合。
        last = delta + gamma * lam * (1 - boundaries[index]) * last
        advantages[index] = last
    return advantages


def main():
    cli = parser(__doc__)
    cli.add_argument("--steps", type=positive_int, default=100000)
    args = cli.parse_args()
    seed_everything(args.seed)
    env = gym.make("CartPole-v1")
    actor, critic = mlp(4, 2), mlp(4, 1)
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.0003)
    run = Run("ppo", args, ["episode", "step", "return", "length"])
    state, _ = env.reset(seed=args.seed)
    step = episode = length = 0
    score = 0.0
    try:
        while step < args.steps:
            states, actions, old_log_probs, rewards = [], [], [], []
            values, next_values, terminals, boundaries = [], [], [], []
            for _ in range(min(1024, args.steps - step)):
                with torch.no_grad():
                    distribution = Categorical(logits=actor(tensor(state)))
                    action = distribution.sample()
                    value = critic(tensor(state)).item()
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                with torch.no_grad():
                    next_value = critic(tensor(next_state)).item()
                states.append(state)
                actions.append(action.item())
                old_log_probs.append(distribution.log_prob(action).item())
                rewards.append(reward)
                values.append(value)
                next_values.append(next_value)
                terminals.append(terminated)
                boundaries.append(terminated or truncated)
                step += 1
                length += 1
                score += reward
                state = next_state
                if terminated or truncated:
                    episode += 1
                    run.log(episode=episode, step=step, **{"return": score}, length=length)
                    state, _ = env.reset()
                    score = 0.0
                    length = 0
            advantages = tensor(gae(rewards, values, next_values, terminals, boundaries))
            returns = advantages + tensor(values)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            observations = tensor(states)
            chosen_actions = torch.tensor(actions)
            old_log_probs = tensor(old_log_probs)
            for _ in range(10):
                for indices in torch.randperm(len(states)).split(64):
                    distribution = Categorical(logits=actor(observations[indices]))
                    ratio = (distribution.log_prob(chosen_actions[indices]) - old_log_probs[indices]).exp()
                    raw = ratio * advantages[indices]
                    clipped = ratio.clamp(0.8, 1.2) * advantages[indices]
                    actor_loss = -torch.minimum(raw, clipped).mean()
                    value_loss = F.mse_loss(critic(observations[indices]).squeeze(1), returns[indices])
                    loss = actor_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 0.5)
                    optimizer.step()
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, run.path / "model.pt")
        evaluate("CartPole-v1", lambda state: int(actor(state).argmax()), args.seed, run.path)
    finally:
        env.close()
        run.close()


if __name__ == "__main__":
    main()
