"""在 Pendulum 上比较 DDPG / TD3 / 固定温度 SAC。"""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from rlstudy.common import Run, parser, positive_int, seed_everything
from rlstudy.deep import ReplayBuffer, evaluate, mlp, soft_update, tensor


class Actor(nn.Module):
    def __init__(self, stochastic):
        super().__init__()
        self.stochastic = stochastic
        self.network = mlp(3, 2 if stochastic else 1, hidden=128)

    def forward(self, states, deterministic=False):
        output = self.network(states)
        if not self.stochastic:
            return 2.0 * torch.tanh(output), None
        mean, log_std = output.chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5, 2).exp())
        latent = mean if deterministic else distribution.rsample()
        unit_action = torch.tanh(latent)
        action = 2.0 * unit_action
        # tanh 和 [-1,1] 到 [-2,2] 的缩放都需要密度变换。
        log_probability = (distribution.log_prob(latent)
                           - torch.log(2.0 * (1 - unit_action.square()) + 1e-6)).sum(-1)
        return action, log_probability


def q_value(network, states, actions):
    return network(torch.cat([states, actions], dim=-1)).squeeze(-1)


def main():
    cli = parser(__doc__)
    cli.add_argument("--algo", choices=["ddpg", "td3", "sac"], default="td3")
    cli.add_argument("--steps", type=positive_int, default=60000)
    args = cli.parse_args()
    seed_everything(args.seed)
    env = gym.make("Pendulum-v1")
    env.action_space.seed(args.seed)
    actor = Actor(stochastic=args.algo == "sac")
    target_actor = deepcopy(actor)
    critics = nn.ModuleList([mlp(4, 1, 128) for _ in range(1 if args.algo == "ddpg" else 2)])
    target_critics = deepcopy(critics)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=0.0003)
    buffer = ReplayBuffer()
    run = Run("continuous", args, ["episode", "step", "return", "length", "critic_loss", "actor_loss"])
    state, _ = env.reset(seed=args.seed)
    score = critic_loss_value = actor_loss_value = 0.0
    episode = length = updates = 0
    try:
        for step in range(1, args.steps + 1):
            if step <= 1000:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = actor(tensor(state).unsqueeze(0))
                    action = action.squeeze(0).numpy()
                if args.algo != "sac":
                    action = np.clip(action + np.random.normal(0, 0.2, size=action.shape), -2, 2)
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.add(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
            length += 1
            if len(buffer) >= 1000:
                updates += 1
                states, actions, rewards, next_states, terminals = buffer.sample(128)
                with torch.no_grad():
                    if args.algo == "sac":
                        next_actions, next_log_probability = actor(next_states)
                    else:
                        next_actions, _ = target_actor(next_states)
                    if args.algo == "td3":
                        noise = (torch.randn_like(next_actions) * 0.4).clamp(-1, 1)
                        next_actions = (next_actions + noise).clamp(-2, 2)
                    next_q = torch.stack([q_value(net, next_states, next_actions)
                                          for net in target_critics]).min(dim=0).values
                    if args.algo == "sac":
                        next_q -= 0.2 * next_log_probability
                    targets = rewards + 0.99 * (1 - terminals) * next_q
                critic_loss = sum(F.mse_loss(q_value(net, states, actions), targets) for net in critics)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                critic_loss_value = critic_loss.item()
                if args.algo != "td3" or updates % 2 == 0:
                    for parameter in critics.parameters():
                        parameter.requires_grad_(False)
                    predicted_actions, log_probability = actor(states)
                    if args.algo == "sac":
                        predicted_q = torch.stack([q_value(net, states, predicted_actions)
                                                   for net in critics]).min(dim=0).values
                        actor_loss = (0.2 * log_probability - predicted_q).mean()
                    else:
                        actor_loss = -q_value(critics[0], states, predicted_actions).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    for parameter in critics.parameters():
                        parameter.requires_grad_(True)
                    actor_loss_value = actor_loss.item()
                    soft_update(target_actor, actor)
                    soft_update(target_critics, critics)
            if terminated or truncated:
                episode += 1
                run.log(episode=episode, step=step, **{"return": score}, length=length,
                        critic_loss=critic_loss_value, actor_loss=actor_loss_value)
                state, _ = env.reset()
                score = 0.0
                length = 0
        torch.save({"actor": actor.state_dict(), "critics": critics.state_dict()}, run.path / "model.pt")
        evaluate("Pendulum-v1", lambda state: actor(state, deterministic=True)[0].squeeze(0).numpy(),
                 args.seed, run.path)
    finally:
        env.close()
        run.close()


if __name__ == "__main__":
    main()
