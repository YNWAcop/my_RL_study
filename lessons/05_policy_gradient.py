"""REINFORCE 与一步 Actor–Critic，逐步对照回报和 TD 误差。"""
import gymnasium as gym
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from rlstudy.common import Run, parser, positive_int, seed_everything
from rlstudy.deep import evaluate, mlp, tensor


def main():
    cli = parser(__doc__)
    cli.add_argument("--algo", choices=["reinforce", "actor_critic"], default="reinforce")
    cli.add_argument("--episodes", type=positive_int, default=1000)
    args = cli.parse_args()
    seed_everything(args.seed)
    env = gym.make("CartPole-v1")
    actor, critic = mlp(4, 2), mlp(4, 1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    run = Run("policy_gradient", args, ["episode", "step", "return", "length"])
    total_steps = 0
    try:
        for episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed + episode)
            log_probabilities, rewards = [], []
            length = 0
            done = False
            while not done:
                state_tensor = tensor(state)
                distribution = Categorical(logits=actor(state_tensor))
                action = distribution.sample()
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                log_probability = distribution.log_prob(action)
                if args.algo == "actor_critic":
                    value = critic(state_tensor).squeeze()
                    with torch.no_grad():
                        target = reward + 0.99 * (1 - terminated) * critic(tensor(next_state)).squeeze()
                    advantage = target - value
                    actor_loss = -log_probability * advantage.detach() - 0.01 * distribution.entropy()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.zero_grad()
                    F.mse_loss(value, target).backward()
                    critic_optimizer.step()
                else:
                    log_probabilities.append(log_probability)
                rewards.append(reward)
                state = next_state
                length += 1
                total_steps += 1
                done = terminated or truncated
            if args.algo == "reinforce":
                # 此处明确优化被 CartPole 的时间上限截断的有限回合目标。
                returns = []
                discounted_return = 0.0
                for reward in reversed(rewards):
                    discounted_return = reward + 0.99 * discounted_return
                    returns.append(discounted_return)
                weights = tensor(list(reversed(returns)))
                time_weights = 0.99 ** torch.arange(length, dtype=torch.float32)
                # 固定缩放保持不同回合的权重一致，不除以随机的回合长度。
                loss = -(torch.stack(log_probabilities) * weights * time_weights).sum() / 100.0
                actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
            run.log(episode=episode + 1, step=total_steps, **{"return": sum(rewards)}, length=length)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, run.path / "model.pt")
        evaluate("CartPole-v1", lambda state: int(actor(state).argmax()), args.seed, run.path)
    finally:
        env.close()
        run.close()


if __name__ == "__main__":
    main()
