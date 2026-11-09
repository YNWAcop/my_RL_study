"""用 Stable-Baselines3 对照手写 DQN / PPO，练习训练、保存和独立评估。"""
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from rlstudy.common import Run, parser, positive_int, seed_everything
from rlstudy.deep import evaluate


def main():
    cli = parser(__doc__)
    cli.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    cli.add_argument("--steps", type=positive_int, default=60000)
    args = cli.parse_args()
    seed_everything(args.seed)
    run = Run("sb3", args, ["episode", "step", "return", "length"])
    env = Monitor(gym.make("CartPole-v1"), filename=str(run.path / "monitor.csv"))
    try:
        if args.algo == "ppo":
            model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                        n_steps=128, batch_size=64, n_epochs=4)
        else:
            model = DQN("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                        learning_starts=256, buffer_size=10000, batch_size=64,
                        train_freq=4, target_update_interval=250)
        model.learn(total_timesteps=args.steps)
        total_steps = 0
        for episode, (score, length) in enumerate(zip(env.get_episode_rewards(), env.get_episode_lengths()), start=1):
            total_steps += length
            run.log(episode=episode, step=total_steps, **{"return": score}, length=length)
        model.save(str(run.path / "model"))
        model = type(model).load(str(run.path / "model"), device="cpu")
        evaluate("CartPole-v1", lambda state: int(model.predict(state.numpy(), deterministic=True)[0][0]),
                 args.seed, run.path)
        print("实际训练步数", model.num_timesteps)
    finally:
        env.close()
        run.close()


if __name__ == "__main__":
    main()
