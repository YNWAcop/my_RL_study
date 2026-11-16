"""加载 Code_2 原课程的 Q 表或 Paddle DQN，直接演示，不进行训练。"""
import argparse
from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LESSONS = {
    "sarsa": ROOT / "projects/Code_2/lesson2/sarsa",
    "q_learning": ROOT / "projects/Code_2/lesson2/q_learning",
    "dqn": ROOT / "projects/Code_2/lesson3/dqn",
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--algo", choices=LESSONS, required=True)
    cli.add_argument("--model", type=Path, help="模型路径；相对路径按当前终端目录解释")
    cli.add_argument("--episodes", type=positive_int, default=3)
    cli.add_argument("--max-steps", type=positive_int, default=1000,
                     help="每回合演示上限，避免未学好的格子策略一直循环")
    cli.add_argument("--delay", type=float, default=None, help="每步额外暂停秒数")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--no-render", action="store_true", help="仅评估，不打开窗口")
    args = cli.parse_args()
    if args.delay is not None and (args.delay < 0 or not float("inf") > args.delay):
        cli.error("delay 必须是非负有限数")
    lesson = LESSONS[args.algo]
    filename = "dqn_model.ckpt" if args.algo == "dqn" else "q_table.npy"
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        candidates = [lesson / filename]
        if args.algo == "dqn":
            candidates.append(ROOT / filename)
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            cli.error("找到多份模型，请用 --model 明确指定：" + ", ".join(map(str, existing)))
        model_path = existing[0] if existing else candidates[0]
    if not model_path.is_file():
        cli.error(f"找不到模型：{model_path}。请先完成训练并保存，或用 --model 指定路径。")

    # 复用原课程的网络、算法、智能体与画面封装；导入 train 不会运行 main。
    sys.path.insert(0, str(lesson))
    course = importlib.import_module("train")
    np.random.seed(args.seed)
    if args.algo == "dqn":
        env = course.gym.make("CartPole-v0")
    else:
        env = course.CliffWalkingWapper(course.gym.make("CliffWalking-v0"))
    returns = []
    try:
        env.seed(args.seed)
        if args.algo == "dqn":
            model = course.Model(env.observation_space.shape[0], env.action_space.n)
            algorithm = course.DQN(model, gamma=course.GAMMA, lr=course.LEARNING_RATE)
            agent = course.Agent(algorithm, act_dim=env.action_space.n)
            agent.restore(str(model_path))
            model.eval()
            context = course.paddle.no_grad
        else:
            agent_class = course.SarsaAgent if args.algo == "sarsa" else course.QLearningAgent
            agent = agent_class(env.observation_space.n, env.action_space.n)
            agent.restore(str(model_path))
            expected = (env.observation_space.n, env.action_space.n)
            if agent.Q.shape != expected or not np.isfinite(agent.Q).all():
                cli.error(f"Q 表应为有限数值矩阵，形状 {expected}；实际形状 {agent.Q.shape}")
            context = nullcontext
        print("已加载：", model_path)
        delay = args.delay if args.delay is not None else (0.0 if args.algo == "dqn" else 0.2)
        with context():
            for episode in range(1, args.episodes + 1):
                obs = env.reset()
                score = 0.0
                done = False
                if not args.no_render:
                    env.render()
                for step in range(1, args.max_steps + 1):
                    action = agent.predict(obs)
                    obs, reward, done, _ = env.step(action)
                    score += reward
                    if not args.no_render:
                        env.render()
                        if delay:
                            time.sleep(delay)
                    if done:
                        break
                returns.append(score)
                ending = "环境回合结束" if done else "达到演示步数上限"
                print(f"回合 {episode}: 步数={step}, 回报={score:.1f}, {ending}", flush=True)
        print(f"平均回报：{sum(returns) / len(returns):.2f}")
    finally:
        env.close()
        # 原 turtle 包装没有实现 close；仅在本进程确实创建画面时关闭。
        if args.algo != "dqn" and getattr(env, "t", None) is not None:
            import turtle
            turtle.bye()


if __name__ == "__main__":
    main()
