from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from snake_drl.agent import DQNAgent
from snake_drl.curriculum import CurriculumManager
from snake_drl.env import SnakeGame
from snake_drl.utils import json_dump, make_run_dir, set_seed


def evaluate(agent: DQNAgent, env: SnakeGame, episodes: int) -> float:
    scores = []
    epsilon_prev = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(episodes):
        state = env.reset()
        while True:
            action = agent.act(state, training=False)
            next_state, _, done = env.step(action)
            state = next_state
            if done:
                scores.append(env.score)
                break

    agent.epsilon = epsilon_prev
    return float(np.mean(scores)) if scores else 0.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Snake Deep RL (DQN + PER + Curriculum)")

    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-steps-per-env-step", type=int, default=1)
    p.add_argument("--target-update", type=int, default=5)

    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.995)

    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--memory-size", type=int, default=100_000)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--grid-min", type=int, default=10)
    p.add_argument("--grid-max", type=int, default=30)
    p.add_argument("--threshold-scores", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--obstacles", action="store_true")

    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=5)

    p.add_argument("--resume", type=str, default=None)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    set_seed(args.seed, deterministic=args.deterministic)

    resume_ckpt_path = Path(args.resume).resolve() if args.resume else None

    if resume_ckpt_path:
        if resume_ckpt_path.parent.name == "checkpoints":
            run_dir = resume_ckpt_path.parent.parent
        else:
            run_dir = resume_ckpt_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(args.log_dir, args.run_name)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    if not resume_ckpt_path:
        json_dump(run_dir / "config.json", config)

    curriculum = None
    if not args.no_curriculum:
        curriculum = CurriculumManager(grid_min=args.grid_min, grid_max=args.grid_max, threshold_scores=tuple(args.threshold_scores))
        grid_w, grid_h = curriculum.get_grid_size()
        use_obstacles = curriculum.get_obstacle_setting()
        env = SnakeGame(grid_width=grid_w, grid_height=grid_h, obstacles=use_obstacles, curriculum_level=curriculum.current_level)
    else:
        env = SnakeGame(grid_width=args.grid_max, grid_height=args.grid_max, obstacles=args.obstacles, curriculum_level=0)

    state_size = int(env.reset().shape[0])
    action_size = 3

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout,
        memory_capacity=args.memory_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.lr,
        device=args.device,
    )

    start_episode = 0
    best_avg_score = float("-inf")

    if resume_ckpt_path:
        resume_payload = agent.load(str(resume_ckpt_path))
        start_episode = int(resume_payload.get("episode", 0)) + 1
        best_avg_score = float(resume_payload.get("best_avg_score", float("-inf")))

        env_cfg = resume_payload.get("config", {}).get("env", {})
        if env_cfg:
            env = SnakeGame(
                grid_width=int(env_cfg.get("grid_width", env.grid_width)),
                grid_height=int(env_cfg.get("grid_height", env.grid_height)),
                obstacles=bool(env_cfg.get("use_obstacles", env.use_obstacles)),
                curriculum_level=int(env_cfg.get("curriculum_level", 0)),
            )

        if curriculum is not None:
            curriculum.current_level = int(env_cfg.get("curriculum_level", curriculum.current_level))

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    metrics_path = run_dir / "metrics.csv"

    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "score",
                "reward_sum",
                "avg_score",
                "epsilon",
                "loss_mean",
                "steps",
                "curriculum_level",
                "grid_width",
                "grid_height",
                "use_obstacles",
            ],
        )
        if write_header:
            csv_writer.writeheader()

        for episode in range(start_episode, args.episodes):
            state = env.reset()
            reward_sum = 0.0
            steps = 0

            loss_sum = 0.0
            loss_count = 0

            while True:
                action = agent.act(state, training=True)
                next_state, reward, done = env.step(action)

                agent.memory.push(state, action, float(reward), next_state, bool(done))

                for _ in range(max(1, args.train_steps_per_env_step)):
                    loss = agent.train_step(args.batch_size)
                    if loss is not None:
                        loss_sum += loss
                        loss_count += 1

                state = next_state
                reward_sum += float(reward)
                steps += 1

                if done:
                    break

            agent.update_metrics(env.score)

            if episode % args.target_update == 0:
                agent.update_target_model()

            if curriculum is not None and curriculum.should_advance(agent.avg_score):
                curriculum.advance_level()
                grid_w, grid_h = curriculum.get_grid_size()
                use_obstacles = curriculum.get_obstacle_setting()
                env = SnakeGame(
                    grid_width=grid_w,
                    grid_height=grid_h,
                    obstacles=use_obstacles,
                    curriculum_level=curriculum.current_level,
                )

            loss_mean = (loss_sum / loss_count) if loss_count else 0.0

            row = {
                "episode": episode,
                "score": env.score,
                "reward_sum": reward_sum,
                "avg_score": agent.avg_score,
                "epsilon": agent.epsilon,
                "loss_mean": loss_mean,
                "steps": steps,
                "curriculum_level": getattr(curriculum, "current_level", 0),
                "grid_width": env.grid_width,
                "grid_height": env.grid_height,
                "use_obstacles": env.use_obstacles,
            }
            csv_writer.writerow(row)
            f.flush()

            writer.add_scalar("train/score", env.score, episode)
            writer.add_scalar("train/avg_score", agent.avg_score, episode)
            writer.add_scalar("train/epsilon", agent.epsilon, episode)
            writer.add_scalar("train/reward_sum", reward_sum, episode)
            writer.add_scalar("train/loss_mean", loss_mean, episode)
            writer.add_scalar("train/steps", steps, episode)

            if args.eval_every and episode > 0 and episode % args.eval_every == 0:
                eval_env = SnakeGame(
                    grid_width=env.grid_width,
                    grid_height=env.grid_height,
                    obstacles=env.use_obstacles,
                    curriculum_level=getattr(curriculum, "current_level", 0),
                )
                eval_score = evaluate(agent, eval_env, episodes=args.eval_episodes)
                writer.add_scalar("eval/mean_score", eval_score, episode)

            if args.save_every and episode > 0 and episode % args.save_every == 0:
                agent.save(
                    str(ckpt_dir / "last.pt"),
                    episode=episode,
                    config={"env": row, "args": config},
                    extra={"best_avg_score": best_avg_score},
                )

            if agent.avg_score > best_avg_score:
                best_avg_score = float(agent.avg_score)
                agent.save(
                    str(ckpt_dir / "best.pt"),
                    episode=episode,
                    config={"env": row, "args": config},
                    extra={"best_avg_score": best_avg_score},
                )

    writer.close()


if __name__ == "__main__":
    main()
