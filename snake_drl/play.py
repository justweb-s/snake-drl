from __future__ import annotations

import argparse

import pygame
import torch

from snake_drl.agent import DQNAgent
from snake_drl.env import SnakeGame


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play Snake with a trained DQN checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--cell-size", type=int, default=20)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu")

    ckpt_args = payload.get("config", {}).get("args", {})
    hidden_size = int(ckpt_args.get("hidden_size", 128))
    dropout = float(ckpt_args.get("dropout", 0.2))
    device = args.device if args.device is not None else ckpt_args.get("device", None)

    env_cfg = payload.get("config", {}).get("env", {})
    grid_w = int(env_cfg.get("grid_width", 30))
    grid_h = int(env_cfg.get("grid_height", 30))
    use_obstacles = bool(env_cfg.get("use_obstacles", False))
    curriculum_level = int(env_cfg.get("curriculum_level", 0))

    env = SnakeGame(
        grid_width=grid_w,
        grid_height=grid_h,
        obstacles=use_obstacles,
        curriculum_level=curriculum_level,
    )

    state_size = int(env.reset().shape[0])
    agent = DQNAgent(
        state_size=state_size,
        action_size=3,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        device=device,
    )
    agent.load(args.checkpoint)

    pygame.init()
    width = env.grid_width * args.cell_size
    height = env.grid_height * args.cell_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake DRL - Play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    agent.epsilon = 0.0

    for _ in range(args.episodes):
        state = env.reset()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state, training=False)
            next_state, _, done = env.step(action)
            state = next_state

            white = env.render(screen, cell_size=args.cell_size, draw_grid=True)
            text = font.render(f"Score: {env.score}", True, white)
            screen.blit(text, (8, 8))

            pygame.display.flip()
            clock.tick(args.fps)

            if done:
                break

    pygame.quit()


if __name__ == "__main__":
    main()
