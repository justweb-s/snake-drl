from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training metrics from metrics.csv")
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    metrics_path = Path(args.metrics)
    out_path = Path(args.out) if args.out else metrics_path.with_suffix(".png")

    episodes = []
    scores = []
    avg_scores = []

    with metrics_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx_episode = header.index("episode")
        idx_score = header.index("score")
        idx_avg = header.index("avg_score")

        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            episodes.append(int(float(parts[idx_episode])))
            scores.append(float(parts[idx_score]))
            avg_scores.append(float(parts[idx_avg]))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, label="score", alpha=0.4)
    plt.plot(episodes, avg_scores, label="avg_score", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Snake DRL Training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    main()
