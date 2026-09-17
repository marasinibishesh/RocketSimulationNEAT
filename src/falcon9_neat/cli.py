"""Command-line interfaces for training, evaluation and visualization."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

from .config import load_config
from .controllers import HeuristicLandingController
from .environment import FalconLandingEnv
from .neat.genome import Genome
from .neat.network import NeatLandingController
from .neat.trainer import Trainer


def _config_path(value: str | None) -> str | None:
    return value if value else None


def train_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a from-scratch NEAT controller for Falcon 9 booster landing.")
    p.add_argument("--config", default=None, help="YAML config; defaults to built-in settings")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None, help="Randomized episodes per genome")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    args = p.parse_args(argv)
    cfg = load_config(_config_path(args.config))
    if args.population is not None:
        cfg.neat.population_size = max(2, args.population)
    if args.episodes is not None:
        cfg.neat.episodes_per_genome = max(1, args.episodes)
    trainer = Trainer(cfg, args.checkpoint_dir)

    def report(s):
        print(
            f"gen={s.generation:03d} best={s.best_fitness:9.2f} mean={s.mean_fitness:9.2f} "
            f"species={s.species_count:2d} topology={s.best_nodes}n/{s.best_connections}c "
            f"landings={100*s.landing_rate:5.1f}%"
        )

    best = trainer.train(args.generations, report)
    path = Path(args.checkpoint_dir) / "best_genome.json"
    best.save(path)
    print(f"Best checkpoint: {path.resolve()}")
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained NEAT checkpoint without rendering.")
    p.add_argument("--config", default=None, help="YAML config; defaults to built-in settings")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--controller", choices=("neat", "heuristic"), default="neat")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.controller == "neat":
        if not args.checkpoint:
            p.error("--checkpoint is required for --controller neat")
        controller = NeatLandingController(Genome.load(args.checkpoint))
    else:
        controller = HeuristicLandingController()
    successes = 0
    returns = []
    for i in range(args.episodes):
        env = FalconLandingEnv(cfg, seed=cfg.neat.seed + 1000 + i)
        obs = env.reset(cfg.neat.seed + 1000 + i)
        total = 0.0
        while True:
            result = env.step(controller.act(obs))
            obs = result.observation
            total += result.reward
            if result.terminated:
                successes += int(result.info["success"])
                print(
                    f"episode={i+1:02d} outcome={result.info['outcome']:12s} "
                    f"return={total:9.1f} miss={result.info['lateral_distance_m']:6.1f}m "
                    f"vz={result.info['vertical_speed_mps']:7.2f}m/s tilt={result.info['tilt_deg']:5.1f}deg"
                )
                break
        returns.append(total)
    print(f"success={successes}/{args.episodes} ({100*successes/max(1,args.episodes):.1f}%) mean_return={np.mean(returns):.1f}")
    return 0


def demo_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the real-time 3-D Falcon 9 landing visualization.")
    p.add_argument("--config", default=None, help="YAML config; defaults to built-in settings")
    p.add_argument("--controller", choices=("heuristic", "neat"), default="heuristic")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--showroom", action="store_true", help="Rotate the static full-stack 70 m class rocket model")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    try:
        from .render.app import run_visualization
    except ImportError as exc:
        raise SystemExit("Panda3D is required for visualization. Install the project normally with `pip install -e .`.") from exc
    run_visualization(cfg, checkpoint=args.checkpoint, controller_kind=args.controller, showroom=args.showroom)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: falcon9-neat {train,evaluate,demo} [options]\n")
        print("commands:\n  train      evolve a NEAT landing controller\n  evaluate   run headless evaluation episodes\n  demo       launch the Panda3D visualization")
        return 0
    command, rest = args[0], args[1:]
    if command == "train":
        return train_main(rest)
    if command == "evaluate":
        return evaluate_main(rest)
    if command == "demo":
        return demo_main(rest)
    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
