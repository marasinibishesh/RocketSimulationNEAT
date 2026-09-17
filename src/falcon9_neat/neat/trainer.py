"""Training loop connecting NEAT genomes to the landing environment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np

from ..config import AppConfig
from ..environment import FalconLandingEnv
from .genome import Genome
from .network import NeatLandingController
from .population import Population


@dataclass(slots=True)
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    species_count: int
    best_nodes: int
    best_connections: int
    landing_rate: float


def evaluate_genome(genome: Genome, config: AppConfig, seeds: list[int]) -> tuple[float, int]:
    controller = NeatLandingController(genome)
    scores: list[float] = []
    landings = 0
    for seed in seeds:
        env = FalconLandingEnv(config, seed=seed)
        obs = env.reset(seed)
        total = 0.0
        while True:
            action = controller.act(obs)
            result = env.step(action)
            total += result.reward
            obs = result.observation
            if result.terminated:
                if result.info["success"]:
                    landings += 1
                break
        scores.append(total)
    # Slightly pessimistic objective: average plus a fraction of the worst case.
    return float(mean(scores) + 0.20 * min(scores)), landings


class Trainer:
    def __init__(self, config: AppConfig, checkpoint_dir: str | Path = "checkpoints"):
        if config.neat.input_count != FalconLandingEnv.observation_size:
            raise ValueError(
                f"NEAT input_count={config.neat.input_count} but environment exposes "
                f"{FalconLandingEnv.observation_size} observations"
            )
        if config.neat.output_count != FalconLandingEnv.action_size:
            raise ValueError(
                f"NEAT output_count={config.neat.output_count} but environment expects "
                f"{FalconLandingEnv.action_size} actions"
            )
        self.config = config
        self.population = Population(config.neat)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[GenerationStats] = []
        self.best_ever: Genome | None = None

    def train(
        self,
        generations: int | None = None,
        callback: Callable[[GenerationStats], None] | None = None,
    ) -> Genome:
        generations = self.config.neat.generations if generations is None else generations
        base_seed = self.config.neat.seed * 10_000
        for _ in range(generations):
            gen = self.population.generation
            # Use common random numbers across generations. This reduces
            # fitness noise while still evaluating every genome on the same
            # randomized scenarios; change the global seed for a new suite.
            seeds = [base_seed + i * 37 for i in range(self.config.neat.episodes_per_genome)]
            landing_count = 0
            for genome in self.population.genomes:
                genome.fitness, landings = evaluate_genome(genome, self.config, seeds)
                landing_count += landings

            best = max(self.population.genomes, key=lambda g: g.fitness)
            if self.best_ever is None or best.fitness > self.best_ever.fitness:
                self.best_ever = best.copy()
                self.best_ever.save(self.checkpoint_dir / "best_genome.json")

            stats = GenerationStats(
                generation=gen,
                best_fitness=best.fitness,
                mean_fitness=float(np.mean([g.fitness for g in self.population.genomes])),
                species_count=len(self.population.species),
                best_nodes=len(best.nodes),
                best_connections=len(best.connections),
                landing_rate=landing_count
                / max(1, len(self.population.genomes) * self.config.neat.episodes_per_genome),
            )
            self.history.append(stats)
            if callback:
                callback(stats)
            if gen + 1 >= generations:
                break
            self.population.reproduce()

        assert self.best_ever is not None
        return self.best_ever
