"""Species bookkeeping for NEAT."""
from __future__ import annotations

from dataclasses import dataclass, field

from .genome import Genome


@dataclass
class Species:
    id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = float("-inf")
    stagnant_generations: int = 0

    def update_stagnation(self) -> None:
        if not self.members:
            return
        current = max(g.fitness for g in self.members)
        if current > self.best_fitness + 1e-9:
            self.best_fitness = current
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1

    @property
    def champion(self) -> Genome:
        return max(self.members, key=lambda g: g.fitness)
