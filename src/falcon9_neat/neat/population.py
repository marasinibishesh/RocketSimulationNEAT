"""NEAT population manager and reproduction logic."""
from __future__ import annotations

import math
import random

from ..config import NeatConfig
from .genome import Genome, MutationRates
from .innovation import InnovationTracker
from .species import Species


class Population:
    def __init__(self, config: NeatConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        interface_nodes = config.input_count + 1 + config.output_count
        self.tracker = InnovationTracker(first_node_id=interface_nodes)
        template = Genome.minimal(config.input_count, config.output_count, self.tracker, self.rng)
        self.genomes = [template.copy() for _ in range(config.population_size)]
        rates = self._rates()
        for g in self.genomes:
            g.mutate_weights(self.rng, rates)
        self.species: list[Species] = []
        self._next_species_id = 0
        self.generation = 0
        self.speciate()

    def _rates(self) -> MutationRates:
        c = self.config
        return MutationRates(
            c.weight_mutation_rate,
            c.weight_perturb_rate,
            c.weight_perturb_sigma,
            c.add_connection_rate,
            c.add_node_rate,
            c.toggle_connection_rate,
        )

    def _distance(self, a: Genome, b: Genome) -> float:
        c = self.config
        return a.compatibility_distance(
            b, c.compatibility_excess, c.compatibility_disjoint, c.compatibility_weight
        )

    def speciate(self) -> None:
        for s in self.species:
            s.members.clear()
        for genome in self.genomes:
            assigned = False
            # Closest compatible representative gives more stable clustering.
            compatible = [
                (self._distance(genome, s.representative), s)
                for s in self.species
                if self._distance(genome, s.representative) < self.config.compatibility_threshold
            ]
            if compatible:
                _, species = min(compatible, key=lambda item: item[0])
                species.members.append(genome)
                assigned = True
            if not assigned:
                species = Species(self._next_species_id, genome.copy(), [genome])
                self._next_species_id += 1
                self.species.append(species)
        self.species = [s for s in self.species if s.members]
        for s in self.species:
            s.representative = self.rng.choice(s.members).copy()

    def _parent(self, pool: list[Genome]) -> Genome:
        low = min(g.fitness for g in pool)
        weights = [max(1e-8, g.fitness - low + 1e-6) for g in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _offspring_counts(self, viable: list[Species]) -> dict[int, int]:
        # Explicit fitness sharing.
        min_fit = min(g.fitness for s in viable for g in s.members)
        totals: dict[int, float] = {}
        for s in viable:
            size = max(1, len(s.members))
            totals[s.id] = sum((g.fitness - min_fit + 1e-6) / size for g in s.members)
        total = sum(totals.values())
        n = self.config.population_size
        if total <= 0.0:
            raw = {s.id: n / len(viable) for s in viable}
        else:
            raw = {s.id: n * totals[s.id] / total for s in viable}
        counts = {sid: int(math.floor(value)) for sid, value in raw.items()}
        remainder = n - sum(counts.values())
        order = sorted(raw, key=lambda sid: raw[sid] - counts[sid], reverse=True)
        for sid in order[:remainder]:
            counts[sid] += 1
        return counts

    def reproduce(self) -> None:
        if not self.genomes:
            raise RuntimeError("Empty population")
        # Track stagnation after genomes have current-generation fitness.
        for s in self.species:
            s.update_stagnation()
        best_species = max(self.species, key=lambda s: s.champion.fitness)
        viable = [
            s
            for s in self.species
            if s.stagnant_generations < self.config.stagnation_generations or s is best_species
        ]
        counts = self._offspring_counts(viable)
        rates = self._rates()
        new_population: list[Genome] = []

        for s in viable:
            quota = counts.get(s.id, 0)
            if quota <= 0:
                continue
            ranked = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            elite_count = min(self.config.elitism, quota, len(ranked))
            for i in range(elite_count):
                elite = ranked[i].copy()
                elite.fitness = float("-inf")
                new_population.append(elite)
            quota -= elite_count
            survivor_count = max(1, int(math.ceil(len(ranked) * self.config.survival_fraction)))
            pool = ranked[:survivor_count]
            for _ in range(quota):
                p1 = self._parent(pool)
                if self.rng.random() < self.config.crossover_rate and len(pool) > 1:
                    if self.rng.random() < self.config.interspecies_mating_rate and len(viable) > 1:
                        other_s = self.rng.choice([x for x in viable if x.id != s.id])
                        p2 = self._parent(sorted(other_s.members, key=lambda g: g.fitness, reverse=True)[:max(1, int(math.ceil(len(other_s.members) * self.config.survival_fraction)))])
                    else:
                        p2 = self._parent(pool)
                    child = Genome.crossover(p1, p2, self.rng)
                else:
                    child = p1.copy()
                child.fitness = float("-inf")
                child.mutate(self.tracker, self.rng, rates)
                new_population.append(child)

        # Numerical allocation fallback.
        champion = max(self.genomes, key=lambda g: g.fitness)
        while len(new_population) < self.config.population_size:
            child = champion.copy()
            child.fitness = float("-inf")
            child.mutate(self.tracker, self.rng, rates)
            new_population.append(child)
        self.genomes = new_population[: self.config.population_size]
        self.generation += 1
        self.speciate()
