"""Genome representation, mutation, crossover and compatibility distance."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable

from .genes import ConnectionGene, NodeGene
from .innovation import InnovationTracker


@dataclass(slots=True)
class MutationRates:
    weight_mutation_rate: float = 0.8
    weight_perturb_rate: float = 0.9
    weight_perturb_sigma: float = 0.35
    add_connection_rate: float = 0.12
    add_node_rate: float = 0.045
    toggle_connection_rate: float = 0.015


class Genome:
    def __init__(self) -> None:
        self.nodes: dict[int, NodeGene] = {}
        self.connections: dict[int, ConnectionGene] = {}
        self.fitness: float = float("-inf")
        self.adjusted_fitness: float = 0.0

    @property
    def input_ids(self) -> list[int]:
        return sorted(n.id for n in self.nodes.values() if n.kind == "input")

    @property
    def bias_ids(self) -> list[int]:
        return sorted(n.id for n in self.nodes.values() if n.kind == "bias")

    @property
    def output_ids(self) -> list[int]:
        return sorted(n.id for n in self.nodes.values() if n.kind == "output")

    @classmethod
    def minimal(
        cls,
        input_count: int,
        output_count: int,
        tracker: InnovationTracker,
        rng: random.Random,
    ) -> "Genome":
        g = cls()
        for i in range(input_count):
            g.nodes[i] = NodeGene(i, "input")
        bias_id = input_count
        g.nodes[bias_id] = NodeGene(bias_id, "bias")
        output_start = input_count + 1
        for j in range(output_count):
            node_id = output_start + j
            g.nodes[node_id] = NodeGene(node_id, "output")
        tracker.next_node_id = max(tracker.next_node_id, output_start + output_count)
        for src in [*range(input_count), bias_id]:
            for dst in range(output_start, output_start + output_count):
                innov = tracker.connection(src, dst)
                g.connections[innov] = ConnectionGene(src, dst, rng.uniform(-1.0, 1.0), True, innov)
        return g

    def copy(self) -> "Genome":
        out = Genome()
        out.nodes = {k: v.copy() for k, v in self.nodes.items()}
        out.connections = {k: v.copy() for k, v in self.connections.items()}
        out.fitness = self.fitness
        out.adjusted_fitness = self.adjusted_fitness
        return out

    def enabled_connections(self) -> list[ConnectionGene]:
        return [c for c in self.connections.values() if c.enabled]

    def _would_create_cycle(self, src: int, dst: int) -> bool:
        """Return True if src->dst would make the feed-forward graph cyclic."""
        adjacency: dict[int, list[int]] = {}
        for c in self.connections.values():
            if c.enabled:
                adjacency.setdefault(c.src, []).append(c.dst)
        stack = [dst]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node == src:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        return False

    def mutate_weights(self, rng: random.Random, rates: MutationRates) -> None:
        for c in self.connections.values():
            if rng.random() < rates.weight_mutation_rate:
                if rng.random() < rates.weight_perturb_rate:
                    c.weight += rng.gauss(0.0, rates.weight_perturb_sigma)
                else:
                    c.weight = rng.uniform(-2.0, 2.0)
                c.weight = max(-5.0, min(5.0, c.weight))

    def mutate_add_connection(self, tracker: InnovationTracker, rng: random.Random) -> bool:
        # Keep the phenotype feed-forward and treat output nodes as terminal.
        sources = sorted(n.id for n in self.nodes.values() if n.kind != "output")
        destinations = [n.id for n in self.nodes.values() if n.kind not in {"input", "bias"}]
        existing = {(c.src, c.dst) for c in self.connections.values()}
        candidates = [
            (a, b)
            for a in sources
            for b in destinations
            if a != b and (a, b) not in existing and not self._would_create_cycle(a, b)
        ]
        if not candidates:
            return False
        src, dst = rng.choice(candidates)
        innov = tracker.connection(src, dst)
        self.connections[innov] = ConnectionGene(src, dst, rng.uniform(-1.5, 1.5), True, innov)
        return True

    def mutate_add_node(self, tracker: InnovationTracker, rng: random.Random) -> bool:
        candidates = self.enabled_connections()
        if not candidates:
            return False
        old = rng.choice(candidates)
        old.enabled = False
        node_id = tracker.node_for_split(old.innovation)
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeGene(node_id, "hidden")
        in_innov = tracker.connection(old.src, node_id)
        out_innov = tracker.connection(node_id, old.dst)
        self.connections[in_innov] = ConnectionGene(old.src, node_id, 1.0, True, in_innov)
        self.connections[out_innov] = ConnectionGene(node_id, old.dst, old.weight, True, out_innov)
        return True

    def mutate_toggle(self, rng: random.Random) -> bool:
        if not self.connections:
            return False
        gene = rng.choice(list(self.connections.values()))
        if gene.enabled:
            gene.enabled = False
        elif not self._would_create_cycle(gene.src, gene.dst):
            gene.enabled = True
        return True

    def mutate(self, tracker: InnovationTracker, rng: random.Random, rates: MutationRates) -> None:
        self.mutate_weights(rng, rates)
        if rng.random() < rates.add_connection_rate:
            self.mutate_add_connection(tracker, rng)
        if rng.random() < rates.add_node_rate:
            self.mutate_add_node(tracker, rng)
        if rng.random() < rates.toggle_connection_rate:
            self.mutate_toggle(rng)

    @staticmethod
    def crossover(a: "Genome", b: "Genome", rng: random.Random) -> "Genome":
        """NEAT crossover using innovation numbers.

        Disjoint/excess genes come from the fitter parent. Matching genes are
        sampled from either parent. Disabled matching genes remain disabled 75%
        of the time if disabled in either parent.
        """
        if b.fitness > a.fitness:
            a, b = b, a
        equal = math.isclose(a.fitness, b.fitness, rel_tol=1e-12, abs_tol=1e-12)
        child = Genome()
        innovations = sorted(set(a.connections) | set(b.connections))
        chosen: list[ConnectionGene] = []
        for innov in innovations:
            ga = a.connections.get(innov)
            gb = b.connections.get(innov)
            gene: ConnectionGene | None = None
            if ga is not None and gb is not None:
                gene = rng.choice([ga, gb]).copy()
                if (not ga.enabled or not gb.enabled) and rng.random() < 0.75:
                    gene.enabled = False
            elif ga is not None:
                gene = ga.copy()
            elif equal and gb is not None and rng.random() < 0.5:
                gene = gb.copy()
            if gene is not None:
                chosen.append(gene)

        used_nodes: set[int] = set()
        for c in chosen:
            used_nodes.add(c.src)
            used_nodes.add(c.dst)
        # Always retain interface nodes; hidden nodes only if referenced.
        for parent in (a, b):
            for node in parent.nodes.values():
                if node.kind in {"input", "bias", "output"} or node.id in used_nodes:
                    child.nodes.setdefault(node.id, node.copy())
        # Combining two acyclic parents can theoretically introduce a cycle.
        # Insert enabled genes one by one and disable only the offending edge.
        child.connections = {}
        for c in chosen:
            if c.enabled and child._would_create_cycle(c.src, c.dst):
                c.enabled = False
            child.connections[c.innovation] = c
        return child

    def compatibility_distance(
        self,
        other: "Genome",
        c_excess: float = 1.0,
        c_disjoint: float = 1.0,
        c_weight: float = 0.4,
    ) -> float:
        if not self.connections and not other.connections:
            return 0.0
        ia, ib = set(self.connections), set(other.connections)
        matching = ia & ib
        max_a = max(ia, default=-1)
        max_b = max(ib, default=-1)
        excess = 0
        disjoint = 0
        for innov in ia ^ ib:
            if innov > max_b and innov in ia:
                excess += 1
            elif innov > max_a and innov in ib:
                excess += 1
            else:
                disjoint += 1
        if matching:
            weight_diff = sum(
                abs(self.connections[i].weight - other.connections[i].weight) for i in matching
            ) / len(matching)
        else:
            weight_diff = 0.0
        n = max(len(ia), len(ib))
        if n < 20:
            n = 1
        return c_excess * excess / n + c_disjoint * disjoint / n + c_weight * weight_diff

    def to_dict(self) -> dict:
        return {
            "fitness": self.fitness,
            "nodes": [
                {"id": n.id, "kind": n.kind} for n in sorted(self.nodes.values(), key=lambda n: n.id)
            ],
            "connections": [
                {
                    "src": c.src,
                    "dst": c.dst,
                    "weight": c.weight,
                    "enabled": c.enabled,
                    "innovation": c.innovation,
                }
                for c in sorted(self.connections.values(), key=lambda c: c.innovation)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        g = cls()
        for item in data["nodes"]:
            n = NodeGene(int(item["id"]), str(item["kind"]))
            g.nodes[n.id] = n
        for item in data["connections"]:
            c = ConnectionGene(
                int(item["src"]),
                int(item["dst"]),
                float(item["weight"]),
                bool(item["enabled"]),
                int(item["innovation"]),
            )
            g.connections[c.innovation] = c
        g.fitness = float(data.get("fitness", float("-inf")))
        return g

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
