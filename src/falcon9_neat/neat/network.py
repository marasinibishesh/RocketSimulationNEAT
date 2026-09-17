"""Feed-forward neural phenotype compiled from a NEAT genome."""
from __future__ import annotations

import math
import numpy as np

from .genome import Genome


def _activation(x: float) -> float:
    # NEAT commonly uses a steepened sigmoid; tanh works well for normalized
    # continuous-control observations and naturally produces signed outputs.
    return math.tanh(1.8 * max(-20.0, min(20.0, x)))


class FeedForwardNetwork:
    def __init__(self, genome: Genome):
        self.input_ids = genome.input_ids
        self.bias_ids = genome.bias_ids
        self.output_ids = genome.output_ids
        enabled = [c for c in genome.connections.values() if c.enabled]
        self.incoming: dict[int, list[tuple[int, float]]] = {}
        adjacency: dict[int, list[int]] = {}
        indegree: dict[int, int] = {nid: 0 for nid in genome.nodes}
        for c in enabled:
            self.incoming.setdefault(c.dst, []).append((c.src, c.weight))
            adjacency.setdefault(c.src, []).append(c.dst)
            indegree[c.dst] = indegree.get(c.dst, 0) + 1
            indegree.setdefault(c.src, 0)

        queue = sorted([nid for nid, degree in indegree.items() if degree == 0])
        topo: list[int] = []
        while queue:
            nid = queue.pop(0)
            topo.append(nid)
            for dst in adjacency.get(nid, ()):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)
                    queue.sort()
        if len(topo) != len(indegree):
            raise ValueError("Genome contains a recurrent cycle; this implementation is feed-forward NEAT.")
        interface = set(self.input_ids + self.bias_ids)
        self.eval_order = [nid for nid in topo if nid not in interface]

    def activate(self, observation: np.ndarray | list[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if len(obs) != len(self.input_ids):
            raise ValueError(f"Expected {len(self.input_ids)} inputs, got {len(obs)}")
        values: dict[int, float] = {nid: float(v) for nid, v in zip(self.input_ids, obs)}
        for nid in self.bias_ids:
            values[nid] = 1.0
        for nid in self.eval_order:
            total = sum(values.get(src, 0.0) * weight for src, weight in self.incoming.get(nid, ()))
            values[nid] = _activation(total)
        return np.array([values.get(nid, 0.0) for nid in self.output_ids], dtype=float)


class NeatLandingController:
    def __init__(self, genome: Genome):
        self.network = FeedForwardNetwork(genome)

    def act(self, observation: np.ndarray) -> np.ndarray:
        raw = self.network.activate(observation)
        if len(raw) != 3:
            raise ValueError("Landing controller requires exactly three network outputs.")
        # tanh -> [0,1] throttle; signed channels remain [-1,1].
        return np.array([(raw[0] + 1.0) * 0.5, raw[1], raw[2]], dtype=float)
