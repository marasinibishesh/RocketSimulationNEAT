"""Gene data structures for a compact NEAT implementation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NodeGene:
    id: int
    kind: str  # input, bias, hidden, output

    def copy(self) -> "NodeGene":
        return NodeGene(self.id, self.kind)


@dataclass(slots=True)
class ConnectionGene:
    src: int
    dst: int
    weight: float
    enabled: bool
    innovation: int

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(self.src, self.dst, self.weight, self.enabled, self.innovation)
