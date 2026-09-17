"""Global innovation bookkeeping shared by a NEAT population."""
from __future__ import annotations


class InnovationTracker:
    def __init__(self, first_node_id: int = 0) -> None:
        self.next_innovation = 0
        self.next_node_id = first_node_id
        self.connection_history: dict[tuple[int, int], int] = {}
        self.split_history: dict[int, int] = {}

    def connection(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self.connection_history:
            self.connection_history[key] = self.next_innovation
            self.next_innovation += 1
        return self.connection_history[key]

    def node_for_split(self, connection_innovation: int) -> int:
        if connection_innovation not in self.split_history:
            self.split_history[connection_innovation] = self.next_node_id
            self.next_node_id += 1
        return self.split_history[connection_innovation]
