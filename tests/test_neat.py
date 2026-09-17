import random
import numpy as np

from falcon9_neat.neat.genome import Genome, MutationRates
from falcon9_neat.neat.innovation import InnovationTracker
from falcon9_neat.neat.network import FeedForwardNetwork


def test_minimal_network_runs():
    rng = random.Random(1)
    tracker = InnovationTracker(first_node_id=6)
    g = Genome.minimal(2, 3, tracker, rng)
    net = FeedForwardNetwork(g)
    out = net.activate(np.array([0.25, -0.5]))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_add_node_grows_topology_without_cycle():
    rng = random.Random(2)
    tracker = InnovationTracker(first_node_id=4)
    g = Genome.minimal(2, 1, tracker, rng)
    before = len(g.nodes)
    assert g.mutate_add_node(tracker, rng)
    assert len(g.nodes) == before + 1
    FeedForwardNetwork(g)  # raises on a cycle


def test_serialization_round_trip(tmp_path):
    rng = random.Random(3)
    tracker = InnovationTracker(first_node_id=4)
    g = Genome.minimal(2, 1, tracker, rng)
    p = tmp_path / "genome.json"
    g.save(p)
    loaded = Genome.load(p)
    assert loaded.to_dict() == g.to_dict()
