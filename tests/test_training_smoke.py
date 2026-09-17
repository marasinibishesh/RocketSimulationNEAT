from falcon9_neat.config import load_config
from falcon9_neat.neat.trainer import Trainer


def test_one_generation_training_smoke(tmp_path):
    cfg = load_config(None)
    cfg.neat.population_size = 4
    cfg.neat.episodes_per_genome = 1
    cfg.environment.max_time_s = 1.0
    trainer = Trainer(cfg, tmp_path)
    best = trainer.train(generations=2)
    assert best.fitness == best.fitness
    assert (tmp_path / "best_genome.json").exists()
