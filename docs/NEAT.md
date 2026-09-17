# NEAT implementation

The project does not use `neat-python`. The neuroevolution implementation lives in `src/falcon9_neat/neat/`.

## Genome

Each genome has node genes and directed connection genes. Every connection contains source/destination node IDs, a weight, an enabled flag and a global innovation number.

The initial topology is minimal: 13 inputs plus one bias are connected directly to 3 outputs.

## Historical markings

`InnovationTracker` gives the same structural connection the same innovation number across a population. When a connection is split by add-node mutation, the tracker also remembers the node ID assigned to that split so homologous structure can be recognized later.

## Mutation

Three structural operations are supported:

1. perturb or replace connection weights
2. add a feed-forward connection if it does not make a cycle
3. split an enabled connection with a new hidden node

A low-rate connection toggle mutation is included as well.

## Compatibility distance and species

Genomes are grouped using the standard NEAT idea of a compatibility distance based on excess genes, disjoint genes and average weight difference among matching genes.

Fitness sharing prevents a large species from automatically dominating a smaller species. Stagnant species are removed after a configurable number of generations, except for the species containing the current global champion.

## Crossover

Matching genes are sampled from either parent. Disjoint and excess genes come from the fitter parent. If matching genes are disabled in either parent, the child usually inherits them disabled.

## Landing phenotype

The feed-forward phenotype uses a tanh activation. The first output is remapped to `[0, 1]` for throttle; pitch/yaw commands remain signed.

## Fitness

Each genome is evaluated over several randomized scenarios. The episode reward includes:

- horizontal progress toward the pad
- descent progress
- lateral distance penalty
- horizontal and excessive vertical speed penalties
- tilt and angular-rate penalties
- small fuel-use penalty
- a dominant terminal bonus for a valid soft landing
- a large terminal penalty for impact, timeout or leaving the operating region

The genome fitness is the episode average plus a fraction of its worst-case score. This intentionally pressures controllers toward robustness rather than one lucky trajectory.

## Practical tuning

For quick development:

```bash
falcon9-train --generations 5 --population 24 --episodes 1
```

For more robust training, increase all three values. If evolution plateaus too early, consider increasing population size, loosening the compatibility threshold, increasing structural-mutation rates slightly, or beginning from an easier scenario curriculum.
