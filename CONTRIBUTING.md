# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Design rules

- Keep physics independent from Panda3D so training remains headless.
- Keep the NEAT implementation independent from the Falcon landing task.
- Add a regression test for changes to dynamics, mutation, crossover, serialization or CLI behavior.
- Document any new physical constants and state whether they are public reference values, measured data or modeling assumptions.
- Do not add proprietary SpaceX CAD, telemetry or copyrighted artwork to the repository.

## Pull requests

Run `pytest` before opening a pull request. For renderer changes, also run both `falcon9-demo --showroom` and `falcon9-demo --controller heuristic` locally on a machine with an OpenGL-capable display.
