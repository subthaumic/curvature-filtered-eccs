# Experiments

Each experiment lives in its own folder:

- `e01_name`
- `e02_name`
- ...

Use `./scripts/new_experiment.sh <name>` to create new experiments.

## Conventions

- Keep code/notebooks in `analysis/`.
- Keep parameters in `config/`.
- Keep reproducibility metadata in `README.md` and `data_manifest.yaml`.
- Keep heavy generated files in `data/`, `results/`, and `runs/` (ignored by git).
- Use the experiment `README.md` as the single place to capture goal, assumptions, and conclusions.

