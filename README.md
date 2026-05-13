# python-template
a template for python data science projects, based on cookiecutter

## Python Environment and Local Package
This project uses `uv` with `pyproject.toml` and `uv.lock`.

Create a `.venv` and sync dependencies:

```bash
uv venv
uv sync
```

Install only runtime dependencies (skip dev tools):

```bash
uv sync --no-dev
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Reusable code lives in `src/pkg` and can be imported across experiments and notebooks.

## Experiment Workflow
Experiments are self-contained folders under `experiments/` with a 2-digit ID:
`e01_name`, `e02_name`, etc.

Create a new experiment scaffold:

```bash
./scripts/new_experiment.sh my_experiment_name
```

This creates:

```bash
experiments/eNN_my_experiment_name/
├── README.md
├── data_manifest.yaml
├── analysis/
├── config/
├── data/
├── results/
└── runs/
```

- `analysis/`: notebooks and scripts.
- `config/`: experiment parameters.
- `data/`: local experiment data (ignored by git).
- `results/`: generated artifacts like figures/tables (ignored by git).
- `runs/`: run-specific outputs and logs (ignored by git).

Large experiment artifacts are ignored by default via `.gitignore`. Keep metadata (`README.md`,
configs, manifests) in git so work remains reproducible.

## Directory Structure
```bash
.
├── LICENSE
├── README.md
├── doc
├── experiments
│   └── README.md
├── pyproject.toml
├── references
├── scripts
│   └── new_experiment.sh
├── src
│   └── pkg
├── tests
└── uv.lock
```
