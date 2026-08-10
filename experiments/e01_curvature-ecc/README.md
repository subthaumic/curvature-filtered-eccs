# e01_curvature-ecc

## Goal
Study Euler characteristic curves based on curvature.

## Status
planned

## Analysis
- `analysis/curvature_ecc.py` parses the `POLYS` section of `.poly` files,
  fan-triangulates polygonal records when needed, builds the curvature sublevel
  filtration, and writes Euler characteristic curve outputs.
- `analysis/generate_dented_sphere.py` generates synthetic dented-sphere
  triangulations at multiple refinements and writes signed mean curvature into
  the `POLYS` alpha slot.
- `analysis/ecc_l1_confusion.py` compares generated ECC curves by pairwise L1
  distance and plots the resulting log-distance matrix and dendrogram.

Run one dataset from a TOML config:

```bash
uv run python experiments/e01_curvature-ecc/analysis/curvature_ecc.py \
  --config experiments/e01_curvature-ecc/config/199.toml
```

Run all exported datasets:

```bash
for config in experiments/e01_curvature-ecc/config/*.toml; do
  uv run python experiments/e01_curvature-ecc/analysis/curvature_ecc.py --config "$config"
done
```

Compare all generated ECC curves from the current export:

```bash
uv run python experiments/e01_curvature-ecc/analysis/ecc_l1_confusion.py \
  --results-dir experiments/e01_curvature-ecc/results \
  --output-dir experiments/e01_curvature-ecc/results/ecc_l1_confusion
```

The CSV contains raw L1 distances; the heatmap and dendrogram use
`log(1 + distance)`. The aggregate output also includes `ecc_curves.png`, which
plots all current-export ECC curves in one image.

Configs are provided for each `.poly` dataset in `curvature_export.zip`:

- `config/199.toml`
- `config/199_smooth.toml`
- `config/201.toml`
- `config/201_smooth.toml`
- `config/210.toml`
- `config/210_smooth.toml`
- `config/gyroid.toml`

Each config sets `input`, `results_dir`, `output_name`, `skip_degenerate`,
`surface_plots`, `histogram_bins`, and optionally `histogram_symlog` and
`plot_percentile`. `histogram_symlog` defaults to `true`; set
`histogram_symlog = false` for a linear count axis. Set `plot_percentile = 98`
or `plot_percentile = 95` to focus the ECC and histogram plots on the central
percentile range of triangle curvature values while leaving CSV outputs
unchanged. Set `z_min = <value>` to discard points with z-coordinate below that
threshold and remove all incident triangles before building the complex.
Configs may also set `smooth_triangle_filtration = true` to replace each
triangle filtration value by the mean over that triangle and all triangles
sharing an edge with it. Relative paths are resolved from the config file.
Outputs are written under `results_dir/output_name`.

Outputs per config:

- `<stem>_filtered_complex.json`: filtered vertices, edges, and triangles.
- `<stem>_ecc_events.csv`: signed simplex events for the ECC.
- `<stem>_ecc_curve.csv`: cumulative Euler characteristic curve.
- `<stem>_ecc_plot.png`: step plot of the cumulative ECC.
- `<stem>_curvature_histogram.csv`: binned triangle-curvature counts.
- `<stem>_curvature_histogram.png`: histogram plot of triangle curvatures
  with a symlog count axis.
- `<stem>_summary.json`: counts, filtration range, and data-quality notes.

- `<stem>_surface_plot.png`: optional 3D plot of the triangulated surface,
  colored by the alpha/curvature values. The current export configs leave this
  disabled to avoid large extra plots.

## Data
- `data/curvature_export.zip` is the canonical local export for this run. It
  is flattened into `data/` and contains `199.poly`, `199_smooth.poly`,
  `201.poly`, `201_smooth.poly`, `210.poly`, `210_smooth.poly`, and
  `gyroid.poly`.
- `data/3D models.zip` contains print-ready STL files and full-resolution OBJ
  ZIP exports for several USNM specimens. It does not contain `.poly` files or
  curvature alpha values, so the current ECC scripts do not use it.
- Document data sources and local paths in `data_manifest.yaml`.

## Notes
- Created via `scripts/new_experiment.sh`.
