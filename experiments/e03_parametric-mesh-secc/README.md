# e03_parametric-mesh-secc

## Goal

Compute smooth Euler characteristic curves (SECCs) for the same parametrized
surfaces, curvature estimators, and mesh resolutions used in `e02`.

## Method

For each raw Euler characteristic curve `chi`, e03 evaluates it on a common
analytic curvature interval and computes

```text
SECC(t) = integral from t_min to t of (chi(s) - mean(chi)) ds.
```

The smooth, fine, medium, and coarse curves for one surface all use the same
interval. Nonconstant analytic curvature ranges receive 5% padding at both
ends. Constant ranges receive an absolute padding of 1 so that constant-
curvature examples still have a nondegenerate integration interval.

The smooth reference retains the analytic/theoretical choices from e02.
Generated meshes use the same discrete estimators as e02: libigl cotangent-
Voronoi signed mean curvature and boundary-aware angle-defect Gaussian
curvature.

Run both curvature types:

```bash
uv run python experiments/e03_parametric-mesh-secc/analysis/parametric_mesh_secc.py \
  --config experiments/e03_parametric-mesh-secc/config/parametric_surfaces.toml
```

## Outputs

Results are written under `results/<curvature>/`:

- `<surface>/<surface>_secc_curves.csv`: the threshold grid, raw ECC,
  centered ECC, and SECC for every mesh level.
- `secc_summary.csv`: normalized L2 and L-infinity distances from each mesh
  SECC to its smooth reference.
- `main_results.png`: the e02-style overview with SECC overlays in the first
  column and smooth/fine/coarse surface renderings in the remaining columns.
- `calculation_cache/`: content-addressed SECC arrays derived from e02's cached
  ECCs. Changing a source curve, interval, or sample count creates a new entry.
- Surface renderings are imported directly from e02's `surface_assets/` cache;
  e03 does not render a duplicate set.

## Notes

- This is a centered cumulative ECC, not kernel smoothing of the curvature or
  of the ECC.
- e03 imports the surface and estimator implementation from e02 and points to
  the combined e02 config, so both experiments stay synchronized. It also
  consumes e02's content-addressed calculation cache rather than recomputing
  meshes, curvature fields, or ECCs.
