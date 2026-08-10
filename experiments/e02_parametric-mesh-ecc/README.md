# e02_parametric-mesh-ecc

## Goal
Probe how Gaussian- and signed-mean-curvature Euler characteristic curves change
under mesh resolution for the parametrized examples in the manuscript.

## Analysis
- `analysis/parametric_mesh_ecc.py` generates triangular meshes for:
  - the doubly elliptical torus,
  - an ellipsoid,
  - the product-of-sines graph over `[0, 2] x [0, 2]`,
  - a bounded helicoid patch.
  - a truncated pseudosphere (tractricoid) patch.
- The combined config computes the two manuscript invariants:
  `gaussian_curvature` and `mean_curvature`. The smooth reference evaluates the
  analytic curvature formula. Generated mesh levels instead estimate curvature
  from their triangulations, so their ECC differences include curvature
  discretization error as well as lower-star-complex resolution effects.
- ECCs are computed with the vertex lower-star filtration: vertices enter at
  their curvature value and edges/triangles enter at the maximum curvature of
  their vertices.
- The config runs three mesh levels: `coarse`, `medium`, and `fine`.
  Individual surfaces may override those segment counts.
- Each surface also gets a `smooth_reference` curve: the script evaluates the
  analytic smooth curvature on a finer parametrization grid and computes a
  numerical lower-star ECC from those values.
- The smooth curves are therefore numerical ECCs of the smooth curvature
  formulas, not symbolic closed-form ECCs in general. For the helicoid, the
  smooth Gaussian-curvature ECC uses the exact manuscript result, a single jump
  at `K = -alpha^2`; the smooth mean-curvature ECC uses the minimal
  surface result, a single jump at `H = 0`.
- The pseudosphere smooth-reference ECC is intentionally numerical: the exact
  curvature formulas are sampled on the reference triangulation, but no
  theoretical ECC override is applied.
- For Gaussian curvature, the smooth pseudosphere reference enters at `K = -1`
  and the cylindrical patch has Euler characteristic zero. Its smooth ECC is
  therefore the singleton `(K, chi) = (-1, 0)`, shown with a circular marker;
  discrete mesh curves expose estimator and boundary error around that ideal.

Run both curvatures with the combined config:

```bash
uv run python experiments/e02_parametric-mesh-ecc/analysis/parametric_mesh_ecc.py \
  --config experiments/e02_parametric-mesh-ecc/config/parametric_surfaces.toml
```

## Outputs
Generated `.poly` meshes are written to `data/<curvature>/` and results are
written under `results/<curvature>/<surface>/<mesh_level>/`. The script scopes
outputs by curvature even for single-curvature configs unless the configured
output directory is already named after that curvature, so Gaussian and mean
runs cannot overwrite each other by accident.

- `<surface>_<level>_vertex_curvature.csv`
- `<surface>_<level>_ecc_events.csv`
- `<surface>_<level>_ecc_curve.csv`
- `<surface>_<level>_curvature_histogram.csv`
- `<surface>_<level>_summary.json`

Each per-curvature `results/<curvature>/` directory also contains:

- `mesh_summary.csv`
- `main_results.png`, the consolidated figure: rows are surfaces and
  columns are `ECC`, `smooth`, `fine`, and `coarse`. Each row has its own
  curvature colorbar at the right of the surface panels.
- `calculation_cache/`, containing content-addressed mesh, curvature, and ECC
  arrays. Unchanged configurations load these instead of recomputing curvature
  and lower-star events.
- `surface_assets/`, containing content-addressed PDF/PNG renderings and JSON
  metadata for the smooth, fine, and coarse panels. Composite figures import
  these images instead of redrawing the 3D surfaces.

Reference outputs are written under
`results/<curvature>/<surface>/smooth_reference/` in the default config:

- `<surface>_smooth_reference_smooth_curvature.csv`
- `<surface>_smooth_reference_ecc_events.csv`
- `<surface>_smooth_reference_ecc_curve.csv`
- `<surface>_smooth_reference_summary.json`

## Notes
- The torus is closed and should finish with Euler characteristic `0`.
- The sine graph and helicoid patch have boundary and should finish with Euler
  characteristic `1`.
- Signed mean curvature on generated meshes follows the manuscript method:
  libigl's cotangent matrix and mixed-Voronoi mass matrix are applied to the
  vertex positions, then projected onto area-weighted vertex normals and halved.
- Gaussian curvature on generated meshes uses angle defect divided by mixed
  Voronoi area. Boundary vertices use a target angle of `pi` rather than
  `2*pi`.
- The reference curve is still numerical: it is not a symbolic ECC. It uses
  smooth curvature values sampled on the configured reference grid, except for
  the helicoid Gaussian- and mean-curvature curves where the closed-form
  ECCs are known.
