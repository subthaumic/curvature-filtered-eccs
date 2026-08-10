# e04 projected-subdivision stability

## Question

Do curvature-based ECCs or their centred cumulative SECCs converge more quickly
and vary less under triangulation choices than the pointwise curvature field?

## Design

Every e02 surface starts from the same `8 x 8` parameter-grid resolution. This
is deliberately coarse, but it gives the product-of-sines surface four cells
per period and avoids sampling all sine factors at zero. Each refinement uses
libigl's `upsample` connectivity: one midpoint is added to every edge and each
triangle is replaced by four children. Midpoint parameters are pushed through
the exact surface parametrisation; ellipsoid midpoints use normalized spherical
midpoints to avoid its coordinate-singular poles.

At every subdivision level, independent alternatives are produced by flipping
an exact prescribed fraction of admissible edges in the parameter domain while
leaving its vertices fixed. A flip is allowed only for a locally convex domain
quadrilateral. Periodic coordinates are locally unwrapped; edges incident to
the ellipsoid's coordinate-singular poles are not flipped. The same random
connectivity realization is used for Gaussian and mean curvature.

There are two nonzero-flip regimes:

- `stress` imposes no embedded-space triangle-quality restriction. It is kept
  as a deliberately adverse thin-triangle test where failure is expected.
- `quality_controlled` uses a relaxed 10-degree minimum-angle floor in embedded
  space. The two replacement triangles must meet that floor; if the original
  local patch is already below 10 degrees, the flip must not reduce its local
  minimum angle. This prevents a flip from introducing or worsening a sliver.

The unflipped refinement family is recorded once as `baseline`. Both regimes
then realize exactly 5% and 10% of all admissible interior edges; the acceptance
filter has ample candidates for every retained surface and level.

For curvature, ECC and SECC the experiment records:

- error to the analytic or smooth-reference target at every subdivision level;
- the median and interquartile range across triangulations;
- a convergence ratio obtained by dividing by the level-zero median error;
- pairwise distances among repeated triangulations, without treating one as
  canonical.

Curvature uses mixed-Voronoi-area weighting and the dimensionless fields
`sqrt(area) H` and `area K`. ECC and SECC use one analytic curvature interval
per surface and curvature, mapped to `u in [0, 1]`; SECC integrates with respect
to this dimensionless coordinate.

## Run

Retained smoke run (four subdivision levels, eight realisations for each
regime and each of the 5% and 10% flip conditions):

```bash
uv run python experiments/e04_triangulation-stability/analysis/subdivision_stability.py \
  --config experiments/e04_triangulation-stability/config/smoke.toml
```

Full run:

```bash
uv run python experiments/e04_triangulation-stability/analysis/subdivision_stability.py \
  --config experiments/e04_triangulation-stability/config/subdivision_stability.toml
```

## Artifacts

- `subdivision_realizations.csv`: per-realisation errors to theory.
- `subdivision_summary.csv`: convergence medians, quartiles and ratios.
- `triangulation_pairwise.csv`: all within-level pairwise distances.
- `triangulation_summary.csv`: pairwise medians, quartiles and ratios to the
  level-zero median for within-method comparison across refinement levels.
- `ecc_jump_scaling.csv`: growth factors and fitted vertex-count exponents for
  ECC jump counts and total variation.
- `main_results.png`: surface-specific convergence curves and IQR ribbons.
- `triangulation_variability.png`: surface-specific pairwise-distance ratios,
  normalized within each method and flip condition at level zero.
- `ecc_jump_diagnostics.png`: exact and 501-grid-resolved ECC jump complexity
  against mesh size.
- `mesh_quality.png`: minimum and first-percentile embedded triangle angles,
  verifying the separation between quality-controlled and stress meshes.
- `calculation_cache/`: content-addressed per-realisation arrays and metadata.

Tables and figures have signature sidecars and are not rewritten on warm runs.
The convergence figure uses a logarithmic error-ratio axis, marks the
level-zero baseline at one, and places finer meshes toward the right.
Exact ECC diagnostics rebuild only the lower-star event sequence from cached
curvature values and deterministic connectivity; they do not repeat curvature
estimation. The sampled diagnostics show which changes remain visible on the
shared 501-point filtration grid used by the distance calculation.

## Current smoke-run reading

The two ensembles are cleanly separated: at the finest smoke level, stress
meshes reach sub-degree minimum angles on the ellipsoid, helicoid and
pseudosphere, whereas quality-controlled flips never reduce the unflipped
family's global minimum angle (and meet the 10-degree floor where the baseline
does). Quality control generally reduces triangulation-induced variability,
especially for curvature, but this smoke run does **not** yet establish the
desired ECC/SECC stability claim. Their pairwise-distance ratios still grow
with refinement on most surfaces. This is a diagnostic result, not a final
conclusion; in particular, a fixed positive flip fraction changes an increasing
number of edges as the mesh is refined.
