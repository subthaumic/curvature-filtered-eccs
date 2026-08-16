# e05 vertex-position perturbation stability

## Question

How stable are pointwise curvature, curvature-filtered Euler characteristic
curves (ECCs), and centred cumulative ECCs (SECCs) when mesh vertex coordinates
are perturbed but mesh connectivity is held fixed?

## Status

Implemented. A retained smoke run covers every configured surface, both
curvature types, all five positive perturbation magnitudes, and a zero-noise
baseline.

## Inputs

The experiment imports the surface definitions, analytic curvature formulas,
and smooth ECC references from `e02_parametric-mesh-ecc`. It uses five
surfaces:

- doubly elliptical torus;
- ellipsoid;
- product-of-sines graph;
- bounded helicoid patch; and
- truncated pseudosphere patch.

Both Gaussian curvature and signed mean curvature are tested. Every surface
starts from an `8 x 8` parameter grid. Finer levels use the projected midpoint
subdivision from `e04`: each edge receives one midpoint, every triangle becomes
four triangles, and new vertices are pushed through the exact surface
parametrization.

## Coordinate perturbation

At a fixed surface and subdivision level, every realization uses exactly the
same face array, vertex count, and edge connectivity as its unperturbed mesh.
Only the three coordinates of each vertex change.

For each realization, the script draws independent standard Gaussian values
for the `x`, `y`, and `z` coordinates. Each coordinate column is centered and
standardized before it is scaled. Consequently, the coordinate-wise RMS
perturbation is exactly

```text
perturbation_scale * unperturbed_median_edge_length.
```

Centering removes rigid translation. The same seeded coordinate realization is
used for Gaussian and mean curvature. Five positive scales are linearly spaced
from `0.0001` through `0.005`:

```text
0.0001, 0.001325, 0.00255, 0.003775, 0.005
```

The zero-noise mesh is recorded once per surface, curvature, and subdivision
level. Every positive scale has independently seeded repetitions.

## Comparisons

The experiment makes three complementary comparisons:

1. **Original mathematical target.** Discrete curvature is compared with the
   analytic curvature at the unperturbed parameter locations. ECC and SECC are
   compared with the smooth reference inherited from `e02` and `e03`.
2. **Fixed-connectivity discrete baseline.** Each perturbed curvature field,
   ECC, and SECC is compared directly with the unperturbed discrete result at
   the same subdivision level. This is the primary coordinate-stability test.
3. **Between perturbations.** All pairs of repeated realizations at the same
   scale are compared without selecting one random perturbation as canonical.

Curvature comparisons use mixed-Voronoi-area weighting and the same
dimensionless fields as `e04`: `sqrt(area) H` and `area K`. ECC and SECC are
evaluated on one analytic interval per surface and curvature, mapped to
`u in [0, 1]`. The primary summaries use L2 distance and retain medians and
interquartile ranges. Per-realization tables additionally retain L1,
L-infinity, and curvature bias where applicable.

The run also records requested versus realized displacement, minimum and
low-percentile triangle angles, face-normal reversals relative to the
unperturbed mesh, degenerate-face fractions, and exact/sampled ECC jump
complexity.

## Run

Retained smoke run:

```bash
uv run python experiments/e05_vertex-perturbation-stability/analysis/vertex_perturbation_stability.py \
  --config experiments/e05_vertex-perturbation-stability/config/smoke.toml
```

The smoke configuration uses three subdivision levels, four repetitions per
positive scale, and 301 filtration samples. Across five surfaces and two
curvatures this produces 630 realization rows and 900 pairwise comparisons.

Full run:

```bash
uv run python experiments/e05_vertex-perturbation-stability/analysis/vertex_perturbation_stability.py
```

The full configuration uses four subdivision levels, twelve repetitions per
positive scale, and 1,001 filtration samples. It is configured for 2,440
realization rows and 13,200 pairwise comparisons. Calculations are
content-addressed, so interrupted or repeated runs reuse completed
realizations. The full configuration is the script default, so running
`vertex_perturbation_stability.py` directly from an IDE such as PyCharm also
starts the full analysis without requiring command-line arguments.

## Artifacts

- `perturbation_realizations.csv`: every baseline and perturbed realization.
- `perturbation_summary.csv`: medians and quartiles by surface, curvature,
  subdivision level, and perturbation scale.
- `perturbation_pairwise.csv`: all within-condition realization pairs.
- `perturbation_pairwise_summary.csv`: pairwise medians and quartiles.
- `main_results.png`: curvature, ECC, and SECC distances to the unperturbed
  fixed-connectivity baseline.
- `reference_errors.png`: errors to analytic curvature and smooth ECC/SECC
  references.
- `perturbation_variability.png`: pairwise variability among coordinate-noise
  realizations.
- `displacement_diagnostics.png`: requested and realized displacement scales.
- `mesh_quality.png`: triangle angles, face-normal reversals, and degeneracy.
- `ecc_jump_diagnostics.png`: exact and sampled ECC jump complexity.
- `ecc_curve_comparison_gaussian.png`: direct smooth-reference, unperturbed,
  and perturbed Gaussian-curvature ECC overlays.
- `ecc_curve_comparison_mean.png`: the corresponding signed-mean-curvature ECC
  overlays.
- `calculation_cache/`: content-addressed arrays and realization metadata.

CSV tables and plots have signature sidecars and are not rewritten on an
unchanged warm run. Cache arrays and signature sidecars are ignored by Git.

## Current smoke-run reading

The realized coordinate RMS follows the requested scale exactly, confirming
that magnitudes are comparable across surfaces and refinement levels.
Curvature distance from the unperturbed baseline increases strongly with noise
and is larger on finer meshes at the same edge-relative noise scale. ECC and
SECC usually change more slowly at the smallest scales, sometimes remaining
unchanged until perturbations reorder filtration events, but they also become
large at the upper scales.

The earlier `0.1` stress endpoint was removed because it overwhelmed the ECC
shape and damaged mesh quality. The current range ends at `0.005` so that the
experiment resolves the transition within the small-coordinate-noise regime.
The smoke run is not, by itself, a final statistical stability conclusion.

## Main insight: why small coordinate noise changes the ECC

The experiment does not change mesh connectivity, so the ordinary Euler
characteristic `V - E + F` and the final value of the filtration remain fixed.
The large changes occur in the intermediate curvature-filtered ECC because two
sensitivity mechanisms are composed.

First, discrete curvature estimation amplifies high-frequency coordinate
noise. Mean curvature is obtained from a cotangent-Laplacian expression and is
therefore derivative-like. Gaussian curvature is an angle defect divided by a
small mixed vertex area. If a typical edge length is `h` and coordinate noise
has size `epsilon * h`, a useful heuristic is that mean-curvature noise can
scale like `epsilon / h`, while Gaussian-curvature noise can be even more
sensitive because an angle error is divided by an area of order `h^2`. This
explains why a fixed edge-relative perturbation can have a larger effect after
mesh refinement.

Second, the lower-star ECC depends on the ordering and equality of vertex
curvature values. A small curvature change can reorder many vertices and move
the entry thresholds of their incident edges and triangles. It can also split
one simultaneous cancellation into many separated filtration events. The ECC
may consequently develop large intermediate oscillations even though all
events still cancel to the same final Euler characteristic.

Constant and nearly constant curvature examples expose this especially
clearly. The helicoid has theoretical mean curvature `H = 0`, and the
pseudosphere has constant Gaussian curvature `K = -1`. Their ideal filtrations
contain large groups of tied events. Independent coordinate perturbations break
those ties, spreading the events over an interval and producing a much more
complicated ECC.

The current noise model also matters. It adds independent perturbations to
every coordinate of every vertex. This is spatially uncorrelated,
high-frequency roughness, which curvature estimators are designed to detect.
It is more adverse than a smooth physical deformation or spatially correlated
measurement error.

The central interpretation is therefore that the current result combines two
effects:

```text
coordinate noise -> curvature-estimator error -> filtration reordering -> ECC change
```

It does not show that mesh topology is unstable, and it does not yet identify
how much of the observed change comes from curvature estimation versus the ECC
construction itself.

## Next steps

1. **Isolate ECC sensitivity.** Keep the mesh geometry fixed and perturb the
   scalar curvature values directly. Comparing this control with the current
   coordinate-noise experiment will separate lower-star filtration sensitivity
   from curvature-estimator sensitivity.
2. **Isolate curvature-estimator sensitivity.** Continue perturbing coordinates
   but report curvature error before constructing the ECC, including its
   dependence on edge length and subdivision level. Fit empirical powers of
   `h` to test the expected noise amplification.
3. **Test realistic perturbation models.** Compare independent coordinate noise
   with normal-direction noise, spatially correlated noise, smooth displacement
   fields, and perturbations projected back to the analytic surface.
4. **Evaluate robust curvature estimates.** Repeat the ECC calculation after
   mesh denoising, neighborhood averaging, or local polynomial curvature
   fitting. This tests whether regularization restores ECC stability without
   erasing genuine curvature structure.
5. **Study curvature ties explicitly.** Record the number and multiplicity of
   near-equal curvature values, then measure how perturbations split those
   groups into ECC events. The helicoid and pseudosphere are the primary control
   cases.
6. **Compare ECC with SECC.** Use baseline, smooth-reference, and pairwise
   distances to determine whether cumulative centering reduces sensitivity to
   event-level oscillations.
7. **Run the full experiment after the controls are fixed.** The current
   five-scale smoke run validates the pipeline and identifies the sensitive
   regime. The twelve-repetition full run should be used for final uncertainty
   estimates only after selecting the most physically meaningful noise and
   curvature-estimation models.

## Reading the direct ECC plots

The two `ecc_curve_comparison_*.png` figures are the most direct visual answer
to the experiment question. In every panel:

- the thick black step curve is the smooth reference ECC;
- the dashed grey step curve is the unperturbed discrete mesh ECC;
- each colored step curve is the pointwise median over repeated coordinate
  perturbations at one of the five magnitudes; and
- the translucent ribbon is the interquartile range across those repetitions.

Rows identify surfaces and columns identify subdivision levels. The horizontal
axis is the physical Gaussian (`K`) or mean (`H`) curvature threshold, and the
vertical axis is the Euler characteristic. A colored curve that remains close
to the dashed grey curve indicates stability to coordinate noise; closeness to
the black curve indicates agreement with the smooth reference. The smooth
reference is numerical for most surfaces and theoretical for the helicoid, as
described in `e02`.
