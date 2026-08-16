from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(
    "experiments/e05_vertex-perturbation-stability/analysis/"
    "vertex_perturbation_stability.py"
)
SPEC = importlib.util.spec_from_file_location("vertex_perturbation_stability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_coordinate_perturbation_is_reproducible_centered_and_scaled() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    first, displacements = module.perturb_vertices(
        vertices, scale=0.03, reference_edge_length=2.0, seed=41
    )
    second, second_displacements = module.perturb_vertices(
        vertices, scale=0.03, reference_edge_length=2.0, seed=41
    )

    assert np.array_equal(first, second)
    assert np.array_equal(displacements, second_displacements)
    assert np.allclose(np.mean(displacements, axis=0), 0.0, atol=1e-15)
    assert np.allclose(
        np.sqrt(np.mean(displacements**2, axis=0)) / 2.0,
        0.03,
    )


def test_zero_perturbation_is_identity() -> None:
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    perturbed, displacements = module.perturb_vertices(
        vertices, scale=0.0, reference_edge_length=1.0, seed=2
    )

    assert np.array_equal(perturbed, vertices)
    assert np.count_nonzero(displacements) == 0
    assert not np.shares_memory(perturbed, vertices)


def test_mesh_change_statistics_detects_face_normal_reversal() -> None:
    reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reversed_coordinates = reference[[0, 2, 1]]
    faces = np.asarray([[0, 1, 2]])

    statistics = module.mesh_change_statistics(
        reference, reversed_coordinates, faces, reference_edge_length=1.0
    )

    assert statistics["normal_reversal_fraction"] == 1.0
    assert statistics["degenerate_face_fraction"] == 0.0


def test_representative_plot_scales_span_configured_range() -> None:
    scales = (0.0001, 0.001325, 0.00255, 0.003775, 0.005)

    selected = module.representative_plot_scales(scales)

    assert selected == scales


@pytest.mark.parametrize(
    "config_name", ["smoke.toml", "vertex_perturbation_stability.toml"]
)
def test_repository_configs_define_five_evenly_spaced_scales(config_name: str) -> None:
    path = SCRIPT.parents[1] / "config" / config_name
    config = module.load_config(path.resolve())

    assert len(config.perturbation_scales) == 5
    assert config.perturbation_scales == tuple(sorted(config.perturbation_scales))
    assert np.allclose(np.diff(config.perturbation_scales), 0.001225)
    assert config.noise_model == module.NOISE_MODEL
    assert config.scale_reference == module.SCALE_REFERENCE


def test_tiny_end_to_end_run_writes_tables_and_plots(tmp_path: Path) -> None:
    source_config = tmp_path / "source.toml"
    source_config.write_text(
        "\n".join(
            [
                'curvatures = ["gaussian_curvature"]',
                'output_dir = "generated_data"',
                'results_dir = "source_results"',
                "histogram_bins = 8",
                "",
                "[reference]",
                'name = "smooth_reference"',
                "u_segments = 8",
                "v_segments = 8",
                "",
                "[[mesh_levels]]",
                'name = "tiny"',
                "u_segments = 4",
                "v_segments = 4",
                "",
                "[[surfaces]]",
                'name = "product_sine_graph"',
                'kind = "product_sine_graph"',
                "domain_size = 2.0",
                "amplitude = 1.0",
                "frequency = 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    config = module.Config(
        source_config=source_config,
        results_dir=results_dir,
        subdivision_levels=2,
        repetitions=2,
        random_seed=7,
        perturbation_scales=(0.0001, 0.0003, 0.001, 0.003, 0.01),
        noise_model=module.NOISE_MODEL,
        scale_reference=module.SCALE_REFERENCE,
        sample_count=21,
        interval_padding_fraction=0.05,
        constant_interval_padding=1.0,
        base_levels={"product_sine_graph": (3, 3)},
    )

    outputs = module.run(config)

    assert all(path.exists() for path in outputs)
    assert all(
        path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for path in outputs
        if path.suffix == ".png"
    )
    assert (results_dir / "ecc_curve_comparison_gaussian.png").exists()
    with (results_dir / "perturbation_realizations.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        realizations = list(csv.DictReader(handle))
    with (results_dir / "perturbation_pairwise.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        pairwise = list(csv.DictReader(handle))

    assert len(realizations) == 22
    assert len(pairwise) == 10
    assert {row["face_count"] for row in realizations} == {"18", "72"}
    assert all(
        float(row["realized_coordinate_std"])
        == pytest.approx(float(row["perturbation_scale"]))
        for row in realizations
    )
