from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "e02_parametric-mesh-ecc"
    / "analysis"
    / "parametric_mesh_ecc.py"
)

spec = importlib.util.spec_from_file_location("parametric_mesh_ecc", SCRIPT_PATH)
parametric_mesh_ecc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = parametric_mesh_ecc
spec.loader.exec_module(parametric_mesh_ecc)


def test_closed_torus_lower_star_ecc_finishes_with_chi_zero() -> None:
    surface = parametric_mesh_ecc.SurfaceSpec(
        name="torus",
        kind="doubly_elliptic_torus",
        parameters={"R": 2.0, "r": 0.5, "a": 1.2, "b": 0.7},
    )
    level = parametric_mesh_ecc.MeshLevel("tiny", u_segments=8, v_segments=8)

    mesh = parametric_mesh_ecc.generate_mesh(surface, level)
    curvature = parametric_mesh_ecc.signed_vertex_mean_curvature(
        mesh.vertices, mesh.faces
    )
    curve, _ = parametric_mesh_ecc.lower_star_ecc(curvature, mesh.faces)

    assert np.isfinite(curvature).all()
    assert parametric_mesh_ecc.euler_characteristic(len(mesh.vertices), mesh.faces) == 0
    assert curve[-1]["chi_after"] == 0


def test_open_product_sine_graph_lower_star_ecc_finishes_with_chi_one() -> None:
    surface = parametric_mesh_ecc.SurfaceSpec(
        name="sine",
        kind="product_sine_graph",
        parameters={"domain_size": 1.0, "amplitude": 0.25, "frequency": 1.0},
    )
    level = parametric_mesh_ecc.MeshLevel("tiny", u_segments=6, v_segments=5)

    mesh = parametric_mesh_ecc.generate_mesh(surface, level)
    curvature = parametric_mesh_ecc.signed_vertex_mean_curvature(
        mesh.vertices, mesh.faces
    )
    curve, _ = parametric_mesh_ecc.lower_star_ecc(curvature, mesh.faces)

    assert np.isfinite(curvature).all()
    assert parametric_mesh_ecc.euler_characteristic(len(mesh.vertices), mesh.faces) == 1
    assert curve[-1]["chi_after"] == 1


def test_helicoid_mesh_mean_curvature_uses_libigl_estimator() -> None:
    surface = parametric_mesh_ecc.SurfaceSpec(
        name="helicoid",
        kind="helicoid_patch",
        parameters={
            "rho_max": 1.0,
            "theta_min": -1.0,
            "theta_max": 1.0,
            "alpha": 1.0,
        },
    )
    level = parametric_mesh_ecc.MeshLevel("tiny", u_segments=4, v_segments=4)
    mesh = parametric_mesh_ecc.generate_mesh(surface, level)

    curvature, source = parametric_mesh_ecc.mesh_level_curvature(
        surface, level, mesh, curvature="mean_curvature"
    )
    curve, _ = parametric_mesh_ecc.lower_star_ecc(curvature, mesh.faces)

    assert source == "libigl_cotangent_voronoi_estimator"
    assert np.isfinite(curvature).all()
    assert np.max(np.abs(curvature)) > 0.0
    assert curve[-1]["chi_after"] == 1


def test_pseudosphere_patch_has_expected_curvature_and_topology() -> None:
    surface = parametric_mesh_ecc.SurfaceSpec(
        name="pseudosphere",
        kind="pseudosphere_patch",
        parameters={"radius": 2.0, "u_min": 0.2, "u_max": 2.0},
    )
    level = parametric_mesh_ecc.MeshLevel("tiny", u_segments=8, v_segments=12)
    mesh = parametric_mesh_ecc.generate_mesh(surface, level)
    parameters = parametric_mesh_ecc.surface_grid_parameters(surface, level)
    gaussian = np.asarray([
        parametric_mesh_ecc.smooth_gaussian_curvature(surface, u, theta)
        for u, theta in parameters
    ])
    mean = np.asarray([
        parametric_mesh_ecc.smooth_signed_mean_curvature(surface, u, theta)
        for u, theta in parameters
    ])

    assert len(parameters) == len(mesh.vertices)
    assert parametric_mesh_ecc.euler_characteristic(len(mesh.vertices), mesh.faces) == 0
    assert np.allclose(gaussian, -0.25)
    assert np.isfinite(mean).all()
    assert parametric_mesh_ecc.theoretical_ecc_curve(
        surface, "gaussian_curvature"
    ) is None


def test_curvature_normalization_is_symmetric_about_zero() -> None:
    negative = parametric_mesh_ecc.zero_centered_norm(np.asarray([-1.0, -1.0]))
    positive = parametric_mesh_ecc.zero_centered_norm(np.asarray([2.0, 3.0]))
    zero = parametric_mesh_ecc.zero_centered_norm(np.asarray([0.0]))

    assert (negative.vmin, negative.vmax) == (-1.0, 1.0)
    assert (positive.vmin, positive.vmax) == (-3.0, 3.0)
    assert (zero.vmin, zero.vmax) == (-1.0, 1.0)
    assert negative(0.0) == positive(0.0) == zero(0.0) == 0.5


def test_ellipsoid_is_closed_and_matches_analytic_curvature() -> None:
    surface = parametric_mesh_ecc.SurfaceSpec(
        name="ellipsoid",
        kind="ellipsoid",
        parameters={"a": 2.0, "b": 1.5, "c": 1.0},
    )
    level = parametric_mesh_ecc.MeshLevel("tiny", u_segments=16, v_segments=10)
    mesh = parametric_mesh_ecc.generate_mesh(surface, level)
    parameters = parametric_mesh_ecc.surface_grid_parameters(surface, level)
    analytic = np.asarray(
        [
            parametric_mesh_ecc.smooth_signed_mean_curvature(surface, *parameter)
            for parameter in parameters
        ]
    )
    discrete = parametric_mesh_ecc.signed_vertex_mean_curvature(
        mesh.vertices, mesh.faces
    )

    assert len(parameters) == len(mesh.vertices)
    assert parametric_mesh_ecc.euler_characteristic(len(mesh.vertices), mesh.faces) == 2
    assert np.all(analytic > 0.0)
    assert np.corrcoef(analytic, discrete)[0, 1] > 0.99


def test_config_run_writes_three_mesh_level_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'curvatures = ["gaussian_curvature"]',
                'output_dir = "data"',
                'results_dir = "results"',
                "histogram_bins = 8",
                "",
                "[reference]",
                'name = "smooth_reference"',
                "u_segments = 10",
                "v_segments = 10",
                "",
                "[[mesh_levels]]",
                'name = "coarse"',
                "u_segments = 4",
                "v_segments = 4",
                "",
                "[[mesh_levels]]",
                'name = "medium"',
                "u_segments = 6",
                "v_segments = 6",
                "",
                "[[mesh_levels]]",
                'name = "fine"',
                "u_segments = 8",
                "v_segments = 8",
                "",
                "[[surfaces]]",
                'name = "helicoid"',
                'kind = "helicoid_patch"',
                "rho_max = 1.0",
                "theta_min = -1.0",
                "theta_max = 1.0",
                "alpha = 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = parametric_mesh_ecc.load_experiment_config(config_path)
    outputs = parametric_mesh_ecc.run_config(config)

    result_root = tmp_path / "results" / "gaussian_curvature"
    data_root = tmp_path / "data" / "gaussian_curvature"
    summary_path = result_root / "mesh_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert [row["mesh_level"] for row in rows] == [
        "smooth_reference",
        "coarse",
        "medium",
        "fine",
    ]
    assert {row["final_ecc_euler_characteristic"] for row in rows} == {"1"}
    assert rows[0]["curve_source"] == "smooth_parametrization_reference"
    assert {row["curve_source"] for row in rows[1:]} == {
        "angle_defect_mixed_voronoi_estimator"
    }
    assert (data_root / "helicoid_coarse.poly").exists()
    assert (
        result_root
        / "helicoid"
        / "smooth_reference"
        / "helicoid_smooth_reference_ecc_curve.csv"
    ).exists()
    assert (result_root / "helicoid" / "fine" / "helicoid_fine_ecc_curve.csv").exists()
    assert (result_root / "main_results.png").exists()
    cached_pdf = next(
        (result_root / "surface_assets").glob("helicoid_smooth_*.pdf")
    )
    assert list((result_root / "surface_assets").glob("helicoid_smooth_*.png"))
    assert list((result_root / "surface_assets").glob("helicoid_fine_*.png"))
    assert list((result_root / "calculation_cache").glob("helicoid_*.npz"))
    cached_mtime = cached_pdf.stat().st_mtime_ns
    parametric_mesh_ecc.run_config(config)
    assert cached_pdf.stat().st_mtime_ns == cached_mtime
    assert outputs


def test_config_run_writes_separate_outputs_for_multiple_curvatures(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'curvatures = ["gaussian_curvature", "mean_curvature"]',
                'output_dir = "data"',
                'results_dir = "results"',
                "histogram_bins = 8",
                "",
                "[reference]",
                'name = "smooth_reference"',
                "u_segments = 5",
                "v_segments = 5",
                "",
                "[[mesh_levels]]",
                'name = "coarse"',
                "u_segments = 4",
                "v_segments = 4",
                "",
                "[[surfaces]]",
                'name = "helicoid"',
                'kind = "helicoid_patch"',
                "rho_max = 1.0",
                "theta_min = -1.0",
                "theta_max = 1.0",
                "alpha = 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = parametric_mesh_ecc.load_experiment_config(config_path)
    outputs = parametric_mesh_ecc.run_config(config)

    assert config.curvatures == ("gaussian_curvature", "mean_curvature")
    assert (tmp_path / "results" / "gaussian_curvature" / "main_results.png").exists()
    assert (tmp_path / "results" / "mean_curvature" / "main_results.png").exists()
    assert (tmp_path / "data" / "mean_curvature" / "helicoid_coarse.poly").exists()
    assert outputs
