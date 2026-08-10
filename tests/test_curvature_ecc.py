from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "e01_curvature-ecc"
    / "analysis"
    / "curvature_ecc.py"
)
GENERATOR_PATH = SCRIPT_PATH.with_name("generate_dented_sphere.py")
FIXTURE_PATH = SCRIPT_PATH.parents[1] / "data" / "test.poly"

spec = importlib.util.spec_from_file_location("curvature_ecc", SCRIPT_PATH)
curvature_ecc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = curvature_ecc
spec.loader.exec_module(curvature_ecc)

generator_spec = importlib.util.spec_from_file_location(
    "generate_dented_sphere", GENERATOR_PATH
)
generate_dented_sphere = importlib.util.module_from_spec(generator_spec)
assert generator_spec.loader is not None
sys.modules[generator_spec.name] = generate_dented_sphere
generator_spec.loader.exec_module(generate_dented_sphere)


def test_builds_sublevel_filtration_and_ecc(tmp_path: Path) -> None:
    poly_path = tmp_path / "example.poly"
    poly_path.write_text(
        "\n".join(
            [
                "POINTS",
                "1: 0 0 0",
                "2: 1 0 0",
                "3: 0 1 0",
                "4: 1 1 0",
                "POLYS",
                "1: 1 2 3 < c(1, 1, 1, 2.0)",
                "2: 2 3 4 < c(1, 1, 1, 5.0)",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    triangles = curvature_ecc.parse_poly_file(poly_path)
    simplices = curvature_ecc.build_filtered_complex(triangles)
    curve = curvature_ecc.grouped_ecc_curve(simplices)

    edge_filtrations = {
        simplex.vertices: simplex.filtration
        for simplex in simplices
        if simplex.dimension == 1
    }
    vertex_filtrations = {
        simplex.vertices: simplex.filtration
        for simplex in simplices
        if simplex.dimension == 0
    }

    assert edge_filtrations == {
        (1, 2): 2.0,
        (1, 3): 2.0,
        (2, 3): 2.0,
        (2, 4): 5.0,
        (3, 4): 5.0,
    }
    assert vertex_filtrations == {
        (1,): 2.0,
        (2,): 2.0,
        (3,): 2.0,
        (4,): 5.0,
    }
    assert curve == [
        {"filtration": 2.0, "delta_chi": 1, "chi_after": 1},
        {"filtration": 5.0, "delta_chi": 0, "chi_after": 1},
    ]


def test_empty_polys_section_gives_empty_curve(tmp_path: Path) -> None:
    poly_path = tmp_path / "empty.poly"
    poly_path.write_text("POINTS\nPOLYS\nEND\n", encoding="utf-8")

    triangles = curvature_ecc.parse_poly_file(poly_path)
    simplices = curvature_ecc.build_filtered_complex(triangles)

    assert triangles == []
    assert simplices == []
    assert curvature_ecc.grouped_ecc_curve(simplices) == []


def test_duplicate_triangle_records_use_minimum_filtration() -> None:
    triangles = [
        curvature_ecc.Triangle(1, (1, 2, 3), 4.0),
        curvature_ecc.Triangle(2, (1, 2, 3), 2.0),
    ]

    simplices = curvature_ecc.build_filtered_complex(triangles)

    assert [simplex for simplex in simplices if simplex.dimension == 2] == [
        curvature_ecc.FilteredSimplex(2, (1, 2, 3), 2.0)
    ]
    assert curvature_ecc.grouped_ecc_curve(simplices) == [
        {"filtration": 2.0, "delta_chi": 1, "chi_after": 1}
    ]


def test_triangle_filtration_smoothing_uses_edge_neighbors() -> None:
    triangles = [
        curvature_ecc.Triangle(1, (1, 2, 3), 0.0),
        curvature_ecc.Triangle(2, (2, 3, 4), 6.0),
        curvature_ecc.Triangle(3, (3, 4, 5), 12.0),
        curvature_ecc.Triangle(4, (6, 7, 8), 30.0),
    ]

    smoothed = curvature_ecc.smooth_triangle_filtrations(triangles)

    assert {triangle.vertices: triangle.filtration for triangle in smoothed} == {
        (1, 2, 3): pytest.approx(3.0),
        (2, 3, 4): pytest.approx(6.0),
        (3, 4, 5): pytest.approx(9.0),
        (6, 7, 8): pytest.approx(30.0),
    }

    simplices = curvature_ecc.build_filtered_complex(
        triangles,
        smooth_triangle_filtration=True,
    )
    triangle_filtrations = {
        simplex.vertices: simplex.filtration
        for simplex in simplices
        if simplex.dimension == 2
    }

    assert triangle_filtrations == {
        (1, 2, 3): pytest.approx(3.0),
        (2, 3, 4): pytest.approx(6.0),
        (3, 4, 5): pytest.approx(9.0),
        (6, 7, 8): pytest.approx(30.0),
    }


def test_alpha_is_read_from_fourth_rgba_component(tmp_path: Path) -> None:
    poly_path = tmp_path / "alpha.poly"
    poly_path.write_text(
        "\n".join(
            [
                "POINTS",
                "POLYS",
                "1: 101 102 103 < c(10, 20, 30, 0.125)",
                "2: 101 103 104 < c(1, 1, 1, -6.63027e-05)",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    triangles = curvature_ecc.parse_poly_file(poly_path)

    assert [triangle.filtration for triangle in triangles] == [
        pytest.approx(0.125),
        pytest.approx(-6.63027e-05),
    ]


def test_polygon_records_are_fan_triangulated(tmp_path: Path) -> None:
    poly_path = tmp_path / "quad.poly"
    poly_path.write_text(
        "\n".join(
            [
                "POINTS",
                "POLYS",
                "7: 1 2 3 4 < c(1, 1, 1, -0.25)",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    triangles = curvature_ecc.parse_poly_file(poly_path)

    assert triangles == [
        curvature_ecc.Triangle(7, (1, 2, 3), -0.25),
        curvature_ecc.Triangle(7, (1, 3, 4), -0.25),
    ]


def test_degenerate_triangles_are_skipped_by_default(tmp_path: Path) -> None:
    poly_path = tmp_path / "degenerate.poly"
    poly_path.write_text(
        "\n".join(
            [
                "POINTS",
                "POLYS",
                "1: 1 1 2 < c(1, 1, 1, 3.0)",
                "2: 1 2 3 < c(1, 1, 1, 4.0)",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = curvature_ecc.parse_poly_data(poly_path)

    assert result.skipped_degenerate_triangles == 1
    assert result.triangles == [curvature_ecc.Triangle(2, (1, 2, 3), 4.0)]
    with pytest.raises(ValueError, match="Degenerate triangle"):
        curvature_ecc.parse_poly_file(poly_path, skip_degenerate=False)


def test_subtriangular_polygon_records_are_skipped_by_default(
    tmp_path: Path,
) -> None:
    poly_path = tmp_path / "subtriangular.poly"
    poly_path.write_text(
        "\n".join(
            [
                "POINTS",
                "POLYS",
                "1: 7",
                "2: 1 2 < c(1, 1, 1, 3.0)",
                "3: 1 2 3 < c(1, 1, 1, 4.0)",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = curvature_ecc.parse_poly_data(poly_path)

    assert result.skipped_degenerate_triangles == 2
    assert result.triangles == [curvature_ecc.Triangle(3, (1, 2, 3), 4.0)]
    with pytest.raises(ValueError, match="Degenerate polygon"):
        curvature_ecc.parse_poly_file(poly_path, skip_degenerate=False)


def test_z_min_filter_removes_low_vertices_and_incident_triangles(
    tmp_path: Path,
) -> None:
    poly_path = tmp_path / "z-filter.poly"
    poly_path.write_text(
        "\n".join(
            [
                "POINTS",
                "1: 0 0 -1",
                "2: 1 0 0",
                "3: 0 1 0.5",
                "4: 1 1 0.75",
                "5: 2 1 1.0",
                "POLYS",
                "1: 1 2 3 < c(1, 1, 1, 2.0)",
                "2: 2 3 4 < c(1, 1, 1, 5.0)",
                "3: 3 4 5 < c(1, 1, 1, 7.0)",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parse_result = curvature_ecc.parse_poly_data(poly_path)
    filtered = curvature_ecc.filter_parse_result_by_z_min(
        poly_path, parse_result, z_min=0.5
    )

    assert filtered.triangles == [curvature_ecc.Triangle(3, (3, 4, 5), 7.0)]
    assert filtered.removed_vertices_below_z_threshold == 2
    assert filtered.triangles_removed_by_z_threshold == 2
    assert filtered.z_min == 0.5


def test_config_run_writes_dented_sphere_outputs(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    input_path = data_dir / "test.poly"
    input_path.write_text(FIXTURE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = config_dir / "test.toml"
    config_path.write_text(
        "\n".join(
            [
                'input = "../data/test.poly"',
                'results_dir = "../results"',
                'output_name = "dented-sphere"',
                "skip_degenerate = true",
                "surface_plots = true",
                "histogram_bins = 12",
                "histogram_symlog = false",
                "plot_percentile = 98",
                "z_min = -1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = curvature_ecc.load_run_config(config_path)
    outputs = curvature_ecc.run_config(config)

    assert config.input_path == input_path.resolve()
    assert config.results_dir == (tmp_path / "results").resolve()
    assert config.output_name == "dented-sphere"
    assert config.output_dir == (tmp_path / "results" / "dented-sphere").resolve()
    assert config.histogram_symlog is False
    assert config.plot_percentile == 98.0
    assert config.z_min == -1.0
    assert all(path.parent == config.output_dir for path in outputs.values())
    assert outputs["ecc_plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert outputs["surface_plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        outputs["curvature_histogram_plot"]
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )
    assert outputs["ecc_events"].read_text(encoding="utf-8").splitlines()[0] == (
        "filtration,sign,dimension,simplex"
    )
    curve_lines = outputs["ecc_curve"].read_text(encoding="utf-8").splitlines()
    assert curve_lines[0] == "filtration,chi"
    assert curve_lines[-1].endswith(",2")

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["simplex_counts"] == {"0": 26, "1": 72, "2": 48}
    assert summary["final_euler_characteristic"] == 2
    assert summary["min_filtration"] == pytest.approx(0.68770407072)
    assert summary["max_filtration"] == pytest.approx(1.14627966299)
    assert summary["plot_percentile"] == 98.0
    assert summary["z_min"] == -1.0
    assert summary["triangles_removed_by_z_threshold"] == 0

    with outputs["curvature_histogram"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert sum(int(row["count"]) for row in rows) == 48
    assert set(rows[0]) == {"bin_left", "bin_right", "count"}


def test_config_rejects_nonpositive_histogram_bins(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        "\n".join(
            [
                'input = "data.poly"',
                'results_dir = "results"',
                'output_name = "bad"',
                "histogram_bins = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="histogram_bins must be positive"):
        curvature_ecc.load_run_config(config_path)


def test_config_loads_triangle_smoothing_option(tmp_path: Path) -> None:
    config_path = tmp_path / "smooth.toml"
    config_path.write_text(
        "\n".join(
            [
                'input = "data.poly"',
                'results_dir = "results"',
                'output_name = "smooth"',
                "smooth_triangle_filtration = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = curvature_ecc.load_run_config(config_path)

    assert config.smooth_triangle_filtration is True


def test_config_defaults_to_symlog_histogram(tmp_path: Path) -> None:
    config_path = tmp_path / "histogram-default.toml"
    config_path.write_text(
        "\n".join(
            [
                'input = "data.poly"',
                'results_dir = "results"',
                'output_name = "histogram-default"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = curvature_ecc.load_run_config(config_path)

    assert config.histogram_symlog is True


def test_central_percentile_limits() -> None:
    values = list(range(101))

    assert curvature_ecc.central_percentile_limits(values, None) is None
    assert curvature_ecc.central_percentile_limits(values, 100) is None
    assert curvature_ecc.central_percentile_limits(values, 80) == pytest.approx(
        (10.0, 90.0)
    )


def test_config_rejects_invalid_plot_percentile(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-percentile.toml"
    config_path.write_text(
        "\n".join(
            [
                'input = "data.poly"',
                'results_dir = "results"',
                'output_name = "bad-percentile"',
                "plot_percentile = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="plot_percentile"):
        curvature_ecc.load_run_config(config_path)


def test_config_rejects_nested_output_name(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-output-name.toml"
    config_path.write_text(
        "\n".join(
            [
                'input = "data.poly"',
                'results_dir = "results"',
                'output_name = "nested/name"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="single directory name"):
        curvature_ecc.load_run_config(config_path)


def test_dented_sphere_generator_approximates_unit_sphere_mean_curvature() -> None:
    mesh = generate_dented_sphere.generate_dented_sphere_mesh(
        lat_segments=12,
        lon_segments=24,
        radius=1.0,
        dent_depth=0.0,
        dent_width=0.48,
    )
    curvature = generate_dented_sphere.triangle_mean_curvature(
        mesh.vertices, mesh.faces
    )

    assert curvature.mean() == pytest.approx(1.0, abs=0.02)
    assert np_median(curvature) == pytest.approx(1.0, abs=0.02)


def test_dented_sphere_generator_writes_poly_and_ecc_config(tmp_path: Path) -> None:
    config_path = tmp_path / "generator.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data"',
                "radius = 1.0",
                "dent_depth = 0.42",
                "dent_width = 0.48",
                "write_ecc_configs = true",
                'ecc_config_dir = "config"',
                'ecc_results_dir = "results"',
                "skip_degenerate = true",
                "histogram_bins = 9",
                "",
                "[[stages]]",
                'name = "tiny"',
                "lat_segments = 4",
                "lon_segments = 8",
                "surface_plots = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = generate_dented_sphere.load_generator_config(config_path)
    outputs = generate_dented_sphere.generate_from_config(config)

    poly_path = tmp_path / "data" / "tiny.poly"
    ecc_config_path = tmp_path / "config" / "tiny.toml"
    assert outputs == [poly_path.resolve(), ecc_config_path.resolve()]
    assert poly_path.exists()
    assert ecc_config_path.read_text(encoding="utf-8").splitlines() == [
        'input = "../data/tiny.poly"',
        'results_dir = "../results"',
        'output_name = "tiny"',
        "skip_degenerate = true",
        "surface_plots = true",
        "histogram_bins = 9",
    ]

    triangles = curvature_ecc.parse_poly_file(poly_path)
    assert len(triangles) == 48
    assert all(np_isfinite(triangle.filtration) for triangle in triangles)


def np_median(values: object) -> float:
    import numpy as np

    return float(np.median(values))


def np_isfinite(value: float) -> bool:
    import numpy as np

    return bool(np.isfinite(value))
