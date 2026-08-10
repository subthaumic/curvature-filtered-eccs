from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(
    "experiments/e04_triangulation-stability/analysis/subdivision_stability.py"
)
SPEC = importlib.util.spec_from_file_location("subdivision_stability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


@pytest.mark.parametrize(
    ("kind", "parameters", "segments", "euler_characteristic"),
    [
        ("doubly_elliptic_torus", {}, (4, 4), 0),
        ("ellipsoid", {"a": 2.0, "b": 1.5, "c": 1.0}, (6, 3), 2),
        ("product_sine_graph", {"domain_size": 2.0}, (3, 3), 1),
        ("helicoid_patch", {}, (3, 3), 1),
        ("pseudosphere_patch", {}, (3, 6), 0),
    ],
)
def test_projected_midpoint_subdivision(
    kind: str,
    parameters: dict[str, float],
    segments: tuple[int, int],
    euler_characteristic: int,
) -> None:
    surface = module.e02.SurfaceSpec(kind, kind, parameters)
    mesh = module.initial_parametric_mesh(surface, *segments)
    subdivided = module.projected_midpoint_subdivision(surface, mesh)
    edge_count = len(module.e02.mesh_edges(mesh.faces))
    assert len(subdivided.faces) == 4 * len(mesh.faces)
    assert len(subdivided.vertices) == len(mesh.vertices) + edge_count
    assert (
        module.e02.euler_characteristic(len(subdivided.vertices), subdivided.faces)
        == euler_characteristic
    )
    assert np.allclose(
        subdivided.vertices, module.push_forward(surface, subdivided.parameters)
    )


def test_domain_flips_realize_requested_fraction() -> None:
    surface = module.e02.SurfaceSpec(
        "product_sine_graph", "product_sine_graph", {"domain_size": 2.0}
    )
    mesh = module.initial_parametric_mesh(surface, 5, 5)
    faces, count, interior_count = module.flip_domain_edges(
        surface, mesh, fraction=0.1, seed=4
    )
    assert count == round(0.1 * interior_count)
    assert module.e02.euler_characteristic(len(mesh.vertices), faces) == 1


def test_zero_domain_flip_is_identity() -> None:
    surface = module.e02.SurfaceSpec("torus", "doubly_elliptic_torus", {})
    mesh = module.initial_parametric_mesh(surface, 4, 4)
    faces, count, _ = module.flip_domain_edges(surface, mesh, fraction=0.0, seed=2)
    assert count == 0
    assert np.array_equal(faces, mesh.faces)


def test_quality_controlled_flips_do_not_degrade_global_minimum_angle() -> None:
    surface = module.e02.SurfaceSpec(
        "product_sine_graph", "product_sine_graph", {"domain_size": 2.0}
    )
    mesh = module.initial_parametric_mesh(surface, 5, 5)
    baseline_minimum = np.min(
        module.triangle_minimum_angles_degrees(mesh.vertices, mesh.faces)
    )
    faces, count, interior_count = module.flip_domain_edges(
        surface,
        mesh,
        fraction=0.1,
        seed=4,
        regime="quality_controlled",
        quality_min_angle_degrees=10.0,
    )
    flipped_minimum = np.min(
        module.triangle_minimum_angles_degrees(mesh.vertices, faces)
    )
    assert count == round(0.1 * interior_count)
    assert flipped_minimum + 1e-10 >= min(baseline_minimum, 10.0)


def test_vectorized_gaussian_curvature_matches_shared_estimator() -> None:
    surface = module.e02.SurfaceSpec(
        "product_sine_graph", "product_sine_graph", {"domain_size": 2.0}
    )
    mesh = module.initial_parametric_mesh(surface, 5, 5)
    curvature, areas = module.vertex_gaussian_curvature_and_areas(
        mesh.vertices, mesh.faces
    )
    expected_curvature = module.e02.vertex_gaussian_curvature(mesh.vertices, mesh.faces)
    expected_areas = np.zeros(len(mesh.vertices), dtype=float)
    for face in mesh.faces:
        a, b, c = (int(value) for value in face)
        va, vb, vc = mesh.vertices[[a, b, c]]
        face_area = 0.5 * np.linalg.norm(np.cross(vb - va, vc - va))
        cotangents = (
            module.e02.cotangent(vb - va, vc - va),
            module.e02.cotangent(va - vb, vc - vb),
            module.e02.cotangent(va - vc, vb - vc),
        )
        module.e02.add_mixed_voronoi_areas(
            expected_areas,
            (a, b, c),
            (va, vb, vc),
            cotangents,
            face_area,
        )
    assert np.allclose(curvature, expected_curvature)
    assert np.allclose(areas, expected_areas)


def test_normalized_secc_returns_to_zero() -> None:
    curve = [
        {"filtration": -1.0, "delta_chi": 1, "chi_after": 1},
        {"filtration": 0.5, "delta_chi": -1, "chi_after": 0},
    ]
    _, secc = module.normalized_curve_arrays(curve, np.linspace(-2.0, 2.0, 101))
    assert secc[0] == 0.0
    assert abs(secc[-1]) < 1e-12


def test_ecc_jump_diagnostics_distinguish_nonzero_jumps() -> None:
    vertex_filtration = np.asarray((0.0, 2.0, 3.0, 1.0))
    faces = np.asarray(((0, 1, 2), (0, 2, 3)))
    curve, _ = module.e02.lower_star_ecc(vertex_filtration, faces)
    sampled = module.e03.evaluate_ecc(curve, np.linspace(-1.0, 4.0, 21))
    diagnostics = module.ecc_jump_diagnostics(vertex_filtration, faces, sampled)
    exact_deltas = [int(row["delta_chi"]) for row in curve]
    sampled_deltas = np.diff(sampled)
    assert diagnostics["ecc_exact_jump_count"] == np.count_nonzero(exact_deltas)
    assert diagnostics["ecc_exact_total_variation"] == np.sum(np.abs(exact_deltas))
    assert diagnostics["ecc_sampled_jump_count"] == np.count_nonzero(sampled_deltas)
    assert diagnostics["ecc_sampled_total_variation"] == np.sum(np.abs(sampled_deltas))
