"""Projected edge-midpoint-refinement experiment for curvature, ECC, and SECC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def import_script(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e02 = import_script(
    "e04_e02",
    HERE.parents[1] / "e02_parametric-mesh-ecc" / "analysis" / "parametric_mesh_ecc.py",
)
e03 = import_script(
    "e04_e03",
    HERE.parents[1]
    / "e03_parametric-mesh-secc"
    / "analysis"
    / "parametric_mesh_secc.py",
)

CACHE_VERSION = 2
ARTIFACT_VERSION = 6
BASELINE_REGIME = "baseline"
FLIP_REGIMES = ("stress", "quality_controlled")
ERROR_METRICS = (
    "curvature_error_l2",
    "ecc_error_l2",
    "secc_error_l2",
)
ECC_DIAGNOSTIC_METRICS = (
    "ecc_exact_jump_count",
    "ecc_exact_total_variation",
    "ecc_sampled_jump_count",
    "ecc_sampled_total_variation",
)
PAIRWISE_METRICS = (
    "curvature_l2",
    "ecc_l2",
    "secc_l2",
)
MESH_QUALITY_METRICS = (
    "minimum_angle_degrees",
    "angle_q01_degrees",
    "angle_q05_degrees",
)


@dataclass(frozen=True)
class Config:
    source_config: Path
    results_dir: Path
    subdivision_levels: int
    repetitions: int
    random_seed: int
    flip_fractions: tuple[float, ...]
    flip_regimes: tuple[str, ...]
    quality_min_angle_degrees: float
    sample_count: int
    interval_padding_fraction: float
    constant_interval_padding: float
    base_levels: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class ParametricMesh:
    parameters: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class DomainFlipContext:
    regime: str
    interior_count: int
    candidates: tuple[tuple[int, int, int, int, int, int], ...]


@dataclass(frozen=True)
class Realization:
    row: dict[str, object]
    curvature_values: np.ndarray
    weights: np.ndarray
    chi: np.ndarray
    secc: np.ndarray


def initial_parametric_mesh(
    surface: object, u_segments: int, v_segments: int
) -> ParametricMesh:
    level = e02.MeshLevel("subdivision_0", u_segments, v_segments)
    mesh = e02.generate_mesh(surface, level)
    parameters = np.asarray(e02.surface_grid_parameters(surface, level), dtype=float)
    if len(parameters) != len(mesh.vertices):
        raise RuntimeError(f"Parameter/vertex mismatch for {surface.name}")
    projected = push_forward(surface, parameters)
    if not np.allclose(projected, mesh.vertices):
        raise RuntimeError(f"Shared parametrization mismatch for {surface.name}")
    return ParametricMesh(parameters, projected, mesh.faces.copy())


def push_forward(surface: object, parameters: np.ndarray) -> np.ndarray:
    return np.asarray([e02.surface_point(surface, *point) for point in parameters])


def periodic_dimensions(surface: object) -> dict[int, tuple[float, float]]:
    if surface.kind == "doubly_elliptic_torus":
        return {0: (0.0, 2.0 * math.pi), 1: (0.0, 2.0 * math.pi)}
    if surface.kind in {"pseudosphere_patch", "ellipsoid"}:
        return {1: (0.0, 2.0 * math.pi)}
    return {}


def circular_mean(values: np.ndarray, lower: float, upper: float) -> float:
    period = upper - lower
    angles = 2.0 * math.pi * (values - lower) / period
    angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    return lower + period * ((angle / (2.0 * math.pi)) % 1.0)


def parameter_edge_midpoint(surface: object, points: np.ndarray) -> np.ndarray:
    if len(points) != 2:
        raise ValueError("An edge midpoint requires exactly two endpoints")
    if surface.kind == "ellipsoid":
        theta, phi = points[:, 0], points[:, 1]
        sphere = np.column_stack(
            (
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
        )
        midpoint = np.sum(sphere, axis=0)
        midpoint /= np.linalg.norm(midpoint)
        return np.asarray(
            (
                math.acos(float(np.clip(midpoint[2], -1.0, 1.0))),
                math.atan2(float(midpoint[1]), float(midpoint[0])) % (2.0 * math.pi),
            )
        )

    adjusted = points.copy()
    result = np.mean(adjusted, axis=0)
    for dimension, (lower, upper) in periodic_dimensions(surface).items():
        result[dimension] = circular_mean(adjusted[:, dimension], lower, upper)
    return result


def lift_parameters(surface: object, points: np.ndarray) -> np.ndarray:
    lifted = points.copy()
    if surface.kind == "ellipsoid":
        pole = np.isclose(lifted[:, 0], 0.0) | np.isclose(lifted[:, 0], math.pi)
        nonpole = ~pole
        if np.any(pole) and np.any(nonpole):
            lifted[pole, 1] = circular_mean(lifted[nonpole, 1], 0.0, 2.0 * math.pi)
    for dimension, (lower, upper) in periodic_dimensions(surface).items():
        period = upper - lower
        anchor = lifted[0, dimension]
        delta = (lifted[:, dimension] - anchor + 0.5 * period) % period - 0.5 * period
        lifted[:, dimension] = anchor + delta
    return lifted


def projected_midpoint_subdivision(
    surface: object, mesh: ParametricMesh
) -> ParametricMesh:
    subdivision_matrix, new_faces = e02.igl.upsample_matrix(
        mesh.faces, len(mesh.parameters)
    )
    new_parameters = np.empty((subdivision_matrix.shape[0], 2), dtype=float)
    new_parameters[: len(mesh.parameters)] = mesh.parameters
    for vertex in range(len(mesh.parameters), subdivision_matrix.shape[0]):
        endpoints = subdivision_matrix.getrow(vertex).indices
        if len(endpoints) != 2:
            raise RuntimeError("libigl returned a non-edge subdivision vertex")
        new_parameters[vertex] = parameter_edge_midpoint(
            surface, mesh.parameters[endpoints]
        )
    new_vertices = push_forward(surface, new_parameters)
    return ParametricMesh(new_parameters, new_vertices, new_faces)


def triangle_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    first, second, third = vertices[face]
    return np.cross(second - first, third - first)


def edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    incidence: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        a, b, c = (int(value) for value in face)
        for edge in ((a, b), (b, c), (c, a)):
            incidence[tuple(sorted(edge))].append(face_index)
    return incidence


def cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def valid_domain_flip(
    surface: object,
    parameters: np.ndarray,
    a: int,
    b: int,
    c: int,
    d: int,
) -> bool:
    points = parameters[[a, b, c, d]]
    if surface.kind == "ellipsoid" and (
        np.any(np.isclose(points[:, 0], 0.0))
        or np.any(np.isclose(points[:, 0], math.pi))
    ):
        return False
    a_point, b_point, c_point, d_point = lift_parameters(surface, points)
    side_c = cross_2d(b_point - a_point, c_point - a_point)
    side_d = cross_2d(b_point - a_point, d_point - a_point)
    side_a = cross_2d(d_point - c_point, a_point - c_point)
    side_b = cross_2d(d_point - c_point, b_point - c_point)
    tolerance = 1e-13
    return side_c * side_d < -tolerance and side_a * side_b < -tolerance


def triangle_minimum_angles_degrees(
    vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    triangles = vertices[np.asarray(faces, dtype=int)]
    opposite_a = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    opposite_b = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    opposite_c = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    numerators = np.column_stack(
        (
            opposite_b**2 + opposite_c**2 - opposite_a**2,
            opposite_a**2 + opposite_c**2 - opposite_b**2,
            opposite_a**2 + opposite_b**2 - opposite_c**2,
        )
    )
    denominators = np.column_stack(
        (
            2.0 * opposite_b * opposite_c,
            2.0 * opposite_a * opposite_c,
            2.0 * opposite_a * opposite_b,
        )
    )
    cosines = np.divide(
        numerators,
        denominators,
        out=np.ones_like(numerators),
        where=denominators > 0,
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return np.min(angles, axis=1)


def quality_controlled_flip(
    mesh: ParametricMesh,
    candidate: tuple[int, int, int, int, int, int],
    minimum_angle_degrees: float,
) -> bool:
    a, b, first_face, second_face, c, d = candidate
    old_faces = mesh.faces[[first_face, second_face]]
    new_faces = np.asarray(((c, d, b), (d, c, a)), dtype=int)
    old_minimum = float(
        np.min(triangle_minimum_angles_degrees(mesh.vertices, old_faces))
    )
    new_minimum = float(
        np.min(triangle_minimum_angles_degrees(mesh.vertices, new_faces))
    )
    required_minimum = min(old_minimum, minimum_angle_degrees)
    return new_minimum + 1e-10 >= required_minimum


def prepare_domain_flips(
    surface: object,
    mesh: ParametricMesh,
    *,
    regime: str = "stress",
    quality_min_angle_degrees: float = 10.0,
) -> DomainFlipContext:
    if regime not in FLIP_REGIMES:
        raise ValueError(f"Unknown flip regime: {regime}")
    incidence = edge_incidence(mesh.faces)
    interior_count = sum(len(faces) == 2 for faces in incidence.values())
    existing_edges = set(incidence)
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for (a, b), incident_faces in incidence.items():
        if len(incident_faces) != 2:
            continue
        first_face, second_face = incident_faces
        c = next(int(v) for v in mesh.faces[first_face] if v not in (a, b))
        d = next(int(v) for v in mesh.faces[second_face] if v not in (a, b))
        if c == d or tuple(sorted((c, d))) in existing_edges:
            continue
        candidate = (a, b, first_face, second_face, c, d)
        if not valid_domain_flip(surface, mesh.parameters, a, b, c, d):
            continue
        if regime == "quality_controlled" and not quality_controlled_flip(
            mesh, candidate, quality_min_angle_degrees
        ):
            continue
        candidates.append(candidate)
    return DomainFlipContext(regime, interior_count, tuple(candidates))


def apply_domain_flips(
    surface: object,
    mesh: ParametricMesh,
    context: DomainFlipContext,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("flip fraction must lie in [0, 1]")
    target = round(fraction * context.interior_count)
    if target == 0:
        return mesh.faces.copy(), 0, context.interior_count

    rng = np.random.default_rng(seed)
    candidates = list(context.candidates)
    rng.shuffle(candidates)
    selected: list[tuple[int, int, int, int, int, int]] = []
    used_faces: set[int] = set()
    new_diagonals: set[tuple[int, int]] = set()
    for candidate in candidates:
        _, _, first_face, second_face, c, d = candidate
        diagonal = tuple(sorted((c, d)))
        if (
            first_face in used_faces
            or second_face in used_faces
            or diagonal in new_diagonals
        ):
            continue
        selected.append(candidate)
        used_faces.update((first_face, second_face))
        new_diagonals.add(diagonal)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(
            f"{surface.name} ({context.regime}): only {len(selected)} of {target} "
            "domain flips available"
        )

    result = mesh.faces.copy()
    for a, b, first_face, second_face, c, d in selected:
        parent_normal = triangle_normal(mesh.vertices, result[first_face])
        parent_normal += triangle_normal(mesh.vertices, result[second_face])
        replacements = np.asarray(((c, d, b), (d, c, a)), dtype=int)
        for replacement in replacements:
            if np.dot(triangle_normal(mesh.vertices, replacement), parent_normal) < 0:
                replacement[[1, 2]] = replacement[[2, 1]]
        result[first_face], result[second_face] = replacements
    return result, len(selected), context.interior_count


def flip_domain_edges(
    surface: object,
    mesh: ParametricMesh,
    *,
    fraction: float,
    seed: int,
    regime: str = "stress",
    quality_min_angle_degrees: float = 10.0,
) -> tuple[np.ndarray, int, int]:
    return apply_domain_flips(
        surface,
        mesh,
        prepare_domain_flips(
            surface,
            mesh,
            regime=regime,
            quality_min_angle_degrees=quality_min_angle_degrees,
        ),
        fraction=fraction,
        seed=seed,
    )


def mixed_areas_and_angle_sums(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Meyer mixed areas and incident-angle sums."""

    triangles = vertices[faces]
    from_a_to_b = triangles[:, 1] - triangles[:, 0]
    from_a_to_c = triangles[:, 2] - triangles[:, 0]
    from_b_to_a = -from_a_to_b
    from_b_to_c = triangles[:, 2] - triangles[:, 1]
    from_c_to_a = -from_a_to_c
    from_c_to_b = -from_b_to_c
    double_areas = np.linalg.norm(np.cross(from_a_to_b, from_a_to_c), axis=1)
    valid = double_areas > 0.0
    valid_faces = faces[valid]
    double_areas = double_areas[valid]
    vectors = (
        (from_a_to_b[valid], from_a_to_c[valid]),
        (from_b_to_a[valid], from_b_to_c[valid]),
        (from_c_to_a[valid], from_c_to_b[valid]),
    )
    cotangents = np.column_stack(
        [
            np.einsum("ij,ij->i", first, second) / double_areas
            for first, second in vectors
        ]
    )
    angle_values = np.column_stack(
        [
            np.arccos(
                np.clip(
                    np.einsum("ij,ij->i", first, second)
                    / (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)),
                    -1.0,
                    1.0,
                )
            )
            for first, second in vectors
        ]
    )

    mixed_areas = np.zeros(len(vertices), dtype=float)
    angle_sums = np.zeros(len(vertices), dtype=float)
    for local in range(3):
        np.add.at(angle_sums, valid_faces[:, local], angle_values[:, local])

    face_areas = 0.5 * double_areas
    obtuse = np.any(cotangents < 0.0, axis=1)
    if np.any(obtuse):
        obtuse_faces = valid_faces[obtuse]
        obtuse_local = np.argmin(cotangents[obtuse], axis=1)
        for local in range(3):
            shares = np.where(obtuse_local == local, 0.5, 0.25)
            np.add.at(
                mixed_areas,
                obtuse_faces[:, local],
                face_areas[obtuse] * shares,
            )

    acute = ~obtuse
    if np.any(acute):
        acute_faces = valid_faces[acute]
        acute_triangles = triangles[valid][acute]
        squared_ab = np.sum(
            (acute_triangles[:, 0] - acute_triangles[:, 1]) ** 2, axis=1
        )
        squared_ac = np.sum(
            (acute_triangles[:, 0] - acute_triangles[:, 2]) ** 2, axis=1
        )
        squared_bc = np.sum(
            (acute_triangles[:, 1] - acute_triangles[:, 2]) ** 2, axis=1
        )
        acute_cotangents = cotangents[acute]
        contributions = np.column_stack(
            (
                (
                    squared_ab * acute_cotangents[:, 2]
                    + squared_ac * acute_cotangents[:, 1]
                )
                / 8.0,
                (
                    squared_ab * acute_cotangents[:, 2]
                    + squared_bc * acute_cotangents[:, 0]
                )
                / 8.0,
                (
                    squared_ac * acute_cotangents[:, 1]
                    + squared_bc * acute_cotangents[:, 0]
                )
                / 8.0,
            )
        )
        for local in range(3):
            np.add.at(mixed_areas, acute_faces[:, local], contributions[:, local])
    return mixed_areas, angle_sums, valid_faces


def vertex_mixed_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    mixed_areas, _, _ = mixed_areas_and_angle_sums(vertices, faces)
    return mixed_areas


def vertex_gaussian_curvature_and_areas(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mixed_areas, angle_sums, valid_faces = mixed_areas_and_angle_sums(vertices, faces)
    face_edges = np.concatenate(
        (
            valid_faces[:, (0, 1)],
            valid_faces[:, (0, 2)],
            valid_faces[:, (1, 2)],
        ),
        axis=0,
    )
    face_edges.sort(axis=1)
    edges, counts = np.unique(face_edges, axis=0, return_counts=True)
    boundary_vertices = np.unique(edges[counts == 1])
    target_angles = np.full(len(vertices), 2.0 * math.pi)
    target_angles[boundary_vertices] = math.pi
    curvature = np.zeros(len(vertices), dtype=float)
    valid_vertices = mixed_areas > 0.0
    curvature[valid_vertices] = (
        target_angles[valid_vertices] - angle_sums[valid_vertices]
    ) / mixed_areas[valid_vertices]
    return curvature, mixed_areas


def weighted_errors(
    values: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    selected = np.isfinite(values) & np.isfinite(target) & (weights > 0)
    error = values[selected] - target[selected]
    normalized = weights[selected] / np.sum(weights[selected])
    return {
        "l1": float(np.sum(normalized * np.abs(error))),
        "l2": float(np.sqrt(np.sum(normalized * error**2))),
        "linf": float(np.max(np.abs(error))),
        "bias": float(np.sum(normalized * error)),
    }


def normalized_curve_arrays(
    curve: list[dict[str, float | int]], physical_thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    chi = e03.evaluate_ecc(curve, physical_thresholds)
    du = 1.0 / (len(physical_thresholds) - 1)
    mean_chi = float(np.sum(chi[:-1]) * du)
    secc = np.zeros_like(chi)
    secc[1:] = np.cumsum((chi[:-1] - mean_chi) * du)
    return chi, secc


def ecc_jump_diagnostics(
    vertex_filtration: np.ndarray,
    faces: np.ndarray,
    sampled_chi: np.ndarray,
) -> dict[str, float | int]:
    face_edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (0, 2)], faces[:, (1, 2)]), axis=0
    )
    face_edges.sort(axis=1)
    edges = np.unique(face_edges, axis=0)
    filtrations = np.concatenate(
        (
            vertex_filtration,
            np.max(vertex_filtration[edges], axis=1),
            np.max(vertex_filtration[faces], axis=1),
        )
    )
    signs = np.concatenate(
        (
            np.ones(len(vertex_filtration), dtype=int),
            -np.ones(len(edges), dtype=int),
            np.ones(len(faces), dtype=int),
        )
    )
    _, inverse = np.unique(filtrations, return_inverse=True)
    exact_deltas = np.bincount(inverse, weights=signs).astype(int)
    sampled_deltas = np.diff(np.asarray(sampled_chi, dtype=float))
    return {
        "ecc_exact_jump_count": int(np.count_nonzero(exact_deltas)),
        "ecc_exact_total_variation": int(np.sum(np.abs(exact_deltas))),
        "ecc_sampled_jump_count": int(np.count_nonzero(sampled_deltas)),
        "ecc_sampled_total_variation": float(np.sum(np.abs(sampled_deltas))),
    }


def curve_errors(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    u = np.linspace(0.0, 1.0, len(difference))
    return {
        "l1": float(np.trapezoid(np.abs(difference), u)),
        "l2": float(np.sqrt(np.trapezoid(difference**2, u))),
        "linf": float(np.max(np.abs(difference))),
    }


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (base, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def mesh_quality_statistics(
    vertices: np.ndarray, faces: np.ndarray
) -> dict[str, float]:
    minimum_angles = triangle_minimum_angles_degrees(vertices, faces)
    return {
        "minimum_angle_degrees": float(np.min(minimum_angles)),
        "angle_q01_degrees": float(np.quantile(minimum_angles, 0.01)),
        "angle_q05_degrees": float(np.quantile(minimum_angles, 0.05)),
    }


def compute_realization(
    surface: object,
    mesh: ParametricMesh,
    flip_context: DomainFlipContext,
    subdivision_level: int,
    curvature: str,
    analytic_values: np.ndarray,
    reference_chi: np.ndarray,
    reference_secc: np.ndarray,
    physical_thresholds: np.ndarray,
    characteristic_length: float,
    config: Config,
    flip_regime: str,
    flip_fraction: float,
    repetition: int,
) -> Realization:
    seed = stable_seed(
        config.random_seed,
        surface.name,
        subdivision_level,
        flip_regime,
        flip_fraction,
        repetition,
    )
    faces, flipped_edges, interior_edges = apply_domain_flips(
        surface,
        mesh,
        flip_context,
        fraction=flip_fraction,
        seed=seed,
    )
    if curvature == "gaussian_curvature":
        values, weights = vertex_gaussian_curvature_and_areas(mesh.vertices, faces)
    else:
        values = e02.mesh_vertex_curvature(mesh.vertices, faces, curvature=curvature)
        weights = vertex_mixed_areas(mesh.vertices, faces)
    curvature_scale = (
        characteristic_length**2
        if curvature == "gaussian_curvature"
        else characteristic_length
    )
    dimensionless_values = curvature_scale * values
    dimensionless_analytic = curvature_scale * analytic_values
    curve, _ = e02.lower_star_ecc(values, faces)
    chi, secc = normalized_curve_arrays(curve, physical_thresholds)
    curvature_error = weighted_errors(
        dimensionless_values, dimensionless_analytic, weights
    )
    ecc_error = curve_errors(chi, reference_chi)
    secc_error = curve_errors(secc, reference_secc)
    quality = mesh_quality_statistics(mesh.vertices, faces)
    diagnostics = ecc_jump_diagnostics(dimensionless_values, faces, chi)
    edges = e02.mesh_edges(faces)
    mean_edge_length = float(
        np.mean([np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]) for a, b in edges])
    )
    row: dict[str, object] = {
        "surface": surface.name,
        "curvature": curvature,
        "subdivision_level": subdivision_level,
        "flip_regime": flip_regime,
        "flip_fraction": flip_fraction,
        "realized_flip_fraction": flipped_edges / interior_edges,
        "repetition": repetition,
        "seed": seed,
        "vertex_count": len(mesh.vertices),
        "face_count": len(faces),
        "flipped_edges": flipped_edges,
        "interior_edges": interior_edges,
        "characteristic_length": characteristic_length,
        "normalized_mean_edge_length": mean_edge_length / characteristic_length,
        "curvature_error_l1": curvature_error["l1"],
        "curvature_error_l2": curvature_error["l2"],
        "curvature_error_linf": curvature_error["linf"],
        "curvature_error_bias": curvature_error["bias"],
        "ecc_error_l1": ecc_error["l1"],
        "ecc_error_l2": ecc_error["l2"],
        "ecc_error_linf": ecc_error["linf"],
        "secc_error_l2": secc_error["l2"],
        "secc_error_linf": secc_error["linf"],
        **quality,
        **diagnostics,
    }
    return Realization(row, dimensionless_values, weights, chi, secc)


def pairwise_row(first: Realization, second: Realization) -> dict[str, object]:
    first_row, second_row = first.row, second.row
    weights = 0.5 * (first.weights + second.weights)
    curvature = weighted_errors(
        first.curvature_values, second.curvature_values, weights
    )
    ecc = curve_errors(first.chi, second.chi)
    secc = curve_errors(first.secc, second.secc)
    return {
        "surface": first_row["surface"],
        "curvature": first_row["curvature"],
        "subdivision_level": first_row["subdivision_level"],
        "flip_regime": first_row["flip_regime"],
        "flip_fraction": first_row["flip_fraction"],
        "realized_flip_fraction": first_row["realized_flip_fraction"],
        "normalized_mean_edge_length": 0.5
        * (
            float(first_row["normalized_mean_edge_length"])
            + float(second_row["normalized_mean_edge_length"])
        ),
        "first_repetition": first_row["repetition"],
        "second_repetition": second_row["repetition"],
        "curvature_l1": curvature["l1"],
        "curvature_l2": curvature["l2"],
        "curvature_linf": curvature["linf"],
        "ecc_l1": ecc["l1"],
        "ecc_l2": ecc["l2"],
        "ecc_linf": ecc["linf"],
        "secc_l2": secc["l2"],
        "secc_linf": secc["linf"],
    }


def cache_stem(cache_dir: Path, payload: dict[str, object]) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return cache_dir / (
        f"{payload['surface']}_s{payload['subdivision_level']}_"
        f"{payload['curvature']}_{key}"
    )


def load_cached_realization(stem: Path) -> Realization | None:
    metadata_path, data_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if not metadata_path.exists() or not data_path.exists():
        return None
    row = json.loads(metadata_path.read_text())
    with np.load(data_path) as data:
        return Realization(
            row,
            data["curvature_values"],
            data["weights"],
            data["chi"],
            data["secc"],
        )


def save_cached_realization(stem: Path, realization: Realization) -> None:
    e02.write_json(stem.with_suffix(".json"), realization.row)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        curvature_values=realization.curvature_values,
        weights=realization.weights,
        chi=realization.chi,
        secc=realization.secc,
    )


def smooth_reference(
    source: object,
    surface: object,
    curvature: str,
    sample_count: int,
    padding_fraction: float,
    constant_padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = (
        e02.curvature_scoped_dir(source.results_dir, curvature) / "calculation_cache"
    )
    computation = e02.cached_level_computation(
        cache_dir,
        surface,
        source.reference_level,
        curvature=curvature,
        reference=True,
    )
    lower, upper = e03.analytic_interval(
        computation.vertex_curvature,
        padding_fraction=padding_fraction,
        constant_padding=constant_padding,
    )
    thresholds = np.linspace(lower, upper, sample_count)
    chi, secc = normalized_curve_arrays(computation.curve, thresholds)
    return thresholds, chi, secc


def run(config: Config) -> list[Path]:
    source = e02.load_experiment_config(config.source_config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.results_dir / "calculation_cache"
    cache_dir.mkdir(exist_ok=True)
    realization_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for surface in source.surfaces:
        u_segments, v_segments = config.base_levels[surface.name]
        meshes = [initial_parametric_mesh(surface, u_segments, v_segments)]
        for _ in range(1, config.subdivision_levels):
            meshes.append(projected_midpoint_subdivision(surface, meshes[-1]))
        flip_contexts = [
            {
                regime: prepare_domain_flips(
                    surface,
                    mesh,
                    regime=regime,
                    quality_min_angle_degrees=config.quality_min_angle_degrees,
                )
                for regime in config.flip_regimes
            }
            for mesh in meshes
        ]

        base_weights = vertex_mixed_areas(meshes[0].vertices, meshes[0].faces)
        characteristic_length = math.sqrt(float(np.sum(base_weights)))
        for curvature in source.curvatures:
            thresholds, reference_chi, reference_secc = smooth_reference(
                source,
                surface,
                curvature,
                config.sample_count,
                config.interval_padding_fraction,
                config.constant_interval_padding,
            )
            for subdivision_level, (mesh, level_flip_contexts) in enumerate(
                zip(meshes, flip_contexts, strict=True)
            ):
                analytic_values = np.asarray(
                    [
                        e02.smooth_curvature(surface, *point, curvature=curvature)
                        for point in mesh.parameters
                    ]
                )
                conditions = [(BASELINE_REGIME, 0.0)] + [
                    (regime, fraction)
                    for regime in config.flip_regimes
                    for fraction in config.flip_fractions
                    if fraction > 0.0
                ]
                for flip_regime, flip_fraction in conditions:
                    repetitions = (
                        1 if flip_regime == BASELINE_REGIME else config.repetitions
                    )
                    context_regime = (
                        config.flip_regimes[0]
                        if flip_regime == BASELINE_REGIME
                        else flip_regime
                    )
                    flip_context = level_flip_contexts[context_regime]
                    realizations: list[Realization] = []
                    for repetition in range(repetitions):
                        payload = {
                            "cache_version": CACHE_VERSION,
                            "surface": surface.name,
                            "surface_parameters": surface.parameters,
                            "base_level": [u_segments, v_segments],
                            "subdivision_level": subdivision_level,
                            "curvature": curvature,
                            "flip_regime": flip_regime,
                            "flip_fraction": flip_fraction,
                            "quality_min_angle_degrees": (
                                config.quality_min_angle_degrees
                                if flip_regime == "quality_controlled"
                                else None
                            ),
                            "repetition": repetition,
                            "seed": config.random_seed,
                            "sample_count": config.sample_count,
                            "threshold_interval": [
                                float(thresholds[0]),
                                float(thresholds[-1]),
                            ],
                        }
                        stem = cache_stem(cache_dir, payload)
                        realization = load_cached_realization(stem)
                        if realization is None:
                            realization = compute_realization(
                                surface,
                                mesh,
                                flip_context,
                                subdivision_level,
                                curvature,
                                analytic_values,
                                reference_chi,
                                reference_secc,
                                thresholds,
                                characteristic_length,
                                config,
                                flip_regime,
                                flip_fraction,
                                repetition,
                            )
                            save_cached_realization(stem, realization)
                        required_cached_metrics = (
                            *ECC_DIAGNOSTIC_METRICS,
                            *MESH_QUALITY_METRICS,
                        )
                        if not all(
                            metric in realization.row
                            for metric in required_cached_metrics
                        ):
                            diagnostic_faces, _, _ = apply_domain_flips(
                                surface,
                                mesh,
                                flip_context,
                                fraction=flip_fraction,
                                seed=int(realization.row["seed"]),
                            )
                            realization.row.update(
                                ecc_jump_diagnostics(
                                    realization.curvature_values,
                                    diagnostic_faces,
                                    realization.chi,
                                )
                            )
                            realization.row.update(
                                mesh_quality_statistics(mesh.vertices, diagnostic_faces)
                            )
                            e02.write_json(stem.with_suffix(".json"), realization.row)
                        realizations.append(realization)
                        realization_rows.append(realization.row)
                    if len(realizations) > 1:
                        pairwise_rows.extend(
                            pairwise_row(first, second)
                            for first, second in itertools.combinations(realizations, 2)
                        )

    summaries = summarize_realizations(realization_rows)
    pairwise_summaries = summarize_pairwise(pairwise_rows)
    add_convergence_ratios(summaries)
    add_pairwise_ratios(pairwise_summaries)
    ecc_jump_scaling = summarize_ecc_jump_scaling(summaries)
    signature = artifact_signature(realization_rows, pairwise_rows)
    outputs = []
    for filename, rows in (
        ("subdivision_realizations.csv", realization_rows),
        ("subdivision_summary.csv", summaries),
        ("triangulation_pairwise.csv", pairwise_rows),
        ("triangulation_summary.csv", pairwise_summaries),
        ("ecc_jump_scaling.csv", ecc_jump_scaling),
    ):
        path = config.results_dir / filename
        if not e02.artifact_is_current(path, signature):
            write_rows(path, rows)
            e02.mark_artifact_current(path, signature)
        outputs.append(path)

    main_path = config.results_dir / "main_results.png"
    if not e02.artifact_is_current(main_path, signature):
        write_convergence_plot(main_path, summaries)
        e02.mark_artifact_current(main_path, signature)
    outputs.append(main_path)
    variability_path = config.results_dir / "triangulation_variability.png"
    if not e02.artifact_is_current(variability_path, signature):
        write_variability_plot(variability_path, pairwise_summaries)
        e02.mark_artifact_current(variability_path, signature)
    outputs.append(variability_path)
    jump_path = config.results_dir / "ecc_jump_diagnostics.png"
    if not e02.artifact_is_current(jump_path, signature):
        write_ecc_jump_plot(jump_path, summaries)
        e02.mark_artifact_current(jump_path, signature)
    outputs.append(jump_path)
    quality_path = config.results_dir / "mesh_quality.png"
    if not e02.artifact_is_current(quality_path, signature):
        write_mesh_quality_plot(
            quality_path, summaries, config.quality_min_angle_degrees
        )
        e02.mark_artifact_current(quality_path, signature)
    outputs.append(quality_path)
    return outputs


def percentile_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values)
    return tuple(float(value) for value in np.quantile(array, (0.25, 0.5, 0.75)))


def summarize_realizations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["surface"],
                row["curvature"],
                row["subdivision_level"],
                row["flip_regime"],
                row["flip_fraction"],
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (surface, curvature, level, regime, fraction), members in grouped.items():
        summary: dict[str, object] = {
            "surface": surface,
            "curvature": curvature,
            "subdivision_level": level,
            "flip_regime": regime,
            "flip_fraction": fraction,
            "realizations": len(members),
            "vertex_count": members[0]["vertex_count"],
            "face_count": members[0]["face_count"],
            "normalized_mean_edge_length": float(
                np.median(
                    [float(member["normalized_mean_edge_length"]) for member in members]
                )
            ),
        }
        for metric in ERROR_METRICS + ECC_DIAGNOSTIC_METRICS + MESH_QUALITY_METRICS:
            q25, median, q75 = percentile_summary(
                [float(member[metric]) for member in members]
            )
            summary[f"{metric}_q25"] = q25
            summary[f"{metric}_median"] = median
            summary[f"{metric}_q75"] = q75
        summaries.append(summary)
    return summaries


def summarize_pairwise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["surface"],
                row["curvature"],
                row["subdivision_level"],
                row["flip_regime"],
                row["flip_fraction"],
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (surface, curvature, level, regime, fraction), members in grouped.items():
        summary: dict[str, object] = {
            "surface": surface,
            "curvature": curvature,
            "subdivision_level": level,
            "flip_regime": regime,
            "flip_fraction": fraction,
            "pairs": len(members),
            "normalized_mean_edge_length": float(
                np.median(
                    [float(member["normalized_mean_edge_length"]) for member in members]
                )
            ),
        }
        for metric in PAIRWISE_METRICS:
            q25, median, q75 = percentile_summary(
                [float(member[metric]) for member in members]
            )
            summary[f"{metric}_q25"] = q25
            summary[f"{metric}_median"] = median
            summary[f"{metric}_q75"] = q75
        summaries.append(summary)
    return summaries


def add_convergence_ratios(summaries: list[dict[str, object]]) -> None:
    baselines: dict[tuple[object, ...], float] = {}
    for row in summaries:
        if row["subdivision_level"] != 0:
            continue
        for metric in ERROR_METRICS:
            baselines[
                (
                    row["surface"],
                    row["curvature"],
                    row["flip_regime"],
                    row["flip_fraction"],
                    metric,
                )
            ] = float(row[f"{metric}_median"])
    for row in summaries:
        for metric in ERROR_METRICS:
            baseline = baselines[
                (
                    row["surface"],
                    row["curvature"],
                    row["flip_regime"],
                    row["flip_fraction"],
                    metric,
                )
            ]
            for statistic in ("q25", "median", "q75"):
                value = float(row[f"{metric}_{statistic}"])
                row[f"{metric}_ratio_{statistic}"] = (
                    value / baseline if baseline > 0 else math.nan
                )


def add_pairwise_ratios(summaries: list[dict[str, object]]) -> None:
    baselines: dict[tuple[object, ...], float] = {}
    for row in summaries:
        if row["subdivision_level"] != 0:
            continue
        for metric in PAIRWISE_METRICS:
            baselines[
                (
                    row["surface"],
                    row["curvature"],
                    row["flip_regime"],
                    row["flip_fraction"],
                    metric,
                )
            ] = float(row[f"{metric}_median"])
    for row in summaries:
        for metric in PAIRWISE_METRICS:
            baseline = baselines[
                (
                    row["surface"],
                    row["curvature"],
                    row["flip_regime"],
                    row["flip_fraction"],
                    metric,
                )
            ]
            for statistic in ("q25", "median", "q75"):
                value = float(row[f"{metric}_{statistic}"])
                row[f"{metric}_ratio_{statistic}"] = (
                    value / baseline if baseline > 0 else math.nan
                )


def summarize_ecc_jump_scaling(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        grouped[
            (
                row["surface"],
                row["curvature"],
                row["flip_regime"],
                row["flip_fraction"],
            )
        ].append(row)

    scaling_rows: list[dict[str, object]] = []
    for (surface, curvature, regime, fraction), members in grouped.items():
        members.sort(key=lambda row: int(row["subdivision_level"]))
        result: dict[str, object] = {
            "surface": surface,
            "curvature": curvature,
            "flip_regime": regime,
            "flip_fraction": fraction,
            "levels": len(members),
            "initial_vertex_count": members[0]["vertex_count"],
            "final_vertex_count": members[-1]["vertex_count"],
        }
        vertex_counts = np.asarray(
            [float(member["vertex_count"]) for member in members]
        )
        for metric in ECC_DIAGNOSTIC_METRICS:
            values = np.asarray(
                [float(member[f"{metric}_median"]) for member in members]
            )
            positive = values > 0
            exponent = (
                float(
                    np.polyfit(
                        np.log(vertex_counts[positive]), np.log(values[positive]), 1
                    )[0]
                )
                if np.count_nonzero(positive) >= 2
                else math.nan
            )
            result[f"{metric}_initial"] = float(values[0])
            result[f"{metric}_final"] = float(values[-1])
            result[f"{metric}_growth_factor"] = (
                float(values[-1] / values[0]) if values[0] > 0 else math.nan
            )
            result[f"{metric}_vertex_power"] = exponent
        scaling_rows.append(result)
    return scaling_rows


def artifact_signature(
    realization_rows: list[dict[str, object]], pairwise_rows: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "realizations": hashlib.sha256(
            json.dumps(realization_rows, sort_keys=True, allow_nan=True).encode()
        ).hexdigest(),
        "pairwise": hashlib.sha256(
            json.dumps(pairwise_rows, sort_keys=True, allow_nan=True).encode()
        ).hexdigest(),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_keys(rows: list[dict[str, object]]) -> list[tuple[str, float]]:
    conditions = {
        (str(row["flip_regime"]), float(row["flip_fraction"])) for row in rows
    }
    regime_order = {BASELINE_REGIME: 0, "quality_controlled": 1, "stress": 2}
    return sorted(
        conditions,
        key=lambda item: (regime_order.get(item[0], 99), item[1]),
    )


def condition_label(regime: str, fraction: float) -> str:
    if regime == BASELINE_REGIME:
        return "unflipped"
    label = "quality" if regime == "quality_controlled" else "stress"
    return f"{label} {100.0 * fraction:g}%"


def condition_style(
    regime: str, fraction: float, fractions: list[float], plt: object
) -> dict[str, object]:
    if regime == BASELINE_REGIME:
        return {"color": "0.25", "linestyle": "-", "marker": "o"}
    colors = plt.get_cmap("viridis")(np.linspace(0.18, 0.82, max(len(fractions), 2)))
    color = colors[fractions.index(fraction)]
    return {
        "color": color,
        "linestyle": "-" if regime == "quality_controlled" else "--",
        "marker": "o" if regime == "quality_controlled" else "x",
    }


def plot_grid(path: Path, rows: list[dict[str, object]], *, pairwise: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfaces = tuple(dict.fromkeys(str(row["surface"]) for row in rows))
    curvatures = ("gaussian_curvature", "mean_curvature")
    metrics = PAIRWISE_METRICS if pairwise else ERROR_METRICS
    titles = (
        (
            "Curvature pairwise ratio",
            "ECC pairwise ratio",
            "SECC pairwise ratio",
        )
        if pairwise
        else ("Curvature error ratio", "ECC error ratio", "SECC error ratio")
    )
    figure_rows = len(surfaces) * len(curvatures)
    fig, axes = plt.subplots(
        figure_rows, 3, figsize=(14, 2.7 * figure_rows), constrained_layout=True
    )
    conditions = condition_keys(rows)
    positive_fractions = sorted(
        {fraction for _, fraction in conditions if fraction > 0.0}
    )
    for curvature_index, curvature in enumerate(curvatures):
        for surface_index, surface in enumerate(surfaces):
            row_index = curvature_index * len(surfaces) + surface_index
            for column, metric in enumerate(metrics):
                ax = axes[row_index, column]
                if row_index == 0:
                    ax.set_title(titles[column])
                for regime, fraction in conditions:
                    selected = [
                        row
                        for row in rows
                        if row["surface"] == surface
                        and row["curvature"] == curvature
                        and row["flip_regime"] == regime
                        and float(row["flip_fraction"]) == fraction
                    ]
                    selected.sort(key=lambda row: int(row["subdivision_level"]))
                    if not selected:
                        continue
                    x = np.asarray(
                        [float(row["normalized_mean_edge_length"]) for row in selected]
                    )
                    prefix = f"{metric}_ratio"
                    median = np.asarray(
                        [float(row[f"{prefix}_median"]) for row in selected]
                    )
                    q25 = np.asarray([float(row[f"{prefix}_q25"]) for row in selected])
                    q75 = np.asarray([float(row[f"{prefix}_q75"]) for row in selected])
                    style = condition_style(regime, fraction, positive_fractions, plt)
                    ax.plot(
                        x,
                        median,
                        label=condition_label(regime, fraction),
                        **style,
                    )
                    ax.fill_between(x, q25, q75, color=style["color"], alpha=0.14)
                ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.grid(alpha=0.25)
                ax.set_xlabel("normalised mean edge length (refinement →)")
                if column == 0:
                    ax.set_ylabel(
                        f"{surface.replace('_', ' ')}\n{e02.curvature_label(curvature)}"
                    )
                if row_index == 0 and column == 2:
                    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_ecc_jump_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfaces = tuple(dict.fromkeys(str(row["surface"]) for row in rows))
    curvatures = ("gaussian_curvature", "mean_curvature")
    metrics = ECC_DIAGNOSTIC_METRICS
    titles = (
        "Exact nonzero jumps",
        "Exact total variation",
        "501-grid changes",
        "501-grid total variation",
    )
    figure_rows = len(surfaces) * len(curvatures)
    fig, axes = plt.subplots(
        figure_rows, 4, figsize=(18, 2.7 * figure_rows), constrained_layout=True
    )
    conditions = condition_keys(rows)
    positive_fractions = sorted(
        {fraction for _, fraction in conditions if fraction > 0.0}
    )
    for curvature_index, curvature in enumerate(curvatures):
        for surface_index, surface in enumerate(surfaces):
            row_index = curvature_index * len(surfaces) + surface_index
            for column, metric in enumerate(metrics):
                ax = axes[row_index, column]
                if row_index == 0:
                    ax.set_title(titles[column])
                for regime, fraction in conditions:
                    selected = [
                        row
                        for row in rows
                        if row["surface"] == surface
                        and row["curvature"] == curvature
                        and row["flip_regime"] == regime
                        and float(row["flip_fraction"]) == fraction
                    ]
                    selected.sort(key=lambda row: int(row["subdivision_level"]))
                    if not selected:
                        continue
                    x = np.asarray([float(row["vertex_count"]) for row in selected])
                    median = np.asarray(
                        [float(row[f"{metric}_median"]) for row in selected]
                    )
                    q25 = np.asarray([float(row[f"{metric}_q25"]) for row in selected])
                    q75 = np.asarray([float(row[f"{metric}_q75"]) for row in selected])
                    style = condition_style(regime, fraction, positive_fractions, plt)
                    ax.plot(
                        x,
                        median,
                        label=condition_label(regime, fraction),
                        **style,
                    )
                    ax.fill_between(x, q25, q75, color=style["color"], alpha=0.14)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(alpha=0.25)
                ax.set_xlabel("vertex count")
                if column == 0:
                    ax.set_ylabel(
                        f"{surface.replace('_', ' ')}\n{e02.curvature_label(curvature)}"
                    )
                if row_index == 0 and column == len(metrics) - 1:
                    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_mesh_quality_plot(
    path: Path,
    rows: list[dict[str, object]],
    quality_min_angle_degrees: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row["curvature"] == "gaussian_curvature"]
    surfaces = tuple(dict.fromkeys(str(row["surface"]) for row in rows))
    metrics = ("minimum_angle_degrees", "angle_q01_degrees")
    titles = ("Smallest triangle angle", "1st percentile triangle angle")
    conditions = condition_keys(rows)
    positive_fractions = sorted(
        {fraction for _, fraction in conditions if fraction > 0.0}
    )
    fig, axes = plt.subplots(
        len(surfaces),
        len(metrics),
        figsize=(10, 2.6 * len(surfaces)),
        constrained_layout=True,
    )
    for surface_index, surface in enumerate(surfaces):
        for column, metric in enumerate(metrics):
            ax = axes[surface_index, column]
            if surface_index == 0:
                ax.set_title(titles[column])
            for regime, fraction in conditions:
                selected = [
                    row
                    for row in rows
                    if row["surface"] == surface
                    and row["flip_regime"] == regime
                    and float(row["flip_fraction"]) == fraction
                ]
                selected.sort(key=lambda row: int(row["subdivision_level"]))
                if not selected:
                    continue
                x = np.asarray(
                    [float(row["normalized_mean_edge_length"]) for row in selected]
                )
                median = np.asarray(
                    [float(row[f"{metric}_median"]) for row in selected]
                )
                q25 = np.asarray([float(row[f"{metric}_q25"]) for row in selected])
                q75 = np.asarray([float(row[f"{metric}_q75"]) for row in selected])
                style = condition_style(regime, fraction, positive_fractions, plt)
                ax.plot(
                    x,
                    median,
                    label=condition_label(regime, fraction),
                    **style,
                )
                ax.fill_between(x, q25, q75, color=style["color"], alpha=0.14)
            ax.axhline(
                quality_min_angle_degrees,
                color="0.6",
                linewidth=0.8,
                linestyle=":",
                zorder=0,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()
            ax.grid(alpha=0.25)
            ax.set_xlabel("normalised mean edge length (refinement →)")
            if column == 0:
                ax.set_ylabel(f"{surface.replace('_', ' ')}\nangle (degrees)")
            if surface_index == 0 and column == len(metrics) - 1:
                ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_convergence_plot(path: Path, rows: list[dict[str, object]]) -> None:
    plot_grid(path, rows, pairwise=False)


def write_variability_plot(path: Path, rows: list[dict[str, object]]) -> None:
    plot_grid(path, rows, pairwise=True)


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())

    def resolved(key: str) -> Path:
        candidate = Path(str(raw[key]))
        return (
            candidate
            if candidate.is_absolute()
            else (path.parent / candidate).resolve()
        )

    base_levels = {
        str(entry["surface"]): (int(entry["u_segments"]), int(entry["v_segments"]))
        for entry in raw.get("base_meshes", [])
    }
    fractions = tuple(float(value) for value in raw["flip_fractions"])
    regimes = tuple(str(value) for value in raw.get("flip_regimes", FLIP_REGIMES))
    config = Config(
        source_config=resolved("source_config"),
        results_dir=resolved("results_dir"),
        subdivision_levels=int(raw["subdivision_levels"]),
        repetitions=int(raw["repetitions"]),
        random_seed=int(raw.get("random_seed", 0)),
        flip_fractions=fractions,
        flip_regimes=regimes,
        quality_min_angle_degrees=float(raw["quality_min_angle_degrees"]),
        sample_count=int(raw.get("sample_count", 1001)),
        interval_padding_fraction=float(raw.get("interval_padding_fraction", 0.05)),
        constant_interval_padding=float(raw.get("constant_interval_padding", 1.0)),
        base_levels=base_levels,
    )
    if config.subdivision_levels < 2 or config.repetitions < 2:
        raise ValueError("Use at least two subdivision levels and repetitions")
    if (
        config.sample_count < 2
        or not fractions
        or any(not 0 <= value <= 1 for value in fractions)
        or set(regimes) != set(FLIP_REGIMES)
        or config.quality_min_angle_degrees <= 0.0
        or config.quality_min_angle_degrees >= 60.0
    ):
        raise ValueError("Invalid samples, flip conditions, or angle threshold")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    for output in run(load_config(args.config.resolve())):
        print(output)


if __name__ == "__main__":
    main()
