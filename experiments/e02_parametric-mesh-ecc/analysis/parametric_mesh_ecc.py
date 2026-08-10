"""Mesh parametrized surfaces and compute curvature lower-star ECCs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import igl
from scipy.sparse.linalg import spsolve

SUPPORTED_CURVATURES = ("gaussian_curvature", "mean_curvature")
CALCULATION_CACHE_VERSION = 1
COMPOSITE_PLOT_VERSION = 1


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class MeshLevel:
    name: str
    u_segments: int
    v_segments: int


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    kind: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class SimplexEvent:
    dimension: int
    vertices: tuple[int, ...]
    filtration: float

    @property
    def sign(self) -> int:
        return 1 if self.dimension % 2 == 0 else -1


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path
    results_dir: Path
    mesh_levels: tuple[MeshLevel, ...]
    reference_level: MeshLevel
    surfaces: tuple[SurfaceSpec, ...]
    curvatures: tuple[str, ...]
    histogram_bins: int
    plot_percentile: float | None


@dataclass(frozen=True)
class SurfaceRendering:
    label: str
    mesh: Mesh
    face_curvature: np.ndarray


@dataclass(frozen=True)
class LevelComputation:
    mesh: Mesh
    vertex_curvature: np.ndarray
    curve: list[dict[str, float | int]]
    events: list[SimplexEvent] | None
    curve_source: str
    cache_hit: bool


@dataclass(frozen=True)
class MainResultRow:
    surface_name: str
    parameter_text: str
    rendering_assets: dict[str, Path]
    curves: list[tuple[str, list[dict[str, float | int]]]]
    renderings: list[SurfaceRendering]


def generate_mesh(surface: SurfaceSpec, level: MeshLevel) -> Mesh:
    if surface.kind == "doubly_elliptic_torus":
        return generate_doubly_elliptic_torus(surface, level)
    if surface.kind == "product_sine_graph":
        return generate_product_sine_graph(surface, level)
    if surface.kind == "helicoid_patch":
        return generate_helicoid_patch(surface, level)
    if surface.kind == "pseudosphere_patch":
        return generate_pseudosphere_patch(surface, level)
    if surface.kind == "ellipsoid":
        return generate_ellipsoid(surface, level)
    raise ValueError(f"Unsupported surface kind {surface.kind!r}")


def surface_point(
    surface: SurfaceSpec, first: float, second: float
) -> tuple[float, float, float]:
    """Evaluate a configured surface parametrization."""

    if surface.kind == "doubly_elliptic_torus":
        radius = float_parameter(surface, "R", 2.0)
        tube_radius = float_parameter(surface, "r", 0.55)
        radial = radius + tube_radius * math.cos(second)
        return (
            float_parameter(surface, "a", 1.35) * radial * math.cos(first),
            radial * math.sin(first),
            float_parameter(surface, "b", 0.75) * math.sin(second),
        )
    if surface.kind == "product_sine_graph":
        amplitude = float_parameter(surface, "amplitude", 1.0)
        frequency = float_parameter(surface, "frequency", 1.0)
        height = amplitude * math.sin(2.0 * math.pi * frequency * first)
        height *= math.sin(2.0 * math.pi * frequency * second)
        return (first, second, height)
    if surface.kind == "helicoid_patch":
        angle = float_parameter(surface, "alpha", 1.0) * second
        return (first * math.cos(angle), first * math.sin(angle), second)
    if surface.kind == "pseudosphere_patch":
        radius = float_parameter(surface, "radius", 1.0)
        sech = 1.0 / math.cosh(first)
        return (
            radius * sech * math.cos(second),
            radius * sech * math.sin(second),
            radius * (first - math.tanh(first)),
        )
    if surface.kind == "ellipsoid":
        return ellipsoid_point(
            float_parameter(surface, "a", 2.0),
            float_parameter(surface, "b", 1.5),
            float_parameter(surface, "c", 1.0),
            first,
            second,
        )
    raise ValueError(f"Unsupported surface kind {surface.kind!r}")


def generate_reference_mesh_and_curvature(
    surface: SurfaceSpec, level: MeshLevel, *, curvature: str
) -> tuple[Mesh, np.ndarray]:
    """Sample the smooth curvature on a high-resolution parameter grid."""

    mesh = generate_mesh(surface, level)
    parameters = surface_grid_parameters(surface, level)
    curvatures = np.asarray(
        [
            smooth_curvature(surface, first, second, curvature=curvature)
            for first, second in parameters
        ],
        dtype=float,
    )
    return mesh, curvatures


def generate_doubly_elliptic_torus(surface: SurfaceSpec, level: MeshLevel) -> Mesh:
    radius = float_parameter(surface, "R", 2.0)
    tube_radius = float_parameter(surface, "r", 0.55)
    a_scale = float_parameter(surface, "a", 1.35)
    b_scale = float_parameter(surface, "b", 0.75)
    if radius <= tube_radius:
        raise ValueError("doubly_elliptic_torus requires R > r")
    if min(radius, tube_radius, a_scale, b_scale) <= 0:
        raise ValueError("doubly_elliptic_torus parameters must be positive")

    def parametrization(phi: float, theta: float) -> tuple[float, float, float]:
        radial = radius + tube_radius * math.cos(theta)
        return (
            a_scale * radial * math.cos(phi),
            radial * math.sin(phi),
            b_scale * math.sin(theta),
        )

    return periodic_grid_mesh(
        level.u_segments,
        level.v_segments,
        parametrization,
        u_min=0.0,
        u_max=2.0 * math.pi,
        v_min=0.0,
        v_max=2.0 * math.pi,
    )


def generate_product_sine_graph(surface: SurfaceSpec, level: MeshLevel) -> Mesh:
    domain_size = float_parameter(surface, "domain_size", 2.0)
    amplitude = float_parameter(surface, "amplitude", 1.0)
    frequency = float_parameter(surface, "frequency", 1.0)
    if domain_size <= 0:
        raise ValueError("product_sine_graph requires positive domain_size")

    def parametrization(x_coord: float, y_coord: float) -> tuple[float, float, float]:
        z_coord = amplitude * math.sin(2.0 * math.pi * frequency * x_coord)
        z_coord *= math.sin(2.0 * math.pi * frequency * y_coord)
        return (x_coord, y_coord, z_coord)

    return open_grid_mesh(
        level.u_segments,
        level.v_segments,
        parametrization,
        u_min=0.0,
        u_max=domain_size,
        v_min=0.0,
        v_max=domain_size,
    )


def generate_helicoid_patch(surface: SurfaceSpec, level: MeshLevel) -> Mesh:
    rho_min = float_parameter(surface, "rho_min", 0.0)
    rho_max = float_parameter(surface, "rho_max", 1.6)
    theta_min = float_parameter(surface, "theta_min", -math.pi)
    theta_max = float_parameter(surface, "theta_max", math.pi)
    alpha = float_parameter(surface, "alpha", 1.0)
    if rho_max <= rho_min:
        raise ValueError("helicoid_patch requires rho_max > rho_min")
    if theta_max <= theta_min:
        raise ValueError("helicoid_patch requires theta_max > theta_min")

    def parametrization(rho: float, theta: float) -> tuple[float, float, float]:
        angle = alpha * theta
        return (rho * math.cos(angle), rho * math.sin(angle), theta)

    return open_grid_mesh(
        level.u_segments,
        level.v_segments,
        parametrization,
        u_min=rho_min,
        u_max=rho_max,
        v_min=theta_min,
        v_max=theta_max,
    )


def generate_pseudosphere_patch(surface: SurfaceSpec, level: MeshLevel) -> Mesh:
    """Generate a truncated tractricoid, avoiding its singular limiting tip."""

    radius = float_parameter(surface, "radius", 1.0)
    u_min = float_parameter(surface, "u_min", 0.15)
    u_max = float_parameter(surface, "u_max", 2.5)
    if radius <= 0:
        raise ValueError("pseudosphere_patch requires positive radius")
    if u_min <= 0 or u_max <= u_min:
        raise ValueError("pseudosphere_patch requires 0 < u_min < u_max")

    def parametrization(u_coord: float, theta: float) -> tuple[float, float, float]:
        sech_u = 1.0 / math.cosh(u_coord)
        return (
            radius * sech_u * math.cos(theta),
            radius * sech_u * math.sin(theta),
            radius * (u_coord - math.tanh(u_coord)),
        )

    return cylindrical_grid_mesh(
        level.u_segments,
        level.v_segments,
        parametrization,
        u_min=u_min,
        u_max=u_max,
        v_min=0.0,
        v_max=2.0 * math.pi,
    )


def generate_ellipsoid(surface: SurfaceSpec, level: MeshLevel) -> Mesh:
    """Generate a closed UV ellipsoid with one vertex at each pole."""

    a_axis = float_parameter(surface, "a", 2.0)
    b_axis = float_parameter(surface, "b", 1.5)
    c_axis = float_parameter(surface, "c", 1.0)
    if min(a_axis, b_axis, c_axis) <= 0:
        raise ValueError("ellipsoid semi-axes must be positive")
    if level.u_segments < 3 or level.v_segments < 2:
        raise ValueError("ellipsoid requires >=3 longitude and >=2 latitude segments")

    vertices = [ellipsoid_point(a_axis, b_axis, c_axis, 0.0, 0.0)]
    for v_index in range(1, level.v_segments):
        theta = math.pi * v_index / level.v_segments
        for u_index in range(level.u_segments):
            phi = 2.0 * math.pi * u_index / level.u_segments
            vertices.append(ellipsoid_point(a_axis, b_axis, c_axis, theta, phi))
    south = len(vertices)
    vertices.append(ellipsoid_point(a_axis, b_axis, c_axis, math.pi, 0.0))

    def ring_index(v_index: int, u_index: int) -> int:
        return 1 + (v_index - 1) * level.u_segments + (u_index % level.u_segments)

    faces: list[tuple[int, int, int]] = []
    for u_index in range(level.u_segments):
        faces.append((0, ring_index(1, u_index), ring_index(1, u_index + 1)))
    for v_index in range(1, level.v_segments - 1):
        for u_index in range(level.u_segments):
            current = ring_index(v_index, u_index)
            next_phi = ring_index(v_index, u_index + 1)
            next_theta = ring_index(v_index + 1, u_index)
            diagonal = ring_index(v_index + 1, u_index + 1)
            faces.append((current, next_theta, next_phi))
            faces.append((next_phi, next_theta, diagonal))
    for u_index in range(level.u_segments):
        faces.append(
            (ring_index(level.v_segments - 1, u_index + 1),
             ring_index(level.v_segments - 1, u_index), south)
        )
    vertex_array = np.asarray(vertices, dtype=float)
    face_array = orient_faces_outward(vertex_array, np.asarray(faces, dtype=int))
    return Mesh(vertex_array, face_array)


def ellipsoid_point(
    a_axis: float, b_axis: float, c_axis: float, theta: float, phi: float
) -> tuple[float, float, float]:
    return (
        a_axis * math.sin(theta) * math.cos(phi),
        b_axis * math.sin(theta) * math.sin(phi),
        c_axis * math.cos(theta),
    )


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    oriented = faces.copy()
    for index, face in enumerate(oriented):
        first, second, third = vertices[face]
        normal = np.cross(second - first, third - first)
        if float(np.dot(normal, (first + second + third) / 3.0)) < 0.0:
            oriented[index, 1], oriented[index, 2] = oriented[index, 2], oriented[index, 1]
    return oriented


def periodic_grid_mesh(
    u_segments: int,
    v_segments: int,
    parametrization: Callable[[float, float], tuple[float, float, float]],
    *,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> Mesh:
    if u_segments < 3 or v_segments < 3:
        raise ValueError("periodic meshes require at least 3 segments per direction")

    vertices = [
        parametrization(
            u_min + (u_max - u_min) * u_index / u_segments,
            v_min + (v_max - v_min) * v_index / v_segments,
        )
        for u_index in range(u_segments)
        for v_index in range(v_segments)
    ]

    def index(u_index: int, v_index: int) -> int:
        return (u_index % u_segments) * v_segments + (v_index % v_segments)

    faces: list[tuple[int, int, int]] = []
    for u_index in range(u_segments):
        for v_index in range(v_segments):
            lower_left = index(u_index, v_index)
            lower_right = index(u_index + 1, v_index)
            upper_left = index(u_index, v_index + 1)
            upper_right = index(u_index + 1, v_index + 1)
            faces.append((lower_left, lower_right, upper_left))
            faces.append((upper_left, lower_right, upper_right))

    return Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int))


def open_grid_mesh(
    u_segments: int,
    v_segments: int,
    parametrization: Callable[[float, float], tuple[float, float, float]],
    *,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> Mesh:
    if u_segments < 1 or v_segments < 1:
        raise ValueError("open meshes require at least 1 segment per direction")

    u_count = u_segments + 1
    v_count = v_segments + 1
    vertices = [
        parametrization(
            u_min + (u_max - u_min) * u_index / u_segments,
            v_min + (v_max - v_min) * v_index / v_segments,
        )
        for u_index in range(u_count)
        for v_index in range(v_count)
    ]

    def index(u_index: int, v_index: int) -> int:
        return u_index * v_count + v_index

    faces: list[tuple[int, int, int]] = []
    for u_index in range(u_segments):
        for v_index in range(v_segments):
            lower_left = index(u_index, v_index)
            lower_right = index(u_index + 1, v_index)
            upper_left = index(u_index, v_index + 1)
            upper_right = index(u_index + 1, v_index + 1)
            faces.append((lower_left, lower_right, upper_left))
            faces.append((upper_left, lower_right, upper_right))

    return Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int))


def cylindrical_grid_mesh(
    u_segments: int,
    v_segments: int,
    parametrization: Callable[[float, float], tuple[float, float, float]],
    *,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> Mesh:
    """Triangulate a grid that is open in u and periodic in v."""

    if u_segments < 1 or v_segments < 3:
        raise ValueError("cylindrical meshes require >=1 u and >=3 v segments")
    u_count = u_segments + 1
    vertices = [
        parametrization(
            u_min + (u_max - u_min) * u_index / u_segments,
            v_min + (v_max - v_min) * v_index / v_segments,
        )
        for u_index in range(u_count)
        for v_index in range(v_segments)
    ]

    def index(u_index: int, v_index: int) -> int:
        return u_index * v_segments + (v_index % v_segments)

    faces: list[tuple[int, int, int]] = []
    for u_index in range(u_segments):
        for v_index in range(v_segments):
            lower_left = index(u_index, v_index)
            lower_right = index(u_index + 1, v_index)
            upper_left = index(u_index, v_index + 1)
            upper_right = index(u_index + 1, v_index + 1)
            faces.append((lower_left, lower_right, upper_left))
            faces.append((upper_left, lower_right, upper_right))
    return Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int))


def surface_grid_parameters(
    surface: SurfaceSpec, level: MeshLevel
) -> list[tuple[float, float]]:
    if surface.kind == "ellipsoid":
        parameters = [(0.0, 0.0)]
        parameters.extend(
            (
                math.pi * v_index / level.v_segments,
                2.0 * math.pi * u_index / level.u_segments,
            )
            for v_index in range(1, level.v_segments)
            for u_index in range(level.u_segments)
        )
        parameters.append((math.pi, 0.0))
        return parameters
    u_min, u_max, v_min, v_max = surface_domain(surface)
    periodic_both = surface.kind == "doubly_elliptic_torus"
    periodic_v = surface.kind == "pseudosphere_patch"
    u_count = level.u_segments if periodic_both else level.u_segments + 1
    v_count = level.v_segments if periodic_both or periodic_v else level.v_segments + 1
    return [
        (
            u_min + (u_max - u_min) * u_index / level.u_segments,
            v_min + (v_max - v_min) * v_index / level.v_segments,
        )
        for u_index in range(u_count)
        for v_index in range(v_count)
    ]


def surface_domain(surface: SurfaceSpec) -> tuple[float, float, float, float]:
    if surface.kind == "doubly_elliptic_torus":
        return (0.0, 2.0 * math.pi, 0.0, 2.0 * math.pi)
    if surface.kind == "product_sine_graph":
        domain_size = float_parameter(surface, "domain_size", 2.0)
        return (0.0, domain_size, 0.0, domain_size)
    if surface.kind == "helicoid_patch":
        return (
            float_parameter(surface, "rho_min", 0.0),
            float_parameter(surface, "rho_max", 1.6),
            float_parameter(surface, "theta_min", -math.pi),
            float_parameter(surface, "theta_max", math.pi),
        )
    if surface.kind == "pseudosphere_patch":
        return (
            float_parameter(surface, "u_min", 0.15),
            float_parameter(surface, "u_max", 2.5),
            0.0,
            2.0 * math.pi,
        )
    if surface.kind == "ellipsoid":
        return (0.0, math.pi, 0.0, 2.0 * math.pi)
    raise ValueError(f"Unsupported surface kind {surface.kind!r}")


def smooth_signed_mean_curvature(
    surface: SurfaceSpec, first: float, second: float
) -> float:
    if surface.kind == "doubly_elliptic_torus":
        return doubly_elliptic_torus_mean_curvature(surface, first, second)
    if surface.kind == "product_sine_graph":
        return product_sine_graph_mean_curvature(surface, first, second)
    if surface.kind == "helicoid_patch":
        return 0.0
    if surface.kind == "pseudosphere_patch":
        return pseudosphere_mean_curvature(surface, first, second)
    if surface.kind == "ellipsoid":
        return ellipsoid_mean_curvature(surface, first, second)
    raise ValueError(f"Unsupported surface kind {surface.kind!r}")


def smooth_curvature(
    surface: SurfaceSpec, first: float, second: float, *, curvature: str
) -> float:
    if curvature == "gaussian_curvature":
        return smooth_gaussian_curvature(surface, first, second)
    if is_mean_curvature(curvature):
        return smooth_signed_mean_curvature(surface, first, second)
    raise ValueError(f"Unsupported curvature kind {curvature!r}")


def smooth_gaussian_curvature(
    surface: SurfaceSpec, first: float, second: float
) -> float:
    if surface.kind == "doubly_elliptic_torus":
        return doubly_elliptic_torus_gaussian_curvature(surface, first, second)
    if surface.kind == "product_sine_graph":
        return product_sine_graph_gaussian_curvature(surface, first, second)
    if surface.kind == "helicoid_patch":
        return helicoid_gaussian_curvature(surface, first, second)
    if surface.kind == "pseudosphere_patch":
        return pseudosphere_gaussian_curvature(surface, first, second)
    if surface.kind == "ellipsoid":
        return ellipsoid_gaussian_curvature(surface, first, second)
    raise ValueError(f"Unsupported surface kind {surface.kind!r}")


def doubly_elliptic_torus_gaussian_curvature(
    surface: SurfaceSpec, phi: float, theta: float
) -> float:
    forms = doubly_elliptic_torus_fundamental_forms(surface, phi, theta)
    return gaussian_curvature_from_forms(*forms)


def doubly_elliptic_torus_mean_curvature(
    surface: SurfaceSpec, phi: float, theta: float
) -> float:
    forms = doubly_elliptic_torus_fundamental_forms(surface, phi, theta)
    return mean_curvature_from_forms(*forms)


def doubly_elliptic_torus_fundamental_forms(
    surface: SurfaceSpec, phi: float, theta: float
) -> tuple[float, float, float, float, float, float]:
    radius = float_parameter(surface, "R", 2.0)
    tube_radius = float_parameter(surface, "r", 0.55)
    a_scale = float_parameter(surface, "a", 1.35)
    b_scale = float_parameter(surface, "b", 0.75)

    radial = radius + tube_radius * math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    x_phi = np.asarray(
        (-a_scale * radial * sin_phi, radial * cos_phi, 0.0), dtype=float
    )
    x_theta = np.asarray(
        (
            -a_scale * tube_radius * sin_theta * cos_phi,
            -tube_radius * sin_theta * sin_phi,
            b_scale * cos_theta,
        ),
        dtype=float,
    )
    x_phi_phi = np.asarray(
        (-a_scale * radial * cos_phi, -radial * sin_phi, 0.0), dtype=float
    )
    x_phi_theta = np.asarray(
        (
            a_scale * tube_radius * sin_theta * sin_phi,
            -tube_radius * sin_theta * cos_phi,
            0.0,
        ),
        dtype=float,
    )
    x_theta_theta = np.asarray(
        (
            -a_scale * tube_radius * cos_theta * cos_phi,
            -tube_radius * cos_theta * sin_phi,
            -b_scale * sin_theta,
        ),
        dtype=float,
    )
    return parametric_fundamental_forms(
        x_phi, x_theta, x_phi_phi, x_phi_theta, x_theta_theta
    )


def product_sine_graph_gaussian_curvature(
    surface: SurfaceSpec, x_coord: float, y_coord: float
) -> float:
    f_x, f_y, f_xx, f_xy, f_yy = product_sine_derivatives(surface, x_coord, y_coord)
    numerator = f_xx * f_yy - f_xy**2
    denominator = (1.0 + f_x**2 + f_y**2) ** 2
    return numerator / denominator


def product_sine_graph_mean_curvature(
    surface: SurfaceSpec, x_coord: float, y_coord: float
) -> float:
    f_x, f_y, f_xx, f_xy, f_yy = product_sine_derivatives(surface, x_coord, y_coord)
    numerator = (1.0 + f_y**2) * f_xx - 2.0 * f_x * f_y * f_xy
    numerator += (1.0 + f_x**2) * f_yy
    denominator = 2.0 * (1.0 + f_x**2 + f_y**2) ** 1.5
    return -numerator / denominator


def product_sine_derivatives(
    surface: SurfaceSpec, x_coord: float, y_coord: float
) -> tuple[float, float, float, float, float]:
    amplitude = float_parameter(surface, "amplitude", 1.0)
    frequency = float_parameter(surface, "frequency", 1.0)
    omega = 2.0 * math.pi * frequency
    sin_x = math.sin(omega * x_coord)
    cos_x = math.cos(omega * x_coord)
    sin_y = math.sin(omega * y_coord)
    cos_y = math.cos(omega * y_coord)

    f_x = amplitude * omega * cos_x * sin_y
    f_y = amplitude * omega * sin_x * cos_y
    f_xx = -amplitude * omega**2 * sin_x * sin_y
    f_yy = -amplitude * omega**2 * sin_x * sin_y
    f_xy = amplitude * omega**2 * cos_x * cos_y
    return (f_x, f_y, f_xx, f_xy, f_yy)


def helicoid_gaussian_curvature(
    surface: SurfaceSpec, rho: float, theta: float
) -> float:
    del theta
    alpha = float_parameter(surface, "alpha", 1.0)
    return -(alpha**2) / (1.0 + (alpha * rho) ** 2) ** 2


def pseudosphere_gaussian_curvature(
    surface: SurfaceSpec, u_coord: float, theta: float
) -> float:
    del u_coord, theta
    radius = float_parameter(surface, "radius", 1.0)
    return -1.0 / radius**2


def pseudosphere_mean_curvature(
    surface: SurfaceSpec, u_coord: float, theta: float
) -> float:
    return mean_curvature_from_forms(
        *pseudosphere_fundamental_forms(surface, u_coord, theta)
    )


def pseudosphere_fundamental_forms(
    surface: SurfaceSpec, u_coord: float, theta: float
) -> tuple[float, float, float, float, float, float]:
    radius = float_parameter(surface, "radius", 1.0)
    sech_u = 1.0 / math.cosh(u_coord)
    tanh_u = math.tanh(u_coord)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    radial_uu = radius * sech_u * (tanh_u**2 - sech_u**2)
    x_u = np.asarray(
        (-radius * sech_u * tanh_u * cos_theta,
         -radius * sech_u * tanh_u * sin_theta,
         radius * tanh_u**2),
        dtype=float,
    )
    x_v = np.asarray(
        (-radius * sech_u * sin_theta, radius * sech_u * cos_theta, 0.0),
        dtype=float,
    )
    x_uu = np.asarray(
        (radial_uu * cos_theta, radial_uu * sin_theta,
         2.0 * radius * tanh_u * sech_u**2),
        dtype=float,
    )
    x_uv = np.asarray(
        (radius * sech_u * tanh_u * sin_theta,
         -radius * sech_u * tanh_u * cos_theta, 0.0),
        dtype=float,
    )
    x_vv = np.asarray(
        (-radius * sech_u * cos_theta, -radius * sech_u * sin_theta, 0.0),
        dtype=float,
    )
    return parametric_fundamental_forms(x_u, x_v, x_uu, x_uv, x_vv)


def ellipsoid_gaussian_curvature(
    surface: SurfaceSpec, theta: float, phi: float
) -> float:
    if math.isclose(math.sin(theta), 0.0, abs_tol=1e-12):
        a_axis = float_parameter(surface, "a", 2.0)
        b_axis = float_parameter(surface, "b", 1.5)
        c_axis = float_parameter(surface, "c", 1.0)
        return c_axis**2 / (a_axis**2 * b_axis**2)
    return gaussian_curvature_from_forms(
        *ellipsoid_fundamental_forms(surface, theta, phi)
    )


def ellipsoid_mean_curvature(
    surface: SurfaceSpec, theta: float, phi: float
) -> float:
    if math.isclose(math.sin(theta), 0.0, abs_tol=1e-12):
        a_axis = float_parameter(surface, "a", 2.0)
        b_axis = float_parameter(surface, "b", 1.5)
        c_axis = float_parameter(surface, "c", 1.0)
        return 0.5 * c_axis * (a_axis**-2 + b_axis**-2)
    return mean_curvature_from_forms(
        *ellipsoid_fundamental_forms(surface, theta, phi)
    )


def ellipsoid_fundamental_forms(
    surface: SurfaceSpec, theta: float, phi: float
) -> tuple[float, float, float, float, float, float]:
    a_axis = float_parameter(surface, "a", 2.0)
    b_axis = float_parameter(surface, "b", 1.5)
    c_axis = float_parameter(surface, "c", 1.0)
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    x_theta = np.asarray(
        (a_axis * cos_theta * cos_phi,
         b_axis * cos_theta * sin_phi,
         -c_axis * sin_theta), dtype=float
    )
    x_phi = np.asarray(
        (-a_axis * sin_theta * sin_phi,
         b_axis * sin_theta * cos_phi, 0.0), dtype=float
    )
    x_theta_theta = np.asarray(
        (-a_axis * sin_theta * cos_phi,
         -b_axis * sin_theta * sin_phi,
         -c_axis * cos_theta), dtype=float
    )
    x_theta_phi = np.asarray(
        (-a_axis * cos_theta * sin_phi,
         b_axis * cos_theta * cos_phi, 0.0), dtype=float
    )
    x_phi_phi = np.asarray(
        (-a_axis * sin_theta * cos_phi,
         -b_axis * sin_theta * sin_phi, 0.0), dtype=float
    )
    return parametric_fundamental_forms(
        x_theta, x_phi, x_theta_theta, x_theta_phi, x_phi_phi
    )


def parametric_fundamental_forms(
    x_u: np.ndarray,
    x_v: np.ndarray,
    x_uu: np.ndarray,
    x_uv: np.ndarray,
    x_vv: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    normal = np.cross(x_u, x_v)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    unit_normal = normal / normal_norm
    first_e = float(np.dot(x_u, x_u))
    first_f = float(np.dot(x_u, x_v))
    first_g = float(np.dot(x_v, x_v))
    second_l = float(np.dot(x_uu, unit_normal))
    second_m = float(np.dot(x_uv, unit_normal))
    second_n = float(np.dot(x_vv, unit_normal))
    return (first_e, first_f, first_g, second_l, second_m, second_n)


def gaussian_curvature_from_forms(
    first_e: float,
    first_f: float,
    first_g: float,
    second_l: float,
    second_m: float,
    second_n: float,
) -> float:
    denominator = first_e * first_g - first_f**2
    if denominator == 0:
        return 0.0
    return (second_l * second_n - second_m**2) / denominator


def mean_curvature_from_forms(
    first_e: float,
    first_f: float,
    first_g: float,
    second_l: float,
    second_m: float,
    second_n: float,
) -> float:
    denominator = 2.0 * (first_e * first_g - first_f**2)
    if denominator == 0:
        return 0.0
    conventional_h = (
        second_l * first_g - 2.0 * second_m * first_f + second_n * first_e
    ) / denominator
    return -conventional_h


def signed_vertex_mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Estimate signed H using libigl's cotangent and Voronoi operators."""

    cotangent_matrix = igl.cotmatrix(vertices, faces)
    mass_matrix = igl.massmatrix(
        vertices, faces, igl.MASSMATRIX_TYPE_VORONOI
    )
    mean_curvature_normal = spsolve(
        mass_matrix, -(cotangent_matrix @ vertices)
    )
    normals = igl.per_vertex_normals(
        vertices, faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
    )
    # Vira's convention is -Delta x = 2 H n.
    return 0.5 * np.einsum("ij,ij->i", mean_curvature_normal, normals)


def vertex_gaussian_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Estimate Gaussian curvature by angle defect over mixed vertex area."""

    vertex_count = len(vertices)
    mixed_areas = np.zeros(vertex_count, dtype=float)
    angle_sums = np.zeros(vertex_count, dtype=float)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)

    for face in faces:
        i, j, k = (int(value) for value in face)
        vi, vj, vk = vertices[[i, j, k]]
        face_normal = np.cross(vj - vi, vk - vi)
        face_area = 0.5 * np.linalg.norm(face_normal)
        if face_area <= 0:
            continue

        angles = triangle_angles(vi, vj, vk)
        for vertex, angle in zip((i, j, k), angles, strict=True):
            angle_sums[vertex] += angle

        cot_i = cotangent(vj - vi, vk - vi)
        cot_j = cotangent(vi - vj, vk - vj)
        cot_k = cotangent(vi - vk, vj - vk)
        add_mixed_voronoi_areas(
            mixed_areas,
            (i, j, k),
            (vi, vj, vk),
            (cot_i, cot_j, cot_k),
            face_area,
        )

        for edge in ((i, j), (i, k), (j, k)):
            edge_counts[tuple(sorted(edge))] += 1

    target_angles = np.full(vertex_count, 2.0 * math.pi)
    boundary_vertices = {
        vertex for edge, count in edge_counts.items() if count == 1 for vertex in edge
    }
    for vertex in boundary_vertices:
        target_angles[vertex] = math.pi

    curvature = np.zeros(vertex_count, dtype=float)
    valid = mixed_areas > 0
    curvature[valid] = (target_angles[valid] - angle_sums[valid]) / mixed_areas[valid]
    return curvature


def triangle_angles(
    first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> tuple[float, float, float]:
    return (
        angle_between(second - first, third - first),
        angle_between(first - second, third - second),
        angle_between(first - third, second - third),
    )


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    return math.acos(max(-1.0, min(1.0, cosine)))


def mesh_vertex_curvature(
    vertices: np.ndarray, faces: np.ndarray, *, curvature: str
) -> np.ndarray:
    if curvature == "gaussian_curvature":
        return vertex_gaussian_curvature(vertices, faces)
    if is_mean_curvature(curvature):
        return signed_vertex_mean_curvature(vertices, faces)
    raise ValueError(f"Unsupported curvature kind {curvature!r}")


def mesh_level_curvature(
    surface: SurfaceSpec, level: MeshLevel, mesh: Mesh, *, curvature: str
) -> tuple[np.ndarray, str]:
    values = mesh_vertex_curvature(mesh.vertices, mesh.faces, curvature=curvature)
    source = (
        "libigl_cotangent_voronoi_estimator"
        if is_mean_curvature(curvature)
        else "angle_defect_mixed_voronoi_estimator"
    )
    return (values, source)


def add_mixed_voronoi_areas(
    areas: np.ndarray,
    face: tuple[int, int, int],
    positions: tuple[np.ndarray, np.ndarray, np.ndarray],
    cotangents: tuple[float, float, float],
    face_area: float,
) -> None:
    i, j, k = face
    vi, vj, vk = positions
    cot_i, cot_j, cot_k = cotangents

    if cot_i < 0.0 or cot_j < 0.0 or cot_k < 0.0:
        obtuse_local = min(range(3), key=lambda local: cotangents[local])
        for local, vertex in enumerate(face):
            areas[vertex] += (
                face_area / 2.0 if local == obtuse_local else face_area / 4.0
            )
        return

    areas[i] += (squared_norm(vi - vj) * cot_k + squared_norm(vi - vk) * cot_j) / 8.0
    areas[j] += (squared_norm(vj - vi) * cot_k + squared_norm(vj - vk) * cot_i) / 8.0
    areas[k] += (squared_norm(vk - vi) * cot_j + squared_norm(vk - vj) * cot_i) / 8.0


def triangle_mean(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return values[faces].mean(axis=1)


def lower_star_ecc(
    vertex_filtration: np.ndarray, faces: np.ndarray
) -> tuple[list[dict[str, float | int]], list[SimplexEvent]]:
    """Compute the ECC of the lower-star filtration induced by vertex values."""

    events: list[SimplexEvent] = []
    for vertex, value in enumerate(vertex_filtration):
        events.append(SimplexEvent(0, (vertex,), float(value)))

    edges = mesh_edges(faces)
    for edge in sorted(edges):
        events.append(SimplexEvent(1, edge, float(max(vertex_filtration[list(edge)]))))

    for triangle in sorted(unique_faces(faces)):
        events.append(
            SimplexEvent(2, triangle, float(max(vertex_filtration[list(triangle)])))
        )

    events.sort(key=lambda event: (event.filtration, event.dimension, event.vertices))
    deltas: dict[float, int] = defaultdict(int)
    for event in events:
        deltas[event.filtration] += event.sign

    chi = 0
    curve: list[dict[str, float | int]] = []
    for filtration in sorted(deltas):
        delta = deltas[filtration]
        chi += delta
        curve.append(
            {
                "filtration": filtration,
                "delta_chi": delta,
                "chi_after": chi,
            }
        )
    return curve, events


def mesh_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        i, j, k = (int(value) for value in face)
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((i, k))))
        edges.add(tuple(sorted((j, k))))
    return edges


def unique_faces(faces: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(sorted(int(value) for value in face)) for face in faces}


def euler_characteristic(vertex_count: int, faces: np.ndarray) -> int:
    return vertex_count - len(mesh_edges(faces)) + len(unique_faces(faces))


def theoretical_ecc_curve(
    surface: SurfaceSpec, curvature: str
) -> list[dict[str, float | int]] | None:
    if curvature == "gaussian_curvature" and surface.kind == "helicoid_patch":
        alpha = float_parameter(surface, "alpha", 1.0)
        return [
            {
                "filtration": -(alpha**2),
                "delta_chi": 1,
                "chi_after": 1,
            }
        ]
    if is_mean_curvature(curvature) and surface.kind == "helicoid_patch":
        return [
            {
                "filtration": 0.0,
                "delta_chi": 1,
                "chi_after": 1,
            }
        ]
    return None


def surface_mesh_level(surface: SurfaceSpec, level: MeshLevel) -> MeshLevel:
    return MeshLevel(
        name=level.name,
        u_segments=int_parameter(surface, f"{level.name}_u_segments", level.u_segments),
        v_segments=int_parameter(surface, f"{level.name}_v_segments", level.v_segments),
    )


def cached_level_computation(
    cache_dir: Path,
    surface: SurfaceSpec,
    level: MeshLevel,
    *,
    curvature: str,
    reference: bool,
) -> LevelComputation:
    """Load or compute mesh, curvature, and ECC data for one level."""

    payload = {
        "cache_version": CALCULATION_CACHE_VERSION,
        "surface": surface.name,
        "surface_kind": surface.kind,
        "surface_parameters": surface.parameters,
        "level": {
            "name": level.name,
            "u_segments": level.u_segments,
            "v_segments": level.v_segments,
        },
        "curvature": curvature,
        "reference": reference,
        "estimator": (
            "analytic_smooth_formula"
            if reference
            else (
                "libigl_cotangent_voronoi_v1"
                if is_mean_curvature(curvature)
                else "angle_defect_mixed_voronoi_v1"
            )
        ),
    }
    cache_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{surface.name}_{level.name}_{cache_key}.npz"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            curve = [
                {
                    "filtration": float(filtration),
                    "delta_chi": int(delta),
                    "chi_after": int(chi),
                }
                for filtration, delta, chi in zip(
                    cached["curve_filtration"],
                    cached["curve_delta"],
                    cached["curve_chi"],
                    strict=True,
                )
            ]
            return LevelComputation(
                mesh=Mesh(cached["vertices"], cached["faces"]),
                vertex_curvature=cached["vertex_curvature"],
                curve=curve,
                events=None,
                curve_source=str(cached["curve_source"].item()),
                cache_hit=True,
            )

    if reference:
        mesh, vertex_curvature = generate_reference_mesh_and_curvature(
            surface, level, curvature=curvature
        )
        sampled_curve, events = lower_star_ecc(vertex_curvature, mesh.faces)
        curve = theoretical_ecc_curve(surface, curvature) or sampled_curve
        curve_source = "smooth_parametrization_reference"
    else:
        mesh = generate_mesh(surface, level)
        vertex_curvature, curve_source = mesh_level_curvature(
            surface, level, mesh, curvature=curvature
        )
        curve, events = lower_star_ecc(vertex_curvature, mesh.faces)

    np.savez_compressed(
        cache_path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_curvature=vertex_curvature,
        curve_filtration=np.asarray([row["filtration"] for row in curve], dtype=float),
        curve_delta=np.asarray([row["delta_chi"] for row in curve], dtype=int),
        curve_chi=np.asarray([row["chi_after"] for row in curve], dtype=int),
        curve_source=np.asarray(curve_source),
    )
    write_json(metadata_path, payload | {"cache_key": cache_key})
    return LevelComputation(
        mesh=mesh,
        vertex_curvature=vertex_curvature,
        curve=curve,
        events=events,
        curve_source=curve_source,
        cache_hit=False,
    )


def computation_events(computation: LevelComputation) -> list[SimplexEvent]:
    if computation.events is not None:
        return computation.events
    _, events = lower_star_ecc(
        computation.vertex_curvature, computation.mesh.faces
    )
    return events


def curve_content_hash(
    curves: list[tuple[str, list[dict[str, float | int]]]],
) -> str:
    return hashlib.sha256(
        json.dumps(curves, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def artifact_is_current(path: Path, payload: dict[str, object]) -> bool:
    metadata_path = path.with_suffix(path.suffix + ".cache.json")
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            return json.load(handle) == payload
    except (OSError, json.JSONDecodeError):
        return False


def mark_artifact_current(path: Path, payload: dict[str, object]) -> None:
    write_json(path.with_suffix(path.suffix + ".cache.json"), payload)


def run_config(config: ExperimentConfig) -> list[Path]:
    outputs: list[Path] = []
    for curvature in config.curvatures:
        output_dir = curvature_scoped_dir(config.output_dir, curvature)
        results_dir = curvature_scoped_dir(config.results_dir, curvature)
        outputs.extend(run_curvature_config(config, curvature, output_dir, results_dir))
    return outputs


def curvature_scoped_dir(path: Path, curvature: str) -> Path:
    if path.name == curvature:
        return path
    return path / curvature


def run_curvature_config(
    config: ExperimentConfig, curvature: str, output_dir: Path, results_dir: Path
) -> list[Path]:
    outputs: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    main_result_rows: list[MainResultRow] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    calculation_cache_dir = results_dir / "calculation_cache"

    for surface in config.surfaces:
        reference_computation = cached_level_computation(
            calculation_cache_dir,
            surface,
            config.reference_level,
            curvature=curvature,
            reference=True,
        )
        reference_mesh = reference_computation.mesh
        reference_curvature = reference_computation.vertex_curvature
        reference_face_curvature = triangle_mean(
            reference_curvature, reference_mesh.faces
        )
        surface_renderings = [
            SurfaceRendering("smooth", reference_mesh, reference_face_curvature)
        ]
        reference_curve = reference_computation.curve
        surface_curves = [("smooth", reference_curve)]
        reference_label = config.reference_level.name
        reference_stem = f"{surface.name}_{reference_label}"
        reference_dir = results_dir / surface.name / reference_label
        reference_dir.mkdir(parents=True, exist_ok=True)

        reference_vertex_path = reference_dir / f"{reference_stem}_smooth_curvature.csv"
        reference_events_path = reference_dir / f"{reference_stem}_ecc_events.csv"
        reference_curve_path = reference_dir / f"{reference_stem}_ecc_curve.csv"
        reference_summary_path = reference_dir / f"{reference_stem}_summary.json"
        reference_summary = run_summary(
            surface,
            config.reference_level,
            reference_mesh,
            reference_curvature,
            reference_curve,
            curve_source="smooth_parametrization_reference",
            curvature=curvature,
        )
        reference_paths = [
            reference_vertex_path,
            reference_events_path,
            reference_curve_path,
            reference_summary_path,
        ]
        if not reference_computation.cache_hit or not all(
            path.exists() for path in reference_paths
        ):
            write_vertex_curvature(
                reference_vertex_path, reference_mesh.vertices, reference_curvature
            )
            write_events(
                reference_events_path, computation_events(reference_computation)
            )
            write_curve(reference_curve_path, reference_curve)
            write_json(reference_summary_path, reference_summary)
        outputs.extend(
            reference_paths
        )
        summary_rows.append(reference_summary)

        for base_level in config.mesh_levels:
            level = surface_mesh_level(surface, base_level)
            stem = f"{surface.name}_{level.name}"
            computation = cached_level_computation(
                calculation_cache_dir,
                surface,
                level,
                curvature=curvature,
                reference=False,
            )
            mesh = computation.mesh
            vertex_curvature = computation.vertex_curvature
            mesh_curve_source = computation.curve_source
            face_curvature = triangle_mean(vertex_curvature, mesh.faces)
            surface_renderings.append(
                SurfaceRendering(level.name, mesh, face_curvature)
            )
            curve = computation.curve
            surface_curves.append((level.name, curve))

            poly_path = output_dir / f"{stem}.poly"
            result_dir = results_dir / surface.name / level.name
            result_dir.mkdir(parents=True, exist_ok=True)

            outputs.append(poly_path)

            vertex_path = result_dir / f"{stem}_vertex_curvature.csv"
            events_path = result_dir / f"{stem}_ecc_events.csv"
            curve_path = result_dir / f"{stem}_ecc_curve.csv"
            summary_path = result_dir / f"{stem}_summary.json"
            histogram_path = result_dir / f"{stem}_curvature_histogram.csv"

            output_group = [
                vertex_path,
                events_path,
                curve_path,
                histogram_path,
                summary_path,
            ]

            summary = run_summary(
                surface,
                level,
                mesh,
                vertex_curvature,
                curve,
                curve_source=mesh_curve_source,
                curvature=curvature,
            )
            all_level_paths = [poly_path, *output_group]
            if not computation.cache_hit or not all(
                path.exists() for path in all_level_paths
            ):
                write_poly(poly_path, mesh, face_curvature)
                write_vertex_curvature(vertex_path, mesh.vertices, vertex_curvature)
                write_events(events_path, computation_events(computation))
                write_curve(curve_path, curve)
                write_histogram(
                    histogram_path, vertex_curvature, bins=config.histogram_bins
                )
                write_json(summary_path, summary)
            summary_rows.append(summary)
            outputs.extend(output_group)

        all_renderings = ordered_surface_renderings(surface_renderings)
        rendering_assets = ensure_surface_rendering_assets(
            results_dir / "surface_assets",
            surface,
            curvature=curvature,
            reference_level=config.reference_level,
            renderings=all_renderings,
            normalization_renderings=main_result_renderings(surface_renderings),
        )
        for asset in rendering_assets.values():
            outputs.extend(asset.with_suffix(suffix) for suffix in (".pdf", ".png", ".json"))
        main_result_rows.append(
            MainResultRow(
                surface_name=surface.name,
                parameter_text=surface_parameter_text(surface),
                rendering_assets=rendering_assets,
                curves=main_result_curves(surface_curves),
                renderings=all_renderings,
            )
        )
    summary_path = results_dir / "mesh_summary.csv"
    write_mesh_summary(summary_path, summary_rows)
    outputs.append(summary_path)

    main_results_path = results_dir / "main_results.png"
    main_signature = {
        "plot_version": COMPOSITE_PLOT_VERSION,
        "kind": "main_results",
        "curvature": curvature,
        "plot_percentile": config.plot_percentile,
        "rows": [
            {
                "surface": row.surface_name,
                "assets": {key: str(value) for key, value in row.rendering_assets.items()},
                "curves": curve_content_hash(row.curves),
            }
            for row in main_result_rows
        ],
    }
    if not artifact_is_current(main_results_path, main_signature):
        write_main_results_plot(
            main_results_path,
            main_result_rows,
            plot_percentile=config.plot_percentile,
            curvature=curvature,
        )
        mark_artifact_current(main_results_path, main_signature)
    outputs.append(main_results_path)

    return outputs


def run_summary(
    surface: SurfaceSpec,
    level: MeshLevel,
    mesh: Mesh,
    vertex_curvature: np.ndarray,
    curve: list[dict[str, float | int]],
    *,
    curve_source: str,
    curvature: str,
) -> dict[str, object]:
    chi = euler_characteristic(len(mesh.vertices), mesh.faces)
    final_chi = int(curve[-1]["chi_after"]) if curve else 0
    return {
        "surface": surface.name,
        "surface_kind": surface.kind,
        "mesh_level": level.name,
        "u_segments": level.u_segments,
        "v_segments": level.v_segments,
        "vertex_count": int(len(mesh.vertices)),
        "edge_count": int(len(mesh_edges(mesh.faces))),
        "triangle_count": int(len(unique_faces(mesh.faces))),
        "mesh_euler_characteristic": int(chi),
        "final_ecc_euler_characteristic": final_chi,
        "unique_filtration_values": len(curve),
        "curvature": curvature,
        "curve_source": curve_source,
        "min_curvature": float(np.min(vertex_curvature)),
        "median_curvature": float(np.median(vertex_curvature)),
        "mean_curvature": float(np.mean(vertex_curvature)),
        "max_curvature": float(np.max(vertex_curvature)),
        "std_curvature": float(np.std(vertex_curvature)),
    }


def write_poly(path: Path, mesh: Mesh, face_curvature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("POINTS\n")
        for index, vertex in enumerate(mesh.vertices, start=1):
            handle.write(
                f"{index}: {format_float(vertex[0])} {format_float(vertex[1])} {format_float(vertex[2])}\n"
            )
        handle.write("POLYS\n")
        for index, (face, curvature) in enumerate(
            zip(mesh.faces, face_curvature, strict=True), start=1
        ):
            a, b, c = (int(vertex) + 1 for vertex in face)
            handle.write(
                f"{index}: {a} {b} {c} < c(1, 1, 1, {format_float(float(curvature))})\n"
            )
        handle.write("END\n")


def write_vertex_curvature(
    path: Path, vertices: np.ndarray, vertex_curvature: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["vertex", "x", "y", "z", "curvature"]
        )
        writer.writeheader()
        for index, (vertex, curvature) in enumerate(
            zip(vertices, vertex_curvature, strict=True)
        ):
            writer.writerow(
                {
                    "vertex": index + 1,
                    "x": format_float(float(vertex[0])),
                    "y": format_float(float(vertex[1])),
                    "z": format_float(float(vertex[2])),
                    "curvature": format_float(float(curvature)),
                }
            )


def write_events(path: Path, events: Iterable[SimplexEvent]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filtration", "sign", "dimension", "simplex"],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "filtration": format_float(event.filtration),
                    "sign": event.sign,
                    "dimension": event.dimension,
                    "simplex": " ".join(str(vertex + 1) for vertex in event.vertices),
                }
            )


def write_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filtration", "delta_chi", "chi"])
        writer.writeheader()
        for row in curve:
            writer.writerow(
                {
                    "filtration": format_float(float(row["filtration"])),
                    "delta_chi": row["delta_chi"],
                    "chi": row["chi_after"],
                }
            )


def write_histogram(path: Path, values: np.ndarray, *, bins: int) -> None:
    counts, edges = np.histogram(values, bins=bins)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        for index, count in enumerate(counts):
            writer.writerow(
                {
                    "bin_left": format_float(float(edges[index])),
                    "bin_right": format_float(float(edges[index + 1])),
                    "count": int(count),
                }
            )


def write_mesh_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_main_results_plot(
    path: Path,
    rows: list[MainResultRow],
    *,
    plot_percentile: float | None,
    curvature: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return

    figure_height = max(4.0 * len(rows), 6.0)
    fig = plt.figure(figsize=(18, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(
        len(rows),
        4,
        width_ratios=[1.35, 1.0, 1.0, 1.0],
        wspace=0.08,
        hspace=0.08,
    )
    column_titles = ["ECC", "smooth", "fine", "coarse"]

    for row_index, row in enumerate(rows):
        ecc_ax = fig.add_subplot(grid[row_index, 0])
        plot_main_result_ecc(
            ecc_ax,
            row.curves,
            title=display_surface_name(row.surface_name),
            parameter_text=row.parameter_text,
            plot_percentile=plot_percentile,
            curvature=curvature,
        )
        if row_index == 0:
            ecc_ax.set_title(column_titles[0], fontweight="bold")

        renderings = row.renderings[:3]
        if not renderings:
            continue
        norm = rendering_row_norm(row.renderings[:3])
        coordinates = np.concatenate(
            [rendering.mesh.vertices for rendering in row.renderings], axis=0
        )
        cmap = plt.get_cmap("coolwarm")
        surface_axes = []

        for column_offset, rendering in enumerate(renderings, start=1):
            if rendering.label in row.rendering_assets:
                ax = fig.add_subplot(grid[row_index, column_offset])
                add_cached_surface_image(ax, row.rendering_assets[rendering.label])
            else:
                ax = fig.add_subplot(grid[row_index, column_offset], projection="3d")
                add_surface_collection(ax, rendering, cmap=cmap, norm=norm)
                set_equal_3d_limits(ax, coordinates)
                ax.view_init(elev=24, azim=-52)
                ax.set_axis_off()
            surface_axes.append(ax)
            if row_index == 0:
                ax.set_title(column_titles[column_offset], fontweight="bold")

        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=surface_axes,
            fraction=0.025,
            pad=0.01,
            shrink=0.72,
        )
        colorbar.set_label(curvature_label(curvature), fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_main_result_ecc(
    ax: object,
    curves: list[tuple[str, list[dict[str, float | int]]]],
    *,
    title: str,
    parameter_text: str,
    plot_percentile: float | None,
    curvature: str,
) -> None:
    all_x_values: list[float] = []
    styles = {
        "smooth": {"color": "black", "linewidth": 2.2, "alpha": 0.75},
        "fine": {"color": "#1f77b4", "linewidth": 1.9, "alpha": 0.72},
        "coarse": {"color": "#d95f02", "linewidth": 1.9, "alpha": 0.72},
    }

    for label, curve in curves:
        if not curve:
            continue
        all_x_values.extend(float(row["filtration"]) for row in curve)
        plot_ecc_step(ax, curve, label=label, **styles.get(label, {"linewidth": 1.8}))

    if all_x_values:
        set_x_limits(ax, all_x_values, plot_percentile)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_ylabel(display_surface_name(title))
    ax.set_xlabel(curvature_label(curvature))
    ax.text(
        0.02,
        0.04,
        parameter_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.15",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.82,
            "linewidth": 0.45,
        },
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_ecc_step(
    ax: object,
    curve: list[dict[str, float | int]],
    *,
    label: str,
    color: str | None = None,
    linewidth: float = 1.8,
    alpha: float = 0.75,
    zorder: int = 3,
) -> None:
    x_values = [float(row["filtration"]) for row in curve]
    y_values = [int(row["chi_after"]) for row in curve]
    if len(x_values) == 1 and y_values == [0]:
        ax.plot(
            x_values,
            y_values,
            linestyle="none",
            marker="o",
            markersize=5.5,
            label=label,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        return
    line = ax.step(
        x_values,
        y_values,
        where="post",
        label=label,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )[0]
    line_color = line.get_color()

    previous_chi = 0
    for x_value, y_value in zip(x_values, y_values, strict=True):
        if y_value != previous_chi:
            ax.vlines(
                x_value,
                previous_chi,
                y_value,
                color=line_color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        previous_chi = y_value

    # A constant filtration on a chi=0 surface produces one ECC sample with
    # delta_chi=0.  A step line has zero extent in both directions in that
    # case, so mark its (filtration, chi) location explicitly rather than
    # silently dropping the numerical curve from the plot.
    if len(x_values) == 1 and int(curve[0]["delta_chi"]) == 0:
        ax.plot(
            [x_values[0]],
            [y_values[0]],
            linestyle="none",
            marker="|",
            markersize=18,
            markeredgewidth=max(2.0, linewidth * 1.4),
            color=line_color,
            alpha=alpha,
            zorder=zorder,
        )


def rendering_row_norm(renderings: list[SurfaceRendering]):
    values = np.concatenate([rendering.face_curvature for rendering in renderings])
    return zero_centered_norm(values)


def zero_centered_norm(values: np.ndarray):
    """Return a diverging normalization whose midpoint is always zero."""

    from matplotlib.colors import Normalize

    max_absolute = float(np.max(np.abs(values))) if len(values) else 0.0
    if max_absolute == 0.0:
        max_absolute = 1.0
    return Normalize(vmin=-max_absolute, vmax=max_absolute)


def ensure_surface_rendering_assets(
    asset_dir: Path,
    surface: SurfaceSpec,
    *,
    curvature: str,
    reference_level: MeshLevel,
    renderings: list[SurfaceRendering],
    normalization_renderings: list[SurfaceRendering] | None = None,
) -> dict[str, Path]:
    return {
        rendering.label: ensure_surface_rendering_asset(
            asset_dir,
            surface,
            curvature=curvature,
            reference_level=reference_level,
            renderings=renderings,
            target_label=rendering.label,
            normalization_renderings=normalization_renderings,
        )
        for rendering in renderings
    }


def ensure_surface_rendering_asset(
    asset_dir: Path,
    surface: SurfaceSpec,
    *,
    curvature: str,
    reference_level: MeshLevel,
    renderings: list[SurfaceRendering],
    target_label: str,
    normalization_renderings: list[SurfaceRendering] | None = None,
) -> Path:
    target = next(
        (rendering for rendering in renderings if rendering.label == target_label), None
    )
    if target is None:
        raise ValueError(f"No {target_label!r} rendering for {surface.name!r}")
    norm = rendering_row_norm(normalization_renderings or renderings)
    coordinates = np.concatenate(
        [rendering.mesh.vertices for rendering in renderings], axis=0
    )
    payload = {
        "asset_kind": "cached_surface_rendering_v2",
        "target_label": target_label,
        "target_content_hash": hashlib.sha256(
            target.mesh.vertices.tobytes()
            + target.mesh.faces.tobytes()
            + target.face_curvature.tobytes()
        ).hexdigest()[:16],
        "surface": surface.name,
        "surface_kind": surface.kind,
        "surface_parameters": surface.parameters,
        "curvature": curvature,
        "reference_level": {
            "name": reference_level.name,
            "u_segments": reference_level.u_segments,
            "v_segments": reference_level.v_segments,
        },
        "color_scale": {"vmin": float(norm.vmin), "vmax": float(norm.vmax)},
        "view": {"elev": 24, "azim": -52},
    }
    payload_json = json.dumps(payload, sort_keys=True)
    cache_key = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]
    asset_dir.mkdir(parents=True, exist_ok=True)
    png_path = asset_dir / f"{surface.name}_{target_label}_{cache_key}.png"
    pdf_path = png_path.with_suffix(".pdf")
    metadata_path = png_path.with_suffix(".json")

    if png_path.exists() and pdf_path.exists() and metadata_path.exists():
        return png_path

    write_cached_surface_asset(
        png_path,
        pdf_path,
        target,
        coordinates=coordinates,
        norm=norm,
    )
    write_json(metadata_path, payload | {"cache_key": cache_key})
    return png_path


def write_cached_surface_asset(
    png_path: Path,
    pdf_path: Path,
    rendering: SurfaceRendering,
    *,
    coordinates: np.ndarray,
    norm,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(4.8, 3.8))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    add_surface_collection(ax, rendering, cmap=cmap, norm=norm)
    set_equal_3d_limits(ax, coordinates)
    ax.view_init(elev=24, azim=-52)
    ax.set_axis_off()
    for path in (pdf_path, png_path):
        fig.savefig(
            path,
            dpi=260,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.01,
        )
    plt.close(fig)


def ordered_surface_renderings(
    renderings: list[SurfaceRendering],
) -> list[SurfaceRendering]:
    by_label = {rendering.label: rendering for rendering in renderings}
    preferred_order = ["smooth", "fine", "coarse", "medium"]
    ordered = [by_label[label] for label in preferred_order if label in by_label]
    ordered.extend(
        rendering
        for rendering in renderings
        if rendering.label not in set(preferred_order)
    )
    return ordered


def main_result_curves(
    curves: list[tuple[str, list[dict[str, float | int]]]],
) -> list[tuple[str, list[dict[str, float | int]]]]:
    by_label = {label: curve for label, curve in curves}
    preferred_order = ["smooth", "fine", "coarse"]
    return [(label, by_label[label]) for label in preferred_order if label in by_label]


def main_result_renderings(
    renderings: list[SurfaceRendering],
) -> list[SurfaceRendering]:
    by_label = {rendering.label: rendering for rendering in renderings}
    preferred_order = ["smooth", "fine", "coarse"]
    return [by_label[label] for label in preferred_order if label in by_label]


def display_surface_name(name: str) -> str:
    return name.replace("_", " ")


def surface_parameter_text(surface: SurfaceSpec) -> str:
    if surface.kind == "doubly_elliptic_torus":
        parameters = [
            ("R", float_parameter(surface, "R", 2.0)),
            ("r", float_parameter(surface, "r", 0.55)),
            ("a", float_parameter(surface, "a", 1.35)),
            ("b", float_parameter(surface, "b", 0.75)),
        ]
    elif surface.kind == "product_sine_graph":
        parameters = [
            ("domain", float_parameter(surface, "domain_size", 2.0)),
            ("A", float_parameter(surface, "amplitude", 1.0)),
            ("f", float_parameter(surface, "frequency", 1.0)),
        ]
    elif surface.kind == "helicoid_patch":
        parameters = [
            (
                "rho",
                format_interval(
                    float_parameter(surface, "rho_min", 0.0),
                    float_parameter(surface, "rho_max", 1.6),
                ),
            ),
            (
                "theta",
                format_interval(
                    float_parameter(surface, "theta_min", -math.pi),
                    float_parameter(surface, "theta_max", math.pi),
                ),
            ),
            ("alpha", float_parameter(surface, "alpha", 1.0)),
        ]
    elif surface.kind == "pseudosphere_patch":
        parameters = [
            ("radius", float_parameter(surface, "radius", 1.0)),
            (
                "u",
                format_interval(
                    float_parameter(surface, "u_min", 0.15),
                    float_parameter(surface, "u_max", 2.5),
                ),
            ),
        ]
    elif surface.kind == "ellipsoid":
        parameters = [
            ("a", float_parameter(surface, "a", 2.0)),
            ("b", float_parameter(surface, "b", 1.5)),
            ("c", float_parameter(surface, "c", 1.0)),
        ]
    else:
        parameters = sorted(surface.parameters.items())

    return ", ".join(
        f"{name}={format_parameter_value(value)}" for name, value in parameters
    )


def format_interval(lower: float, upper: float) -> str:
    return f"[{format_parameter_value(lower)}, {format_parameter_value(upper)}]"


def format_parameter_value(value: object) -> str:
    if isinstance(value, float):
        return format_parameter_float(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def format_parameter_float(value: float) -> str:
    if math.isclose(value, math.pi):
        return "pi"
    if math.isclose(value, -math.pi):
        return "-pi"
    if math.isclose(value, 0.0):
        return "0"
    return f"{value:.4g}"


def curvature_label(curvature: str) -> str:
    labels = {
        "gaussian_curvature": "Gaussian curvature",
        "mean_curvature": "Mean curvature",
    }
    return labels.get(curvature, curvature.replace("_", " "))


def is_mean_curvature(curvature: str) -> bool:
    return curvature == "mean_curvature"


def add_surface_collection(
    ax: object, rendering: SurfaceRendering, *, cmap, norm
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    face_count = len(rendering.mesh.faces)
    linewidth = surface_edge_linewidth(rendering.label, face_count)
    edgecolor = (0.12, 0.12, 0.12, 0.38) if linewidth > 0 else "none"
    collection = Poly3DCollection(
        rendering.mesh.vertices[rendering.mesh.faces],
        facecolors=[cmap(norm(value)) for value in rendering.face_curvature],
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=0.94,
    )
    ax.add_collection3d(collection)


def add_cached_surface_image(ax: object, image_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = plt.imread(image_path)
    ax.imshow(image)
    ax.set_axis_off()


def surface_edge_linewidth(label: str, face_count: int) -> float:
    if label == "smooth":
        return 0.0
    if face_count <= 1500:
        return 0.35
    if face_count <= 6000:
        return 0.14
    return 0.04


def set_x_limits(ax: object, values: list[float], percentile: float | None) -> None:
    limits = central_percentile_limits(values, percentile)
    lower, upper = limits if limits is not None else (min(values), max(values))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0)
        lower -= padding
        upper += padding
    else:
        padding = max((upper - lower) * 0.035, 1e-9)
        lower -= padding
        upper += padding
    ax.set_xlim(lower, upper)


def central_percentile_limits(
    values: list[float], percentile: float | None
) -> tuple[float, float] | None:
    if percentile is None or percentile >= 100.0 or not values:
        return None
    tail = (100.0 - percentile) / 2.0
    lower, upper = np.percentile(values, [tail, 100.0 - tail])
    if lower == upper:
        return None
    return (float(lower), float(upper))


def set_equal_3d_limits(ax: object, coordinates: np.ndarray) -> None:
    centers = coordinates.mean(axis=0)
    spans = np.ptp(coordinates, axis=0)
    radius = float(max(spans) / 2.0)
    if radius == 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def add_edge_weight(
    weights: dict[tuple[int, int], float], first: int, second: int, weight: float
) -> None:
    edge = tuple(sorted((first, second)))
    weights[edge] = weights.get(edge, 0.0) + weight


def cotangent(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(np.cross(first, second))
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    normalized = np.zeros_like(values)
    nonzero = norms > 0
    normalized[nonzero] = values[nonzero] / norms[nonzero, None]
    return normalized


def squared_norm(value: np.ndarray) -> float:
    return float(np.dot(value, value))


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    curvatures = parse_curvatures(raw_config)

    mesh_levels = tuple(
        parse_mesh_level(raw_level) for raw_level in raw_config.get("mesh_levels", [])
    )
    if not mesh_levels:
        raise ValueError(f"Config {path} must define at least one [[mesh_levels]]")

    raw_reference = raw_config.get("reference")
    if raw_reference is None:
        reference_level = default_reference_level(mesh_levels)
    else:
        reference_level = parse_mesh_level(raw_reference)

    surfaces = tuple(
        parse_surface_spec(raw_surface)
        for raw_surface in raw_config.get("surfaces", [])
    )
    if not surfaces:
        raise ValueError(f"Config {path} must define at least one [[surfaces]]")

    histogram_bins = int(raw_config.get("histogram_bins", 40))
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")

    plot_percentile = optional_float(raw_config, "plot_percentile")
    if plot_percentile is not None and not (0.0 < plot_percentile <= 100.0):
        raise ValueError("plot_percentile must lie in (0, 100]")

    return ExperimentConfig(
        output_dir=config_path(raw_config, "output_dir", path),
        results_dir=config_path(raw_config, "results_dir", path),
        mesh_levels=mesh_levels,
        reference_level=reference_level,
        surfaces=surfaces,
        curvatures=curvatures,
        histogram_bins=histogram_bins,
        plot_percentile=plot_percentile,
    )


def parse_curvatures(raw_config: dict[str, object]) -> tuple[str, ...]:
    raw_curvatures = raw_config.get("curvatures")
    if (
        not isinstance(raw_curvatures, list)
        or not raw_curvatures
        or not all(isinstance(value, str) for value in raw_curvatures)
    ):
        raise ValueError("'curvatures' must be a non-empty list of curvature names")

    unsupported = sorted(set(raw_curvatures) - set(SUPPORTED_CURVATURES))
    if unsupported:
        supported = "', '".join(SUPPORTED_CURVATURES)
        raise ValueError(
            f"Unsupported curvature(s): {', '.join(unsupported)}. "
            f"Supported values are '{supported}'."
        )
    return tuple(dict.fromkeys(raw_curvatures))


def default_reference_level(mesh_levels: tuple[MeshLevel, ...]) -> MeshLevel:
    return MeshLevel(
        name="smooth_reference",
        u_segments=max(level.u_segments for level in mesh_levels) * 2,
        v_segments=max(level.v_segments for level in mesh_levels) * 2,
    )


def parse_mesh_level(raw_level: object) -> MeshLevel:
    if not isinstance(raw_level, dict):
        raise ValueError(f"Mesh level must be a TOML table, got {raw_level!r}")
    name = stem_value(raw_level.get("name"), "mesh level name")
    u_segments = int(raw_level.get("u_segments", 0))
    v_segments = int(raw_level.get("v_segments", 0))
    if u_segments <= 0 or v_segments <= 0:
        raise ValueError(f"Mesh level {name!r} must have positive segment counts")
    return MeshLevel(name=name, u_segments=u_segments, v_segments=v_segments)


def parse_surface_spec(raw_surface: object) -> SurfaceSpec:
    if not isinstance(raw_surface, dict):
        raise ValueError(f"Surface must be a TOML table, got {raw_surface!r}")
    name = stem_value(raw_surface.get("name"), "surface name")
    kind = raw_surface.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"Surface {name!r} must define a non-empty kind")
    parameters = {
        key: value for key, value in raw_surface.items() if key not in {"name", "kind"}
    }
    return SurfaceSpec(name=name, kind=kind, parameters=parameters)


def stem_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty {label}, got {value!r}")
    if Path(value).name != value:
        raise ValueError(f"Expected {label} to be a single path stem, got {value!r}")
    return value


def config_path(config: dict[str, object], key: str, config_path_: Path) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config {config_path_} must define a non-empty {key!r}")
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path_.parent / path).resolve()


def optional_float(config: dict[str, object], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric, got {value!r}")
    return float(value)


def float_parameter(surface: SurfaceSpec, key: str, default: float) -> float:
    value = surface.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Surface {surface.name!r} parameter {key!r} must be numeric")
    return float(value)


def int_parameter(surface: SurfaceSpec, key: str, default: int) -> int:
    value = surface.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Surface {surface.name!r} parameter {key!r} must be an int")
    if value <= 0:
        raise ValueError(f"Surface {surface.name!r} parameter {key!r} must be positive")
    return value


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.12g}"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate parametrized meshes and curvature ECCs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML config for the parametrized surface experiment.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_config(load_experiment_config(args.config))
    for output in outputs:
        print(display_path(output))


if __name__ == "__main__":
    main()
