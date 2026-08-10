"""Generate dented-sphere POLYS fixtures with discrete mean curvature.

The surface is a radial graph over the unit sphere with a Gaussian dent at the
north pole. Meshes are generated as latitude-longitude triangulations at one or
more refinement stages. Triangle weights are signed mean curvature values,
computed from the triangulation via the cotangent Laplacian.
"""

from __future__ import annotations

import argparse
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StageConfig:
    name: str
    lat_segments: int
    lon_segments: int
    surface_plots: bool


@dataclass(frozen=True)
class GeneratorConfig:
    output_dir: Path
    radius: float
    dent_depth: float
    dent_width: float
    stages: tuple[StageConfig, ...]
    write_ecc_configs: bool
    ecc_config_dir: Path
    ecc_results_dir: Path
    skip_degenerate: bool
    histogram_bins: int


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def dented_radius(
    theta: np.ndarray | float, *, radius: float, dent_depth: float, dent_width: float
) -> np.ndarray | float:
    """Radius of a sphere with a Gaussian north-pole dent."""

    return radius - dent_depth * np.exp(-((theta / dent_width) ** 2))


def generate_dented_sphere_mesh(
    *,
    lat_segments: int,
    lon_segments: int,
    radius: float,
    dent_depth: float,
    dent_width: float,
) -> Mesh:
    """Generate a closed latitude-longitude triangulation of a dented sphere."""

    if lat_segments < 3:
        raise ValueError(f"lat_segments must be at least 3, got {lat_segments}")
    if lon_segments < 3:
        raise ValueError(f"lon_segments must be at least 3, got {lon_segments}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if dent_width <= 0:
        raise ValueError(f"dent_width must be positive, got {dent_width}")
    if not 0 <= dent_depth < radius:
        raise ValueError(
            f"dent_depth must satisfy 0 <= dent_depth < radius, got {dent_depth}"
        )

    vertices: list[tuple[float, float, float]] = []
    north_radius = dented_radius(
        0.0, radius=radius, dent_depth=dent_depth, dent_width=dent_width
    )
    vertices.append((0.0, 0.0, float(north_radius)))

    rings: list[list[int]] = []
    for lat_index in range(1, lat_segments):
        theta = math.pi * lat_index / lat_segments
        ring: list[int] = []
        ring_radius = dented_radius(
            theta, radius=radius, dent_depth=dent_depth, dent_width=dent_width
        )
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for lon_index in range(lon_segments):
            phi = 2.0 * math.pi * lon_index / lon_segments
            ring.append(len(vertices))
            vertices.append(
                (
                    float(ring_radius * sin_theta * math.cos(phi)),
                    float(ring_radius * sin_theta * math.sin(phi)),
                    float(ring_radius * cos_theta),
                )
            )
        rings.append(ring)

    south_radius = dented_radius(
        math.pi, radius=radius, dent_depth=dent_depth, dent_width=dent_width
    )
    south_index = len(vertices)
    vertices.append((0.0, 0.0, float(-south_radius)))

    faces: list[tuple[int, int, int]] = []
    first_ring = rings[0]
    for lon_index in range(lon_segments):
        faces.append(
            (
                0,
                first_ring[lon_index],
                first_ring[(lon_index + 1) % lon_segments],
            )
        )

    for ring_index in range(len(rings) - 1):
        upper = rings[ring_index]
        lower = rings[ring_index + 1]
        for lon_index in range(lon_segments):
            upper_left = upper[lon_index]
            upper_right = upper[(lon_index + 1) % lon_segments]
            lower_left = lower[lon_index]
            lower_right = lower[(lon_index + 1) % lon_segments]
            faces.append((upper_left, lower_left, upper_right))
            faces.append((upper_right, lower_left, lower_right))

    last_ring = rings[-1]
    for lon_index in range(lon_segments):
        faces.append(
            (
                last_ring[lon_index],
                south_index,
                last_ring[(lon_index + 1) % lon_segments],
            )
        )

    vertices_array = np.asarray(vertices, dtype=float)
    faces_array = orient_faces_outward(vertices_array, np.asarray(faces, dtype=int))
    return Mesh(vertices=vertices_array, faces=faces_array)


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient faces so normals point away from the origin."""

    oriented = faces.copy()
    for index, face in enumerate(oriented):
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        centroid = triangle.mean(axis=0)
        if np.dot(normal, centroid) < 0:
            oriented[index, 1], oriented[index, 2] = (
                oriented[index, 2],
                oriented[index, 1],
            )
    return oriented


def signed_vertex_mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Estimate signed vertex mean curvature with cotangent-Laplacian weights."""

    vertex_count = len(vertices)
    areas = np.zeros(vertex_count, dtype=float)
    normal_sums = np.zeros((vertex_count, 3), dtype=float)
    weights: dict[tuple[int, int], float] = {}

    for face in faces:
        i, j, k = (int(value) for value in face)
        vi, vj, vk = vertices[[i, j, k]]
        face_normal = np.cross(vj - vi, vk - vi)
        face_area = 0.5 * np.linalg.norm(face_normal)
        if face_area == 0:
            continue

        for vertex in (i, j, k):
            areas[vertex] += face_area / 3.0
            normal_sums[vertex] += face_normal

        add_edge_weight(weights, j, k, cotangent(vj - vi, vk - vi))
        add_edge_weight(weights, i, k, cotangent(vi - vj, vk - vj))
        add_edge_weight(weights, i, j, cotangent(vi - vk, vj - vk))

    laplace_sums = np.zeros((vertex_count, 3), dtype=float)
    for (i, j), weight in weights.items():
        laplace_sums[i] += weight * (vertices[j] - vertices[i])
        laplace_sums[j] += weight * (vertices[i] - vertices[j])

    normals = normalize_rows(normal_sums)
    laplace = np.zeros_like(laplace_sums)
    valid_area = areas > 0
    laplace[valid_area] = laplace_sums[valid_area] / (2.0 * areas[valid_area, None])

    return -0.5 * np.einsum("ij,ij->i", laplace, normals)


def triangle_mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Assign each triangle the mean of its three vertex mean curvatures."""

    vertex_curvature = signed_vertex_mean_curvature(vertices, faces)
    return vertex_curvature[faces].mean(axis=1)


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


def write_poly(path: Path, mesh: Mesh, face_curvature: np.ndarray) -> None:
    """Write a mesh with triangle mean curvature in the POLYS alpha slot."""

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


def write_ecc_config(path: Path, stage: StageConfig, config: GeneratorConfig) -> None:
    """Write a curvature_ecc.py TOML config for one generated stage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    input_path = relative_config_path(config.output_dir / f"{stage.name}.poly", path)
    results_dir = relative_config_path(config.ecc_results_dir, path)
    text = "\n".join(
        [
            f'input = "{input_path}"',
            f'results_dir = "{results_dir}"',
            f'output_name = "{stage.name}"',
            f"skip_degenerate = {toml_bool(config.skip_degenerate)}",
            f"surface_plots = {toml_bool(stage.surface_plots)}",
            f"histogram_bins = {config.histogram_bins}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def generate_from_config(config: GeneratorConfig) -> list[Path]:
    outputs: list[Path] = []
    for stage in config.stages:
        mesh = generate_dented_sphere_mesh(
            lat_segments=stage.lat_segments,
            lon_segments=stage.lon_segments,
            radius=config.radius,
            dent_depth=config.dent_depth,
            dent_width=config.dent_width,
        )
        curvature = triangle_mean_curvature(mesh.vertices, mesh.faces)
        poly_path = config.output_dir / f"{stage.name}.poly"
        write_poly(poly_path, mesh, curvature)
        outputs.append(poly_path)

        if config.write_ecc_configs:
            ecc_config_path = config.ecc_config_dir / f"{stage.name}.toml"
            write_ecc_config(ecc_config_path, stage, config)
            outputs.append(ecc_config_path)

    return outputs


def load_generator_config(path: Path) -> GeneratorConfig:
    with path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    stages = tuple(parse_stage_config(stage) for stage in raw_config.get("stages", []))
    if not stages:
        raise ValueError(f"Config {path} must define at least one [[stages]] entry")

    return GeneratorConfig(
        output_dir=config_path_value(raw_config, "output_dir", path),
        radius=float(raw_config.get("radius", 1.0)),
        dent_depth=float(raw_config.get("dent_depth", 0.42)),
        dent_width=float(raw_config.get("dent_width", 0.48)),
        stages=stages,
        write_ecc_configs=bool(raw_config.get("write_ecc_configs", True)),
        ecc_config_dir=config_path_value(raw_config, "ecc_config_dir", path),
        ecc_results_dir=config_path_value(raw_config, "ecc_results_dir", path),
        skip_degenerate=bool(raw_config.get("skip_degenerate", True)),
        histogram_bins=int(raw_config.get("histogram_bins", 30)),
    )


def parse_stage_config(raw_stage: object) -> StageConfig:
    if not isinstance(raw_stage, dict):
        raise ValueError(f"Stage entries must be tables, got {raw_stage!r}")

    name = raw_stage.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Stage must define a non-empty name, got {raw_stage!r}")
    if Path(name).name != name:
        raise ValueError(f"Stage name must be a single file stem, got {name!r}")

    lat_segments = int(raw_stage.get("lat_segments", 0))
    lon_segments = int(raw_stage.get("lon_segments", 0))
    surface_plots = bool(raw_stage.get("surface_plots", False))
    return StageConfig(
        name=name,
        lat_segments=lat_segments,
        lon_segments=lon_segments,
        surface_plots=surface_plots,
    )


def config_path_value(config: dict[str, object], key: str, config_path: Path) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config {config_path} must define a non-empty {key!r} path")

    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def relative_config_path(target: Path, config_path: Path) -> str:
    """Return target as a POSIX path relative to a TOML config file."""

    return Path(
        os.path.relpath(target.resolve(), start=config_path.parent.resolve())
    ).as_posix()


def toml_bool(value: bool) -> str:
    return "true" if value else "false"


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
        description="Generate dented-sphere POLYS fixtures with discrete mean curvature."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the dented-sphere generator TOML config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = generate_from_config(load_generator_config(args.config))
    for output in outputs:
        print(display_path(output))


if __name__ == "__main__":
    main()
