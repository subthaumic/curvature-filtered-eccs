"""Build curvature sublevel filtrations and Euler characteristic curves.

The input files contain weighted polygons in a ``POLYS`` section:

    polygon_id: v1 v2 v3 [v4 ...] < c(r, g, b, alpha)

Polygonal records with more than three vertices are fan-triangulated. The alpha
channel is treated as the triangle curvature. The filtration is the
lower-star-style sublevel filtration induced by triangle curvature:

* triangle filtration = its alpha value
* edge filtration = minimum filtration of incident triangles
* vertex filtration = minimum filtration of incident edges

The Euler characteristic data is written in two forms: signed simplex events
and the cumulative Euler characteristic curve:

    chi(t) = sum(sign of events with filtration_value <= t)

where vertices and triangles have sign +1 and edges have sign -1.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
POLY_RE = re.compile(
    r"""
    ^\s*
    (?P<id>\d+)\s*:\s*
    (?P<vertices>\d+(?:\s+\d+)*)
    \s*<\s*c\(
    (?P<rgba>[^)]*)
    \)
    \s*$
    """,
    re.VERBOSE,
)
POLY_VERTEX_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?P<id>\d+)\s*:\s*
    (?P<vertices>\d+(?:\s+\d+)*)
    (?:\s*<.*)?
    \s*$
    """,
    re.VERBOSE,
)
RGBA_RE = re.compile(
    rf"""
    ^\s*
    (?P<red>{FLOAT_PATTERN})\s*,\s*
    (?P<green>{FLOAT_PATTERN})\s*,\s*
    (?P<blue>{FLOAT_PATTERN})\s*,\s*
    (?P<alpha>{FLOAT_PATTERN})
    \s*$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Triangle:
    """Weighted triangle from the POLYS section."""

    triangle_id: int
    vertices: tuple[int, int, int]
    filtration: float


@dataclass(frozen=True)
class PolyParseResult:
    """Parsed POLYS records plus records excluded from the complex."""

    triangles: list[Triangle]
    skipped_degenerate_triangles: int
    removed_vertices_below_z_threshold: int = 0
    triangles_removed_by_z_threshold: int = 0
    z_min: float | None = None


@dataclass(frozen=True)
class FilteredSimplex:
    """A simplex with its filtration value."""

    dimension: int
    vertices: tuple[int, ...]
    filtration: float

    @property
    def sign(self) -> int:
        return 1 if self.dimension % 2 == 0 else -1


@dataclass(frozen=True)
class FilteredComponents:
    """Compact vertex, edge, and triangle filtrations for one run."""

    triangles: list[Triangle]
    edge_filtrations: dict[tuple[int, int], float]
    vertex_filtrations: dict[tuple[int], float]


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one curvature ECC run."""

    input_path: Path
    results_dir: Path
    output_name: str
    skip_degenerate: bool = True
    include_surface_plot: bool = False
    histogram_bins: int = 30
    histogram_symlog: bool = True
    plot_percentile: float | None = None
    z_min: float | None = None
    smooth_triangle_filtration: bool = False

    @property
    def output_dir(self) -> Path:
        return self.results_dir / self.output_name


def parse_poly_file(path: Path, *, skip_degenerate: bool = True) -> list[Triangle]:
    """Parse weighted triangles from the POLYS section of a .poly file."""

    return parse_poly_data(path, skip_degenerate=skip_degenerate).triangles


def parse_poly_data(path: Path, *, skip_degenerate: bool = True) -> PolyParseResult:
    """Parse weighted POLYS records and triangulate polygons when needed."""

    triangles: list[Triangle] = []
    skipped_degenerate_triangles = 0
    in_polys = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "POLYS":
                in_polys = True
                continue
            if in_polys and line == "END":
                break
            if not in_polys or not line:
                continue

            match = POLY_RE.match(line)
            if match is None:
                vertex_match = POLY_VERTEX_PREFIX_RE.match(line)
                if vertex_match is not None:
                    polygon_vertices = [
                        int(vertex)
                        for vertex in vertex_match.group("vertices").split()
                    ]
                    if len(polygon_vertices) < 3:
                        if not skip_degenerate:
                            raise ValueError(
                                f"Degenerate polygon in {path}:{line_number}: vertices={tuple(polygon_vertices)!r}"
                            )
                        skipped_degenerate_triangles += 1
                        continue
                raise ValueError(
                    f"Malformed POLYS line in {path}:{line_number}: {line!r}"
                )

            alpha = parse_alpha_channel(match.group("rgba"), path, line_number)

            polygon_vertices = [
                int(vertex) for vertex in match.group("vertices").split()
            ]
            if len(polygon_vertices) < 3:
                if not skip_degenerate:
                    raise ValueError(
                        f"Degenerate polygon in {path}:{line_number}: vertices={tuple(polygon_vertices)!r}"
                    )
                skipped_degenerate_triangles += 1
                continue

            polygon_triangles = fan_triangulate(polygon_vertices)
            for vertices in polygon_triangles:
                simplex_vertices = tuple(sorted(vertices))
                if len(set(simplex_vertices)) != 3:
                    if not skip_degenerate:
                        raise ValueError(
                            f"Degenerate triangle in {path}:{line_number}: vertices={simplex_vertices!r}"
                        )
                    skipped_degenerate_triangles += 1
                    continue

                triangles.append(
                    Triangle(
                        triangle_id=int(match.group("id")),
                        vertices=simplex_vertices,
                        filtration=alpha,
                    )
                )

    if not in_polys:
        raise ValueError(f"No POLYS section found in {path}")

    return PolyParseResult(
        triangles=triangles,
        skipped_degenerate_triangles=skipped_degenerate_triangles,
    )


def fan_triangulate(vertices: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate one POLYS record using its first vertex as the fan root."""

    if len(vertices) < 3:
        return []
    return [
        (vertices[0], vertices[index], vertices[index + 1])
        for index in range(1, len(vertices) - 1)
    ]


def parse_alpha_channel(rgba_text: str, path: Path, line_number: int) -> float:
    """Extract the alpha/curvature value from an explicit RGBA tuple."""

    match = RGBA_RE.match(rgba_text)
    if match is None:
        raise ValueError(
            f"Expected RGBA weight in {path}:{line_number}, got {rgba_text!r}"
        )
    return float(match.group("alpha"))


def parse_point_data(path: Path) -> dict[int, tuple[float, float, float]]:
    """Parse vertex coordinates from the POINTS section."""

    points: dict[int, tuple[float, float, float]] = {}
    in_points = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "POINTS":
                in_points = True
                continue
            if in_points and line == "POLYS":
                break
            if not in_points or not line:
                continue

            try:
                point_id, coordinates = line.split(":", maxsplit=1)
                xyz = tuple(float(value) for value in coordinates.split())
            except ValueError as exc:
                raise ValueError(
                    f"Malformed POINTS line in {path}:{line_number}: {line!r}"
                ) from exc

            if len(xyz) != 3:
                raise ValueError(
                    f"Expected 3 coordinates in {path}:{line_number}, got {line!r}"
                )

            points[int(point_id)] = xyz

    if not in_points:
        raise ValueError(f"No POINTS section found in {path}")

    return points


def filter_parse_result_by_z_min(
    source: Path, parse_result: PolyParseResult, z_min: float | None
) -> PolyParseResult:
    """Remove vertices below z_min and all triangles incident to them."""

    if z_min is None:
        return parse_result

    points = parse_point_data(source)
    removed_vertices = {
        point_id for point_id, coordinates in points.items() if coordinates[2] < z_min
    }

    kept_triangles: list[Triangle] = []
    removed_triangles = 0
    for triangle in parse_result.triangles:
        missing_vertices = [
            vertex for vertex in triangle.vertices if vertex not in points
        ]
        if missing_vertices:
            raise ValueError(
                f"Missing point coordinates for vertices {missing_vertices} in {source}"
            )
        if any(vertex in removed_vertices for vertex in triangle.vertices):
            removed_triangles += 1
            continue
        kept_triangles.append(triangle)

    return PolyParseResult(
        triangles=kept_triangles,
        skipped_degenerate_triangles=parse_result.skipped_degenerate_triangles,
        removed_vertices_below_z_threshold=len(removed_vertices),
        triangles_removed_by_z_threshold=removed_triangles,
        z_min=z_min,
    )


def edge_faces(vertices: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    """Return sorted edge faces of a sorted triangle."""

    v0, v1, v2 = vertices
    return ((v0, v1), (v0, v2), (v1, v2))


def build_filtered_complex(
    triangles: Iterable[Triangle], *, smooth_triangle_filtration: bool = False
) -> list[FilteredSimplex]:
    """Build vertex, edge, and triangle filtration values."""

    components = build_filtered_components(
        triangles,
        smooth_triangle_filtration=smooth_triangle_filtration,
    )
    simplices: list[FilteredSimplex] = []
    simplices.extend(
        FilteredSimplex(dimension=0, vertices=vertices, filtration=value)
        for vertices, value in components.vertex_filtrations.items()
    )
    simplices.extend(
        FilteredSimplex(dimension=1, vertices=edge, filtration=value)
        for edge, value in components.edge_filtrations.items()
    )
    simplices.extend(
        FilteredSimplex(
            dimension=2, vertices=triangle.vertices, filtration=triangle.filtration
        )
        for triangle in components.triangles
    )

    return sorted(simplices, key=simplex_sort_key)


def build_filtered_components(
    triangles: Iterable[Triangle], *, smooth_triangle_filtration: bool = False
) -> FilteredComponents:
    """Build compact vertex, edge, and triangle filtration values."""

    triangles = minimal_triangle_simplices(triangles)
    if smooth_triangle_filtration:
        triangles = smooth_triangle_filtrations(triangles)

    edge_filtrations: dict[tuple[int, int], float] = {}

    for triangle in triangles:
        for edge in edge_faces(triangle.vertices):
            current = edge_filtrations.get(edge)
            if current is None or triangle.filtration < current:
                edge_filtrations[edge] = triangle.filtration

    vertex_filtrations: dict[tuple[int], float] = {}
    for edge, edge_filtration in edge_filtrations.items():
        for vertex in edge:
            key = (vertex,)
            current = vertex_filtrations.get(key)
            if current is None or edge_filtration < current:
                vertex_filtrations[key] = edge_filtration

    return FilteredComponents(
        triangles=triangles,
        edge_filtrations=edge_filtrations,
        vertex_filtrations=vertex_filtrations,
    )


def minimal_triangle_simplices(triangles: Iterable[Triangle]) -> list[Triangle]:
    """Collapse duplicate triangle records to the earliest filtration value."""

    by_vertices: dict[tuple[int, int, int], Triangle] = {}
    for triangle in triangles:
        current = by_vertices.get(triangle.vertices)
        if current is None or triangle.filtration < current.filtration:
            by_vertices[triangle.vertices] = triangle
    return list(by_vertices.values())


def smooth_triangle_filtrations(triangles: list[Triangle]) -> list[Triangle]:
    """Average each triangle filtration over its closed edge-neighborhood."""

    edge_to_triangle_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, triangle in enumerate(triangles):
        for edge in edge_faces(triangle.vertices):
            edge_to_triangle_indices[edge].append(index)

    neighbor_indices = [{index} for index in range(len(triangles))]
    for indices in edge_to_triangle_indices.values():
        for index in indices:
            neighbor_indices[index].update(indices)

    return [
        Triangle(
            triangle_id=triangle.triangle_id,
            vertices=triangle.vertices,
            filtration=sum(triangles[index].filtration for index in indices)
            / len(indices),
        )
        for triangle, indices in zip(triangles, neighbor_indices, strict=True)
    ]


def simplex_sort_key(simplex: FilteredSimplex) -> tuple[float, int, tuple[int, ...]]:
    """Sort by filtration, then dimension, then vertex labels."""

    return (simplex.filtration, simplex.dimension, simplex.vertices)


def grouped_ecc_curve(
    simplices: Iterable[FilteredSimplex],
) -> list[dict[str, float | int]]:
    """Aggregate signed simplex events into ECC jumps."""

    deltas: dict[float, int] = defaultdict(int)
    for simplex in simplices:
        deltas[simplex.filtration] += simplex.sign

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
    return curve


def grouped_ecc_curve_components(
    components: FilteredComponents,
) -> list[dict[str, float | int]]:
    """Aggregate signed simplex events from compact component maps."""

    deltas: dict[float, int] = defaultdict(int)
    for filtration in components.vertex_filtrations.values():
        deltas[filtration] += 1
    for filtration in components.edge_filtrations.values():
        deltas[filtration] -= 1
    for triangle in components.triangles:
        deltas[triangle.filtration] += 1

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
    return curve


def simplex_record(simplex: FilteredSimplex) -> dict[str, object]:
    """JSON-serializable representation of a filtered simplex."""

    return {
        "dimension": simplex.dimension,
        "vertices": list(simplex.vertices),
        "filtration": simplex.filtration,
        "sign": simplex.sign,
    }


def simplex_record_from_values(
    dimension: int, vertices: tuple[int, ...], filtration: float
) -> dict[str, object]:
    """JSON-serializable representation of a simplex tuple."""

    return {
        "dimension": dimension,
        "vertices": list(vertices),
        "filtration": filtration,
        "sign": simplex_sign(dimension),
    }


def simplex_sign(dimension: int) -> int:
    return 1 if dimension % 2 == 0 else -1


def iter_component_simplices(
    components: FilteredComponents,
) -> Iterable[tuple[int, tuple[int, ...], float]]:
    """Yield compact simplex entries without allocating a combined simplex list."""

    for vertices, filtration in components.vertex_filtrations.items():
        yield (0, vertices, filtration)
    for vertices, filtration in components.edge_filtrations.items():
        yield (1, vertices, filtration)
    for triangle in components.triangles:
        yield (2, triangle.vertices, triangle.filtration)


def iter_sorted_component_simplices(
    components: FilteredComponents,
) -> Iterable[tuple[int, tuple[int, ...], float]]:
    """Yield compact simplex entries sorted like FilteredSimplex records."""

    vertices = sorted(
        (filtration, 0, simplex_vertices)
        for simplex_vertices, filtration in components.vertex_filtrations.items()
    )
    edges = sorted(
        (filtration, 1, simplex_vertices)
        for simplex_vertices, filtration in components.edge_filtrations.items()
    )
    triangles = sorted(
        (triangle.filtration, 2, triangle.vertices) for triangle in components.triangles
    )
    for filtration, dimension, simplex_vertices in heapq.merge(
        vertices, edges, triangles
    ):
        yield (dimension, simplex_vertices, filtration)


def write_filtered_complex(
    path: Path, source: Path, simplices: list[FilteredSimplex]
) -> None:
    payload = {
        "source": display_path(source),
        "simplices": [simplex_record(simplex) for simplex in simplices],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_filtered_complex_components(
    path: Path, source: Path, components: FilteredComponents
) -> None:
    """Stream the filtered complex JSON without building one giant payload."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        handle.write(f'  "source": {json.dumps(display_path(source))},\n')
        handle.write('  "simplices": [\n')
        first = True
        for dimension, vertices, filtration in iter_component_simplices(components):
            if not first:
                handle.write(",\n")
            first = False
            handle.write("    ")
            json.dump(
                simplex_record_from_values(dimension, vertices, filtration),
                handle,
            )
        if not first:
            handle.write("\n")
        handle.write("  ]\n")
        handle.write("}\n")


def write_events(path: Path, simplices: list[FilteredSimplex]) -> None:
    """Write signed simplex events for the Euler characteristic curve."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filtration",
                "sign",
                "dimension",
                "simplex",
            ],
        )
        writer.writeheader()
        for simplex in simplices:
            simplex_label = " ".join(str(vertex) for vertex in simplex.vertices)
            writer.writerow(
                {
                    "filtration": format_float(simplex.filtration),
                    "sign": simplex.sign,
                    "dimension": simplex.dimension,
                    "simplex": simplex_label,
                }
            )


def write_events_components(path: Path, components: FilteredComponents) -> None:
    """Write signed simplex events from compact components."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filtration",
                "sign",
                "dimension",
                "simplex",
            ],
        )
        writer.writeheader()
        for dimension, vertices, filtration in iter_sorted_component_simplices(
            components
        ):
            simplex_label = " ".join(str(vertex) for vertex in vertices)
            writer.writerow(
                {
                    "filtration": format_float(filtration),
                    "sign": simplex_sign(dimension),
                    "dimension": dimension,
                    "simplex": simplex_label,
                }
            )


def write_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filtration", "chi"])
        writer.writeheader()
        for row in curve:
            writer.writerow(
                {
                    "filtration": format_float(float(row["filtration"])),
                    "chi": row["chi_after"],
                }
            )


def write_curve_plot(
    path: Path,
    curve: list[dict[str, float | int]],
    *,
    title: str,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Write a step plot of the cumulative Euler characteristic curve."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    if curve:
        filtrations = [float(row["filtration"]) for row in curve]
        chi_values = [int(row["chi_after"]) for row in curve]
        ax.step(filtrations, chi_values, where="post", linewidth=1.8)
        if len(filtrations) <= 5000:
            ax.scatter(filtrations, chi_values, s=10)
        lower, upper = (
            x_limits if x_limits is not None else (min(filtrations), max(filtrations))
        )
        if lower == upper:
            padding = max(abs(lower) * 0.05, 1.0)
            lower -= padding
            upper += padding
        ax.set_xlim(lower, upper)
    else:
        ax.text(
            0.5,
            0.5,
            "No simplices",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("filtration value")
    ax.set_ylabel("Euler characteristic")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_curvature_histogram(
    path: Path, triangles: Iterable[Triangle], *, bins: int
) -> None:
    """Write binned triangle-curvature counts."""

    rows = curvature_histogram_rows(triangles, bins=bins)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "bin_left": format_float(float(row["bin_left"])),
                    "bin_right": format_float(float(row["bin_right"])),
                    "count": row["count"],
                }
            )


def write_curvature_histogram_plot(
    path: Path,
    triangles: Iterable[Triangle],
    *,
    bins: int,
    title: str,
    use_symlog: bool,
    plot_percentile: float | None,
) -> None:
    """Write a PNG histogram of triangle curvature values."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = triangle_curvatures(triangles)
    x_limits = central_percentile_limits(values, plot_percentile)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    if values:
        ax.hist(
            values,
            bins=bins,
            range=x_limits,
            color="#4f7cac",
            edgecolor="0.2",
            linewidth=0.7,
        )
        if use_symlog:
            ax.set_yscale("symlog", linthresh=1)
    else:
        ax.text(
            0.5,
            0.5,
            "No triangle curvatures",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(title)
    ax.set_xlabel("triangle curvature alpha")
    ylabel = "triangle count (symlog)" if use_symlog else "triangle count"
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def central_percentile_limits(
    values: Iterable[float], percentile: float | None
) -> tuple[float, float] | None:
    """Return the central percentile interval, or None for the full range."""

    if percentile is None or percentile >= 100:
        return None

    values = list(values)
    if not values:
        return None

    import numpy as np

    tail = (100.0 - percentile) / 2.0
    lower, upper = np.percentile(values, [tail, 100.0 - tail])
    if lower == upper:
        return None
    return (float(lower), float(upper))


def curvature_histogram_rows(
    triangles: Iterable[Triangle], *, bins: int
) -> list[dict[str, float | int]]:
    """Return equal-width histogram rows for triangle curvature values."""

    if bins <= 0:
        raise ValueError(f"Histogram bins must be positive, got {bins}")

    values = triangle_curvatures(triangles)
    if not values:
        return []

    import numpy as np

    counts, edges = np.histogram(values, bins=bins)
    return [
        {
            "bin_left": float(edges[index]),
            "bin_right": float(edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def triangle_curvatures(triangles: Iterable[Triangle]) -> list[float]:
    """Return curvature values for unique triangle simplices."""

    return [triangle.filtration for triangle in minimal_triangle_simplices(triangles)]


def write_surface_plot(path: Path, source: Path, triangles: Iterable[Triangle]) -> None:
    """Write a curvature-colored 3D plot of a triangulated surface."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = parse_point_data(source)
    triangles = minimal_triangle_simplices(triangles)
    faces: list[list[tuple[float, float, float]]] = []
    curvatures: list[float] = []

    for triangle in triangles:
        missing = [vertex for vertex in triangle.vertices if vertex not in points]
        if missing:
            raise ValueError(
                f"Missing point coordinates for vertices {missing} in {source}"
            )
        faces.append([points[vertex] for vertex in triangle.vertices])
        curvatures.append(triangle.filtration)

    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    if faces:
        lower = min(curvatures)
        upper = max(curvatures)
        if lower == upper:
            lower -= 1.0
            upper += 1.0

        cmap = plt.get_cmap("coolwarm")
        norm = Normalize(vmin=lower, vmax=upper)
        face_colors = [cmap(norm(value)) for value in curvatures]
        collection = Poly3DCollection(
            faces,
            facecolors=face_colors,
            edgecolors="0.18",
            linewidths=0.55,
            alpha=0.92,
        )
        ax.add_collection3d(collection)
        set_equal_3d_limits(ax, [coordinate for face in faces for coordinate in face])

        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            shrink=0.7,
            pad=0.08,
        )
        colorbar.set_label("curvature alpha")
    else:
        ax.text2D(
            0.5, 0.5, "No triangles", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    ax.set_title(f"Triangulated surface: {source.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-52)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def set_equal_3d_limits(
    ax: object, coordinates: list[tuple[float, float, float]]
) -> None:
    """Set equal x/y/z limits for a 3D matplotlib axis."""

    xs = [coordinate[0] for coordinate in coordinates]
    ys = [coordinate[1] for coordinate in coordinates]
    zs = [coordinate[2] for coordinate in coordinates]
    centers = [(min(values) + max(values)) / 2.0 for values in (xs, ys, zs)]
    radius = (
        max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
        )
        / 2.0
    )
    if radius == 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def write_summary(
    path: Path,
    source: Path,
    parse_result: PolyParseResult,
    simplices: list[FilteredSimplex],
    curve: list[dict[str, float | int]],
    *,
    plot_percentile: float | None,
    smooth_triangle_filtration: bool,
) -> None:
    counts_by_dimension = {
        str(dimension): sum(
            1 for simplex in simplices if simplex.dimension == dimension
        )
        for dimension in (0, 1, 2)
    }
    triangle_simplex_count = counts_by_dimension["2"]
    final_chi = int(curve[-1]["chi_after"]) if curve else 0
    filtration_values = [simplex.filtration for simplex in simplices]
    payload = {
        "source": display_path(source),
        "valid_triangle_records": len(parse_result.triangles),
        "skipped_degenerate_triangle_records": parse_result.skipped_degenerate_triangles,
        "z_min": parse_result.z_min,
        "vertices_removed_by_z_threshold": parse_result.removed_vertices_below_z_threshold,
        "triangles_removed_by_z_threshold": parse_result.triangles_removed_by_z_threshold,
        "duplicate_triangle_records_collapsed": len(parse_result.triangles)
        - triangle_simplex_count,
        "simplex_counts": counts_by_dimension,
        "unique_filtration_values": len(curve),
        "final_euler_characteristic": final_chi,
        "min_filtration": min(filtration_values) if filtration_values else None,
        "max_filtration": max(filtration_values) if filtration_values else None,
        "plot_percentile": plot_percentile,
        "smooth_triangle_filtration": smooth_triangle_filtration,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_summary_components(
    path: Path,
    source: Path,
    parse_result: PolyParseResult,
    components: FilteredComponents,
    curve: list[dict[str, float | int]],
    *,
    plot_percentile: float | None,
    smooth_triangle_filtration: bool,
) -> None:
    counts_by_dimension = {
        "0": len(components.vertex_filtrations),
        "1": len(components.edge_filtrations),
        "2": len(components.triangles),
    }
    final_chi = int(curve[-1]["chi_after"]) if curve else 0
    min_filtration, max_filtration = component_filtration_range(components)
    payload = {
        "source": display_path(source),
        "valid_triangle_records": len(parse_result.triangles),
        "skipped_degenerate_triangle_records": parse_result.skipped_degenerate_triangles,
        "z_min": parse_result.z_min,
        "vertices_removed_by_z_threshold": parse_result.removed_vertices_below_z_threshold,
        "triangles_removed_by_z_threshold": parse_result.triangles_removed_by_z_threshold,
        "duplicate_triangle_records_collapsed": len(parse_result.triangles)
        - len(components.triangles),
        "simplex_counts": counts_by_dimension,
        "unique_filtration_values": len(curve),
        "final_euler_characteristic": final_chi,
        "min_filtration": min_filtration,
        "max_filtration": max_filtration,
        "plot_percentile": plot_percentile,
        "smooth_triangle_filtration": smooth_triangle_filtration,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def component_filtration_range(
    components: FilteredComponents,
) -> tuple[float | None, float | None]:
    minimum: float | None = None
    maximum: float | None = None
    for _, _, filtration in iter_component_simplices(components):
        minimum = filtration if minimum is None else min(minimum, filtration)
        maximum = filtration if maximum is None else max(maximum, filtration)
    return (minimum, maximum)


def process_file(
    path: Path,
    output_dir: Path,
    *,
    skip_degenerate: bool = True,
    include_surface_plot: bool = False,
    histogram_bins: int = 30,
    histogram_symlog: bool = True,
    plot_percentile: float | None = None,
    z_min: float | None = None,
    smooth_triangle_filtration: bool = False,
) -> dict[str, Path]:
    parse_result = parse_poly_data(path, skip_degenerate=skip_degenerate)
    parse_result = filter_parse_result_by_z_min(path, parse_result, z_min)
    components = build_filtered_components(
        parse_result.triangles,
        smooth_triangle_filtration=smooth_triangle_filtration,
    )
    curve = grouped_ecc_curve_components(components)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    outputs = {
        "filtered_complex": output_dir / f"{stem}_filtered_complex.json",
        "ecc_events": output_dir / f"{stem}_ecc_events.csv",
        "ecc_curve": output_dir / f"{stem}_ecc_curve.csv",
        "ecc_plot": output_dir / f"{stem}_ecc_plot.png",
        "curvature_histogram": output_dir / f"{stem}_curvature_histogram.csv",
        "curvature_histogram_plot": output_dir / f"{stem}_curvature_histogram.png",
        "summary": output_dir / f"{stem}_summary.json",
    }
    if include_surface_plot:
        outputs["surface_plot"] = output_dir / f"{stem}_surface_plot.png"

    write_filtered_complex_components(outputs["filtered_complex"], path, components)
    write_events_components(outputs["ecc_events"], components)
    write_curve(outputs["ecc_curve"], curve)
    triangle_plot_values = [triangle.filtration for triangle in components.triangles]
    write_curve_plot(
        outputs["ecc_plot"],
        curve,
        title=f"ECC: {path.name}",
        x_limits=central_percentile_limits(triangle_plot_values, plot_percentile),
    )
    write_curvature_histogram(
        outputs["curvature_histogram"],
        parse_result.triangles,
        bins=histogram_bins,
    )
    write_curvature_histogram_plot(
        outputs["curvature_histogram_plot"],
        parse_result.triangles,
        bins=histogram_bins,
        title=f"Triangle curvature histogram: {path.name}",
        use_symlog=histogram_symlog,
        plot_percentile=plot_percentile,
    )
    if include_surface_plot:
        write_surface_plot(outputs["surface_plot"], path, parse_result.triangles)
    write_summary_components(
        outputs["summary"],
        path,
        parse_result,
        components,
        curve,
        plot_percentile=plot_percentile,
        smooth_triangle_filtration=smooth_triangle_filtration,
    )
    return outputs


def run_config(config: RunConfig) -> dict[str, Path]:
    """Run the analysis described by a TOML config."""

    return process_file(
        config.input_path,
        config.output_dir,
        skip_degenerate=config.skip_degenerate,
        include_surface_plot=config.include_surface_plot,
        histogram_bins=config.histogram_bins,
        histogram_symlog=config.histogram_symlog,
        plot_percentile=config.plot_percentile,
        z_min=config.z_min,
        smooth_triangle_filtration=config.smooth_triangle_filtration,
    )


def load_run_config(path: Path) -> RunConfig:
    """Load a curvature ECC run config from TOML."""

    with path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    input_path = config_path_value(raw_config, "input", path)
    results_dir = config_path_value(raw_config, "results_dir", path)
    output_name = config_output_name(raw_config, path)
    skip_degenerate = bool(raw_config.get("skip_degenerate", True))
    include_surface_plot = bool(raw_config.get("surface_plots", False))
    histogram_bins = int(raw_config.get("histogram_bins", 30))
    histogram_symlog = bool(raw_config.get("histogram_symlog", True))
    plot_percentile = config_optional_float(raw_config, "plot_percentile")
    z_min = config_optional_float(raw_config, "z_min")
    smooth_triangle_filtration = bool(
        raw_config.get("smooth_triangle_filtration", False)
    )

    if histogram_bins <= 0:
        raise ValueError(f"histogram_bins must be positive, got {histogram_bins}")
    if plot_percentile is not None and not (0.0 < plot_percentile <= 100.0):
        raise ValueError(
            f"plot_percentile must be in the interval (0, 100], got {plot_percentile}"
        )

    return RunConfig(
        input_path=input_path,
        results_dir=results_dir,
        output_name=output_name,
        skip_degenerate=skip_degenerate,
        include_surface_plot=include_surface_plot,
        histogram_bins=histogram_bins,
        histogram_symlog=histogram_symlog,
        plot_percentile=plot_percentile,
        z_min=z_min,
        smooth_triangle_filtration=smooth_triangle_filtration,
    )


def config_path_value(config: dict[str, object], key: str, config_path: Path) -> Path:
    """Resolve a required config path relative to the config file."""

    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config {config_path} must define a non-empty {key!r} path")

    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def config_optional_float(config: dict[str, object], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value {key!r} must be numeric, got {value!r}")
    return float(value)


def config_output_name(config: dict[str, object], config_path: Path) -> str:
    """Read and validate the required result subdirectory name."""

    value = config.get("output_name")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Config {config_path} must define a non-empty 'output_name' value"
        )

    output_name = Path(value)
    if output_name.is_absolute() or output_name.name != value:
        raise ValueError(
            f"Config {config_path} output_name must be a single directory name, got {value!r}"
        )
    if value in {".", ".."}:
        raise ValueError(
            f"Config {config_path} output_name must be a normal directory name, got {value!r}"
        )
    return value


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.12g}"


def display_path(path: Path) -> str:
    """Display workspace-local paths without machine-specific prefixes."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a curvature filtered simplicial complex and Euler characteristic curve."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML config file for one dataset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_run_config(args.config)
    outputs = run_config(config)
    print(f"Processed {display_path(config.input_path)}")
    for kind, output_path in outputs.items():
        print(f"  {kind}: {display_path(output_path)}")


if __name__ == "__main__":
    main()
