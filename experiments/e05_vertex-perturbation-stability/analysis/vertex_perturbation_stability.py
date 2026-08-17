"""Coordinate-perturbation stability experiment for curvature, ECC, and SECC."""

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
DEFAULT_CONFIG_PATH = HERE.parent / "config" / ("vertex_perturbation_" "stability.toml")


def import_script(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e04 = import_script(
    "e05_e04",
    HERE.parents[1]
    / "e04_triangulation-stability"
    / "analysis"
    / "subdivision_stability.py",
)
e02 = e04.e02

CACHE_VERSION = 1
ARTIFACT_VERSION = 5
NOISE_MODEL = "centered_standardized_gaussian_coordinates"
SCALE_REFERENCE = "unperturbed_median_edge_length"

REFERENCE_ERROR_METRICS = (
    "curvature_reference_error_l2",
    "ecc_reference_error_l2",
    "secc_reference_error_l2",
)
BASELINE_DISTANCE_METRICS = (
    "curvature_baseline_l2",
    "ecc_baseline_l2",
    "secc_baseline_l2",
)
PAIRWISE_METRICS = (
    "curvature_l2",
    "ecc_l2",
    "secc_l2",
)
DISPLACEMENT_METRICS = (
    "realized_coordinate_std",
    "vertex_displacement_rms",
    "median_vertex_displacement",
    "maximum_vertex_displacement",
)
MESH_QUALITY_METRICS = (
    "minimum_angle_degrees",
    "angle_q01_degrees",
    "angle_q05_degrees",
    "normal_reversal_fraction",
    "degenerate_face_fraction",
)
ECC_DIAGNOSTIC_METRICS = e04.ECC_DIAGNOSTIC_METRICS
SUMMARY_METRICS = (
    REFERENCE_ERROR_METRICS
    + BASELINE_DISTANCE_METRICS
    + DISPLACEMENT_METRICS
    + MESH_QUALITY_METRICS
    + ECC_DIAGNOSTIC_METRICS
)


@dataclass(frozen=True)
class Config:
    source_config: Path
    results_dir: Path
    subdivision_levels: int
    repetitions: int
    random_seed: int
    perturbation_scales: tuple[float, ...]
    noise_model: str
    scale_reference: str
    sample_count: int
    interval_padding_fraction: float
    constant_interval_padding: float
    base_levels: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class Realization:
    row: dict[str, object]
    curvature_values: np.ndarray
    weights: np.ndarray
    chi: np.ndarray
    secc: np.ndarray


def mesh_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = np.asarray(sorted(e02.mesh_edges(faces)), dtype=int)
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def median_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    lengths = mesh_edge_lengths(vertices, faces)
    value = float(np.median(lengths))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("The reference mesh must have a positive median edge length")
    return value


def perturb_vertices(
    vertices: np.ndarray,
    *,
    scale: float,
    reference_edge_length: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply zero-mean coordinate noise with exactly controlled axis-wise RMS."""

    if scale < 0.0 or reference_edge_length <= 0.0:
        raise ValueError(
            "Perturbation scale must be nonnegative and mesh scale positive"
        )
    displacements = np.zeros_like(vertices, dtype=float)
    if scale == 0.0:
        return np.asarray(vertices, dtype=float).copy(), displacements

    generator = np.random.default_rng(seed)
    standardized = generator.standard_normal(np.asarray(vertices).shape)
    standardized -= np.mean(standardized, axis=0, keepdims=True)
    axis_rms = np.sqrt(np.mean(standardized**2, axis=0))
    if np.any(axis_rms == 0.0):
        raise ValueError("At least two vertices are required for coordinate noise")
    standardized /= axis_rms
    displacements = scale * reference_edge_length * standardized
    return np.asarray(vertices, dtype=float) + displacements, displacements


def displacement_statistics(
    displacements: np.ndarray, reference_edge_length: float
) -> dict[str, float]:
    vertex_norms = np.linalg.norm(displacements, axis=1) / reference_edge_length
    return {
        "realized_coordinate_std": float(
            np.sqrt(np.mean(displacements**2)) / reference_edge_length
        ),
        "vertex_displacement_rms": float(np.sqrt(np.mean(vertex_norms**2))),
        "median_vertex_displacement": float(np.median(vertex_norms)),
        "maximum_vertex_displacement": float(np.max(vertex_norms)),
    }


def mesh_change_statistics(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    reference_edge_length: float,
) -> dict[str, float]:
    triangles = np.asarray(faces, dtype=int)
    reference = reference_vertices[triangles]
    candidate = candidate_vertices[triangles]
    reference_cross = np.cross(
        reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0]
    )
    candidate_cross = np.cross(
        candidate[:, 1] - candidate[:, 0], candidate[:, 2] - candidate[:, 0]
    )
    reference_norm = np.linalg.norm(reference_cross, axis=1)
    candidate_norm = np.linalg.norm(candidate_cross, axis=1)
    tolerance = 1e-12 * reference_edge_length**2
    valid = (reference_norm > tolerance) & (candidate_norm > tolerance)
    dot_products = np.einsum("ij,ij->i", reference_cross, candidate_cross)
    reversed_faces = valid & (dot_products <= 0.0)
    return {
        "normal_reversal_fraction": float(np.mean(reversed_faces)),
        "degenerate_face_fraction": float(np.mean(candidate_norm <= tolerance)),
    }


def curvature_and_weights(
    vertices: np.ndarray, faces: np.ndarray, curvature: str
) -> tuple[np.ndarray, np.ndarray]:
    if curvature == "gaussian_curvature":
        values, weights = e04.vertex_gaussian_curvature_and_areas(vertices, faces)
    else:
        values = e02.mesh_vertex_curvature(vertices, faces, curvature=curvature)
        weights = e04.vertex_mixed_areas(vertices, faces)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Coordinate perturbation produced non-finite curvature")
    return values, weights


def zero_distance() -> dict[str, float]:
    return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "bias": 0.0}


def compute_realization(
    surface: object,
    mesh: object,
    subdivision_level: int,
    curvature: str,
    analytic_values: np.ndarray,
    reference_chi: np.ndarray,
    reference_secc: np.ndarray,
    physical_thresholds: np.ndarray,
    characteristic_length: float,
    reference_edge_length: float,
    config: Config,
    perturbation_scale: float,
    repetition: int,
    seed: int,
    baseline: Realization | None,
) -> Realization:
    vertices, displacements = perturb_vertices(
        mesh.vertices,
        scale=perturbation_scale,
        reference_edge_length=reference_edge_length,
        seed=seed,
    )
    values, weights = curvature_and_weights(vertices, mesh.faces, curvature)
    curvature_scale = (
        characteristic_length**2
        if curvature == "gaussian_curvature"
        else characteristic_length
    )
    dimensionless_values = curvature_scale * values
    dimensionless_analytic = curvature_scale * analytic_values
    curve, _ = e02.lower_star_ecc(values, mesh.faces)
    chi, secc = e04.normalized_curve_arrays(curve, physical_thresholds)

    curvature_reference = e04.weighted_errors(
        dimensionless_values, dimensionless_analytic, weights
    )
    ecc_reference = e04.curve_errors(chi, reference_chi)
    secc_reference = e04.curve_errors(secc, reference_secc)

    if baseline is None:
        curvature_baseline = zero_distance()
        ecc_baseline = zero_distance()
        secc_baseline = zero_distance()
    else:
        baseline_weights = 0.5 * (weights + baseline.weights)
        curvature_baseline = e04.weighted_errors(
            dimensionless_values, baseline.curvature_values, baseline_weights
        )
        ecc_baseline = e04.curve_errors(chi, baseline.chi)
        secc_baseline = e04.curve_errors(secc, baseline.secc)

    quality = e04.mesh_quality_statistics(vertices, mesh.faces)
    quality.update(
        mesh_change_statistics(
            mesh.vertices, vertices, mesh.faces, reference_edge_length
        )
    )
    displacement = displacement_statistics(displacements, reference_edge_length)
    diagnostics = e04.ecc_jump_diagnostics(dimensionless_values, mesh.faces, chi)
    mean_edge_length = float(np.mean(mesh_edge_lengths(vertices, mesh.faces)))
    row: dict[str, object] = {
        "surface": surface.name,
        "curvature": curvature,
        "subdivision_level": subdivision_level,
        "perturbation_scale": perturbation_scale,
        "repetition": repetition,
        "seed": seed,
        "noise_model": config.noise_model,
        "scale_reference": config.scale_reference,
        "vertex_count": len(vertices),
        "face_count": len(mesh.faces),
        "characteristic_length": characteristic_length,
        "reference_median_edge_length": reference_edge_length,
        "normalized_mean_edge_length": mean_edge_length / characteristic_length,
        **displacement,
        "curvature_reference_error_l1": curvature_reference["l1"],
        "curvature_reference_error_l2": curvature_reference["l2"],
        "curvature_reference_error_linf": curvature_reference["linf"],
        "curvature_reference_error_bias": curvature_reference["bias"],
        "ecc_reference_error_l1": ecc_reference["l1"],
        "ecc_reference_error_l2": ecc_reference["l2"],
        "ecc_reference_error_linf": ecc_reference["linf"],
        "secc_reference_error_l2": secc_reference["l2"],
        "secc_reference_error_linf": secc_reference["linf"],
        "curvature_baseline_l1": curvature_baseline["l1"],
        "curvature_baseline_l2": curvature_baseline["l2"],
        "curvature_baseline_linf": curvature_baseline["linf"],
        "curvature_baseline_bias": curvature_baseline["bias"],
        "ecc_baseline_l1": ecc_baseline["l1"],
        "ecc_baseline_l2": ecc_baseline["l2"],
        "ecc_baseline_linf": ecc_baseline["linf"],
        "secc_baseline_l2": secc_baseline["l2"],
        "secc_baseline_linf": secc_baseline["linf"],
        **quality,
        **diagnostics,
    }
    return Realization(row, dimensionless_values, weights, chi, secc)


def pairwise_row(first: Realization, second: Realization) -> dict[str, object]:
    first_row, second_row = first.row, second.row
    weights = 0.5 * (first.weights + second.weights)
    curvature = e04.weighted_errors(
        first.curvature_values, second.curvature_values, weights
    )
    ecc = e04.curve_errors(first.chi, second.chi)
    secc = e04.curve_errors(first.secc, second.secc)
    return {
        "surface": first_row["surface"],
        "curvature": first_row["curvature"],
        "subdivision_level": first_row["subdivision_level"],
        "perturbation_scale": first_row["perturbation_scale"],
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
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return cache_dir / digest


def load_cached_realization(stem: Path) -> Realization | None:
    metadata_path = stem.with_suffix(".json")
    array_path = stem.with_suffix(".npz")
    if not metadata_path.exists() or not array_path.exists():
        return None
    try:
        row = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(array_path) as arrays:
            return Realization(
                row,
                arrays["curvature_values"],
                arrays["weights"],
                arrays["chi"],
                arrays["secc"],
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def save_cached_realization(stem: Path, realization: Realization) -> None:
    e02.write_json(stem.with_suffix(".json"), realization.row)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        curvature_values=realization.curvature_values,
        weights=realization.weights,
        chi=realization.chi,
        secc=realization.secc,
    )


def realization_payload(
    surface: object,
    config: Config,
    base_level: tuple[int, int],
    subdivision_level: int,
    curvature: str,
    perturbation_scale: float,
    repetition: int,
    seed: int,
    thresholds: np.ndarray,
) -> dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "surface": surface.name,
        "surface_parameters": surface.parameters,
        "base_level": list(base_level),
        "subdivision_level": subdivision_level,
        "curvature": curvature,
        "perturbation_scale": perturbation_scale,
        "repetition": repetition,
        "seed": seed,
        "noise_model": config.noise_model,
        "scale_reference": config.scale_reference,
        "sample_count": config.sample_count,
        "threshold_interval": [float(thresholds[0]), float(thresholds[-1])],
    }


def percentile_summary(values: list[float]) -> tuple[float, float, float]:
    return tuple(
        float(value) for value in np.quantile(np.asarray(values), (0.25, 0.5, 0.75))
    )


def summarize_realizations(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["surface"],
                row["curvature"],
                row["subdivision_level"],
                row["perturbation_scale"],
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (surface, curvature, level, scale), members in grouped.items():
        summary: dict[str, object] = {
            "surface": surface,
            "curvature": curvature,
            "subdivision_level": level,
            "perturbation_scale": scale,
            "realizations": len(members),
            "vertex_count": members[0]["vertex_count"],
            "face_count": members[0]["face_count"],
            "reference_median_edge_length": members[0]["reference_median_edge_length"],
        }
        for metric in SUMMARY_METRICS:
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
                row["perturbation_scale"],
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (surface, curvature, level, scale), members in grouped.items():
        summary: dict[str, object] = {
            "surface": surface,
            "curvature": curvature,
            "subdivision_level": level,
            "perturbation_scale": scale,
            "pairs": len(members),
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


def artifact_signature(
    realization_rows: list[dict[str, object]], pairwise_rows: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "realizations": hashlib.sha256(
            json.dumps(realization_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "pairwise": hashlib.sha256(
            json.dumps(pairwise_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def axis_linthreshold(values: np.ndarray) -> float:
    positive = np.abs(values[np.isfinite(values) & (values != 0.0)])
    return max(float(np.min(positive)) * 0.5, 1e-14) if len(positive) else 1e-14


def write_metric_grid(
    path: Path,
    rows: list[dict[str, object]],
    metrics: tuple[str, ...],
    titles: tuple[str, ...],
    *,
    curvature_filter: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if curvature_filter is not None:
        rows = [row for row in rows if row["curvature"] == curvature_filter]
    surfaces = tuple(dict.fromkeys(str(row["surface"]) for row in rows))
    curvatures = tuple(dict.fromkeys(str(row["curvature"]) for row in rows))
    levels = sorted({int(row["subdivision_level"]) for row in rows})
    figure_rows = len(surfaces) * len(curvatures)
    fig, raw_axes = plt.subplots(
        figure_rows,
        len(metrics),
        figsize=(4.5 * len(metrics), 2.65 * figure_rows),
        constrained_layout=True,
    )
    axes = np.asarray(raw_axes, dtype=object).reshape(figure_rows, len(metrics))
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, max(len(levels), 2)))

    for curvature_index, curvature in enumerate(curvatures):
        for surface_index, surface in enumerate(surfaces):
            row_index = curvature_index * len(surfaces) + surface_index
            for column, (metric, title) in enumerate(zip(metrics, titles, strict=True)):
                ax = axes[row_index, column]
                if row_index == 0:
                    ax.set_title(title)
                panel_values: list[float] = []
                panel_scales: list[float] = []
                for color, level in zip(colors, levels, strict=False):
                    selected = [
                        row
                        for row in rows
                        if row["surface"] == surface
                        and row["curvature"] == curvature
                        and int(row["subdivision_level"]) == level
                    ]
                    selected.sort(key=lambda row: float(row["perturbation_scale"]))
                    if not selected:
                        continue
                    x = np.asarray(
                        [float(row["perturbation_scale"]) for row in selected]
                    )
                    median = np.asarray(
                        [float(row[f"{metric}_median"]) for row in selected]
                    )
                    q25 = np.asarray([float(row[f"{metric}_q25"]) for row in selected])
                    q75 = np.asarray([float(row[f"{metric}_q75"]) for row in selected])
                    ax.plot(x, median, color=color, marker="o", label=f"level {level}")
                    ax.fill_between(x, q25, q75, color=color, alpha=0.14)
                    panel_scales.extend(x.tolist())
                    panel_values.extend(q25.tolist() + median.tolist() + q75.tolist())
                scale_array = np.asarray(panel_scales)
                value_array = np.asarray(panel_values)
                if len(scale_array):
                    if np.any(scale_array == 0.0):
                        ax.set_xscale(
                            "symlog", linthresh=axis_linthreshold(scale_array)
                        )
                        ticks = [0.0] + [
                            value
                            for value in sorted(set(scale_array))
                            if value > 0.0
                            and math.isclose(
                                math.log10(value), round(math.log10(value))
                            )
                        ]
                        ax.set_xticks(ticks)
                        ax.set_xticklabels(
                            [
                                (
                                    "0"
                                    if value == 0.0
                                    else f"$10^{{{round(math.log10(value))}}}$"
                                )
                                for value in ticks
                            ]
                        )
                    else:
                        ax.set_xscale("log")
                if len(value_array):
                    nonzero = value_array[
                        np.isfinite(value_array) & (value_array != 0.0)
                    ]
                    if len(nonzero):
                        if np.all(value_array[np.isfinite(value_array)] > 0.0):
                            ax.set_yscale("log")
                        else:
                            ax.set_yscale(
                                "symlog", linthresh=axis_linthreshold(value_array)
                            )
                    elif metric.endswith("fraction"):
                        ax.set_ylim(-0.0025, 0.05)
                    else:
                        ax.set_ylim(-0.5, 0.5)
                ax.grid(alpha=0.25)
                ax.set_xlabel("coordinate std / median edge length")
                if column == 0:
                    label = surface.replace("_", " ")
                    if len(curvatures) > 1:
                        label += f"\n{e02.curvature_label(curvature)}"
                    ax.set_ylabel(label)
                if row_index == 0 and column == len(metrics) - 1:
                    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_displacement_plot(path: Path, rows: list[dict[str, object]]) -> None:
    write_metric_grid(
        path,
        rows,
        DISPLACEMENT_METRICS,
        (
            "Realized coordinate RMS",
            "Vertex displacement RMS",
            "Median vertex displacement",
            "Maximum vertex displacement",
        ),
        curvature_filter="gaussian_curvature",
    )


def write_mesh_quality_plot(path: Path, rows: list[dict[str, object]]) -> None:
    metrics = (
        "minimum_angle_degrees",
        "angle_q01_degrees",
        "normal_reversal_fraction",
        "degenerate_face_fraction",
    )
    write_metric_grid(
        path,
        rows,
        metrics,
        (
            "Smallest triangle angle",
            "1st percentile triangle angle",
            "Face-normal reversals",
            "Degenerate faces",
        ),
        curvature_filter="gaussian_curvature",
    )


def representative_plot_scales(
    scales: tuple[float, ...], maximum: int = 5
) -> tuple[float, ...]:
    count = min(maximum, len(scales))
    indices = np.unique(np.rint(np.linspace(0, len(scales) - 1, count)).astype(int))
    return tuple(scales[int(index)] for index in indices)


def write_ecc_curve_comparison(
    path: Path,
    curve_groups: dict[tuple[str, str, int, float], list[np.ndarray]],
    reference_curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    curvature: str,
    scales: tuple[float, ...],
    subdivision_levels: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surfaces = [
        surface
        for surface, candidate_curvature in reference_curves
        if candidate_curvature == curvature
    ]
    fig, raw_axes = plt.subplots(
        len(surfaces),
        subdivision_levels,
        figsize=(5.0 * subdivision_levels, 3.0 * len(surfaces)),
        constrained_layout=True,
    )
    axes = np.asarray(raw_axes, dtype=object).reshape(len(surfaces), subdivision_levels)
    colors = plt.get_cmap("plasma")(np.linspace(0.12, 0.85, len(scales)))

    for surface_index, surface in enumerate(surfaces):
        thresholds, reference_chi = reference_curves[(surface, curvature)]
        for subdivision_level in range(subdivision_levels):
            ax = axes[surface_index, subdivision_level]
            ax.step(
                thresholds,
                reference_chi,
                where="post",
                color="black",
                linewidth=2.4,
                label="smooth reference",
                zorder=5,
            )
            baseline = curve_groups[(surface, curvature, subdivision_level, 0.0)][0]
            ax.step(
                thresholds,
                baseline,
                where="post",
                color="0.45",
                linestyle="--",
                linewidth=1.8,
                label="unperturbed mesh",
                zorder=4,
            )
            for color, scale in zip(colors, scales, strict=True):
                curves = np.stack(
                    curve_groups[(surface, curvature, subdivision_level, scale)]
                )
                q25, median, q75 = np.quantile(curves, (0.25, 0.5, 0.75), axis=0)
                ax.step(
                    thresholds,
                    median,
                    where="post",
                    color=color,
                    linewidth=1.35,
                    label=f"noise {scale:g}",
                )
                ax.fill_between(
                    thresholds,
                    q25,
                    q75,
                    step="post",
                    color=color,
                    alpha=0.12,
                )
            ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=0)
            ax.grid(alpha=0.22)
            if surface_index == 0:
                ax.set_title(f"subdivision level {subdivision_level}")
            if subdivision_level == 0:
                ax.set_ylabel(f"{surface.replace('_', ' ')}\nEuler characteristic χ")
            if surface_index == len(surfaces) - 1:
                symbol = "K" if curvature == "gaussian_curvature" else "H"
                ax.set_xlabel(f"curvature threshold {symbol}")

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 7),
        fontsize=9,
    )
    fig.suptitle(
        f"{e02.curvature_label(curvature)} ECC: reference and coordinate perturbations\n"
        "noise = coordinate std / median edge length; colored curves are medians "
        "with interquartile ribbons",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config: Config) -> list[Path]:
    source = e02.load_experiment_config(config.source_config)
    missing = sorted(
        {surface.name for surface in source.surfaces} - set(config.base_levels)
    )
    if missing:
        raise ValueError(f"Missing base mesh entries for: {', '.join(missing)}")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.results_dir / "calculation_cache"
    cache_dir.mkdir(exist_ok=True)
    realization_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    plotted_scales = representative_plot_scales(config.perturbation_scales)
    curve_groups: dict[tuple[str, str, int, float], list[np.ndarray]] = defaultdict(
        list
    )
    reference_curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for surface in source.surfaces:
        base_level = config.base_levels[surface.name]
        meshes = [e04.initial_parametric_mesh(surface, *base_level)]
        for _ in range(1, config.subdivision_levels):
            meshes.append(e04.projected_midpoint_subdivision(surface, meshes[-1]))
        base_weights = e04.vertex_mixed_areas(meshes[0].vertices, meshes[0].faces)
        characteristic_length = math.sqrt(float(np.sum(base_weights)))

        for curvature in source.curvatures:
            thresholds, reference_chi, reference_secc = e04.smooth_reference(
                source,
                surface,
                curvature,
                config.sample_count,
                config.interval_padding_fraction,
                config.constant_interval_padding,
            )
            reference_curves[(surface.name, curvature)] = (
                thresholds.copy(),
                reference_chi.copy(),
            )
            for subdivision_level, mesh in enumerate(meshes):
                analytic_values = np.asarray(
                    [
                        e02.smooth_curvature(surface, *point, curvature=curvature)
                        for point in mesh.parameters
                    ]
                )
                reference_edge_length = median_edge_length(mesh.vertices, mesh.faces)

                baseline_seed = e04.stable_seed(
                    config.random_seed, surface.name, subdivision_level, 0.0, 0
                )
                baseline_payload = realization_payload(
                    surface,
                    config,
                    base_level,
                    subdivision_level,
                    curvature,
                    0.0,
                    0,
                    baseline_seed,
                    thresholds,
                )
                baseline_stem = cache_stem(cache_dir, baseline_payload)
                baseline = load_cached_realization(baseline_stem)
                if baseline is None:
                    baseline = compute_realization(
                        surface,
                        mesh,
                        subdivision_level,
                        curvature,
                        analytic_values,
                        reference_chi,
                        reference_secc,
                        thresholds,
                        characteristic_length,
                        reference_edge_length,
                        config,
                        0.0,
                        0,
                        baseline_seed,
                        None,
                    )
                    save_cached_realization(baseline_stem, baseline)
                realization_rows.append(baseline.row)
                curve_groups[(surface.name, curvature, subdivision_level, 0.0)].append(
                    baseline.chi
                )

                for perturbation_scale in config.perturbation_scales:
                    realizations: list[Realization] = []
                    for repetition in range(config.repetitions):
                        seed = e04.stable_seed(
                            config.random_seed,
                            surface.name,
                            subdivision_level,
                            perturbation_scale,
                            repetition,
                        )
                        payload = realization_payload(
                            surface,
                            config,
                            base_level,
                            subdivision_level,
                            curvature,
                            perturbation_scale,
                            repetition,
                            seed,
                            thresholds,
                        )
                        stem = cache_stem(cache_dir, payload)
                        realization = load_cached_realization(stem)
                        if realization is None:
                            realization = compute_realization(
                                surface,
                                mesh,
                                subdivision_level,
                                curvature,
                                analytic_values,
                                reference_chi,
                                reference_secc,
                                thresholds,
                                characteristic_length,
                                reference_edge_length,
                                config,
                                perturbation_scale,
                                repetition,
                                seed,
                                baseline,
                            )
                            save_cached_realization(stem, realization)
                        realizations.append(realization)
                        realization_rows.append(realization.row)
                        if perturbation_scale in plotted_scales:
                            curve_groups[
                                (
                                    surface.name,
                                    curvature,
                                    subdivision_level,
                                    perturbation_scale,
                                )
                            ].append(realization.chi)
                    pairwise_rows.extend(
                        pairwise_row(first, second)
                        for first, second in itertools.combinations(realizations, 2)
                    )

    summaries = summarize_realizations(realization_rows)
    pairwise_summaries = summarize_pairwise(pairwise_rows)
    signature = artifact_signature(realization_rows, pairwise_rows)
    outputs: list[Path] = []
    for filename, rows in (
        ("perturbation_realizations.csv", realization_rows),
        ("perturbation_summary.csv", summaries),
        ("perturbation_pairwise.csv", pairwise_rows),
        ("perturbation_pairwise_summary.csv", pairwise_summaries),
    ):
        path = config.results_dir / filename
        if not e02.artifact_is_current(path, signature):
            write_rows(path, rows)
            e02.mark_artifact_current(path, signature)
        outputs.append(path)

    plot_specs = (
        (
            "main_results.png",
            summaries,
            BASELINE_DISTANCE_METRICS,
            ("Curvature vs baseline", "ECC vs baseline", "SECC vs baseline"),
            None,
        ),
        (
            "reference_errors.png",
            summaries,
            REFERENCE_ERROR_METRICS,
            (
                "Curvature reference error",
                "ECC reference error",
                "SECC reference error",
            ),
            None,
        ),
        (
            "perturbation_variability.png",
            pairwise_summaries,
            PAIRWISE_METRICS,
            (
                "Curvature pairwise distance",
                "ECC pairwise distance",
                "SECC pairwise distance",
            ),
            None,
        ),
        (
            "ecc_jump_diagnostics.png",
            summaries,
            ECC_DIAGNOSTIC_METRICS,
            (
                "Exact nonzero jumps",
                "Exact total variation",
                "Sampled-grid changes",
                "Sampled total variation",
            ),
            None,
        ),
    )
    for filename, rows, metrics, titles, curvature_filter in plot_specs:
        path = config.results_dir / filename
        if not e02.artifact_is_current(path, signature):
            write_metric_grid(
                path,
                rows,
                metrics,
                titles,
                curvature_filter=curvature_filter,
            )
            e02.mark_artifact_current(path, signature)
        outputs.append(path)

    for filename, writer in (
        ("displacement_diagnostics.png", write_displacement_plot),
        ("mesh_quality.png", write_mesh_quality_plot),
    ):
        path = config.results_dir / filename
        if not e02.artifact_is_current(path, signature):
            writer(path, summaries)
            e02.mark_artifact_current(path, signature)
        outputs.append(path)

    for curvature in source.curvatures:
        short_name = curvature.removesuffix("_curvature")
        path = config.results_dir / f"ecc_curve_comparison_{short_name}.png"
        if not e02.artifact_is_current(path, signature):
            write_ecc_curve_comparison(
                path,
                curve_groups,
                reference_curves,
                curvature=curvature,
                scales=plotted_scales,
                subdivision_levels=config.subdivision_levels,
            )
            e02.mark_artifact_current(path, signature)
        outputs.append(path)
    return outputs


def load_config(path: Path) -> Config:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    def resolved(key: str) -> Path:
        candidate = Path(str(raw[key]))
        return (
            candidate
            if candidate.is_absolute()
            else (path.parent / candidate).resolve()
        )

    scales = tuple(float(value) for value in raw["perturbation_scales"])
    base_levels = {
        str(entry["surface"]): (int(entry["u_segments"]), int(entry["v_segments"]))
        for entry in raw.get("base_meshes", [])
    }
    config = Config(
        source_config=resolved("source_config"),
        results_dir=resolved("results_dir"),
        subdivision_levels=int(raw["subdivision_levels"]),
        repetitions=int(raw["repetitions"]),
        random_seed=int(raw.get("random_seed", 0)),
        perturbation_scales=scales,
        noise_model=str(raw.get("noise_model", NOISE_MODEL)),
        scale_reference=str(raw.get("scale_reference", SCALE_REFERENCE)),
        sample_count=int(raw.get("sample_count", 1001)),
        interval_padding_fraction=float(raw.get("interval_padding_fraction", 0.05)),
        constant_interval_padding=float(raw.get("constant_interval_padding", 1.0)),
        base_levels=base_levels,
    )
    if config.subdivision_levels < 2 or config.repetitions < 2:
        raise ValueError("Use at least two subdivision levels and repetitions")
    if not 5 <= len(scales) <= 10:
        raise ValueError("Configure between five and ten positive perturbation scales")
    if scales != tuple(sorted(set(scales))) or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0 for value in scales
    ):
        raise ValueError(
            "Perturbation scales must be unique, increasing, and in (0, 1]"
        )
    if config.noise_model != NOISE_MODEL or config.scale_reference != SCALE_REFERENCE:
        raise ValueError("Unsupported coordinate-noise model or scale reference")
    if (
        config.sample_count < 2
        or config.interval_padding_fraction < 0.0
        or config.constant_interval_padding <= 0.0
        or not base_levels
        or any(min(level) < 2 for level in base_levels.values())
    ):
        raise ValueError("Invalid sampling, interval, or base-mesh configuration")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Run configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    for output in run(load_config(args.config.resolve())):
        print(output)


if __name__ == "__main__":
    main()
