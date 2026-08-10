"""Compute centered cumulative smooth Euler characteristic curves for e02."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
E02_SCRIPT = (
    HERE.parents[1]
    / "e02_parametric-mesh-ecc"
    / "analysis"
    / "parametric_mesh_ecc.py"
)
SPEC = importlib.util.spec_from_file_location("parametric_mesh_ecc", E02_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load e02 analysis module from {E02_SCRIPT}")
mesh_ecc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mesh_ecc
SPEC.loader.exec_module(mesh_ecc)

CALCULATION_CACHE_VERSION = 1


@dataclass(frozen=True)
class SECCConfig:
    source_config: Path
    results_dir: Path
    sample_count: int
    interval_padding_fraction: float
    constant_interval_padding: float


@dataclass(frozen=True)
class SECCCurve:
    label: str
    threshold: np.ndarray
    euler_characteristic: np.ndarray
    centered_euler_characteristic: np.ndarray
    secc: np.ndarray


@dataclass(frozen=True)
class SECCRow:
    surface: object
    curves: list[SECCCurve]
    renderings: list[object]
    rendering_assets: dict[str, Path]


def evaluate_ecc(
    curve: list[dict[str, float | int]], thresholds: np.ndarray
) -> np.ndarray:
    """Evaluate a right-continuous ECC step function on fixed thresholds."""

    if not curve:
        return np.zeros_like(thresholds, dtype=float)
    events = np.asarray([float(row["filtration"]) for row in curve], dtype=float)
    values = np.asarray([float(row["chi_after"]) for row in curve], dtype=float)
    indices = np.searchsorted(events, thresholds, side="right") - 1
    evaluated = np.zeros_like(thresholds, dtype=float)
    valid = indices >= 0
    evaluated[valid] = values[indices[valid]]
    return evaluated


def smooth_euler_characteristic_curve(
    curve: list[dict[str, float | int]],
    *,
    lower: float,
    upper: float,
    sample_count: int,
    label: str,
) -> SECCCurve:
    """Compute the centered cumulative ECC on a common filtration interval."""

    if not upper > lower:
        raise ValueError("SECC interval must have positive length")
    if sample_count < 2:
        raise ValueError("SECC sample_count must be at least 2")
    thresholds = np.linspace(lower, upper, sample_count)
    chi = evaluate_ecc(curve, thresholds)
    widths = np.diff(thresholds)
    mean_chi = float(np.sum(chi[:-1] * widths) / (upper - lower))
    centered = chi - mean_chi
    secc = np.zeros_like(thresholds)
    secc[1:] = np.cumsum(centered[:-1] * widths)
    return SECCCurve(label, thresholds, chi, centered, secc)


def analytic_interval(
    values: np.ndarray,
    *,
    padding_fraction: float,
    constant_padding: float,
) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span == 0.0:
        return (lower - constant_padding, upper + constant_padding)
    padding = span * padding_fraction
    return (lower - padding, upper + padding)


def secc_distances(reference: SECCCurve, candidate: SECCCurve) -> dict[str, float]:
    difference = candidate.secc - reference.secc
    interval_length = float(reference.threshold[-1] - reference.threshold[0])
    l2 = math.sqrt(
        float(np.trapezoid(difference**2, reference.threshold)) / interval_length
    )
    return {
        "secc_l2": l2,
        "secc_linf": float(np.max(np.abs(difference))),
    }


def cached_secc_curves(
    cache_dir: Path,
    *,
    surface: object,
    curvature: str,
    raw_curves: list[tuple[str, list[dict[str, float | int]]]],
    lower: float,
    upper: float,
    sample_count: int,
) -> list[SECCCurve]:
    payload = {
        "cache_version": CALCULATION_CACHE_VERSION,
        "surface": surface.name,
        "surface_parameters": surface.parameters,
        "curvature": curvature,
        "lower": lower,
        "upper": upper,
        "sample_count": sample_count,
        "raw_curves": mesh_ecc.curve_content_hash(raw_curves),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    stem = cache_dir / f"{surface.name}_{key}"
    data_path = stem.with_suffix(".npz")
    metadata_path = stem.with_suffix(".json")
    if data_path.exists() and metadata_path.exists():
        with np.load(data_path) as cached:
            labels = json.loads(str(cached["labels"].item()))
            return [
                SECCCurve(
                    label,
                    cached[f"threshold_{index}"],
                    cached[f"chi_{index}"],
                    cached[f"centered_{index}"],
                    cached[f"secc_{index}"],
                )
                for index, label in enumerate(labels)
            ]

    curves = [
        smooth_euler_characteristic_curve(
            curve,
            lower=lower,
            upper=upper,
            sample_count=sample_count,
            label=label,
        )
        for label, curve in raw_curves
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {"labels": json.dumps([curve.label for curve in curves])}
    for index, curve in enumerate(curves):
        arrays[f"threshold_{index}"] = curve.threshold
        arrays[f"chi_{index}"] = curve.euler_characteristic
        arrays[f"centered_{index}"] = curve.centered_euler_characteristic
        arrays[f"secc_{index}"] = curve.secc
    np.savez_compressed(data_path, **arrays)
    mesh_ecc.write_json(metadata_path, {**payload, "cache_key": key})
    return curves


def run(config: SECCConfig) -> list[Path]:
    source = mesh_ecc.load_experiment_config(config.source_config)
    outputs: list[Path] = []
    for curvature in source.curvatures:
        results_dir = mesh_ecc.curvature_scoped_dir(config.results_dir, curvature)
        source_results_dir = mesh_ecc.curvature_scoped_dir(
            source.results_dir, curvature
        )
        calculation_cache_dir = source_results_dir / "calculation_cache"
        secc_cache_dir = results_dir / "calculation_cache"
        results_dir.mkdir(parents=True, exist_ok=True)
        rows: list[SECCRow] = []
        summaries: list[dict[str, object]] = []

        for surface in source.surfaces:
            reference_computation = mesh_ecc.cached_level_computation(
                calculation_cache_dir,
                surface,
                source.reference_level,
                curvature=curvature,
                reference=True,
            )
            reference_mesh = reference_computation.mesh
            reference_values = reference_computation.vertex_curvature
            reference_curve = reference_computation.curve
            lower, upper = analytic_interval(
                reference_values,
                padding_fraction=config.interval_padding_fraction,
                constant_padding=config.constant_interval_padding,
            )
            raw_curves = [("smooth", reference_curve)]
            curve_sources: dict[str, str] = {}
            renderings = [
                mesh_ecc.SurfaceRendering(
                    "smooth",
                    reference_mesh,
                    mesh_ecc.triangle_mean(reference_values, reference_mesh.faces),
                )
            ]

            for base_level in source.mesh_levels:
                level = mesh_ecc.surface_mesh_level(surface, base_level)
                computation = mesh_ecc.cached_level_computation(
                    calculation_cache_dir,
                    surface,
                    level,
                    curvature=curvature,
                    reference=False,
                )
                mesh = computation.mesh
                vertex_values = computation.vertex_curvature
                curve_source = computation.curve_source
                raw_curve = computation.curve
                raw_curves.append((level.name, raw_curve))
                curve_sources[level.name] = curve_source
                renderings.append(
                    mesh_ecc.SurfaceRendering(
                        level.name,
                        mesh,
                        mesh_ecc.triangle_mean(vertex_values, mesh.faces),
                    )
                )
            curves = cached_secc_curves(
                secc_cache_dir,
                surface=surface,
                curvature=curvature,
                raw_curves=raw_curves,
                lower=lower,
                upper=upper,
                sample_count=config.sample_count,
            )
            for candidate in curves[1:]:
                summaries.append(
                    {
                        "surface": surface.name,
                        "curvature": curvature,
                        "mesh_level": candidate.label,
                        "curve_source": curve_sources[candidate.label],
                        "interval_lower": lower,
                        "interval_upper": upper,
                        **secc_distances(curves[0], candidate),
                    }
                )

            surface_dir = results_dir / surface.name
            surface_dir.mkdir(parents=True, exist_ok=True)
            curve_path = surface_dir / f"{surface.name}_secc_curves.csv"
            write_secc_curves(curve_path, curves)
            outputs.append(curve_path)

            all_renderings = mesh_ecc.ordered_surface_renderings(renderings)
            rendering_assets = mesh_ecc.ensure_surface_rendering_assets(
                source_results_dir / "surface_assets",
                surface,
                curvature=curvature,
                reference_level=source.reference_level,
                renderings=all_renderings,
                normalization_renderings=mesh_ecc.main_result_renderings(renderings),
            )
            for asset in rendering_assets.values():
                outputs.extend(
                    asset.with_suffix(suffix)
                    for suffix in (".pdf", ".png", ".json")
                )
            rows.append(
                SECCRow(
                    surface=surface,
                    curves=select_main_curves(curves),
                    renderings=all_renderings,
                    rendering_assets=rendering_assets,
                )
            )

        summary_path = results_dir / "secc_summary.csv"
        write_summary(summary_path, summaries)
        outputs.append(summary_path)
        main_path = results_dir / "main_results.png"
        main_signature = {
            "version": mesh_ecc.COMPOSITE_PLOT_VERSION,
            "kind": "secc_main_results",
            "curvature": curvature,
            "rows": [secc_row_signature(row) for row in rows],
        }
        if not mesh_ecc.artifact_is_current(main_path, main_signature):
            write_main_results(main_path, rows, curvature=curvature)
            mesh_ecc.mark_artifact_current(main_path, main_signature)
        outputs.append(main_path)
    return outputs


def select_main_curves(curves: list[SECCCurve]) -> list[SECCCurve]:
    by_label = {curve.label: curve for curve in curves}
    return [by_label[label] for label in ("smooth", "fine", "coarse") if label in by_label]


def secc_row_signature(row: SECCRow) -> dict[str, object]:
    digest = hashlib.sha256()
    for curve in row.curves:
        digest.update(curve.label.encode("utf-8"))
        digest.update(np.ascontiguousarray(curve.threshold).tobytes())
        digest.update(np.ascontiguousarray(curve.secc).tobytes())
    return {
        "surface": row.surface.name,
        "curves": digest.hexdigest()[:16],
        "rendering_assets": {
            label: str(path) for label, path in sorted(row.rendering_assets.items())
        },
    }


def write_secc_curves(path: Path, curves: list[SECCCurve]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["threshold"]
        for curve in curves:
            fieldnames.extend(
                [
                    f"{curve.label}_chi",
                    f"{curve.label}_centered_chi",
                    f"{curve.label}_secc",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, threshold in enumerate(curves[0].threshold):
            row: dict[str, float] = {"threshold": float(threshold)}
            for curve in curves:
                row[f"{curve.label}_chi"] = float(curve.euler_characteristic[index])
                row[f"{curve.label}_centered_chi"] = float(
                    curve.centered_euler_characteristic[index]
                )
                row[f"{curve.label}_secc"] = float(curve.secc[index])
            writer.writerow(row)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "surface",
        "curvature",
        "mesh_level",
        "curve_source",
        "interval_lower",
        "interval_upper",
        "secc_l2",
        "secc_linf",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_main_results(path: Path, rows: list[SECCRow], *, curvature: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, max(4.0 * len(rows), 6.0)), constrained_layout=True)
    grid = fig.add_gridspec(
        len(rows), 4, width_ratios=[1.35, 1.0, 1.0, 1.0], wspace=0.08, hspace=0.08
    )
    titles = ["SECC", "smooth", "fine", "coarse"]

    for row_index, row in enumerate(rows):
        curve_ax = fig.add_subplot(grid[row_index, 0])
        plot_secc(curve_ax, row)
        if row_index == 0:
            curve_ax.set_title(titles[0], fontweight="bold")

        norm = mesh_ecc.rendering_row_norm(row.renderings[:3])
        coordinates = np.concatenate(
            [rendering.mesh.vertices for rendering in row.renderings], axis=0
        )
        cmap = plt.get_cmap("coolwarm")
        surface_axes = []
        for offset, rendering in enumerate(row.renderings[:3], start=1):
            if rendering.label in row.rendering_assets:
                ax = fig.add_subplot(grid[row_index, offset])
                mesh_ecc.add_cached_surface_image(
                    ax, row.rendering_assets[rendering.label]
                )
            else:
                ax = fig.add_subplot(grid[row_index, offset], projection="3d")
                mesh_ecc.add_surface_collection(ax, rendering, cmap=cmap, norm=norm)
                mesh_ecc.set_equal_3d_limits(ax, coordinates)
                ax.view_init(elev=24, azim=-52)
                ax.set_axis_off()
            surface_axes.append(ax)
            if row_index == 0:
                ax.set_title(titles[offset], fontweight="bold")
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=surface_axes,
            fraction=0.025,
            pad=0.01,
            shrink=0.72,
        )
        colorbar.set_label(mesh_ecc.curvature_label(curvature), fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_secc(ax: object, row: SECCRow) -> None:
    styles = {
        "smooth": {"color": "black", "linewidth": 2.2, "alpha": 0.78},
        "fine": {"color": "#1f77b4", "linewidth": 1.9, "alpha": 0.76},
        "coarse": {"color": "#d95f02", "linewidth": 1.9, "alpha": 0.76},
    }
    for curve in row.curves:
        ax.plot(curve.threshold, curve.secc, label=curve.label, **styles[curve.label])
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_ylabel(mesh_ecc.display_surface_name(row.surface.name))
    ax.set_xlabel("curvature threshold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.02,
        0.04,
        mesh_ecc.surface_parameter_text(row.surface),
        transform=ax.transAxes,
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.82},
    )


def load_config(path: Path) -> SECCConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    def relative_path(key: str) -> Path:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Config {path} must define {key!r}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()

    sample_count = int(raw.get("sample_count", 1001))
    padding_fraction = float(raw.get("interval_padding_fraction", 0.05))
    constant_padding = float(raw.get("constant_interval_padding", 1.0))
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if padding_fraction < 0.0 or constant_padding <= 0.0:
        raise ValueError("SECC interval padding values must be nonnegative/positive")
    return SECCConfig(
        source_config=relative_path("source_config"),
        results_dir=relative_path("results_dir"),
        sample_count=sample_count,
        interval_padding_fraction=padding_fraction,
        constant_interval_padding=constant_padding,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute SECC versions of e02 curves.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output in run(load_config(args.config)):
        print(mesh_ecc.display_path(output))


if __name__ == "__main__":
    main()
