"""Compare Euler characteristic curves by pairwise L1 distance."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def is_synthetic_test_label(label: str) -> bool:
    """Return whether a result label belongs to the dented-sphere test data."""

    return label == "test" or label.startswith("test_")


@dataclass(frozen=True)
class EccCurve:
    label: str
    path: Path
    filtration: np.ndarray
    chi: np.ndarray


def discover_curve_files(results_dir: Path) -> list[tuple[str, Path]]:
    """Find one ECC curve CSV per immediate result subdirectory."""

    curve_files: list[tuple[str, Path]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if is_synthetic_test_label(child.name):
            continue
        matches = sorted(child.glob("*_ecc_curve.csv"))
        if len(matches) == 1:
            curve_files.append((child.name, matches[0]))
        else:
            for match in matches:
                curve_files.append((f"{child.name}/{match.stem}", match))
    return curve_files


def read_ecc_curve(label: str, path: Path) -> EccCurve:
    """Read a cumulative ECC curve CSV."""

    filtrations: list[float] = []
    chi_values: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filtration", "chi"]:
            raise ValueError(f"Expected filtration,chi columns in {path}")
        for row in reader:
            filtrations.append(float(row["filtration"]))
            chi_values.append(int(row["chi"]))

    return EccCurve(
        label=label,
        path=path,
        filtration=np.asarray(filtrations, dtype=float),
        chi=np.asarray(chi_values, dtype=np.int64),
    )


def common_domain(curves: list[EccCurve]) -> tuple[float, float]:
    """Return the finite comparison interval shared by the matrix."""

    nonempty = [curve for curve in curves if curve.filtration.size > 0]
    if not nonempty:
        return (0.0, 0.0)
    return (
        min(float(curve.filtration[0]) for curve in nonempty),
        max(float(curve.filtration[-1]) for curve in nonempty),
    )


def value_after(curve: EccCurve, x: float) -> float:
    """Evaluate the right-continuous step curve at x."""

    index = int(np.searchsorted(curve.filtration, x, side="right")) - 1
    if index < 0:
        return 0.0
    return float(curve.chi[index])


def values_after(curve: EccCurve, x_values: np.ndarray) -> np.ndarray:
    """Vectorized right-continuous step-curve evaluation."""

    indices = np.searchsorted(curve.filtration, x_values, side="right") - 1
    values = np.zeros(len(x_values), dtype=curve.chi.dtype)
    valid = indices >= 0
    values[valid] = curve.chi[indices[valid]]
    return values


def global_knots(curves: list[EccCurve], domain: tuple[float, float]) -> np.ndarray:
    """Return all jump locations needed to compare every curve."""

    lower, upper = domain
    knots = [
        np.asarray([lower, upper], dtype=float),
        *(
            curve.filtration[(lower < curve.filtration) & (curve.filtration < upper)]
            for curve in curves
        ),
    ]
    return np.unique(np.concatenate(knots))


def l1_distance(
    first: EccCurve, second: EccCurve, *, domain: tuple[float, float] | None = None
) -> float:
    """Integrate |first - second| over a finite interval."""

    if domain is None:
        domain = common_domain([first, second])

    lower, upper = domain
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"Domain must be finite, got {domain}")
    if upper <= lower:
        return 0.0

    knots = np.concatenate(
        [
            np.asarray([lower, upper], dtype=float),
            first.filtration[(lower < first.filtration) & (first.filtration < upper)],
            second.filtration[
                (lower < second.filtration) & (second.filtration < upper)
            ],
        ]
    )
    knots = np.unique(knots)

    lefts = knots[:-1]
    widths = np.diff(knots)
    differences = np.abs(values_after(first, lefts) - values_after(second, lefts))
    return float(np.sum(differences * widths))


def l1_distance_matrix(curves: list[EccCurve]) -> np.ndarray:
    """Compute the symmetric L1 distance matrix on a common finite domain."""

    domain = common_domain(curves)
    matrix = np.zeros((len(curves), len(curves)), dtype=float)
    lower, upper = domain
    if upper <= lower:
        return matrix

    knots = global_knots(curves, domain)
    lefts = knots[:-1]
    widths = np.diff(knots)
    values = [values_after(curve, lefts).astype(float) for curve in curves]

    for row in range(len(curves)):
        for column in range(row + 1, len(curves)):
            distance = float(np.sum(np.abs(values[row] - values[column]) * widths))
            matrix[row, column] = distance
            matrix[column, row] = distance
    return matrix


def write_ecc_plot(path: Path, labels: list[str], curves: list[EccCurve]) -> None:
    """Write a plot of the cumulative Euler characteristic curve."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

    global_lower, global_upper = (0,0)

    for label, curve in zip(labels, curves, strict=True):
        ax.plot(curve.filtration, curve.chi, label=label)

        lower, upper = (
            np.percentile(curve.filtration, [1.0, 99.0]) if curve.filtration.size > 0 else (0.0, 0.0)
        )

        global_lower = min(global_lower, lower)
        global_upper = max(global_upper, upper)

        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_title(f"ECC curves")
        ax.set_xlabel("filtration value")
        ax.set_ylabel("Euler characteristic")
        ax.grid(True, alpha=0.25)

    if global_lower == global_upper:
        padding = max(abs(global_lower) * 0.05, 1.0)
        global_lower -= padding
        global_upper += padding

    ax.set_xlim(global_lower, global_upper)


    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_title("Euler characteristic curves")
    ax.set_xlabel("filtration value")
    ax.set_ylabel("Euler characteristic")
    ax.grid(True, alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

def write_distance_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *(format_float(float(value)) for value in row)])


def log_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Transform distances for plots while keeping zero distances finite."""

    if np.any(matrix < 0):
        raise ValueError("Distance matrix contains negative entries")
    return np.log1p(matrix)


def write_confusion_plot(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_matrix = log_distance_matrix(matrix)
    size = max(6.0, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(plot_matrix, cmap="viridis")
    ax.set_title("ECC log L1 distance")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("curve")
    ax.set_ylabel("curve")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("log(1 + L1 distance)")

    if len(labels) <= 12:
        max_value = float(np.max(plot_matrix)) if plot_matrix.size else 0.0
        text_color_threshold = max_value / 2.0
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = float(plot_matrix[row, column])
                color = "white" if value > text_color_threshold else "black"
                ax.text(
                    column,
                    row,
                    format_plot_float(value),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_dendrogram_plot(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = max(4.5, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, height), constrained_layout=True)

    if len(labels) >= 2:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        plot_matrix = log_distance_matrix(matrix)
        condensed_distances = squareform(plot_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method="average")
        dendrogram(
            linkage_matrix,
            labels=labels,
            orientation="right",
            ax=ax,
            leaf_font_size=9,
        )
        ax.set_xlabel("log(1 + L1 distance)")
    elif labels:
        ax.text(
            0.5,
            0.5,
            labels[0],
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        ax.text(
            0.5,
            0.5,
            "No curves",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    ax.set_title("ECC log L1 dendrogram")
    ax.grid(True, axis="x", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_outputs(results_dir: Path, output_dir: Path) -> dict[str, Path]:
    curve_specs = discover_curve_files(results_dir)
    if not curve_specs:
        raise ValueError(f"No *_ecc_curve.csv files found under {results_dir}")

    outputs = {
        "ecc_plot": output_dir / "ecc_curves.png",
        "distance_matrix": output_dir / "ecc_l1_distance_matrix.csv",
        "confusion_plot": output_dir / "ecc_l1_confusion_matrix.png",
        "dendrogram_plot": output_dir / "ecc_l1_dendrogram.png",
    }

    curves = [read_ecc_curve(label, path) for label, path in curve_specs]
    labels = [curve.label for curve in curves]

    write_ecc_plot(outputs["ecc_plot"], labels, curves)

    matrix = l1_distance_matrix(curves)

    write_distance_matrix(outputs["distance_matrix"], labels, matrix)
    write_confusion_plot(outputs["confusion_plot"], labels, matrix)
    write_dendrogram_plot(outputs["dendrogram_plot"], labels, matrix)
    return outputs


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.12g}"


def format_plot_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.1f}"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute pairwise L1 distances between ECC curves and plot a confusion matrix."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/e01_curvature-ecc/results"),
        help="Directory containing one subdirectory per ECC run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/e01_curvature-ecc/results/ecc_l1_confusion"),
        help="Directory for the distance matrix CSV and confusion matrix PNG.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_outputs(args.results_dir, args.output_dir)
    for kind, path in outputs.items():
        print(f"{kind}: {display_path(path)}")


if __name__ == "__main__":
    main()
