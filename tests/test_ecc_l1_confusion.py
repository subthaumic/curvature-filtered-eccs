from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "e01_curvature-ecc"
    / "analysis"
    / "ecc_l1_confusion.py"
)

spec = importlib.util.spec_from_file_location("ecc_l1_confusion", SCRIPT_PATH)
ecc_l1_confusion = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ecc_l1_confusion
spec.loader.exec_module(ecc_l1_confusion)


def test_l1_distance_between_step_curves() -> None:
    first = ecc_l1_confusion.EccCurve(
        label="first",
        path=Path("first.csv"),
        filtration=ecc_l1_confusion.np.asarray([0.0, 2.0]),
        chi=ecc_l1_confusion.np.asarray([1.0, 1.0]),
    )
    second = ecc_l1_confusion.EccCurve(
        label="second",
        path=Path("second.csv"),
        filtration=ecc_l1_confusion.np.asarray([1.0, 2.0]),
        chi=ecc_l1_confusion.np.asarray([3.0, 3.0]),
    )

    distance = ecc_l1_confusion.l1_distance(first, second, domain=(0.0, 2.0))

    assert distance == pytest.approx(3.0)


def test_log_distance_matrix_uses_linear_extension_at_zero() -> None:
    matrix = ecc_l1_confusion.np.asarray([[0.0, ecc_l1_confusion.math.e - 1.0]])

    transformed = ecc_l1_confusion.log_distance_matrix(matrix)

    assert transformed[0, 0] == 0.0
    assert transformed[0, 1] == pytest.approx(1.0)


def test_plot_float_uses_one_decimal_place() -> None:
    assert ecc_l1_confusion.format_plot_float(0.0) == "0.0"
    assert ecc_l1_confusion.format_plot_float(12.345) == "12.3"


def test_build_outputs_from_result_subdirectories(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    output_dir = results_dir / "comparison"
    write_curve(results_dir / "alpha" / "alpha_ecc_curve.csv", [(0.0, 1), (2.0, 1)])
    write_curve(results_dir / "beta" / "beta_ecc_curve.csv", [(1.0, 3), (2.0, 3)])
    write_curve(results_dir / "test" / "test_ecc_curve.csv", [(0.0, 50)])
    write_curve(results_dir / "test_fine" / "test_fine_ecc_curve.csv", [(0.0, 100)])
    write_curve(results_dir / "stale_ecc_curve.csv", [(0.0, 100)])

    outputs = ecc_l1_confusion.build_outputs(results_dir, output_dir)

    assert outputs["confusion_plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert outputs["dendrogram_plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with outputs["distance_matrix"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["label", "alpha", "beta"]
    assert rows[1][0] == "alpha"
    assert rows[2][0] == "beta"
    assert float(rows[1][2]) == pytest.approx(3.0)
    assert float(rows[2][1]) == pytest.approx(3.0)
    assert float(rows[1][1]) == 0.0


def write_curve(path: Path, rows: list[tuple[float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filtration", "chi"])
        writer.writerows(rows)
