from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "e03_parametric-mesh-secc"
    / "analysis"
    / "parametric_mesh_secc.py"
)
spec = importlib.util.spec_from_file_location("parametric_mesh_secc", SCRIPT_PATH)
parametric_mesh_secc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = parametric_mesh_secc
spec.loader.exec_module(parametric_mesh_secc)


def test_evaluate_ecc_is_right_continuous() -> None:
    curve = [
        {"filtration": -1.0, "delta_chi": 1, "chi_after": 1},
        {"filtration": 1.0, "delta_chi": -1, "chi_after": 0},
    ]
    thresholds = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = parametric_mesh_secc.evaluate_ecc(curve, thresholds)
    assert np.array_equal(values, np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]))


def test_secc_is_centered_and_returns_to_zero() -> None:
    curve = [
        {"filtration": -1.0, "delta_chi": 1, "chi_after": 1},
        {"filtration": 1.0, "delta_chi": -1, "chi_after": 0},
    ]
    result = parametric_mesh_secc.smooth_euler_characteristic_curve(
        curve, lower=-2.0, upper=2.0, sample_count=401, label="test"
    )
    assert result.secc[0] == 0.0
    assert abs(result.secc[-1]) < 1e-12
    assert np.max(result.secc) > 0.0


def test_constant_interval_gets_absolute_padding() -> None:
    assert parametric_mesh_secc.analytic_interval(
        np.asarray([-1.0, -1.0]),
        padding_fraction=0.05,
        constant_padding=1.0,
    ) == (-2.0, 0.0)
