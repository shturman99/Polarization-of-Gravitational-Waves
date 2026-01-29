import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sympy")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hijij import HijijParams, hijij


def test_hijij_finite_value():
    params = HijijParams(
        k1_max=3.0,
        om1_max=3.0,
        u_reg=0.1,
        abs_tol=1e-3,
        rel_tol=1e-3,
        method="scipy_quad",
    )
    value = hijij(1.0, 0.5, params)
    assert np.isfinite(value)


def test_hijij_deterministic():
    params = HijijParams(
        k1_max=2.0,
        om1_max=2.0,
        u_reg=0.1,
        abs_tol=1e-3,
        rel_tol=1e-3,
        method="scipy_quad",
    )
    value1 = hijij(1.0, 0.5, params)
    value2 = hijij(1.0, 0.5, params)
    assert value1 == pytest.approx(value2, rel=1e-10, abs=1e-12)


def test_progress_tracker_prints(capsys):
    params = HijijParams(
        k1_max=1.0,
        om1_max=1.0,
        u_reg=0.1,
        abs_tol=1e-3,
        rel_tol=1e-3,
        progress_enabled=True,
        progress_print_step_percent=50.0,
        progress_target=10,
        progress_max_prints=2,
        method="scipy_quad",
    )
    _ = hijij(1.0, 0.5, params)
    captured = capsys.readouterr()
    assert "%" in captured.out


def test_analytic_mu_attempt_fallback():
    params = HijijParams(
        k1_max=2.0,
        om1_max=2.0,
        u_reg=0.1,
        abs_tol=1e-3,
        rel_tol=1e-3,
        use_analytic_mu=True,
        method="scipy_quad",
    )
    value = hijij(1.0, 0.5, params)
    assert np.isfinite(value)
