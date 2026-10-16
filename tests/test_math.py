"""Kontrollrechnungen: Minimum-Varianz-Gewichte und Kalman-Update gegen numerische Lösung."""
import numpy as np

from marketlab.analysts import min_variance_weights
from marketlab.kalman import kalman_step, static_fusion


def test_static_fusion_is_min_variance():
    r1, v1, r2, v2, rho = 0.05, 0.03 ** 2, 0.02, 0.045 ** 2, 0.2
    x, v, w1, _ = static_fusion(r1, v1, r2, v2, rho)
    ws = np.linspace(0, 1, 100001)
    c = rho * np.sqrt(v1 * v2)
    var = ws ** 2 * v1 + (1 - ws) ** 2 * v2 + 2 * ws * (1 - ws) * c
    assert abs(ws[var.argmin()] - w1) < 1e-3
    assert v < min(v1, v2)


def test_kalman_diffuse_prior_equals_static():
    r1, v1, r2, v2, rho = 0.05, 0.03 ** 2, 0.02, 0.045 ** 2, 0.2
    x, v, _, _ = static_fusion(r1, v1, r2, v2, rho)
    k = kalman_step({"x": 0.0, "P": 1e6}, r1, v1, r2, v2, rho, {"process_std": 0, "measurement_inflation": 1})
    assert abs(k["x"] - x) < 1e-6 and abs(k["P"] - v) / v < 1e-3


def test_mv_weights_bruteforce():
    sig, rho = np.array([0.03, 0.04, 0.06]), 0.3
    w = min_variance_weights(sig, rho)
    R = np.full((3, 3), rho)
    np.fill_diagonal(R, 1)
    C = np.outer(sig, sig) * R
    grid = [(a, b, 1 - a - b) for a in np.linspace(0, 1, 201) for b in np.linspace(0, 1, 201) if a + b <= 1]
    best = np.array(min(grid, key=lambda t: np.array(t) @ C @ np.array(t)))
    assert np.abs(best - w).max() < 0.01
