"""Treffsicherheit einer Quelle im Testzeitraum (nicht zur Kalibrierung verwendet).

Regression der realisierten Rendite auf die Prognose:  y = a + b·r + e.
  b ≈ 1: Prognose ist gut kalibriert,  b ≈ 0: Prognose trägt keine Information.
Die Quelle wird danach als  r_kal = a + b·r  verwendet; ihr Messrauschen ist
  σ_v² = Var(e) − σε²   (Testfehler abzüglich des unvermeidbaren Kursrauschens).
Eine Quelle ohne Treffsicherheit liefert damit nur noch den Mittelwert mit großer
Unsicherheit und bekommt in der Kalman-Fusion automatisch kaum Gewicht.
"""
from __future__ import annotations

import numpy as np


def fit(pred, real, b_max: float = 1.5) -> dict:
    p = np.asarray(pred, dtype=float)
    y = np.asarray(real, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 100 or np.var(p) <= 0:
        return {"n": int(len(p)), "a": float(np.mean(y)) if len(y) else 0.0, "b": 0.0,
                "var_e": float(np.var(y)) if len(y) > 1 else 0.13 ** 2, "corr": None, "b_raw": None}
    b_raw = float(np.cov(p, y)[0, 1] / np.var(p, ddof=1))
    b = float(np.clip(b_raw, 0.0, b_max))
    a = float(np.mean(y) - b * np.mean(p))
    e = y - (a + b * p)
    return {"n": int(len(p)), "a": a, "b": b, "b_raw": b_raw, "var_e": float(np.var(e, ddof=1)),
            "corr": float(np.corrcoef(p, y)[0, 1]), "mean_real": float(np.mean(y))}


def measurement_var(var_e: float, sigma_eps2: float, floor_share: float) -> float:
    return float(max(var_e - sigma_eps2, floor_share * var_e))
