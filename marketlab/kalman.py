"""Kalman-Fusion von Analysten- (r₁) und Chartinformation (r₂) mit korrelierten Messfehlern.

Zustand x: wahre erwartete Rendite über den Horizont (Random Walk, Prozessrauschen q²).
Messung z = [r₁, r₂]ᵀ = H·x + v,  H = [1, 1]ᵀ,  v ~ N(0, R),
R = [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]].

Zwei Ausgaben je Tag:
  • statische Fusion (ein Messzeitpunkt): x_s, σ_s² – die ehrliche Unsicherheit der heutigen
    Gesamtinformation, immer ≤ min(σ₁², σ₂²); Gewichte = Minimum-Varianz-Kombination.
  • zeitliches Filter: x̂ – geglättet über die Tage. Weil sich die Messungen von Tag zu Tag
    kaum ändern (gleiche Kursziele, träge Indikatoren), wird R im Filter um den Faktor
    `measurement_inflation` erhöht; sonst würde dieselbe Information täglich erneut gezählt
    und die Varianz künstlich gegen null laufen.
Rangkriterium (Chance/Risiko): x̂ / σ_s.
"""
from __future__ import annotations

import numpy as np


def static_fusion(r1, v1, r2, v2, rho):
    """Minimum-Varianz-Kombination zweier korrelierter Schätzer."""
    if r1 is None or v1 is None:
        return r2, v2, 0.0, 1.0
    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    c = rho * s1 * s2
    den = v1 + v2 - 2 * c
    w1 = (v2 - c) / den if den > 1e-12 else 0.5
    w1 = float(np.clip(w1, 0.0, 1.0))
    w2 = 1.0 - w1
    x = w1 * r1 + w2 * r2
    var = w1 ** 2 * v1 + w2 ** 2 * v2 + 2 * w1 * w2 * c
    return float(x), float(var), w1, w2


def kalman_step(state: dict | None, r1, v1, r2, v2, rho, kcfg: dict) -> dict:
    """Ein Tagesschritt. state = {"x": .., "P": ..} oder None (Neuaufnahme)."""
    xs, vs, w1, w2 = static_fusion(r1, v1, r2, v2, rho)
    if state is None:
        return {"x": xs, "P": vs, "x_static": xs, "var_static": vs, "w1": w1, "w2": w2, "gain": None}
    q = kcfg["process_std"] ** 2
    infl = kcfg["measurement_inflation"]
    x_pred, P_pred = state["x"], state["P"] + q
    if r1 is None or v1 is None:
        H = np.array([[1.0]])
        z = np.array([r2])
        R = np.array([[v2]]) * infl
    else:
        H = np.array([[1.0], [1.0]])
        z = np.array([r1, r2])
        c = rho * np.sqrt(v1 * v2)
        R = np.array([[v1, c], [c, v2]]) * infl
    # Informationsform (numerisch stabil, auch bei sehr unsicherem Vorwissen):
    # P⁺ = (P⁻⁻¹ + HᵀR⁻¹H)⁻¹,  x⁺ = P⁺ (P⁻⁻¹ x⁻ + HᵀR⁻¹z),  Kalman-Gain K = P⁺HᵀR⁻¹
    Rinv = np.linalg.inv(R)
    info = 1.0 / P_pred + (H.T @ Rinv @ H).item()
    P_new = 1.0 / info
    x_new = float(P_new * (x_pred / P_pred + (H.T @ Rinv @ z).item()))
    K = P_new * (H.T @ Rinv)
    return {"x": x_new, "P": P_new, "x_static": xs, "var_static": vs, "w1": w1, "w2": w2,
            "gain": [float(k) for k in K.ravel()]}
