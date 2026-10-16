"""Analystenquelle: Top-5-Häuser nach dem Prinzip der minimalen Varianz, sonst Konsens-Rückfall.

Kalibrierung (monatlich, US-Titel mit Häuser-Historie):
  Für jedes historische Kursziel T eines Hauses f am Tag d wird die implizite Prognose
  über den Horizont h gebildet: fc = ln(T/P_d) · h/252. Fehler e = ln(P_{d+h}/P_d) − fc.
  Je Haus: Verzerrung b_f (geschrumpft) und Fehlervarianz σ_f².
  Global: Korrelation der Fehler verschiedener Häuser (ρ_ff) für denselben Titel.

Aktuell (täglich):
  Top-5 = Häuser mit aktuellem Ziel und kleinster σ_f². Gewichte = Minimum-Varianz-Portfolio
  über die Fehler-Kovarianzmatrix Σ = D·R·D (R: konstante Korrelation ρ_ff), long-only.
  Varianz der kombinierten Prognose = wᵀΣw.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("marketlab.analysts")


def calibrate(target_hist: dict[str, pd.DataFrame], close: pd.DataFrame, h: int, acfg: dict) -> dict:
    rows = []
    for tk, df in target_hist.items():
        if df is None or df.empty or tk not in close.columns:
            continue
        p = close[tk].dropna()
        if len(p) < h + 10:
            continue
        idx = p.index
        pos = idx.searchsorted(df["date"].values, side="right") - 1
        for (d, firm, tgt), i in zip(df[["date", "firm", "target"]].itertuples(index=False), pos):
            if i < 0 or i + h >= len(p):
                continue
            p0, ph = float(p.iloc[i]), float(p.iloc[i + h])
            fc = np.log(tgt / p0) * h / 252
            rows.append((tk, idx[i], firm, fc, np.log(ph / p0) - fc))
    if not rows:
        log.warning("Keine auswertbare Analysten-Historie – nur Konsens-Rückfall möglich")
        return {"firms": {}, "global_bias": 0.0, "global_var": 0.12 ** 2, "rho_ff": 0.5, "pair_cov": None, "n": 0}
    df = pd.DataFrame(rows, columns=["ticker", "date", "firm", "fc", "err"])
    g_bias, g_var = float(df["err"].mean()), float(df["err"].var())
    k = acfg["min_firm_obs"]
    firms = {}
    for f, g in df.groupby("firm"):
        n = len(g)
        if n < k:
            continue
        b = (n * g["err"].mean() + k * g_bias) / (n + k)
        v = (n * g["err"].var(ddof=1) + k * g_var) / (n + k)
        firms[f] = {"n": n, "bias": float(b), "var": float(v)}
    pair_cov = _pair_correlation(df)
    df["err_adj"] = df["err"] - df["firm"].map(lambda f: firms.get(f, {}).get("bias", g_bias))
    log.info("Analysten kalibriert: %s Häuser, %s Prognosen", len(firms), len(df))
    return {"firms": firms, "global_bias": g_bias, "global_var": g_var, "pair_cov": pair_cov,
            "rho_ff": 0.5, "n": int(len(df)), "obs": df}


def _pair_correlation(df: pd.DataFrame, window_days: int = 30) -> float:
    """Korrelation der Prognosefehler verschiedener Häuser zum selben Titel (±30 Tage)."""
    a, b = [], []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("date")
        d = g["date"].values
        e = g["err"].values
        f = g["firm"].values
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if (d[j] - d[i]) / np.timedelta64(1, "D") > window_days:
                    break
                if f[i] != f[j]:
                    a.append(e[i])
                    b.append(e[j])
    if len(a) < 30:
        return None
    return float(np.cov(a, b)[0, 1])


def min_variance_weights(sig: np.ndarray, rho: float) -> np.ndarray:
    """Long-only Minimum-Varianz-Gewichte bei konstanter Fehlerkorrelation."""
    n = len(sig)
    if n == 1:
        return np.array([1.0])
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1.0)
    cov = np.outer(sig, sig) * R
    active = np.ones(n, dtype=bool)
    while True:
        sub = cov[np.ix_(active, active)]
        w_sub = np.linalg.solve(sub, np.ones(active.sum()))
        w_sub = w_sub / w_sub.sum()
        if (w_sub >= 0).all():
            w = np.zeros(n)
            w[active] = w_sub
            return w
        # negativstes Gewicht entfernen und neu lösen (aktive-Mengen-Verfahren)
        idx = np.where(active)[0]
        active[idx[np.argmin(w_sub)]] = False


def measurement(ticker: str, price: float, info: dict, hist: pd.DataFrame | None,
                calib: dict, h: int, acfg: dict, asof: pd.Timestamp) -> dict:
    """Analystenmessung r₁, σ₁² mit Methodenkennzeichnung."""
    firms = calib["firms"]
    if hist is not None and not hist.empty and firms:
        recent = hist[hist["date"] >= asof - pd.Timedelta(days=acfg["target_lookback_days"])]
        latest = recent.sort_values("date").groupby("firm").tail(1)
        latest = latest[latest["firm"].isin(firms)]
        if len(latest) >= 2:
            latest = latest.assign(var=latest["firm"].map(lambda f: firms[f]["var"]))
            top = latest.nsmallest(acfg["top_n"], "var")
            sig = np.sqrt(top["var"].values)
            w = min_variance_weights(sig, calib["rho_ff"])
            fc = np.log(top["target"].values / price) * h / 252
            fc_adj = fc + np.array([firms[f]["bias"] for f in top["firm"]])
            R = np.full((len(sig), len(sig)), calib["rho_ff"])
            np.fill_diagonal(R, 1.0)
            var = float(w @ (np.outer(sig, sig) * R) @ w)
            r1 = float(np.exp(w @ fc_adj) - 1)
            return {"r1": r1, "var1": var, "method": "Minimum-Varianz",
                    "detail": [{"firm": f, "target": float(t), "weight": float(x)}
                               for f, t, x in zip(top["firm"], top["target"], w)]}
    # --- Rückfall: Konsens-Median, Unsicherheit aus globaler Fehlervarianz + Kursziel-Streuung
    med = info.get("targetMedianPrice") or info.get("targetMeanPrice")
    if not med or not price:
        return {"r1": None, "var1": None, "method": "keine Analystendaten", "detail": []}
    hi, lo, mean = info.get("targetHighPrice"), info.get("targetLowPrice"), info.get("targetMeanPrice") or med
    rel_std = acfg["fallback_min_rel_std"]
    if hi and lo and mean:
        rel_std = max(rel_std, (hi - lo) / (4 * mean))
    fc = np.log(med / price) * h / 252 + calib["global_bias"]
    var = calib["global_var"] * acfg["horizon_mismatch_penalty"] + (rel_std * h / 252) ** 2
    n = info.get("numberOfAnalystOpinions")
    return {"r1": float(np.exp(fc) - 1), "var1": float(var), "method": "Konsens-Rückfall",
            "detail": [{"firm": f"Konsens ({n or '?'} Analysten)", "target": float(med), "weight": 1.0}]}


def estimate_noise(calib_a: dict, tech_lookup, ncfg: dict) -> dict:
    """Trennt gemeinsames Kursrauschen von den Messfehlern der beiden Quellen.

    Modell: realisierte Rendite y = x + ε (ε: unvorhersehbar, für alle Quellen gleich),
    Messungen r_A = x + v_A, r_T = x + v_T. Die beobachteten Prognosefehler e = y − r sind daher
      Var(e_A) = σε² + σ_vA²,  Var(e_T) = σε² + σ_vT²,  Cov(e_A, e_T) = σε² + ρ_v·σ_vA·σ_vT.
    Mit zwei Quellen ist ρ_v nicht identifizierbar; es wird als Annahme gesetzt (Konfiguration,
    im Dashboard ausgewiesen). Daraus folgt σε² eindeutig (Bisektion).
    """
    obs = calib_a.get("obs")
    ea, et = [], []
    if obs is not None and not obs.empty:
        for row in obs.itertuples(index=False):
            r2 = tech_lookup(row.ticker, row.date)
            if r2 is None:
                continue
            realized = row.err + row.fc
            ea.append(row.err_adj)
            et.append(realized - np.log1p(r2))
    rho_v = ncfg["rho_v"]
    if len(ea) < 50:
        return {"identified": False, "sigma_eps2": None, "rho_v": rho_v, "n": len(ea)}
    va, vt, c = float(np.var(ea, ddof=1)), float(np.var(et, ddof=1)), float(np.cov(ea, et)[0, 1])
    hi = min(va, vt) * 0.99
    f = lambda s: s + rho_v * np.sqrt(max(va - s, 0) * max(vt - s, 0)) - c
    lo_, hi_ = 0.0, hi
    if f(hi_) < 0:      # Kovarianz größer als erklärbar -> Messfehler fast null
        s = hi_
    elif f(lo_) > 0:    # kein gemeinsames Rauschen messbar
        s = 0.0
    else:
        for _ in range(60):
            m = (lo_ + hi_) / 2
            lo_, hi_ = (m, hi_) if f(m) < 0 else (lo_, m)
        s = (lo_ + hi_) / 2
    return {"identified": True, "sigma_eps2": float(s), "rho_v": rho_v, "n": len(ea),
            "var_eA": va, "var_eT": vt, "cov": c,
            "sd_vA": float(np.sqrt(max(va - s, 0))), "sd_vT": float(np.sqrt(max(vt - s, 0)))}


def to_measurement_noise(calib_a: dict, sigma_eps2: float, floor_share: float) -> None:
    """Fehlervarianzen der Häuser in Messrauschen σ_v² umrechnen (in place)."""
    for f in calib_a["firms"].values():
        f["var_raw"] = f["var"]
        f["var"] = max(f["var"] - sigma_eps2, floor_share * f["var"])
    calib_a["global_var_raw"] = calib_a["global_var"]
    calib_a["global_var"] = max(calib_a["global_var"] - sigma_eps2, floor_share * calib_a["global_var"])
    pc = calib_a.get("pair_cov")
    if pc is not None and calib_a["firms"]:
        mean_v = float(np.mean([f["var"] for f in calib_a["firms"].values()]))
        calib_a["rho_ff"] = float(np.clip((pc - sigma_eps2) / mean_v, 0.0, 0.95))
