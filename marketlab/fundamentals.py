"""Qualitätswert, Gewinnstabilität, Fair Value und Gewinnerwartung."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Estimates, Financials


def _lin(x: float | None, bad: float, good: float) -> float:
    """Linearer Score 0–100 zwischen bad und good; fehlend = neutral 50."""
    if x is None or not np.isfinite(x):
        return 50.0
    return float(np.clip((x - bad) / (good - bad) * 100.0, 0.0, 100.0))


def earnings_stability(eps: list[float]) -> float | None:
    """Variationskoeffizient der EPS-Residuen um den linearen Trend (kleiner = stabiler).

    Negative Gewinne in der Historie gelten als maximal instabil.
    """
    e = np.asarray([x for x in eps if x is not None and np.isfinite(x)], dtype=float)
    if len(e) < 3:
        return None
    if (e <= 0).any():
        return 1.0
    t = np.arange(len(e))
    slope, icpt = np.polyfit(t, e, 1)
    resid = e - (slope * t + icpt)
    return float(np.std(resid, ddof=1) / np.mean(np.abs(e)))


def cagr(values: list[float]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if len(v) < 2 or v[0] <= 0 or v[-1] <= 0:
        return None
    return float((v[-1] / v[0]) ** (1 / (len(v) - 1)) - 1)


def quality_metrics(info: dict, fin: Financials) -> dict:
    rev = info.get("totalRevenue")
    fcf = info.get("freeCashflow")
    ebitda = info.get("ebitda")
    debt, cash = info.get("totalDebt") or 0.0, info.get("totalCash") or 0.0
    nde = None
    if ebitda is not None:
        nde = (debt - cash) / ebitda if ebitda > 0 else 99.0
    return {
        "roe": info.get("returnOnEquity"),
        "operating_margin": info.get("operatingMargins"),
        "fcf_margin": (fcf / rev) if (fcf is not None and rev) else None,
        "revenue_growth": cagr(fin.revenue),
        "net_debt_ebitda": nde,
        "earnings_stability": earnings_stability(fin.eps),
    }


def quality_score(metrics: dict, qcfg: dict) -> float:
    scores = [_lin(metrics.get(k), v["bad"], v["good"]) for k, v in qcfg.items()]
    return float(np.mean(scores))


def expected_growth(est: Estimates, fin: Financials) -> dict:
    """Erwartetes Gewinnwachstum (12 Monate) und Schätzungsrevisionen."""
    g = [x for x in (est.growth_0y, est.growth_1y) if x is not None]
    g12 = float(np.mean(g)) if g else cagr(fin.eps)
    rev30 = rev90 = None
    if est.eps_now_1y and est.eps_30d_1y:
        rev30 = est.eps_now_1y / est.eps_30d_1y - 1
    if est.eps_now_1y and est.eps_90d_1y:
        rev90 = est.eps_now_1y / est.eps_90d_1y - 1
    return {"growth_12m": g12, "revision_30d": rev30, "revision_90d": rev90}


def fair_value(fin: Financials, prices: pd.Series, growth_12m: float | None) -> dict:
    """Fair Value = historisches Durchschnitts-KGV × erwarteter Gewinn je Aktie.

    KGV-Historie und Gewinnerwartung stammen aus derselben EPS-Quelle (Jahresabschlüsse),
    dadurch bleibt die Rechnung auch bei abweichender Bilanzwährung konsistent.
    """
    if not fin.eps or prices is None or prices.dropna().empty:
        return {"avg_pe": None, "fair_value": None}
    p = prices.dropna()
    pes = []
    for eps, d in zip(fin.eps, fin.eps_dates):
        if eps is None or eps <= 0:
            continue
        sub = p[p.index <= d]
        if sub.empty:
            continue
        pes.append(float(sub.iloc[-1]) / eps)
    # Ausreißer relativ zum Median entfernen (absolute Grenzen würden z. B. an
    # Pence-Notierungen in London scheitern, wo das "KGV" um Faktor 100 skaliert ist)
    if pes:
        med = float(np.median(pes))
        pes = [x for x in pes if med / 3 <= x <= med * 3]
    last_eps = fin.eps[-1]
    if not pes or last_eps is None or last_eps <= 0:
        return {"avg_pe": None, "fair_value": None}
    g = growth_12m if growth_12m is not None else 0.0
    g = float(np.clip(g, -0.5, 0.5))
    avg_pe = float(np.mean(pes))
    return {"avg_pe": avg_pe, "fair_value": avg_pe * last_eps * (1 + g)}
