"""Ein- und Ausstiegsregeln.

Zweistufig: `evaluate` sammelt je Titel die technischen Ereignisse und den Stopp; `decide`
entscheidet anschließend auf Basis des Vorsprungs gegenüber dem Watchlist-Median
(Chance/Risiko = Vorsprung / Schätzfehler), sobald dieser für alle Titel feststeht.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _d(x: float, digits: int = 2, pct: bool = False) -> str:
    """Zahl mit Vorzeichen und deutschem Dezimalkomma."""
    v = x * 100 if pct else x
    return f"{v:+.{digits}f}".replace(".", ",") + (" %" if pct else "")


def monthly_std(close: pd.Series) -> float | None:
    r = close.dropna().pct_change().dropna().iloc[-252:]
    if len(r) < 60:
        return None
    return float(r.std() * np.sqrt(21))


def evaluate(tk: str, ind: dict[str, pd.DataFrame], fused: dict, scfg: dict) -> dict:
    c = ind["close"][tk].dropna()
    price = float(c.iloc[-1])
    sma50, sma200 = ind["sma50"][tk], ind["sma200"][tk]
    trend = ind["trend"][tk].iloc[-1]
    n = scfg["gd200_break_days"]
    # GD-200-Bruch als Ereignis: n Schlusskurse in Folge darunter, nachdem der Kurs in den
    # letzten 10 Tagen noch darüber lag (ein dauerhafter Abwärtstrend sperrt nur den Einstieg)
    cl = ind["close"][tk]
    below = False
    if sma200.notna().iloc[-10:].all():
        below = bool((cl.iloc[-n:] < sma200.iloc[-n:]).all() and (cl.iloc[-10:-n] > sma200.iloc[-10:-n]).any())
    diff = (sma50 - sma200).dropna()
    death_cross = bool(len(diff) > 6 and (diff.iloc[-6] > 0) and (diff.iloc[-1] < 0))
    events = []
    if below:
        events.append(f"Schlusskurs {n} Tage in Folge unter GD 200")
    if death_cross:
        events.append("GD 50 kreuzt GD 200 nach unten")
    ms = monthly_std(c)
    stop = price * (1 - scfg["stop_monthly_std"] * ms) if ms else None
    return {"price": price, "stop": stop, "monthly_std": ms, "events": events,
            "trend": None if pd.isna(trend) else int(trend),
            # vorläufig, wird in decide() überschrieben
            "signal": "Halten", "reasons": [], "cr": 0.0}


def decide(r, scfg: dict) -> dict:
    """r: Ergebniszeile mit cr (Vorsprung/σ), excess, trend, events."""
    cr, ex = float(r["cr"]), float(r["excess"])
    events = list(r["events"]) if isinstance(r["events"], (list, tuple)) else []
    reasons = []
    if cr < scfg["exit_cr"]:
        reasons.append(f"Rückstand {_d(ex, 1, True)} ggü. Watchlist-Median, Chance/Risiko {_d(cr)} "
                       f"unter {_d(scfg['exit_cr'])}")
    reasons += events
    if reasons:
        return {"signal": "Ausstieg", "reasons": reasons}
    if cr > scfg["entry_cr"] and ex > scfg["entry_min_excess"] and r["trend"] == 1:
        return {"signal": "Einstieg", "reasons": [
            f"Vorsprung {_d(ex, 1, True)} ggü. Watchlist-Median, Chance/Risiko {_d(cr)} über "
            f"{_d(scfg['entry_cr'])}, Aufwärtstrend intakt"]}
    if cr > scfg["entry_cr"] and ex > scfg["entry_min_excess"]:
        return {"signal": "Halten", "reasons": [
            "Vorsprung ausreichend, aber Trendfilter (Kurs > GD 200, GD 50 > GD 200) nicht erfüllt"]}
    return {"signal": "Halten", "reasons": [
        f"Vorsprung {_d(ex, 1, True)} ggü. Watchlist-Median, Chance/Risiko {_d(cr)} – "
        f"kein belastbarer Unterschied"]}
