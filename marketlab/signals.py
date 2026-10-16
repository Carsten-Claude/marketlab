"""Ein- und Ausstiegsregeln: Kalman-Chance/Risiko plus harte Sicherheitsnetze."""
from __future__ import annotations

import numpy as np
import pandas as pd


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

    x, sd = fused["x"], np.sqrt(fused["var_static"])
    cr = x / sd if sd > 0 else 0.0
    ms = monthly_std(c)
    stop = price * (1 - scfg["stop_monthly_std"] * ms) if ms else None

    reasons, signal = [], "Halten"
    if cr < scfg["exit_cr"]:
        reasons.append(f"Chance/Risiko {cr:+.2f} unter Ausstiegsschwelle {scfg['exit_cr']:+.2f}")
    if below:
        reasons.append(f"Schlusskurs {n} Tage in Folge unter GD 200")
    if death_cross:
        reasons.append("GD 50 kreuzt GD 200 nach unten")
    if reasons:
        signal = "Ausstieg"
    elif cr > scfg["entry_cr"] and x > scfg["entry_min_return"] and trend == 1:
        signal = "Einstieg"
        reasons.append(f"Chance/Risiko {cr:+.2f} über {scfg['entry_cr']:+.2f}, "
                       f"erwartete Rendite {x:+.1%}, Aufwärtstrend intakt")
    else:
        if trend != 1 and cr > scfg["entry_cr"]:
            reasons.append("Chance/Risiko gut, aber Trendfilter (Kurs > GD 200, GD 50 > GD 200) nicht erfüllt")
        else:
            reasons.append(f"Chance/Risiko {cr:+.2f} zwischen den Schwellen")
    return {"price": price, "cr": float(cr), "signal": signal, "reasons": reasons, "stop": stop,
            "monthly_std": ms, "trend": None if pd.isna(trend) else int(trend)}
