"""Technische Indikatoren, Signalzustände und deren Kalibrierung auf erwartete Rendite.

Die Chartanalyse wird so zu einer echten Messung im Kalman-Sinn: Jeder Signalzustand
(Trend × RSI × Bollinger × relatives Momentum) bekommt aus der Kurshistorie eine
erwartete Rendite r₂ über den Horizont und deren Fehlervarianz σ₂².
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND = {1: "auf", 0: "gemischt", -1: "ab"}
RSI_EDGES = [-np.inf, 35, 50, 70, np.inf]
RSI_LABEL = ["<35", "35–50", "50–70", ">70"]
BB_EDGES = [-np.inf, 0.2, 0.8, np.inf]
BB_LABEL = ["unten", "Mitte", "oben"]


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def indicators(close: pd.DataFrame, index_close: pd.DataFrame, tcfg: dict) -> dict[str, pd.DataFrame]:
    """close: Datum × Ticker; index_close: Datum × Ticker (bereits passender Regionalindex je Spalte)."""
    sma_f = close.rolling(tcfg["sma_fast"], min_periods=tcfg["sma_fast"]).mean()
    sma_s = close.rolling(tcfg["sma_slow"], min_periods=tcfg["sma_slow"]).mean()
    mid = close.rolling(tcfg["bb_period"]).mean()
    sd = close.rolling(tcfg["bb_period"]).std()
    upper, lower = mid + tcfg["bb_std"] * sd, mid - tcfg["bb_std"] * sd
    pct_b = (close - lower) / (upper - lower)
    m = tcfg["momentum_days"]
    mom = close / close.shift(m) - 1
    idx_mom = index_close / index_close.shift(m) - 1
    rel_mom = mom - idx_mom.reindex_like(mom)
    trend = pd.DataFrame(0, index=close.index, columns=close.columns)
    trend[(close > sma_s) & (sma_f > sma_s)] = 1
    trend[(close < sma_s) & (sma_f < sma_s)] = -1
    trend[sma_s.isna()] = np.nan
    return {"close": close, "sma50": sma_f, "sma200": sma_s, "rsi": rsi(close, tcfg["rsi_period"]),
            "pct_b": pct_b, "bb_upper": upper, "bb_lower": lower, "mom": mom, "rel_mom": rel_mom,
            "trend": trend}


def state_key(trend, rsi_v, pct_b, rel_mom) -> str | None:
    if any(pd.isna(x) for x in (trend, rsi_v, pct_b, rel_mom)):
        return None
    r = int(np.digitize(rsi_v, RSI_EDGES[1:-1]))
    b = int(np.digitize(pct_b, BB_EDGES[1:-1]))
    return f"{int(trend)}|{r}|{b}|{int(rel_mom > 0)}"


def describe_key(key: str) -> str:
    t, r, b, m = key.split("|")
    return (f"Trend {TREND[int(t)]}, RSI {RSI_LABEL[int(r)]}, Bollinger {BB_LABEL[int(b)]}, "
            f"rel. Momentum {'positiv' if m == '1' else 'negativ'}")


def state_frame(ind: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Signalzustand je Datum × Ticker als String-Schlüssel."""
    tr, rs, pb, rm = (ind[k] for k in ("trend", "rsi", "pct_b", "rel_mom"))
    r = pd.DataFrame(np.digitize(rs.fillna(-1), RSI_EDGES[1:-1]), index=rs.index, columns=rs.columns)
    b = pd.DataFrame(np.digitize(pb.fillna(-9), BB_EDGES[1:-1]), index=pb.index, columns=pb.columns)
    key = (tr.fillna(9).astype(int).astype(str) + "|" + r.astype(str) + "|" + b.astype(str) + "|"
           + (rm > 0).astype(int).astype(str))
    valid = tr.notna() & rs.notna() & pb.notna() & rm.notna()
    return key.where(valid)


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    return close.shift(-h) / close - 1


def calibrate(states: pd.DataFrame, fwd: pd.DataFrame, tcfg: dict) -> dict:
    """Erwartete Rendite und Fehlervarianz je Zustand, mit Schrumpfung bei dünner Datenlage."""
    step = tcfg["sample_every"]
    s = states.iloc[::step].stack()
    f = fwd.iloc[::step].stack()
    df = pd.concat([s.rename("key"), f.rename("ret")], axis=1).dropna()
    if df.empty:
        return {"buckets": {}, "global": {"n": 0, "mean": 0.0, "var": 0.12 ** 2}}
    df["trend"] = df["key"].str.split("|").str[0]
    glob = {"n": int(len(df)), "mean": float(df["ret"].mean()), "var": float(df["ret"].var())}
    parent = df.groupby("trend")["ret"].agg(["count", "mean", "var"])
    k = tcfg["min_bucket_obs"]
    out = {}
    for key, g in df.groupby("key"):
        n, mu, var = len(g), float(g["ret"].mean()), float(g["ret"].var(ddof=1)) if len(g) > 1 else glob["var"]
        p = parent.loc[key.split("|")[0]]
        mu_s = (n * mu + k * p["mean"]) / (n + k)
        var_s = (n * var + k * p["var"]) / (n + k)
        out[key] = {"n": n, "mean": float(mu_s), "var": float(var_s), "raw_mean": mu}
    return {"buckets": out, "global": glob}


def technical_measurement(key: str | None, calib: dict) -> tuple[float, float, int]:
    """r₂, σ₂², Stichprobengröße für einen aktuellen Signalzustand."""
    g = calib["global"]
    if key is None or key not in calib["buckets"]:
        return g["mean"], g["var"], 0
    b = calib["buckets"][key]
    return b["mean"], b["var"], b["n"]


def to_measurement_noise(calib: dict, sigma_eps2: float, floor_share: float) -> None:
    """Zustands-Varianz = gemeinsames Kursrauschen σε² + Messrauschen σ_vT² -> nur σ_vT² behalten."""
    for b in calib["buckets"].values():
        b["var_raw"] = b["var"]
        b["var"] = max(b["var"] - sigma_eps2, floor_share * b["var"])
    g = calib["global"]
    g["var_raw"] = g["var"]
    g["var"] = max(g["var"] - sigma_eps2, floor_share * g["var"])
