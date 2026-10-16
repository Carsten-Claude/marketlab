"""Monatslauf (Universum, Qualitätstor, Kalibrierung) und Tageslauf (Messungen, Kalman, Signale)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from . import analysts, fundamentals, kalman, signals, technical
from .data import get_provider
from .universe import build_universe, region_name, region_suffix

log = logging.getLogger("marketlab")
ROOT = os.path.join(os.path.dirname(__file__), "..")
STATE = os.path.join(ROOT, "state")


def load_cfg() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_json(name: str, obj) -> None:
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)


def _load_json(name: str, default=None):
    p = os.path.join(STATE, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _index_frame(close: pd.DataFrame, idx_close: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Für jeden Ticker die Kursreihe des passenden Regionalindex (für relatives Momentum)."""
    out = {}
    for tk in close.columns:
        ix = cfg["region_index"].get(region_suffix(tk), "^GSPC")
        col = idx_close[ix] if ix in idx_close.columns else idx_close.get("^GSPC")
        out[tk] = col if col is not None else pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out).reindex(close.index).ffill()


def _prices_with_index(prov, tickers: list[str], years: int, cfg: dict):
    indices = sorted(set(cfg["region_index"].values()))
    px = prov.prices(tickers + indices, years)
    px = px.dropna(axis=1, how="all")
    idx_close = px[[c for c in indices if c in px.columns]]
    close = px[[c for c in tickers if c in px.columns]].ffill(limit=5)
    return close, _index_frame(close, idx_close, cfg)


# ---------------------------------------------------------------------------
def run_monthly(use_web: bool = True) -> None:
    cfg = load_cfg()
    prov = get_provider()
    h = cfg["horizon_days"]
    uni = build_universe(cfg, use_web=use_web)

    # 1) Größenfilter und Qualitätswert
    rows, fund = [], {}
    for i, r in enumerate(uni.itertuples(index=False)):
        info = prov.info(r.ticker)
        mcap = info.get("marketCap")
        if not mcap:
            continue
        mcap_usd = mcap * prov.fx_to_usd(info.get("currency"))
        if not np.isfinite(mcap_usd) or mcap_usd < cfg["universe"]["min_market_cap_usd"]:
            continue
        fin = prov.financials(r.ticker)
        met = fundamentals.quality_metrics(info, fin)
        q = fundamentals.quality_score(met, cfg["quality"])
        rows.append({"ticker": r.ticker, "name": info.get("shortName") or r.ticker, "index": r.index,
                     "region": region_name(r.ticker), "currency": info.get("currency"),
                     "market_cap_usd": mcap_usd, "quality": q, **met})
        fund[r.ticker] = {"eps": fin.eps, "eps_dates": [d.isoformat() for d in fin.eps_dates]}
        if i % 50 == 0:
            log.info("Fundamentaldaten %s/%s", i, len(uni))
    big = pd.DataFrame(rows)
    if big.empty:
        raise RuntimeError("Keine Fundamentaldaten erhalten – Datenquelle prüfen")
    wl = (big[big["quality"] >= cfg["universe"]["min_quality_score"]]
          .sort_values("quality", ascending=False).head(cfg["universe"]["watchlist_size"]))
    log.info("Größenfilter: %s Titel, Watchlist: %s", len(big), len(wl))

    # 2) Technik-Kalibrierung über alle Titel des Größenfilters (breitere Datenbasis),
    #    mit Trennung in Lern- und Testzeitraum (70/30) für den ehrlichen Rückblick-Test
    close, idx_close = _prices_with_index(prov, big["ticker"].tolist(), cfg["history_years"], cfg)
    ind = technical.indicators(close, idx_close, cfg["technical"])
    states = technical.state_frame(ind)
    fwd = technical.forward_returns(close, h)
    cut = close.index[int(len(close.index) * 0.7)]
    calib_learn = technical.calibrate(states[states.index < cut], fwd[fwd.index < cut], cfg["technical"])
    backtest = _backtest_technical(states[states.index >= cut], fwd[fwd.index >= cut], calib_learn, cfg)
    calib_t = technical.calibrate(states, fwd, cfg["technical"])   # final: alle Daten

    # 3) Analysten-Kalibrierung (Häuser-Historie, praktisch nur US-Titel)
    hist = {tk: prov.analyst_targets(tk) for tk in big["ticker"] if region_suffix(tk) == ""}
    calib_a = analysts.calibrate(hist, close, h, cfg["analysts"])

    def tech_lookup(tk, d):
        if tk not in states.columns:
            return None
        s = states[tk]
        pos = s.index.searchsorted(d, side="right") - 1
        if pos < 0:
            return None
        return technical.technical_measurement(s.iloc[pos], calib_t)[0] if pd.notna(s.iloc[pos]) else None

    noise = analysts.estimate_noise(calib_a, tech_lookup, cfg["noise"])
    if noise["identified"]:
        s_eps = noise["sigma_eps2"]
    else:
        s_eps = cfg["noise"]["default_eps_share"] * calib_t["global"]["var"]
        noise["sigma_eps2"] = s_eps
    technical.to_measurement_noise(calib_t, s_eps, cfg["noise"]["floor_share"])
    analysts.to_measurement_noise(calib_a, s_eps, cfg["noise"]["floor_share"])
    rho = cfg["noise"]["rho_v"]
    log.info("Rauschmodell: σε = %.3f (%s), ρ_v = %.2f, ρ_ff = %.2f", s_eps ** 0.5,
             "geschätzt" if noise["identified"] else "Annahme", rho, calib_a["rho_ff"])

    os.makedirs(STATE, exist_ok=True)
    wl.to_csv(os.path.join(STATE, "watchlist.csv"), index=False)
    _save_json("fundamentals.json", {k: v for k, v in fund.items() if k in set(wl["ticker"])})
    _save_json("calibration.json", {
        "created": datetime.now(timezone.utc).isoformat(), "provider": prov.name, "horizon_days": h,
        "technical": calib_t, "analysts": {k: v for k, v in calib_a.items() if k != "obs"},
        "rho": rho, "noise": noise, "backtest": backtest, "universe_size": int(len(uni)), "size_filter": int(len(big)),
    })


def _backtest_technical(states, fwd, calib, cfg) -> dict:
    """Testzeitraum: Trägt die im Lernzeitraum kalibrierte Technik-Messung Information?"""
    step = cfg["technical"]["sample_every"]
    s = states.iloc[::step].stack()
    f = fwd.iloc[::step].stack()
    df = pd.concat([s.rename("key"), f.rename("ret")], axis=1).dropna()
    if len(df) < 100:
        return {"n": int(len(df))}
    m = df["key"].map(lambda k: technical.technical_measurement(k, calib))
    df["pred"] = m.str[0]
    df["cr"] = m.map(lambda t: t[0] / np.sqrt(t[1]))
    q = df["cr"].quantile(0.8)
    top = df[df["cr"] >= q]
    return {
        "n": int(len(df)),
        "corr_pred_real": float(df[["pred", "ret"]].corr().iloc[0, 1]),
        "mean_all": float(df["ret"].mean()), "hit_all": float((df["ret"] > 0).mean()),
        "mean_top": float(top["ret"].mean()), "hit_top": float((top["ret"] > 0).mean()),
        "n_top": int(len(top)),
        "period": [str(states.index.min().date()), str(states.index.max().date())],
    }


# ---------------------------------------------------------------------------
def run_daily() -> dict:
    cfg = load_cfg()
    prov = get_provider()
    h = cfg["horizon_days"]
    calib = _load_json("calibration.json")
    if calib is None:
        raise RuntimeError("Keine Kalibrierung vorhanden – zuerst den Monatslauf starten")
    wl = pd.read_csv(os.path.join(STATE, "watchlist.csv"))
    fund = _load_json("fundamentals.json", {})
    kstate = _load_json("kalman_state.json", {})
    rho = calib["rho"]
    asof = pd.Timestamp.today().normalize()

    close, idx_close = _prices_with_index(prov, wl["ticker"].tolist(), 2, cfg)
    ind = technical.indicators(close, idx_close, cfg["technical"])
    states = technical.state_frame(ind)

    results = []
    for w in wl.itertuples(index=False):
        tk = w.ticker
        if tk not in close.columns or close[tk].dropna().empty:
            continue
        price = float(close[tk].dropna().iloc[-1])
        info = prov.info(tk)
        hist = prov.analyst_targets(tk) if region_suffix(tk) == "" else None
        a = analysts.measurement(tk, price, info, hist, calib["analysts"], h, cfg["analysts"], asof)
        key = states[tk].iloc[-1] if tk in states.columns else None
        key = key if isinstance(key, str) else None
        r2, v2, n2 = technical.technical_measurement(key, calib["technical"])
        st = kstate.get(tk)
        if st and st.get("date") == str(asof.date()):   # Wiederholungslauf am selben Tag
            st = st.get("prev")
        fused = kalman.kalman_step(st, a["r1"], a["var1"], r2, v2, rho, cfg["kalman"])
        kstate[tk] = {"x": fused["x"], "P": fused["P"], "date": str(asof.date()),
                      "prev": {"x": st["x"], "P": st["P"]} if st else None}
        sig = signals.evaluate(tk, ind, fused, cfg["signals"])

        est = prov.estimates(tk)
        f = fund.get(tk, {})
        from .data import Financials
        fin = Financials(eps=f.get("eps", []), eps_dates=[pd.Timestamp(d) for d in f.get("eps_dates", [])])
        grow = fundamentals.expected_growth(est, fin)
        fv = fundamentals.fair_value(fin, close[tk], grow["growth_12m"])
        fv_gap = (fv["fair_value"] / price - 1) if fv["fair_value"] else None

        results.append({
            "ticker": tk, "name": w.name, "region": w.region, "currency": info.get("currency") or w.currency,
            "quality": w.quality, "price": price, "signal": sig["signal"], "reasons": sig["reasons"],
            "cr": sig["cr"], "x": fused["x"], "sd": float(np.sqrt(fused["var_static"])),
            "sd_total": float(np.sqrt(fused["var_static"] + calib["noise"]["sigma_eps2"])),
            "x_static": fused["x_static"], "w_analyst": fused["w1"], "w_tech": fused["w2"],
            "r1": a["r1"], "sd1": float(np.sqrt(a["var1"])) if a["var1"] else None,
            "analyst_method": a["method"], "analyst_detail": a["detail"],
            "r2": r2, "sd2": float(np.sqrt(v2)), "tech_state": technical.describe_key(key) if key else "n/a",
            "tech_n": n2, "stop": sig["stop"], "trend": sig["trend"],
            "rsi": _last(ind["rsi"], tk), "pct_b": _last(ind["pct_b"], tk),
            "sma50": _last(ind["sma50"], tk), "sma200": _last(ind["sma200"], tk),
            "rel_mom": _last(ind["rel_mom"], tk),
            "fair_value": fv["fair_value"], "fv_gap": fv_gap, **grow,
        })
    res = pd.DataFrame(results).sort_values("cr", ascending=False).reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)
    _save_json("kalman_state.json", kstate)
    _append_log(res, asof)
    track = _track_record(close, h)
    return {"results": res, "calib": calib, "asof": asof, "cfg": cfg, "track": track}


def _track_record(close: pd.DataFrame, h: int) -> dict:
    """Erfolgskontrolle: Was brachten frühere Einstiegssignale nach h Handelstagen?"""
    p = os.path.join(STATE, "signal_log.csv")
    if not os.path.exists(p):
        return {"n": 0}
    log_df = pd.read_csv(p)
    ent = log_df[log_df["signal"] == "Einstieg"].copy()
    # nur das erste Einstiegssignal einer Serie zählen (sonst täglich doppelt)
    ent = ent.sort_values("date").drop_duplicates(["ticker"], keep="first") if len(ent) else ent
    rets, preds = [], []
    for r in ent.itertuples(index=False):
        if r.ticker not in close.columns:
            continue
        s = close[r.ticker].dropna()
        pos = s.index.searchsorted(pd.Timestamp(r.date), side="left")
        if pos + h >= len(s):
            continue
        rets.append(float(s.iloc[pos + h] / s.iloc[pos] - 1))
        preds.append(float(r.x))
    if not rets:
        return {"n": 0, "pending": int(len(ent))}
    return {"n": len(rets), "mean": float(np.mean(rets)), "hit": float(np.mean(np.array(rets) > 0)),
            "mean_pred": float(np.mean(preds)), "pending": int(len(ent) - len(rets))}


def _last(df: pd.DataFrame, tk: str):
    if tk not in df.columns:
        return None
    s = df[tk].dropna()
    return float(s.iloc[-1]) if len(s) else None


def _append_log(res: pd.DataFrame, asof: pd.Timestamp) -> None:
    """Signal-Protokoll für die spätere Erfolgsmessung des Agenten."""
    p = os.path.join(STATE, "signal_log.csv")
    cols = ["rank", "ticker", "signal", "price", "x", "sd", "cr", "r1", "r2", "w_analyst", "analyst_method"]
    out = res[cols].copy()
    out.insert(0, "date", str(asof.date()))
    if os.path.exists(p):
        old = pd.read_csv(p)
        old = old[old["date"] != str(asof.date())]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(p, index=False)
