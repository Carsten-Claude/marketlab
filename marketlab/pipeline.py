"""Monatslauf (Universum, Qualitätstor, Kalibrierung) und Tageslauf (Messungen, Kalman, Signale)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from . import analysts, depot, fundamentals, kalman, signals, skill, technical
from .data import get_provider
from .universe import build_universe, region_name, region_suffix

log = logging.getLogger("marketlab")
ROOT = os.path.join(os.path.dirname(__file__), "..")
STATE = os.path.join(ROOT, "state")
MODEL_VERSION = 4   # bei Änderung der Messdefinitionen erhöhen -> Kalman-Zustand startet neu


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
    ucfg = cfg["universe"]
    stable = big["earnings_stability"].notna() & (big["earnings_stability"] <= ucfg["max_earnings_cv"])
    log.info("Stabilitätstor: %s von %s Titeln mit belegten, stabilen Gewinnen", int(stable.sum()), len(big))
    # Mehrere Aktiengattungen eines Unternehmens (z. B. GOOGL/GOOG) nur einmal: größte Marktkapitalisierung bleibt
    big = big.sort_values("market_cap_usd", ascending=False).drop_duplicates("name", keep="first")
    stable = big["earnings_stability"].notna() & (big["earnings_stability"] <= ucfg["max_earnings_cv"])
    wl = (big[stable & (big["quality"] >= ucfg["min_quality_score"])]
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

    # 3) Analysten-Kalibrierung (Häuser-Historie, praktisch nur US-Titel) + Testzeitraum
    hist = {tk: prov.analyst_targets(tk) for tk in big["ticker"] if region_suffix(tk) == ""}
    calib_a = analysts.calibrate(hist, close, h, cfg["analysts"])
    skill_a = analysts.skill_test(calib_a.get("obs"), cut, cfg["analysts"])
    obs_raw = calib_a.get("obs")   # Rohdaten für Zusatzauswertungen (P2-Test), vor möglicher Wiederverwendung

    # Datenausfall absichern: zu wenige Analystendaten -> letzte gute Analysten-Kalibrierung behalten
    prev = _load_json("calibration.json")
    reused = None
    min_obs = cfg["analysts"].get("min_total_obs", 2000)
    if calib_a["n"] < min_obs and prev and prev.get("analysts", {}).get("n", 0) >= min_obs:
        reused = prev.get("analysts_reused") or str(prev["created"])[:10]
        log.warning("Nur %s Analystenprognosen erhalten – verwende Analysten-Kalibrierung vom %s", calib_a["n"], reused)

    def tech_lookup(tk, d):
        if tk not in states.columns:
            return None
        s_ = states[tk]
        pos = s_.index.searchsorted(d, side="right") - 1
        if pos < 0 or pd.isna(s_.iloc[pos]):
            return None
        return technical.raw_measurement(s_.iloc[pos], calib_t)[0]

    # 4) Rauschmodell: gemeinsames Kursrauschen σε² von den Messfehlern trennen
    noise = analysts.estimate_noise(calib_a, tech_lookup, cfg["noise"])
    if noise["identified"]:
        s_eps = noise["sigma_eps2"]
    else:
        s_eps = cfg["noise"]["default_eps_share"] * calib_t["global"]["var"]
        noise["sigma_eps2"] = s_eps
    fl = cfg["noise"]["floor_share"]

    # 5) Treffsicherheit -> kalibrierte Messungen und Messrauschen beider Quellen
    sk_t = backtest.get("skill") or skill.fit([], [])
    calib_t["skill"] = {**sk_t, "var_v": skill.measurement_var(sk_t["var_e"], s_eps, fl)}
    if reused:
        calib_a = prev["analysts"]
        noise = prev.get("noise", noise)
        s_eps = noise["sigma_eps2"]
        calib_t["skill"] = {**sk_t, "var_v": skill.measurement_var(sk_t["var_e"], s_eps, fl)}
    else:
        analysts.apply_skill(calib_a, skill_a, s_eps, fl)
        calib_a["rank"] = analysts.rank_calibrate(calib_a.get("obs"), close, cut, h, cfg["analysts"])
    calib_a.setdefault("rank", {"ok": False})
    calib_a["rank"]["mu"] = float(np.log1p(calib_t["global"]["mean"]))
    rho = cfg["noise"]["rho_v"]
    log.info("Rauschmodell: σε = %.3f (%s) · Technik b = %.2f, σ_v = %.3f · Analysten b = %.2f, σ_v = %.3f · ρ_ff = %.2f",
             s_eps ** 0.5, "geschätzt" if noise["identified"] else "Annahme",
             calib_t["skill"]["b"], calib_t["skill"]["var_v"] ** 0.5,
             calib_a["skill"]["b"], calib_a["skill"]["var_v"] ** 0.5, calib_a["rho_ff"])

    # 6) Horizonttest: Treffsicherheit beider Quellen für alternative Horizonte
    horizon_test = _horizon_test(close, states, hist, cut, cfg)

    # 7) P2-Test (Chartanalyse: Informationsquelle oder Filter?) – abgesichert, ohne Einfluss auf das Modell
    _p2_test(obs_raw, close, states, cut, h, cfg, noise)

    os.makedirs(STATE, exist_ok=True)
    wl.to_csv(os.path.join(STATE, "watchlist.csv"), index=False)
    _save_json("fundamentals.json", {k: v for k, v in fund.items() if k in set(wl["ticker"])})
    _save_json("calibration.json", {
        "created": datetime.now(timezone.utc).isoformat(), "provider": prov.name, "horizon_days": h,
        "version": MODEL_VERSION, "analysts_reused": reused,
        "technical": calib_t, "analysts": {k: v for k, v in calib_a.items() if k != "obs"},
        "rho": rho, "noise": noise, "backtest": backtest, "horizon_test": horizon_test, "universe_size": int(len(uni)), "size_filter": int(len(big)),
    })


def _p2_test(obs, close, states, cut, h, cfg, noise) -> None:
    from . import p2test
    try:
        os.makedirs(STATE, exist_ok=True)
        if obs is not None and len(obs) >= cfg["analysts"].get("min_total_obs", 2000):
            obs.to_csv(os.path.join(STATE, "analyst_obs.csv.gz"), index=False)   # spätere Tests ohne Neuabruf
        elif os.path.exists(os.path.join(STATE, "analyst_obs.csv.gz")):
            obs = pd.read_csv(os.path.join(STATE, "analyst_obs.csv.gz"), parse_dates=["date"])
            log.info("P2-Test: Analystenprognosen aus gespeichertem Bestand (%s)", len(obs))
        r = p2test.run(obs, close, states, cut, h, cfg, noise)
        r["created"] = datetime.now(timezone.utc).isoformat()
        _save_json("p2_test.json", r)
        with open(os.path.join(STATE, "p2_test.md"), "w", encoding="utf-8") as f:
            f.write(p2test.report(r))
    except Exception as exc:  # noqa: BLE001 – Auswertung darf den Monatslauf nie gefährden
        log.exception("P2-Test fehlgeschlagen: %s", exc)


def _horizon_test(close, states, hist, cut, cfg) -> list[dict]:
    """Treffsicherheit (Testzeitraum) von Technik und Analysten je Horizont.

    Spreizung = Ø Rendite der besten 20 % minus Ø aller Fälle, zusätzlich auf ein Jahr hochgerechnet,
    damit Horizonte vergleichbar sind.
    """
    out = []
    for weeks in cfg.get("horizon_test_weeks", [8, 13, 26]):
        h = int(weeks * 5)
        fwd = technical.forward_returns(close, h)
        cl = technical.calibrate(states[states.index < cut], fwd[fwd.index < cut], cfg["technical"])
        bt = _backtest_technical(states[states.index >= cut], fwd[fwd.index >= cut], cl, cfg)
        ca = analysts.calibrate(hist, close, h, cfg["analysts"])
        sa = analysts.skill_test(ca.get("obs"), cut, cfg["analysts"])
        sk_t = bt.get("skill") or {}
        row = {"weeks": weeks, "h": h,
               "tech_corr": sk_t.get("corr"), "tech_b": sk_t.get("b_raw"), "tech_n": bt.get("n"),
               "tech_spread": (bt["mean_top"] - bt["mean_all"]) if "mean_top" in bt else None,
               "ana_corr": sa.get("corr"), "ana_b": sa.get("b_raw"), "ana_n": sa.get("n"),
               "ana_spread": sa.get("spread")}
        for k in ("tech_spread", "ana_spread"):
            row[k + "_pa"] = (row[k] * 52 / weeks) if row[k] is not None else None
        log.info("Horizonttest %s Wochen: Technik corr %s b %s · Analysten corr %s b %s", weeks,
                 _f(row["tech_corr"]), _f(row["tech_b"]), _f(row["ana_corr"]), _f(row["ana_b"]))
        out.append(row)
    return out


def _f(x):
    return "–" if x is None else f"{x:+.3f}"


def _backtest_technical(states, fwd, calib, cfg) -> dict:
    """Testzeitraum: Trägt die im Lernzeitraum kalibrierte Technik-Messung Information?"""
    step = cfg["technical"]["sample_every"]
    s = states.iloc[::step].stack()
    f = fwd.iloc[::step].stack()
    df = pd.concat([s.rename("key"), f.rename("ret")], axis=1).dropna()
    if len(df) < 100:
        return {"n": int(len(df))}
    df["pred"] = df["key"].map(lambda k: technical.raw_measurement(k, calib)[0])
    q = df["pred"].quantile(0.8)
    top = df[df["pred"] >= q]
    return {
        "n": int(len(df)),
        "corr_pred_real": float(df[["pred", "ret"]].corr().iloc[0, 1]),
        "mean_all": float(df["ret"].mean()), "hit_all": float((df["ret"] > 0).mean()),
        "mean_top": float(top["ret"].mean()), "hit_top": float((top["ret"] > 0).mean()),
        "n_top": int(len(top)), "skill": skill.fit(df["pred"].values, df["ret"].values),
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
    wl = pd.read_csv(os.path.join(STATE, "watchlist.csv")).drop_duplicates("name", keep="first")
    fund = _load_json("fundamentals.json", {})
    kstate = _load_json("kalman_state.json", {})
    if kstate.get("_version") != calib.get("version"):
        log.info("Modellversion geändert – Kalman-Zustand startet neu")
        kstate = {}
    kstate["_version"] = calib.get("version")
    rho = calib["rho"]
    asof = pd.Timestamp.today().normalize()

    extra = [t for t in depot.open_tickers() if t not in set(wl["ticker"])]   # Depot-Titel außerhalb der Watchlist
    close, idx_close = _prices_with_index(prov, wl["ticker"].tolist() + extra, 2, cfg)
    # Stichtag = letzter Handelstag in den Kursdaten (nicht das Kalenderdatum): Wochenend- und
    # Frühläufe werden so dem Tag zugeordnet, dessen Kurse sie tatsächlich verwenden.
    asof = pd.Timestamp(close.index[-1]).normalize()
    run_date = pd.Timestamp.today().normalize()
    ind = technical.indicators(close, idx_close, cfg["technical"])
    states = technical.state_frame(ind)

    results = []
    for w in wl.itertuples(index=False):   # nur Watchlist-Titel werden bewertet
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
        cv = getattr(w, "earnings_stability", None)
        fv_ok = cv is not None and not pd.isna(cv) and cv <= cfg["universe"]["fair_value_max_cv"]
        fv_gap = (fv["fair_value"] / price - 1) if (fv["fair_value"] and fv_ok) else None

        results.append({
            "ticker": tk, "name": w.name, "region": w.region, "sector": info.get("sector") or "", "currency": info.get("currency") or w.currency,
            "quality": w.quality, "price": price, "signal": sig["signal"], "reasons": sig["reasons"],
            "cr": sig["cr"], "x": fused["x"], "sd": float(np.sqrt(fused["var_static"])),
            "sd_total": float(np.sqrt(fused["var_static"] + calib["noise"]["sigma_eps2"])),
            "x_static": fused["x_static"], "w_analyst": fused["w1"], "w_tech": fused["w2"],
            "r1": a["r1"], "sd1": float(np.sqrt(a["var1"])) if a["var1"] else None,
            "analyst_method": a["method"], "analyst_detail": a["detail"], "quintile": a.get("quintile"),
            "r2": r2, "sd2": float(np.sqrt(v2)), "tech_state": technical.describe_key(key) if key else "n/a",
            "tech_n": n2, "stop": sig["stop"], "trend": sig["trend"], "events": sig["events"],
            "rsi": _last(ind["rsi"], tk), "pct_b": _last(ind["pct_b"], tk),
            "sma50": _last(ind["sma50"], tk), "sma200": _last(ind["sma200"], tk),
            "rel_mom": _last(ind["rel_mom"], tk),
            "fair_value": fv["fair_value"] if fv_ok else None, "fv_gap": fv_gap, "fv_ok": bool(fv_ok), **grow,
        })
    res = pd.DataFrame(results)
    # Vorsprung gegenüber dem Watchlist-Median: entfernt die gemeinsame Marktdrift, die beide
    # Quellen jedem Titel gleichermaßen zuschreiben. Signale beruhen nur auf dem Vorsprung.
    mu = float(res["x"].median())
    res["mu"] = mu
    res["excess"] = res["x"] - mu
    res["cr"] = res["excess"] / res["sd"]
    prev = _previous_signals(asof)
    dec = [signals.decide(r, cfg["signals"], prev.get(r["ticker"])) for _, r in res.iterrows()]
    res["signal"] = [d["signal"] for d in dec]
    res["reasons"] = [d["reasons"] for d in dec]
    res = res.sort_values(["cr", "quality"], ascending=False).reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)
    _save_json("kalman_state.json", kstate)
    _append_log(res, asof)
    track = _track_record(close, h, wl["ticker"].tolist())
    depots = depot.update(res, close, asof)
    return {"results": res, "calib": calib, "asof": asof, "cfg": cfg, "track": track, "prev": prev, "depots": depots,
            "run_date": run_date}


def _track_record(close: pd.DataFrame, h: int, universe: list[str]) -> dict:
    """Erfolgskontrolle: Einstiegssignale nach h Handelstagen, verglichen mit der Ø Watchlist."""
    p = os.path.join(STATE, "signal_log.csv")
    if not os.path.exists(p):
        return {"n": 0}
    log_df = pd.read_csv(p)
    # nur Signale des aktuellen Modells (ab Version 4, Rang-Kalibrierung) auswerten
    ver = log_df["version"] if "version" in log_df.columns else pd.Series(np.nan, index=log_df.index)
    log_df = log_df[ver.fillna(0) >= 4]
    ent = log_df[log_df["signal"] == "Einstieg"].copy()
    # nur das erste Einstiegssignal einer Serie zählen (sonst täglich doppelt)
    ent = ent.sort_values("date").drop_duplicates(["ticker"], keep="first") if len(ent) else ent
    cols = [t for t in universe if t in close.columns]
    fwd = close[cols].shift(-h) / close[cols] - 1
    rets, bench = [], []
    for r in ent.itertuples(index=False):
        if r.ticker not in close.columns:
            continue
        s = close[r.ticker].dropna()
        pos = s.index.searchsorted(pd.Timestamp(r.date), side="left")
        if pos + h >= len(s):
            continue
        rets.append(float(s.iloc[pos + h] / s.iloc[pos] - 1))
        d = s.index[pos]
        bench.append(float(fwd.loc[d].mean()) if d in fwd.index else np.nan)
    if not rets:
        return {"n": 0, "pending": int(len(ent))}
    rets, bench = np.array(rets), np.array(bench)
    return {"n": len(rets), "mean": float(rets.mean()), "hit": float((rets > 0).mean()),
            "bench": float(np.nanmean(bench)), "excess": float(np.nanmean(rets - bench)),
            "hit_excess": float(np.nanmean(rets > bench)), "pending": int(len(ent) - len(rets))}


def _last(df: pd.DataFrame, tk: str):
    if tk not in df.columns:
        return None
    s = df[tk].dropna()
    return float(s.iloc[-1]) if len(s) else None


def _previous_signals(asof: pd.Timestamp) -> dict:
    """Signale des letzten Handelstags vor `asof` (nur aktuelles Modell) – Grundlage der Hysterese."""
    p = os.path.join(STATE, "signal_log.csv")
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    if "version" not in d.columns:
        return {}
    d = d[(d["version"].fillna(0) >= 4) & (d["date"] < str(asof.date()))]
    if d.empty:
        return {}
    last = d[d["date"] == d["date"].max()]
    return dict(zip(last["ticker"], last["signal"]))


def _append_log(res: pd.DataFrame, asof: pd.Timestamp) -> None:
    """Signal-Protokoll für die spätere Erfolgsmessung des Agenten."""
    p = os.path.join(STATE, "signal_log.csv")
    cols = ["rank", "ticker", "signal", "price", "x", "mu", "excess", "sd", "cr", "r1", "r2", "w_analyst", "analyst_method"]
    cols += ["trend", "quintile", "sector"]          # Messgrößen für die Auswertung (keine Signalwirkung)
    out = res.reindex(columns=cols).copy()
    out["version"] = MODEL_VERSION
    out.insert(0, "date", str(asof.date()))
    if os.path.exists(p):
        old = pd.read_csv(p)
        old = old[old["date"] != str(asof.date())]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(p, index=False)
