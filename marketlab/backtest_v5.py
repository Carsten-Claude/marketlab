"""Rückrechnung Modellversion 5 gegen eine V4-nahe Regel (Historie, ohne neue Analystenabrufe).

Datenbasis: gespeicherte Analystenprognosen (state/analyst_obs.csv.gz) und Kurse. Aus jeder Prognose
wird das Kursziel rekonstruiert; je Titel und Tag gilt das jüngste Kursziel jedes Hauses (max. 6 Monate).
Kurspotenzial pA = Ø (log(Kursziel/Kurs) · h/252 + Verzerrung des Hauses aus dem Lernzeitraum).
Oberstes Fünftel F5: pA über der 80-%-Grenze des Lernzeitraums (wie im Live-Modell feste Grenzen).

Regeln (je Titel höchstens eine Position, Ein-/Ausstieg zum Schlusskurs, gleiche Beträge, ohne Gebühren):
  R0  V4-nah     Einstieg F5 + Aufwärtstrend; Ende, sobald F5 (aktueller Kurs) oder Aufwärtstrend fehlt
  R1  A+B        Einstieg F5 + Aufwärtstrend; Ende nur, wenn F5 mit Einstiegskurs verloren geht
  R2  A+B+Sperre wie R1; nach GD-200-Bruch/Todeskreuz Wiedereinstieg erst nach 5 Schlusskursen über GD 200
  R3  Referenz   Einstieg wie R1, feste Haltedauer h Tage (prüft die Einstiegsregel allein)
Für R0–R2 gelten zusätzlich Ausstiegsereignisse (GD-200-Bruch an 2 Tagen, Todeskreuz) und der Stopp.
Messlatte: gleichgewichtete Rendite aller Titel mit Analystendaten (selbes Universum).

Vorbehalt: Universum = heutige Großwerte (Überlebenden-Verzerrung); betrifft alle Regeln gleich,
der Regelvergleich bleibt aussagekräftig, absolute Vorsprünge sind eher zu hoch.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from . import pipeline, technical
from .data import get_provider

log = logging.getLogger("marketlab.backtest")
STATE = pipeline.STATE
RULES = {"R0": "V4-nah", "R4": "V4-nah+Sperre", "R1": "A+B", "R2": "A+B+Sperre", "R3": "Referenz: feste Haltedauer"}
COST_RT = 0.002   # angenommene Kosten je Kauf+Verkauf (Spread, Gebühr): 0,2 %


def _panel(obs: pd.DataFrame, close: pd.DataFrame, cut: pd.Timestamp, h: int, cfg: dict):
    o = obs.copy()
    o["date"] = pd.to_datetime(o["date"])
    learn = o[o["date"] < cut]
    g, k = learn["err"].mean(), cfg["analysts"]["min_firm_obs"]
    bias = {f: (len(x) * x["err"].mean() + k * g) / (len(x) + k) for f, x in learn.groupby("firm")}
    o["bias"] = o["firm"].map(bias).fillna(g)
    LTm, Bm = {}, {}
    for tk, x in o.groupby("ticker"):
        if tk not in close.columns:
            continue
        p = close[tk]
        pos = p.index.searchsorted(x["date"].values, side="right") - 1
        ok = pos >= 0
        x, pos = x[ok], pos[ok]
        x = x.assign(d=p.index[pos], p0=p.values[pos])
        x = x[np.isfinite(x["p0"]) & (x["p0"] > 0)]
        if x.empty:
            continue
        x = x.assign(lt=np.log(x["p0"]) + x["fc"] * 252 / h)     # log(Kursziel)
        # je Haus der jüngste Wert, gültig bis 126 Handelstage
        LT = x.pivot_table(index="d", columns="firm", values="lt", aggfunc="last").reindex(close.index).ffill(limit=126)
        B = x.pivot_table(index="d", columns="firm", values="bias", aggfunc="last").reindex(close.index).ffill(limit=126)
        LTm[tk] = LT.mean(axis=1)
        Bm[tk] = B.where(LT.notna()).mean(axis=1)
    tick = sorted(LTm)
    mean_lt, mean_b = pd.DataFrame(LTm), pd.DataFrame(Bm)
    pA = (mean_lt - np.log(close[tick])) * h / 252 + mean_b
    return pA, mean_lt, mean_b, tick


def _sim(rule: str, tick, close, ind, pA, mean_lt, mean_b, e4, start, end, h, scfg):
    trades = []
    held = pd.DataFrame(False, index=close.index, columns=tick)
    n = scfg["gd200_break_days"]
    for tk in tick:
        c = close[tk].values
        s200 = ind["sma200"][tk].values
        s50 = ind["sma50"][tk].values
        tr = ind["trend"][tk].values
        pa = pA[tk].values
        lt, mb = mean_lt[tk].values, mean_b[tk].values
        r = pd.Series(c).pct_change().values
        dates = close.index
        i0, i1 = dates.searchsorted(start), dates.searchsorted(end, side="right")
        pos, entry, stop, lock, days_in = None, None, None, False, 0
        for i in range(max(i0, 260), i1):
            if not np.isfinite(c[i]) or not np.isfinite(s200[i]):
                continue
            above = c[i - n + 1:i + 1] > s200[i - n + 1:i + 1]
            brk = bool((~above).all() and (c[i - 10:i - n + 1] > s200[i - 10:i - n + 1]).any())
            dc = bool(np.isfinite(s50[i - 5]) and s50[i - 5] > s200[i - 5] and s50[i] < s200[i])
            f5 = np.isfinite(pa[i]) and pa[i] >= e4
            up = tr[i] == 1
            if lock and (c[i - 4:i + 1] > s200[i - 4:i + 1]).all():
                lock = False
            if pos is None:
                if f5 and up and not (rule in ("R2", "R4") and lock):
                    rr = r[max(i - 251, 1):i + 1]
                    rr = rr[np.isfinite(rr)]
                    ms = rr.std() * np.sqrt(21) if len(rr) >= 60 else np.nan
                    pos, entry, days_in = i, c[i], 0
                    stop = c[i] * (1 - scfg["stop_monthly_std"] * ms) if np.isfinite(ms) else -np.inf
                continue
            days_in += 1
            held.iat[i, tick.index(tk)] = True
            reason = None
            if rule == "R3":
                if days_in >= h:
                    reason = "Haltedauer"
            else:
                if brk or dc:
                    reason, lock = "Ausstiegssignal", True
                elif c[i] <= stop:
                    reason = "Stopp"
                elif rule in ("R0", "R4") and not (f5 and up):
                    reason = "Signal beendet"
                elif rule in ("R1", "R2"):
                    pa_entry = (lt[i] - np.log(entry)) * h / 252 + mb[i] if np.isfinite(lt[i]) else np.nan
                    if not (np.isfinite(pa_entry) and pa_entry >= e4):
                        reason = "Signal beendet"
            if reason or i == i1 - 1:
                trades.append((tk, dates[pos], dates[i], entry, c[i], reason or "offen am Ende", days_in))
                pos = None
    t = pd.DataFrame(trades, columns=["ticker", "entry", "exit", "p_in", "p_out", "reason", "days"])
    return t, held


def _stats(t, held, close, tick, start, end, bench_ret):
    if t.empty:
        return {"trades": 0}
    lr = close[tick]
    t = t.copy()
    t["ret"] = t["p_out"] / t["p_in"] - 1
    eq = (1 + bench_ret).cumprod()
    t["bench"] = eq.reindex(t["exit"]).values / eq.reindex(t["entry"]).values - 1
    t["ex"] = t["ret"] - t["bench"]
    t = t.sort_values(["ticker", "entry"])
    prev_exit = t.groupby("ticker")["exit"].shift()
    gap = [np.busday_count(a.date(), b.date()) if pd.notna(a) else 999 for a, b in zip(prev_exit, t["entry"])]
    prev_px = t.groupby("ticker")["p_out"].shift()
    t["whipsaw"] = (np.array(gap) <= 10) & (t["p_in"] > prev_px)
    t["whip_cost"] = np.where(t["whipsaw"], t["p_in"] / prev_px - 1, 0.0)
    blk = t["entry"].dt.to_period("M")
    per = t.groupby(blk)["ex"].mean()
    se = per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 2 else np.nan
    # Portfolio: gleichgewichtete Tagesrendite gehaltener Titel (Haltetag i nutzt Rendite i)
    dr = lr.pct_change()
    h_ = held.loc[start:end, tick].shift(0)
    b = bench_ret.loc[start:end]
    port = (dr.loc[start:end].where(h_)).mean(axis=1).fillna(b)   # ohne Position: Messlatte (Vorsprung 0)
    exd = port - b
    eqp = (1 + port).cumprod()
    dd = float((eqp / eqp.cummax() - 1).min())
    yrs = len(port) / 252
    mex = exd.groupby(exd.index.to_period("M")).mean()
    se_pa = float(mex.std(ddof=1) / np.sqrt(len(mex)) * 252) if len(mex) > 2 else np.nan
    return {"trades": int(len(t)), "per_year": float(len(t) / yrs) if yrs else None,
            "avg_days": float(t["days"].mean()), "median_days": float(t["days"].median()),
            "ret": float(t["ret"].mean()), "excess": float(t["ex"].mean()), "excess_se_month": float(se),
            "hit": float((t["ex"] > 0).mean()),
            "whipsaws": int(t["whipsaw"].sum()), "whip_cost_sum": float(t["whip_cost"].sum()),
            "reasons": t["reason"].value_counts().to_dict(),
            "avg_positions": float(h_.sum(axis=1).mean()),
            "port_excess_pa": float(exd.mean() * 252), "port_excess_pa_se": se_pa,
            # Kosten: Trades/Jahr × Kosten je Trade / Ø Positionen (gleichgewichtetes Depot)
            "cost_pa": float(len(t) / yrs * COST_RT / max(float(h_.sum(axis=1).mean()), 1.0)) if yrs else None,
            "whip_per_year": float(t["whipsaw"].sum() / yrs) if yrs else None, "port_te_pa": float(exd.std() * np.sqrt(252)),
            "port_ir": float(exd.mean() / exd.std() * np.sqrt(252)) if exd.std() > 0 else None,
            "port_maxdd": dd, "bench_pa": float(b.mean() * 252), "port_pa": float(port.mean() * 252)}


def run() -> dict:
    cfg = pipeline.load_cfg()
    h = cfg["horizon_days"]
    obs = pd.read_csv(os.path.join(STATE, "analyst_obs.csv.gz"))
    tickers = sorted(obs["ticker"].unique())
    prov = get_provider()
    close, idx = pipeline._prices_with_index(prov, tickers, cfg["history_years"], cfg)
    ind = technical.indicators(close, idx, cfg["technical"])
    cut = close.index[int(len(close.index) * 0.7)]
    pA, mean_lt, mean_b, tick = _panel(obs, close, cut, h, cfg)
    end = min(pd.Timestamp(obs["date"].max()), close.index[-1])
    learn_vals = pA.loc[:cut].stack().dropna()
    e4 = float(np.quantile(learn_vals, 0.8))
    univ = pA.notna()
    bench_ret = close[tick].pct_change().where(univ).mean(axis=1).fillna(0.0)
    start_all = close.index[260]
    out = {"cut": str(cut.date()), "end": str(end.date()), "start": str(start_all.date()), "titles": len(tick),
           "e4": e4, "rules": RULES, "periods": {}}
    for label, a, b_ in (("Testzeitraum", cut, end), ("Gesamt", start_all, end)):
        res = {}
        for rule in RULES:
            t, held = _sim(rule, tick, close, ind, pA, mean_lt, mean_b, e4, a, b_, h, cfg["signals"])
            res[rule] = _stats(t, held, close, tick, a, b_, bench_ret)
        out["periods"][label] = res
        log.info("Rückrechnung %s: %s", label, {k: (v.get("trades"), round(v.get("excess", 0) * 100, 2)) for k, v in res.items()})
    return out


def report(r: dict) -> str:
    p = lambda x, d=2: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f} %".replace(".", ",")
    f = lambda x, d=1: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}".replace(".", ",")
    lines = [f"# Rückrechnung Modellversion 5\n",
             f"Universum {r['titles']} Titel mit Analystendaten · Lern-/Testgrenze {r['cut']} · Ende {r['end']} · "
             f"F5-Grenze Kurspotenzial {p(r['e4'])} (8 Wochen)\n",
             "Regeln: " + " · ".join(f"{k} {v}" for k, v in r["rules"].items()) + "\n"]
    for per, res in r["periods"].items():
        lines += [f"\n## {per}\n",
                  "| Regel | Vorsprung p. a. (Depot) | ± | t | nach Kosten | Trades/Jahr | Ø Haltedauer (Tage) | Treffer je Trade | Hin-und-Her/Jahr | Kosten Hin-und-Her | Ø Positionen | Max. Rückgang Depot |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, s in res.items():
            if not s.get("trades"):
                lines.append(f"| {k} {r['rules'][k]} | 0 | | | | | | | | | |")
                continue
            se = s.get("port_excess_pa_se")
            tv = s["port_excess_pa"] / se if se and np.isfinite(se) and se > 0 else None
            lines.append(f"| {k} {r['rules'][k]} | {p(s['port_excess_pa'], 1)} | {p(se, 1)} | {f(tv)} | {p(s['port_excess_pa'] - (s.get('cost_pa') or 0), 1)} | {f(s['per_year'], 0)} | {f(s['avg_days'])} | "
                         f"{s['hit']:.0%} | {f(s['whip_per_year'])} | {p(s['whip_cost_sum'], 1)} | {f(s['avg_positions'])} | "
                         f"{p(s['port_maxdd'], 1)} |")
        lines.append("\nAusstiegsgründe: " + " · ".join(
            f"{k}: " + ", ".join(f"{a} {b}" for a, b in s.get("reasons", {}).items()) for k, s in res.items()))
    lines.append("\nHin-und-Her: Wiedereinstieg im selben Titel innerhalb von 10 Handelstagen zu höherem Kurs; "
                 "Kosten = Summe der Mehrpreise. Vorsprung p. a.: gleichgewichtetes Depot der gehaltenen Titel minus Messlatte; "
                 "Tage ohne Position zählen mit der Messlatte (Vorsprung 0). ± = Standardfehler über Monatsblöcke. "
                 "Treffer: Anteil der Trades mit Vorsprung ggü. Messlatte im selben Zeitraum. "
                 "Nach Kosten: 0,2 % je Kauf und Verkauf, umgelegt auf das gleichgewichtete Depot.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    res = run()
    with open(os.path.join(STATE, "backtest_v5.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(STATE, "backtest_v5.md"), "w", encoding="utf-8") as fh:
        fh.write(report(res))
    print(report(res))
