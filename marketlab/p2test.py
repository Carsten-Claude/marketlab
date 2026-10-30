"""Prüfung P2: Ist die Chartanalyse eine eigenständige Informationsquelle oder nur ein Filter?

Läuft im Monatslauf mit (abgesichert, ohne Einfluss auf die Kalibrierung). Datenbasis: historische
Kursziele der Häuser (Analysten) mit Chart-Zustand am selben Tag und realisierter Rendite über den
Horizont h. Alle Prognosen des Testzeitraums verwenden nur Parameter aus dem Lernzeitraum.

Vorab festgelegte Entscheidungsregel
  Test 1 (Zusatzinformation): Koeffizient der Technik > 0 mit t ≥ 2 – im Lern- UND im Testzeitraum.
  Test 2 (Praxis): bestes Fünftel der korrigierten Fusion schlägt bestes Fünftel „nur Analysten“
          im Testzeitraum (Differenz > 0).
  -> beide bestanden: Prognosequelle; sonst nur Filter, sofern Test 3 (Trendfilter senkt Risiko
     innerhalb des obersten Analysten-Fünftels, ohne Rendite zu kosten) dafür spricht.

Standardfehler: Cluster nach Zeitblöcken der Länge h (überlappende Fenster sind nicht unabhängig).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import technical

log = logging.getLogger("marketlab.p2test")


def _ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS mit clusterrobusten Standardfehlern (Sandwich, Cluster = Zeitblöcke)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for g in np.unique(groups):
        m = groups == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
        G += 1
    n, k = X.shape
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def _block_diff(df: pd.DataFrame, a: str, b: str) -> dict:
    """Mittelwert und Standardfehler einer Differenz über Zeitblöcke."""
    per = df.groupby("block").apply(lambda g: g[a].mean() - g[b].mean(), include_groups=False).dropna()
    if len(per) < 3:
        return {"diff": None, "se": None, "t": None, "blocks": int(len(per))}
    m, se = float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per)))
    return {"diff": m, "se": se, "t": m / se if se > 0 else None, "blocks": int(len(per))}


def _top_vs_rest(df: pd.DataFrame, score: str, share: float = 0.2) -> pd.DataFrame:
    """Je Tag die besten `share` nach Score markieren (Querschnitt, ohne Blick in die Zukunft)."""
    r = df.groupby("date")[score].rank(pct=True, method="first")
    return df.assign(top=r > 1 - share)


def run(obs: pd.DataFrame, close: pd.DataFrame, states: pd.DataFrame, cut: pd.Timestamp, h: int,
        cfg: dict, noise: dict) -> dict:
    if obs is None or len(obs) < cfg["analysts"].get("min_total_obs", 2000):
        return {"ok": False, "reason": f"zu wenige Analystenprognosen ({0 if obs is None else len(obs)})"}
    tcfg, acfg = cfg["technical"], cfg["analysts"]

    # --- Realisierte Rendite und Vorsprung ggü. Ø aller Titel (gemeinsame Marktbewegung entfernt)
    lr = np.log(close).diff(h).shift(-h)
    bench = lr.mean(axis=1, skipna=True)
    o = obs.copy()
    o["date"] = pd.to_datetime(o["date"])
    o["y"] = o["err"] + o["fc"]
    learn_o = o[o["date"] < cut]
    g_bias = learn_o["err"].mean()
    k = acfg["min_firm_obs"]
    bias = {f: (len(g) * g["err"].mean() + k * g_bias) / (len(g) + k) for f, g in learn_o.groupby("firm")}
    o["pA"] = o["fc"] + o["firm"].map(bias).fillna(g_bias)
    # eine Zeile je Titel und Tag (mehrere Häuser -> Mittel der verzerrungskorrigierten Prognosen)
    d = o.groupby(["ticker", "date"]).agg(pA=("pA", "mean"), y=("y", "first")).reset_index()
    d["bench"] = bench.reindex(d["date"].values).values
    d = d.dropna(subset=["bench"])
    d["ex"] = d["y"] - d["bench"]

    # --- Chart-Zustand am selben Tag, Zustandsmittel nur aus dem Lernzeitraum
    fwd = technical.forward_returns(close, h)
    cal_l = technical.calibrate(states[states.index < cut], fwd[fwd.index < cut], tcfg)
    mu_t = cal_l["global"]["mean"]
    keys, pT = [], []
    for tk, dt in zip(d["ticker"], d["date"]):
        key = None
        if tk in states.columns:
            s_ = states[tk]
            pos = s_.index.searchsorted(dt, side="right") - 1
            if pos >= 0 and isinstance(s_.iloc[pos], str):
                key = s_.iloc[pos]
        keys.append(key)
        pT.append(technical.raw_measurement(key, cal_l)[0] - mu_t if key else np.nan)
    d["key"], d["pT"] = keys, pT
    d = d.dropna(subset=["pT"])
    d["trend_up"] = d["key"].str.split("|").str[0] == "1"

    # Kursverlauf innerhalb des Horizonts: größter Rückschlag (Risiko-Maß für Test 3)
    dd = []
    for tk, dt in zip(d["ticker"], d["date"]):
        p = close[tk].dropna()
        i = p.index.searchsorted(dt, side="left")
        w = p.iloc[i:i + h + 1]
        dd.append(float(w.min() / w.iloc[0] - 1) if len(w) > 1 else np.nan)
    d["max_dd"] = dd

    d["block"] = ((d["date"] - d["date"].min()).dt.days // max(int(h * 7 / 5), 1)).astype(int)
    d["test"] = d["date"] >= cut
    L, T = d[~d["test"]], d[d["test"]]
    if len(L) < cfg.get("p2_min_cases", 300) or len(T) < cfg.get("p2_min_cases", 300) // 2:
        return {"ok": False, "reason": f"zu wenige gemeinsame Fälle (Lernen {len(L)}, Test {len(T)})"}

    # --- Analysten-Fünftel (Grenzen aus dem Lernzeitraum)
    edges = list(np.quantile(L["pA"], [0.2, 0.4, 0.6, 0.8]))
    d["q"] = np.digitize(d["pA"], edges)
    d["F5"] = (d["q"] == 4).astype(float)
    L, T = d[~d["test"]], d[d["test"]]
    m_learn = L.groupby("q")["ex"].mean().reindex(range(5)).fillna(0.0)

    # --- Test 1: Zusatzinformation der Technik bei bekanntem Analysten-Fünftel
    sdT = float(L["pT"].std())
    def reg(sub):
        X = np.column_stack([np.ones(len(sub)), sub["F5"].values, sub["pT"].values / sdT])
        b, se = _ols_cluster(sub["ex"].values, X, sub["block"].values)
        return {"n": int(len(sub)), "blocks": int(sub["block"].nunique()),
                "b_F5": float(b[1]), "t_F5": float(b[1] / se[1]),
                "b_tech_per_sd": float(b[2]), "t_tech": float(b[2] / se[2])}
    t1 = {"learn": reg(L), "test": reg(T)}
    t1["passed"] = bool(t1["learn"]["b_tech_per_sd"] > 0 and t1["learn"]["t_tech"] >= 2
                        and t1["test"]["b_tech_per_sd"] > 0 and t1["test"]["t_tech"] >= 2)

    # --- Test 2: korrigierte Fusion (gleich definierte Messrauschen beider Quellen) vs. nur Analysten
    eA = d["q"].map(m_learn)
    XL = np.column_stack([np.ones(len(L)), L["pT"].values])
    bT = float(np.linalg.lstsq(XL, L["ex"].values, rcond=None)[0][1])
    eT = bT * d["pT"]
    sA, sT, rho = noise.get("sd_vA"), noise.get("sd_vT"), noise.get("rho_v", 0.2)
    if not sA or not sT:
        return {"ok": False, "reason": "Rauschmodell nicht identifiziert"}
    c = rho * sA * sT
    wA = float(np.clip((sT ** 2 - c) / (sA ** 2 + sT ** 2 - 2 * c), 0, 1))
    d["sA"] = eA
    d["sF"] = wA * eA + (1 - wA) * eT
    d["sT"] = eT
    res2 = {}
    T = d[d["test"]]
    for name, col in (("analysten", "sA"), ("fusion", "sF"), ("technik", "sT")):
        x = _top_vs_rest(T, col)
        res2[name] = {"top_ex": float(x.loc[x["top"], "ex"].mean()), "rest_ex": float(x.loc[~x["top"], "ex"].mean()),
                      "spread": float(x.loc[x["top"], "ex"].mean() - x.loc[~x["top"], "ex"].mean())}
        T = T.assign(**{f"top_{name}": x["top"]})
    # Differenz der Spreizungen (Fusion − Analysten) je Zeitblock, dann Mittel und Standardfehler
    def _spread(g, col):
        return g.loc[g[col], "ex"].mean() - g.loc[~g[col], "ex"].mean()
    per = T.groupby("block").apply(lambda g: _spread(g, "top_fusion") - _spread(g, "top_analysten"),
                                   include_groups=False).dropna()
    res2["fusion_minus_analysten"] = ({"diff": float(per.mean()), "se": float(per.std(ddof=1) / np.sqrt(len(per))),
                                       "t": float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))) if per.std(ddof=1) > 0 else None,
                                       "blocks": int(len(per))} if len(per) >= 3 else
                                      {"diff": None, "se": None, "t": None, "blocks": int(len(per))})
    res2["w_analysten"] = wA
    res2["sd_vA"], res2["sd_vT"] = float(sA), float(sT)
    diff = res2["fusion"]["spread"] - res2["analysten"]["spread"]
    res2["passed"] = bool(diff > 0)

    # --- Test 3: Trendfilter innerhalb des obersten Analysten-Fünftels (Gesamtzeitraum)
    f5 = d[d["q"] == 4]
    up, dn = f5[f5["trend_up"]], f5[~f5["trend_up"]]
    tmp = f5.assign(u=np.where(f5["trend_up"], f5["ex"], np.nan), v=np.where(~f5["trend_up"], f5["ex"], np.nan),
                    du=np.where(f5["trend_up"], f5["max_dd"], np.nan), dv=np.where(~f5["trend_up"], f5["max_dd"], np.nan))
    t3 = {"n_up": int(len(up)), "n_other": int(len(dn)),
          "ex_up": float(up["ex"].mean()), "ex_other": float(dn["ex"].mean()),
          "sd_up": float(up["y"].std()), "sd_other": float(dn["y"].std()),
          "dd_up": float(up["max_dd"].mean()), "dd_other": float(dn["max_dd"].mean()),
          "share_dd10_up": float((up["max_dd"] < -0.10).mean()), "share_dd10_other": float((dn["max_dd"] < -0.10).mean()),
          "ex_diff": _block_diff(tmp, "u", "v"), "dd_diff": _block_diff(tmp, "du", "dv")}
    ed, ddd = t3["ex_diff"], t3["dd_diff"]
    t3["passed"] = bool(ddd["diff"] is not None and ddd["diff"] > 0 and (ed["t"] is None or ed["t"] > -2))

    verdict = ("Prognosequelle" if t1["passed"] and res2["passed"]
               else "nur Filter" if t3["passed"] else "weder Prognosequelle noch belegter Filter")
    log.info("P2-Test: Technik t = %.2f (Lernen) / %.2f (Test) · Fusion − Analysten %s · Urteil: %s",
             t1["learn"]["t_tech"], t1["test"]["t_tech"], diff, verdict)
    return {"ok": True, "h": h, "cut": str(cut.date()), "n": int(len(d)),
            "period": [str(d["date"].min().date()), str(d["date"].max().date())],
            "test1": t1, "test2": res2, "test3": t3, "verdict": verdict}


def report(r: dict) -> str:
    """Kurzbericht (Markdown, deutsch)."""
    if not r.get("ok"):
        return f"# P2-Test\n\nNicht durchgeführt: {r.get('reason')}\n"
    p = lambda x: "–" if x is None else f"{x * 100:+.2f} %".replace(".", ",")
    n = lambda x: "–" if x is None else f"{x:+.2f}".replace(".", ",")
    t1, t2, t3 = r["test1"], r["test2"], r["test3"]
    ok = lambda b: "bestanden" if b else "nicht bestanden"
    return f"""# P2-Test: Chartanalyse als Informationsquelle

Zeitraum {r['period'][0]} bis {r['period'][1]} · Lern-/Testgrenze {r['cut']} · {r['n']} Fälle (Titel × Tag) · Horizont {r['h']} Handelstage

**Urteil: {r['verdict']}**

## Test 1 – Zusatzinformation ({ok(t1['passed'])})
Vorsprung je Fall erklärt durch oberstes Analysten-Fünftel (F5) und Technik-Prognose (je 1 Standardabweichung).

| | Fälle | Blöcke | F5 | t | Technik je σ | t |
|---|---|---|---|---|---|---|
| Lernzeitraum | {t1['learn']['n']} | {t1['learn']['blocks']} | {p(t1['learn']['b_F5'])} | {n(t1['learn']['t_F5'])} | {p(t1['learn']['b_tech_per_sd'])} | {n(t1['learn']['t_tech'])} |
| Testzeitraum | {t1['test']['n']} | {t1['test']['blocks']} | {p(t1['test']['b_F5'])} | {n(t1['test']['t_F5'])} | {p(t1['test']['b_tech_per_sd'])} | {n(t1['test']['t_tech'])} |

## Test 2 – Praxis, Testzeitraum ({ok(t2['passed'])})
Bestes Fünftel je Tag gegenüber dem Rest. Fusion mit gleich definiertem Messrauschen (σ_vA {p(t2['sd_vA'])}, σ_vT {p(t2['sd_vT'])}, Analystengewicht {t2['w_analysten']:.0%}).

| Auswahl nach | Vorsprung bestes Fünftel | Rest | Spreizung |
|---|---|---|---|
| nur Analysten | {p(t2['analysten']['top_ex'])} | {p(t2['analysten']['rest_ex'])} | {p(t2['analysten']['spread'])} |
| korrigierte Fusion | {p(t2['fusion']['top_ex'])} | {p(t2['fusion']['rest_ex'])} | {p(t2['fusion']['spread'])} |
| nur Technik | {p(t2['technik']['top_ex'])} | {p(t2['technik']['rest_ex'])} | {p(t2['technik']['spread'])} |

Spreizung Fusion minus Analysten, Mittel über Zeitblöcke: {p(t2['fusion_minus_analysten']['diff'])}, t = {n(t2['fusion_minus_analysten']['t'])}

## Test 3 – Trendfilter innerhalb F5 ({ok(t3['passed'])})

| | Fälle | Vorsprung | Schwankung | Ø größter Rückschlag | Anteil Rückschlag > 10 % |
|---|---|---|---|---|---|
| Aufwärtstrend | {t3['n_up']} | {p(t3['ex_up'])} | {p(t3['sd_up'])} | {p(t3['dd_up'])} | {t3['share_dd10_up']:.0%} |
| übrige | {t3['n_other']} | {p(t3['ex_other'])} | {p(t3['sd_other'])} | {p(t3['dd_other'])} | {t3['share_dd10_other']:.0%} |

Differenz Vorsprung: {p(t3['ex_diff']['diff'])} (t = {n(t3['ex_diff']['t'])}) · Differenz Rückschlag: {p(t3['dd_diff']['diff'])} (t = {n(t3['dd_diff']['t'])})
"""
