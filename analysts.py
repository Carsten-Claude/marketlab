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


def rank_calibrate(obs: pd.DataFrame, close: pd.DataFrame, cut: pd.Timestamp, h: int, acfg: dict) -> dict:
    """Rang-Kalibrierung der Analystenquelle (Fünftel statt Gerade).

    1. Vorsprung jeder historischen Prognose: realisierte Log-Rendite minus Ø Log-Rendite aller
       Titel des Universums im selben Zeitfenster (gemeinsame Marktbewegung entfernt).
    2. Fünftel-Grenzen aus den verzerrungskorrigierten Prognosen des Lernzeitraums.
    3. Validierung: Fünftel-Mittel des Lernzeitraums sagen den Vorsprung im Testzeitraum voraus?
       Steigung b (0 = keine Information, 1 = Stufung hält voll).
    4. Endgültige Stufen: Fünftel-Mittel über den Gesamtzeitraum × min(max(b, 0), 1);
       Unsicherheit je Fünftel über nicht überlappende Blöcke der Länge h (Blockmittel-Streuung).
    """
    from . import skill
    nq = acfg.get("rank_buckets", 5)
    empty = {"ok": False, "edges": [], "m": [0.0] * nq, "se": [0.05] * nq, "b": 0.0}
    if obs is None or obs.empty:
        return empty
    lr = np.log(close).diff(h).shift(-h)              # ln(P_{d+h}/P_d)
    bench = lr.mean(axis=1, skipna=True).reindex(pd.to_datetime(obs["date"]), method="ffill").values
    df = obs.assign(y=obs["err"] + obs["fc"], bench=bench).dropna(subset=["bench"])
    df["ex"] = df["y"] - df["bench"]
    learn, test = df[df["date"] < cut], df[df["date"] >= cut]
    if len(learn) < 200 or len(test) < 100:
        return empty
    g_bias = learn["err"].mean()
    k = acfg["min_firm_obs"]
    bias = {f: (len(g) * g["err"].mean() + k * g_bias) / (len(g) + k) for f, g in learn.groupby("firm")}
    df["pred"] = df["fc"] + df["firm"].map(bias).fillna(g_bias)
    learn, test = df[df["date"] < cut], df[df["date"] >= cut]
    edges = list(np.quantile(learn["pred"], np.linspace(0, 1, nq + 1)[1:-1]))
    df["q"] = np.digitize(df["pred"], edges)
    learn, test = df[df["date"] < cut], df[df["date"] >= cut]
    m_learn = learn.groupby("q")["ex"].mean().reindex(range(nq)).fillna(0.0)
    m_test = test.groupby("q")["ex"].mean().reindex(range(nq))
    val = skill.fit(test["q"].map(m_learn).values, test["ex"].values)
    shrink = float(np.clip(val["b_raw"] if val["b_raw"] is not None else 0.0, 0.0, 1.0))
    # Blockmittel über nicht überlappende Fenster der Länge h (Kalendertage ≈ h·7/5)
    block = ((df["date"] - df["date"].min()).dt.days // max(int(h * 7 / 5), 1))
    df["block"] = block
    m, se, n = [], [], []
    for q in range(nq):
        g = df[df["q"] == q]
        bm = g.groupby("block")["ex"].mean()
        mean = float(g["ex"].mean()) if len(g) else 0.0
        s_ = float(bm.std(ddof=1) / np.sqrt(len(bm))) if len(bm) > 2 else 0.05
        m.append(mean * shrink)
        se.append(max(s_, 1e-3))
        n.append(int(len(g)))
    log.info("Rang-Kalibrierung Analysten: b = %.2f · Fünftel %s", val["b_raw"] or 0,
             " / ".join(f"{x * 100:+.1f}%" for x in m))
    return {"ok": True, "edges": [float(e) for e in edges], "m": m, "se": se, "n": n, "b": shrink,
            "b_raw": val["b_raw"], "corr": val["corr"],
            "m_learn": [float(x) for x in m_learn], "m_test": [None if pd.isna(x) else float(x) for x in m_test],
            "n_blocks": int(df["block"].nunique())}


def _bucket(rank: dict, fc: float) -> int:
    return int(np.digitize(fc, rank["edges"])) if rank.get("edges") else 0


def measurement(ticker: str, price: float, info: dict, hist: pd.DataFrame | None,
                calib: dict, h: int, acfg: dict, asof: pd.Timestamp) -> dict:
    """Analystenmessung r₁, σ₁² mit Methodenkennzeichnung.

    Die (Minimum-Varianz- bzw. Konsens-)Prognose bestimmt nur noch das Fünftel; r₁ ist die im
    Backtest gemessene Erwartung dieses Fünftels (Marktmittel + Vorsprung), σ₁ deren Unsicherheit.
    """
    firms = calib["firms"]
    rank = calib.get("rank") or {}
    mu = rank.get("mu", 0.0)
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
            fc_adj = float(w @ (fc + np.array([firms[f]["bias"] for f in top["firm"]])))
            q = _bucket(rank, fc_adj)
            return {"r1": float(np.exp(mu + rank["m"][q]) - 1) if rank.get("ok") else None,
                    "var1": float(rank["se"][q] ** 2) if rank.get("ok") else None,
                    "method": "Minimum-Varianz", "quintile": q + 1, "fc": fc_adj,
                    "detail": [{"firm": f, "target": float(t), "weight": float(x)}
                               for f, t, x in zip(top["firm"], top["target"], w)]}
    # --- Rückfall: Konsens-Median; Fünftel-Stufen aus US-Häusern übertragen, Unsicherheit erhöht
    med = info.get("targetMedianPrice") or info.get("targetMeanPrice")
    if not med or not price or not rank.get("ok"):
        return {"r1": None, "var1": None, "method": "keine Analystendaten", "quintile": None, "detail": []}
    fc = float(np.log(med / price) * h / 252 + calib["global_bias"])
    q = _bucket(rank, fc)
    n = info.get("numberOfAnalystOpinions")
    return {"r1": float(np.exp(mu + rank["m"][q]) - 1),
            "var1": float(rank["se"][q] ** 2 * acfg.get("transfer_penalty", 2.0)),
            "method": "Konsens-Rückfall", "quintile": q + 1, "fc": fc,
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


def skill_test(obs: pd.DataFrame, cut: pd.Timestamp, acfg: dict) -> dict:
    """Treffsicherheit im Testzeitraum: Häuser-Verzerrung nur aus dem Lernzeitraum."""
    from . import skill
    if obs is None or obs.empty:
        return skill.fit([], [])
    learn, test = obs[obs["date"] < cut], obs[obs["date"] >= cut]
    if learn.empty or test.empty:
        return skill.fit([], [])
    g_bias = learn["err"].mean()
    k = acfg["min_firm_obs"]
    bias = {}
    for f, g in learn.groupby("firm"):
        n = len(g)
        bias[f] = (n * g["err"].mean() + k * g_bias) / (n + k)
    t = test[test["firm"].isin(bias)]
    pred = t["fc"] + t["firm"].map(bias)
    real = t["err"] + t["fc"]
    res = skill.fit(pred.values, real.values)
    if len(pred) >= 100:
        q = np.quantile(pred.values, 0.8)
        res["spread"] = float(real.values[pred.values >= q].mean() - real.values.mean())
    res["period"] = [str(t["date"].min().date()), str(t["date"].max().date())] if len(t) else None
    return res


def apply_skill(calib_a: dict, sk: dict, sigma_eps2: float, floor_share: float) -> None:
    """Häuser-Fehlervarianzen in kalibriertes Messrauschen umrechnen (in place).

    Relative Güte der Häuser (für die Minimum-Varianz-Gewichte) bleibt erhalten, das Niveau
    wird so skaliert, dass ein einzelnes Haus das im Testzeitraum gemessene Messrauschen hat.
    """
    from . import skill
    var_v_single = skill.measurement_var(sk["var_e"], sigma_eps2, floor_share)
    firms = calib_a["firms"]
    for f in firms.values():
        f["var_raw"] = f["var"]
        f["var_v"] = max(f["var"] - sigma_eps2, floor_share * f["var"])
    if firms:
        n = np.array([f["n"] for f in firms.values()], dtype=float)
        v = np.array([f["var_v"] for f in firms.values()])
        mean_v = float((n * v).sum() / n.sum())
        pc = calib_a.get("pair_cov")
        if pc is not None:
            calib_a["rho_ff"] = float(np.clip((pc - sigma_eps2) / mean_v, 0.0, 0.95))
        scale = var_v_single / mean_v
        for f in firms.values():
            f["var"] = f["var_v"] * scale
    else:
        scale = 1.0
    calib_a["global_var_raw"] = calib_a["global_var"]
    calib_a["global_var"] = var_v_single
    calib_a["skill"] = {**{k: v for k, v in sk.items()}, "var_v": var_v_single, "scale": scale}
