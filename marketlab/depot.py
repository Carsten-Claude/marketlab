"""Musterdepots zur Erfolgskontrolle.

• Signal-Depot (automatisch): jedes neue Einstiegssignal wird zum Schlusskurs des Signaltags
  aufgenommen; Ausbuchung bei Ausstiegssignal, Stopp, Ende des Einstiegssignals oder wenn der
  Titel die Watchlist verlässt. Neutrale Messlatte für den Algorithmus.
• Mein Depot (manuell): Buchungen per Klick (GitHub-Issue „Depot: Kauf TICKER“ / „Depot: Verkauf
  TICKER“), Ausführung zum letzten verfügbaren Schlusskurs.

Gleiche Beträge je Position, ohne Gebühren. Vergleichsgröße: gleichgewichtete Watchlist im
selben Zeitraum (Rendite in lokaler Währung).
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
STATE = os.path.join(ROOT, "state")
SIGNAL = os.path.join(STATE, "depot_signal.csv")
MANUAL = os.path.join(STATE, "depot_manual.csv")
COLS = ["ticker", "name", "entry_date", "entry_price", "stop", "exit_date", "exit_price", "exit_reason"]
REQ_COLS = ["requested", "action", "ticker", "issue", "status"]
REQUESTS = os.path.join(STATE, "depot_requests.csv")
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-&=^]{0,19}$")


def _load(path: str, cols: list[str]) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"ticker": str})
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols].astype(object)
    return pd.DataFrame(columns=cols).astype(object)


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def open_tickers() -> list[str]:
    out = set()
    for p in (SIGNAL, MANUAL):
        d = _load(p, COLS)
        out |= set(d.loc[d["exit_date"].isna(), "ticker"])
    req = _load(REQUESTS, REQ_COLS)
    out |= set(req.loc[req["status"] == "offen", "ticker"])
    return sorted(out)


# ---------------------------------------------------------------- Klick-Anfragen
def add_request(title: str, issue: str | int | None) -> str:
    """Titel „Depot: Kauf TICKER“ oder „Depot: Verkauf TICKER“ als offene Anfrage vormerken."""
    m = re.match(r"^\s*Depot:\s*(Kauf|Verkauf)\s+(\S+)\s*$", title or "")
    if not m:
        raise ValueError(f"Titel nicht erkannt: {title!r}")
    action, tk = m.group(1), m.group(2).upper()
    if not TICKER_RE.match(tk):
        raise ValueError(f"Ungültiges Kürzel: {tk!r}")
    req = _load(REQUESTS, REQ_COLS)
    req.loc[len(req)] = [pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"), action, tk, issue, "offen"]
    _save(req, REQUESTS)
    return f"{action} {tk}"


# ---------------------------------------------------------------- Tagesverarbeitung
def _last(close: pd.DataFrame, tk: str):
    if tk not in close.columns:
        return None, None
    s = close[tk].dropna()
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (None, None)


def update(res: pd.DataFrame, close: pd.DataFrame, asof: pd.Timestamp) -> dict:
    today = str(asof.date())
    info = res.set_index("ticker")
    sig = _load(SIGNAL, COLS)
    man = _load(MANUAL, COLS)

    # 1) Signal-Depot: Ausbuchungen
    for i, p in sig[sig["exit_date"].isna()].iterrows():
        tk = p["ticker"]
        price, _ = _last(close, tk)
        reason = None
        if tk not in info.index:
            reason = "nicht mehr in der Watchlist"
        else:
            s = info.loc[tk, "signal"]
            if s == "Ausstieg":
                reason = "Ausstiegssignal: " + "; ".join(info.loc[tk, "reasons"])
            elif price is not None and pd.notna(p["stop"]) and price <= float(p["stop"]):
                reason = "Stopp erreicht"
            elif s != "Einstieg":
                reason = "Einstiegssignal beendet"
        if reason and price is not None:
            sig.loc[i, ["exit_date", "exit_price", "exit_reason"]] = [today, price, reason]

    # 2) Signal-Depot: neue Einstiegssignale
    held = set(sig.loc[sig["exit_date"].isna(), "ticker"])
    for tk, r in info[info["signal"] == "Einstieg"].iterrows():
        if tk not in held:
            sig.loc[len(sig)] = [tk, r["name"], today, float(r["price"]), r.get("stop"), np.nan, np.nan, np.nan]

    # 3) Mein Depot: offene Klick-Anfragen ausführen
    req = _load(REQUESTS, REQ_COLS)
    for i, q in req[req["status"] == "offen"].iterrows():
        tk = q["ticker"]
        price, _ = _last(close, tk)
        open_rows = man[(man["ticker"] == tk) & man["exit_date"].isna()]
        if price is None:
            req.loc[i, "status"] = "abgelehnt: kein Kurs verfügbar"
        elif q["action"] == "Kauf":
            if len(open_rows):
                req.loc[i, "status"] = "abgelehnt: bereits im Depot"
            else:
                name = info.loc[tk, "name"] if tk in info.index else tk
                stop = info.loc[tk, "stop"] if tk in info.index else np.nan
                man.loc[len(man)] = [tk, name, today, price, stop, np.nan, np.nan, np.nan]
                req.loc[i, "status"] = f"gebucht: Kauf zu {price:.2f}"
        else:
            if not len(open_rows):
                req.loc[i, "status"] = "abgelehnt: nicht im Depot"
            else:
                man.loc[open_rows.index[0], ["exit_date", "exit_price", "exit_reason"]] = [today, price, "manuell entfernt"]
                req.loc[i, "status"] = f"gebucht: Verkauf zu {price:.2f}"

    _save(sig, SIGNAL)
    _save(man, MANUAL)
    _save(req, REQUESTS)
    return {"signal": evaluate(sig, close, res), "manual": evaluate(man, close, res),
            "requests": req.tail(20).to_dict("records")}


def evaluate(book: pd.DataFrame, close: pd.DataFrame, res: pd.DataFrame) -> dict:
    """Rendite je Position und Vorsprung ggü. gleichgewichteter Watchlist im selben Zeitraum."""
    wl = [t for t in res["ticker"] if t in close.columns]
    sigs = res.set_index("ticker")["signal"].to_dict()
    cur = res.set_index("ticker")["currency"].to_dict() if "currency" in res.columns else {}
    rows = []
    for _, p in book.iterrows():
        tk = p["ticker"]
        start = pd.Timestamp(p["entry_date"])
        is_open = pd.isna(p["exit_date"])
        if is_open:
            price, _ = _last(close, tk)
            end = close.index[-1]
        else:
            price, end = float(p["exit_price"]), pd.Timestamp(p["exit_date"])
        ret = (price / float(p["entry_price"]) - 1) if price else np.nan
        win = close.loc[(close.index >= start) & (close.index <= end), wl]
        bench = np.nan
        if len(win) >= 2:
            b = (win.ffill().iloc[-1] / win.bfill().iloc[0] - 1).dropna()
            bench = float(b.mean()) if len(b) else np.nan
        days = int(np.busday_count(start.date(), pd.Timestamp(end).date())) if pd.notna(end) else 0
        rows.append({"ticker": tk, "name": p["name"], "entry_date": p["entry_date"], "entry_price": float(p["entry_price"]),
                     "price": price, "ret": ret, "bench": bench, "excess": ret - bench if pd.notna(bench) else np.nan,
                     "days": days, "open": is_open, "exit_date": p["exit_date"], "exit_reason": p["exit_reason"],
                     "stop": p["stop"], "signal_now": sigs.get(tk, "–"), "currency": cur.get(tk, "")})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"positions": [], "n_open": 0, "n_closed": 0}

    def stats(d):
        d = d.dropna(subset=["excess"])
        if d.empty:
            return None
        return {"n": int(len(d)), "ret": float(d["ret"].mean()), "excess": float(d["excess"].mean()),
                "hit": float((d["excess"] > 0).mean())}
    return {"positions": df.to_dict("records"), "n_open": int(df["open"].sum()), "n_closed": int((~df["open"]).sum()),
            "all": stats(df), "open_stats": stats(df[df["open"]]), "closed_stats": stats(df[~df["open"]])}
