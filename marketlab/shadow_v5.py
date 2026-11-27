"""Schattenbetrieb Modellversion 5, parallel zu Version 4.

Seit 27.11.2026 Regel R4 (Entscheidung nach Rückrechnung und drei Wochen Live-Betrieb):
  Einstieg  oberstes Analysten-Fünftel (aktueller Kurs) UND Aufwärtstrend, keine Sperre aktiv
  Ende      Fünftel oder Aufwärtstrend fehlt · Ausstiegssignal · Stopp · Titel verlässt die Watchlist
  Sperre    wie unten
Die Positionen der Regel R2 (05.–26.11.2026) liegen in depot_v5_r2.csv (am Umstellungstag ausgebucht).

Bisherige Regel R2 (A + B + Sperre):

Version 4 bleibt maßgeblich; dieses Depot dient dem Live-Vergleich bis zur Erfolgskontrolle.
  Einstieg  oberstes Analysten-Fünftel (aktuelle Kursziele, aktueller Kurs) UND Aufwärtstrend,
            keine Sperre aktiv
  Ende      F5 mit Einstiegskurs verloren (B) · Ausstiegssignal (GD-200-Bruch 2 Tage, Todeskreuz) · Stopp
            · Titel verlässt die Watchlist
  Sperre    nach Ausstiegssignal oder Ausstieg unter GD 200: Wiedereinstieg erst nach 5 Schlusskursen
            über GD 200
Ein-/Ausstieg zum Schlusskurs, gleiche Beträge, ohne Gebühren; Vergleich mit der gleichgewichteten Watchlist.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import depot
from .analysts import _bucket

V5 = os.path.join(depot.STATE, "depot_v5.csv")
V5_R2 = os.path.join(depot.STATE, "depot_v5_r2.csv")
RULE = os.path.join(depot.STATE, "v5_rule.txt")
CURRENT_RULE = "R4"
LOCKS = os.path.join(depot.STATE, "v5_locks.json")
START = os.path.join(depot.STATE, "v5_start.txt")


def _f5_entry(rank: dict, fc, price: float, entry: float, h: int) -> bool:
    """Fünftel mit Einstiegskurs: fc = log(Ziel/Kurs)·h/252 → log(Ziel/Einstand)·h/252."""
    if fc is None or not np.isfinite(fc) or not rank.get("edges"):
        return False
    return _bucket(rank, float(fc) + np.log(price / entry) * h / 252) == len(rank["edges"])


def update(res: pd.DataFrame, close: pd.DataFrame, ind: dict, asof: pd.Timestamp, rank: dict, h: int) -> dict:
    today = str(asof.date())
    rule_now = open(RULE).read().strip() if os.path.exists(RULE) else "R2"
    if rule_now != CURRENT_RULE and os.path.exists(V5):
        # Regelwechsel: bisheriges Depot archivieren, offene Positionen zum Tageskurs ausbuchen, neu beginnen
        old_book = depot._load(V5, depot.COLS)
        for i, p in old_book[old_book["exit_date"].isna()].iterrows():
            price, _ = depot._last(close, p["ticker"])
            if price is not None:
                old_book.loc[i, ["exit_date", "exit_price", "exit_reason"]] = [today, price, f"Regelwechsel auf {CURRENT_RULE}"]
        depot._save(old_book, V5_R2)
        os.remove(V5)
        if os.path.exists(START):
            os.remove(START)
    with open(RULE, "w") as f:
        f.write(CURRENT_RULE)
    if not os.path.exists(START):
        with open(START, "w") as f:
            f.write(today)
    start = open(START).read().strip()
    info = res.set_index("ticker")
    book = depot._load(V5, depot.COLS)
    try:
        locks = json.load(open(LOCKS))
    except (OSError, ValueError):
        locks = {}

    def above5(tk) -> bool:
        if tk not in close.columns:
            return False
        c, s = close[tk].iloc[-5:], ind["sma200"][tk].iloc[-5:]
        return bool(s.notna().all() and (c > s).all())

    # 1) Sperren lösen
    for tk in list(locks):
        if above5(tk):
            locks.pop(tk)

    # 2) Ausbuchungen
    for i, p in book[book["exit_date"].isna()].iterrows():
        tk = p["ticker"]
        price, _ = depot._last(close, tk)
        if price is None:
            continue
        reason, lock = None, False
        if tk not in info.index:
            reason = "nicht mehr in der Watchlist"
        else:
            r = info.loc[tk]
            ev = r["events"] if isinstance(r["events"], (list, tuple)) else []
            if ev:
                reason, lock = "Ausstiegssignal: " + "; ".join(ev), True
            elif pd.notna(p["stop"]) and price <= float(p["stop"]):
                reason = "Stopp erreicht"
            elif not (r.get("quintile") == 5 and r.get("trend") == 1):
                reason = "Einstiegssignal beendet (Fünftel oder Aufwärtstrend fehlt)"
            sma = r.get("sma200")
            if reason and sma is not None and pd.notna(sma) and price < float(sma):
                lock = True
        if reason:
            book.loc[i, ["exit_date", "exit_price", "exit_reason"]] = [today, price, reason]
            if lock:
                locks[tk] = today

    # 3) Neue Einstiege
    held = set(book.loc[book["exit_date"].isna(), "ticker"])
    cand = info[(info["quintile"] == 5) & (info["trend"] == 1)]
    for tk, r in cand.iterrows():
        if tk in held or tk in locks:
            continue
        book.loc[len(book)] = [tk, r["name"], today, float(r["price"]), r.get("stop"), np.nan, np.nan, np.nan]

    depot._save(book, V5)
    with open(LOCKS, "w") as f:
        json.dump(locks, f, ensure_ascii=False, indent=1)

    out = depot.evaluate(book, close, res)
    # Vergleich auf gleicher Zeitbasis: V4-Signal-Depot nur mit Einstiegen seit Start des Schattenbetriebs
    sig = depot._load(depot.SIGNAL, depot.COLS)
    sig = sig[pd.to_datetime(sig["entry_date"]) >= pd.Timestamp(start)]
    out["v4_same_period"] = depot.evaluate(sig, close, res) if len(sig) else {"positions": [], "n_open": 0, "n_closed": 0}
    out["locks"] = locks
    out["start"] = start
    out["rule"] = CURRENT_RULE
    if os.path.exists(V5_R2):
        out["r2_archive"] = depot.evaluate(depot._load(V5_R2, depot.COLS), close, res)
    return out
