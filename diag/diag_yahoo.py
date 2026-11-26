"""Diagnose Tageslauf (läuft auf GitHub, ohne Speichern)."""
import logging, sys, traceback
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("yfinance").setLevel(logging.WARNING)
import pandas as pd
from marketlab import pipeline
orig = pipeline._prices_with_index
def wrapped(*a, **k):
    close, idx = orig(*a, **k)
    print("CLOSE", close.shape, list(close.index[-3:]))
    print("leere Spalten", int(close.isna().all().sum()))
    print("letzte Zeile NaN-Anteil", float(close.iloc[-1].isna().mean()))
    return close, idx
pipeline._prices_with_index = wrapped
try:
    run = pipeline.run_daily()
    res = run["results"]
    print("RES", res.shape, list(res.columns)[:6])
except Exception:
    traceback.print_exc()
