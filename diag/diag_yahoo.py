"""Diagnose Kursabruf (läuft auf GitHub)."""
import logging, sys
import pandas as pd
import yfinance as yf
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
print("yfinance", yf.__version__, "pandas", pd.__version__)
df = yf.download(["AAPL", "MSFT", "ANET"], period="1mo", interval="1d", auto_adjust=True, progress=False, threads=True, group_by="column")
print("download shape", None if df is None else df.shape)
if df is not None and not df.empty:
    print("columns", list(df.columns)[:8]); print("index", df.index[-3:], df.index.dtype)
    print(df.tail(3))
t = yf.Ticker("AAPL")
try:
    print("history", t.history(period="5d").tail(2))
except Exception as e:
    print("history error", repr(e))
try:
    i = t.info; print("info keys", len(i or {}), (i or {}).get("currentPrice"))
except Exception as e:
    print("info error", repr(e))
sys.path.insert(0, ".")
from marketlab import pipeline
from marketlab.data import get_provider
cfg = pipeline.load_cfg()
wl = pd.read_csv("state/watchlist.csv")
close, idx = pipeline._prices_with_index(get_provider(), wl["ticker"].tolist()[:20], 2, cfg)
print("pipeline close", close.shape, close.index[-3:] if len(close) else None)
