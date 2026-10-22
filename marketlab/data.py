"""Datenschicht: Yahoo Finance (yfinance) oder synthetische Testdaten.

Alle Zugriffe laufen über einen Provider mit identischer Schnittstelle, damit der
Gesamtablauf ohne Netz getestet werden kann (MARKETLAB_SYNTHETIC=1).
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("marketlab.data")

INFO_KEYS = [
    "shortName", "longName", "currency", "financialCurrency", "marketCap",
    "returnOnEquity", "operatingMargins", "freeCashflow", "totalRevenue",
    "totalDebt", "totalCash", "ebitda", "forwardEps", "trailingEps",
    "targetMeanPrice", "targetMedianPrice", "targetHighPrice", "targetLowPrice",
    "numberOfAnalystOpinions", "recommendationMean", "sector",
]


def _retry(fn, *args, tries: int = 3, pause: float = 2.0, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 – Netzwerk/Parsing, wir loggen und versuchen erneut
            last = exc
            time.sleep(pause * (i + 1))
    log.warning("Abruf fehlgeschlagen nach %s Versuchen: %s", tries, last)
    return None


@dataclass
class Estimates:
    growth_0y: float | None = None        # erwartetes EPS-Wachstum laufendes GJ
    growth_1y: float | None = None        # erwartetes EPS-Wachstum nächstes GJ
    eps_now_1y: float | None = None       # Konsens nächstes GJ heute
    eps_30d_1y: float | None = None       # ... vor 30 Tagen
    eps_90d_1y: float | None = None       # ... vor 90 Tagen


@dataclass
class Financials:
    revenue: list[float] = field(default_factory=list)       # alt -> neu
    eps: list[float] = field(default_factory=list)           # Jahres-EPS, alt -> neu
    eps_dates: list[pd.Timestamp] = field(default_factory=list)
    quarterly_eps: list[float] = field(default_factory=list)  # alt -> neu


class YahooProvider:
    name = "yahoo"

    def __init__(self):
        import yfinance as yf  # erst hier, damit der Synthetik-Modus ohne yfinance läuft
        self.yf = yf
        self._fx: dict[str, float] = {}

    def prices(self, tickers: list[str], years: int) -> pd.DataFrame:
        frames = []
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i + 100]
            df = _retry(self.yf.download, chunk, period=f"{years}y", interval="1d",
                        auto_adjust=True, progress=False, threads=True, group_by="column")
            if df is None or df.empty:
                continue
            close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            frames.append(close)
            time.sleep(1.0)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out.sort_index()

    def info(self, ticker: str) -> dict:
        t = self.yf.Ticker(ticker)
        raw = _retry(lambda: t.info) or {}
        return {k: raw.get(k) for k in INFO_KEYS}

    def financials(self, ticker: str) -> Financials:
        t = self.yf.Ticker(ticker)
        fin = Financials()
        inc = _retry(lambda: t.income_stmt)
        if inc is not None and not inc.empty:
            inc = inc[sorted(inc.columns)]
            if "Total Revenue" in inc.index:
                fin.revenue = [float(x) for x in inc.loc["Total Revenue"].dropna()]
            for row in ("Diluted EPS", "Basic EPS"):
                if row in inc.index:
                    s = inc.loc[row].dropna()
                    fin.eps = [float(x) for x in s]
                    fin.eps_dates = [pd.Timestamp(d) for d in s.index]
                    break
        q = _retry(lambda: t.quarterly_income_stmt)
        if q is not None and not q.empty:
            q = q[sorted(q.columns)]
            for row in ("Diluted EPS", "Basic EPS"):
                if row in q.index:
                    fin.quarterly_eps = [float(x) for x in q.loc[row].dropna()]
                    break
        return fin

    def estimates(self, ticker: str) -> Estimates:
        t = self.yf.Ticker(ticker)
        est = Estimates()
        ee = _retry(lambda: t.earnings_estimate)
        if ee is not None and not ee.empty and "growth" in ee.columns:
            est.growth_0y = _num(ee["growth"].get("0y"))
            est.growth_1y = _num(ee["growth"].get("+1y"))
        tr = _retry(lambda: t.eps_trend)
        if tr is not None and not tr.empty and "+1y" in tr.index:
            row = tr.loc["+1y"]
            est.eps_now_1y = _num(row.get("current"))
            est.eps_30d_1y = _num(row.get("30daysAgo"))
            est.eps_90d_1y = _num(row.get("90daysAgo"))
        return est

    def analyst_targets(self, ticker: str) -> pd.DataFrame:
        """Kursziel-Historie einzelner Häuser: Spalten date, firm, target."""
        time.sleep(0.4)   # Abrufe gemächlich takten (Yahoo drosselt bei zu vielen Anfragen)
        t = self.yf.Ticker(ticker)
        ud = _retry(lambda: t.upgrades_downgrades)
        if ud is None or ud.empty:
            return pd.DataFrame(columns=["date", "firm", "target"])
        ud = ud.reset_index()
        date_col = "GradeDate" if "GradeDate" in ud.columns else ud.columns[0]
        tgt_col = next((c for c in ("currentPriceTarget", "priceTarget") if c in ud.columns), None)
        if tgt_col is None or "Firm" not in ud.columns:
            return pd.DataFrame(columns=["date", "firm", "target"])
        out = pd.DataFrame({
            "date": pd.to_datetime(ud[date_col]).dt.tz_localize(None),
            "firm": ud["Firm"].astype(str),
            "target": pd.to_numeric(ud[tgt_col], errors="coerce"),
        })
        return out[out["target"] > 0].sort_values("date").reset_index(drop=True)

    def fx_to_usd(self, currency: str | None) -> float:
        if not currency or currency == "USD":
            return 1.0
        cur = currency
        factor = 1.0
        if cur in ("GBp", "GBX"):
            cur, factor = "GBP", 0.01
        if cur in ("ZAc", "ILA"):
            cur, factor = {"ZAc": "ZAR", "ILA": "ILS"}[cur], 0.01
        if cur not in self._fx:
            df = _retry(self.yf.download, f"{cur}USD=X", period="5d", progress=False, auto_adjust=True)
            rate = float(df["Close"].dropna().iloc[-1].squeeze()) if df is not None and not df.empty else np.nan
            self._fx[cur] = rate
        return self._fx[cur] * factor


def _num(x) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Synthetische Daten: reproduzierbar je Ticker, mit eingebauter Struktur
# (Analystenziele und Trends tragen echte, aber verrauschte Information),
# damit Kalibrierung und Kalman-Fusion sinnvoll getestet werden können.
# ---------------------------------------------------------------------------
FIRMS = ["Morgan Stanley", "Goldman Sachs", "JP Morgan", "UBS", "Barclays", "Citigroup",
         "Deutsche Bank", "BofA Securities", "Jefferies", "Bernstein", "Evercore ISI", "Wells Fargo"]


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, end: str | None = None):
        self.end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        self._cache: dict[str, pd.Series] = {}
        # Haus-spezifische Prognosegüte (fest, reproduzierbar)
        rng = np.random.default_rng(7)
        self.firm_noise = {f: float(s) for f, s in zip(FIRMS, rng.uniform(0.08, 0.35, len(FIRMS)))}

    @staticmethod
    def _rng(ticker: str, salt: str = "") -> np.random.Generator:
        h = int(hashlib.sha256((ticker + salt).encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h)

    def _series(self, ticker: str, years: int = 6) -> pd.Series:
        if ticker in self._cache:
            return self._cache[ticker]
        rng = self._rng(ticker)
        idx = pd.bdate_range(end=self.end, periods=252 * years)
        n = len(idx)
        # langsam wandernde Drift (Regimes) -> Trends sind teilweise prognostisch
        drift = np.cumsum(rng.normal(0, 0.00004, n)) + rng.normal(0.0003, 0.0003)
        drift = np.clip(drift, -0.0015, 0.002)
        vol = rng.uniform(0.012, 0.025)
        rets = drift + rng.normal(0, vol, n)
        price = rng.uniform(20, 400) * np.exp(np.cumsum(rets))
        s = pd.Series(price, index=idx, name=ticker)
        self._cache[ticker] = s
        return s

    def prices(self, tickers: list[str], years: int) -> pd.DataFrame:
        frames = {t: self._series(t, max(years, 6)).iloc[-252 * years:] for t in tickers}
        return pd.DataFrame(frames)

    def info(self, ticker: str) -> dict:
        rng = self._rng(ticker, "info")
        s = self._series(ticker)
        price = float(s.iloc[-1])
        future_hint = float(s.iloc[-1] / s.iloc[-60] - 1)  # Analysten folgen teils dem Momentum
        mean_t = price * (1 + rng.normal(0.08, 0.10) + 0.5 * future_hint)
        rev = rng.uniform(5e9, 200e9)
        ebitda = rev * rng.uniform(0.08, 0.40)
        return {
            "shortName": f"Synth {ticker}", "longName": f"Synthetic {ticker}",
            "currency": "USD", "financialCurrency": "USD",
            "marketCap": rng.uniform(5e9, 500e9),
            "returnOnEquity": rng.normal(0.17, 0.08), "operatingMargins": rng.normal(0.18, 0.08),
            "freeCashflow": rev * rng.normal(0.12, 0.06), "totalRevenue": rev,
            "totalDebt": ebitda * rng.uniform(0, 4), "totalCash": ebitda * rng.uniform(0, 1.5),
            "ebitda": ebitda, "forwardEps": price / rng.uniform(12, 35), "trailingEps": price / rng.uniform(14, 40),
            "targetMeanPrice": mean_t, "targetMedianPrice": mean_t * rng.uniform(0.97, 1.03),
            "targetHighPrice": mean_t * rng.uniform(1.10, 1.45), "targetLowPrice": mean_t * rng.uniform(0.60, 0.92),
            "numberOfAnalystOpinions": int(rng.integers(3, 40)), "recommendationMean": rng.uniform(1.5, 3.2),
            "sector": "Synthetic",
        }

    def financials(self, ticker: str) -> Financials:
        rng = self._rng(ticker, "fin")
        g = rng.normal(0.07, 0.06)
        rev0 = rng.uniform(5e9, 100e9)
        eps0 = rng.uniform(1, 15)
        s = self._series(ticker)
        years = [self.end - pd.DateOffset(years=k) for k in (4, 3, 2, 1)]
        return Financials(
            revenue=[rev0 * (1 + g) ** k for k in range(4)],
            eps=[eps0 * (1 + g) ** k * (1 + rng.normal(0, 0.12)) for k in range(4)],
            eps_dates=years,
            quarterly_eps=[eps0 / 4 * (1 + rng.normal(0, 0.15)) for _ in range(8)],
        ) if len(s) else Financials()

    def estimates(self, ticker: str) -> Estimates:
        rng = self._rng(ticker, "est")
        now = rng.uniform(2, 20)
        return Estimates(growth_0y=rng.normal(0.08, 0.10), growth_1y=rng.normal(0.10, 0.08),
                         eps_now_1y=now, eps_30d_1y=now * (1 - rng.normal(0.005, 0.02)),
                         eps_90d_1y=now * (1 - rng.normal(0.01, 0.04)))

    def analyst_targets(self, ticker: str) -> pd.DataFrame:
        if "." in ticker:  # wie in der Realität: Häuser-Historie nur für US-Titel
            return pd.DataFrame(columns=["date", "firm", "target"])
        rng = self._rng(ticker, "ud")
        s = self._series(ticker)
        rows = []
        for _ in range(int(rng.integers(25, 60))):
            i = int(rng.integers(0, len(s) - 1))
            firm = FIRMS[int(rng.integers(0, len(FIRMS)))]
            j = min(i + 252, len(s) - 1)
            realized = float(np.log(s.iloc[j] / s.iloc[i]))
            fc = 0.5 * realized + 0.04 + rng.normal(0, self.firm_noise[firm])
            rows.append((s.index[i], firm, float(s.iloc[i] * np.exp(fc))))
        return pd.DataFrame(rows, columns=["date", "firm", "target"]).sort_values("date").reset_index(drop=True)

    def fx_to_usd(self, currency: str | None) -> float:
        return 1.0


def get_provider():
    if os.environ.get("MARKETLAB_SYNTHETIC") == "1":
        return SyntheticProvider(os.environ.get("MARKETLAB_SYNTHETIC_END"))
    return YahooProvider()
