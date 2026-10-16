"""Ausgangsuniversum: Indexmitglieder von Wikipedia plus Seed-Liste als Rückfall/Ergänzung."""
from __future__ import annotations

import io
import logging
import os
import re

import pandas as pd

log = logging.getLogger("marketlab.universe")

SEED = os.path.join(os.path.dirname(__file__), "..", "data", "universe_seed.csv")


def _clean(sym: str, suffix: str, dot_to_dash: bool) -> str | None:
    s = str(sym).strip().upper()
    s = re.sub(r"\[.*?\]", "", s).strip()
    if not s or s == "NAN" or " " in s:
        return None
    if dot_to_dash:
        s = s.replace(".", "-")
    if suffix and not s.endswith(suffix.upper()):
        # schon ein anderes Suffix vorhanden (z. B. Wikipedia listet "ADS.DE") -> übernehmen
        if re.search(r"\.[A-Z]{1,3}$", s):
            return s
        s = s + suffix
    return s


def from_wikipedia(sources: list[dict]) -> pd.DataFrame:
    import requests

    rows = []
    headers = {"User-Agent": "Mozilla/5.0 (MarketLab private research)"}
    for src in sources:
        try:
            html = requests.get(src["url"], headers=headers, timeout=30).text
            tables = pd.read_html(io.StringIO(html))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: Indexliste nicht lesbar (%s)", src["name"], exc)
            continue
        found = 0
        for tbl in tables:
            cols = [str(c).strip() for c in tbl.columns]
            tbl.columns = cols
            col = next((c for c in src["columns"] if c in cols), None)
            if col is None or len(tbl) < 15:
                continue
            for sym in tbl[col]:
                t = _clean(sym, src.get("suffix", ""), src.get("dot_to_dash", False))
                if t:
                    rows.append((t, src["name"]))
                    found += 1
            break
        log.info("%s: %s Titel", src["name"], found)
    return pd.DataFrame(rows, columns=["ticker", "index"])


def build_universe(cfg: dict, use_web: bool = True) -> pd.DataFrame:
    seed = pd.read_csv(SEED)
    web = from_wikipedia(cfg["index_sources"]) if use_web else pd.DataFrame(columns=["ticker", "index"])
    uni = pd.concat([web, seed], ignore_index=True).drop_duplicates("ticker", keep="first")
    log.info("Universum: %s Titel (%s aus Wikipedia, Rest Seed-Liste)", len(uni), len(web))
    return uni.reset_index(drop=True)


def region_suffix(ticker: str) -> str:
    m = re.search(r"(\.[A-Z]{1,3})$", ticker)
    return m.group(1) if m else ""


REGION_NAME = {
    "": "USA", ".DE": "Deutschland", ".PA": "Frankreich", ".L": "Großbritannien", ".SW": "Schweiz",
    ".AS": "Niederlande", ".MC": "Spanien", ".MI": "Italien", ".ST": "Schweden", ".CO": "Dänemark",
    ".T": "Japan", ".HK": "Hongkong", ".TW": "Taiwan", ".NS": "Indien",
}


def region_name(ticker: str) -> str:
    return REGION_NAME.get(region_suffix(ticker), region_suffix(ticker).lstrip("."))
