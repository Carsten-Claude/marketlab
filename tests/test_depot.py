"""Musterdepot: Buchungen, Ausbuchung, Vorsprung ggü. Watchlist."""
import numpy as np
import pandas as pd
import pytest

from marketlab import depot


@pytest.fixture()
def tmpstate(tmp_path, monkeypatch):
    monkeypatch.setattr(depot, "SIGNAL", str(tmp_path / "s.csv"))
    monkeypatch.setattr(depot, "MANUAL", str(tmp_path / "m.csv"))
    monkeypatch.setattr(depot, "REQUESTS", str(tmp_path / "r.csv"))
    return tmp_path


def _close(prices):
    idx = pd.bdate_range("2026-10-01", periods=len(next(iter(prices.values()))))
    return pd.DataFrame(prices, index=idx)


def _res(sig):
    return pd.DataFrame([{"ticker": t, "name": t, "signal": s, "reasons": ["x"], "price": 0.0, "stop": 50.0,
                          "currency": "USD"} for t, s in sig.items()])


def test_signal_depot_open_and_close(tmpstate):
    close = _close({"A": [100, 110], "B": [100, 100]})
    d = depot.update(_res({"A": "Einstieg", "B": "Halten"}).assign(price=[110, 100]), close, close.index[-1])
    assert d["signal"]["n_open"] == 1
    close2 = _close({"A": [100, 110, 121], "B": [100, 100, 100]})
    d = depot.update(_res({"A": "Halten", "B": "Halten"}).assign(price=[121, 100]), close2, close2.index[-1])
    p = d["signal"]["positions"][0]
    assert not p["open"] and p["exit_reason"] == "Einstiegssignal beendet"
    assert abs(p["ret"] - 0.10) < 1e-9          # 110 -> 121


def test_manual_buy_sell_and_rejections(tmpstate):
    close = _close({"A": [100, 110], "B": [100, 120]})
    res = _res({"A": "Einstieg", "B": "Halten"}).assign(price=[110, 120])
    depot.add_request("Depot: Kauf A", 1)
    depot.add_request("Depot: Verkauf B", 2)
    depot.add_request("Depot: Kauf ZZZ", 3)
    d = depot.update(res, close, close.index[-1])
    st = [r["status"] for r in d["requests"]]
    assert st[0].startswith("gebucht: Kauf") and st[1].startswith("abgelehnt") and st[2].startswith("abgelehnt")
    depot.add_request("Depot: Verkauf A", 4)
    d = depot.update(res, close, close.index[-1])
    assert d["manual"]["n_closed"] == 1


def test_rejects_bad_titles(tmpstate):
    for t in ["Kauf A", "Depot: Kaufen A", "Depot: Kauf A; rm -rf /", "Depot: Kauf $(x)"]:
        with pytest.raises(ValueError):
            depot.add_request(t, 9)


def test_excess_vs_watchlist(tmpstate):
    close = _close({"A": [100, 120], "B": [100, 100], "C": [100, 110]})
    res = _res({"A": "Einstieg", "B": "Halten", "C": "Halten"}).assign(price=[120, 100, 110])
    sig = pd.DataFrame([["A", "A", str(close.index[0].date()), 100.0, 50.0, np.nan, np.nan, np.nan]], columns=depot.COLS)
    e = depot.evaluate(sig, close, res)["positions"][0]
    assert abs(e["ret"] - 0.20) < 1e-9 and abs(e["bench"] - 0.10) < 1e-9 and abs(e["excess"] - 0.10) < 1e-9
