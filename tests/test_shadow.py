import numpy as np
import pandas as pd

from marketlab import depot, shadow_v5


def _setup(tmp_path, monkeypatch):
    for name in ("SIGNAL", "MANUAL", "REQUESTS"):
        monkeypatch.setattr(depot, name, str(tmp_path / f"{name}.csv"))
    monkeypatch.setattr(shadow_v5, "V5", str(tmp_path / "v5.csv"))
    monkeypatch.setattr(shadow_v5, "LOCKS", str(tmp_path / "locks.json"))
    monkeypatch.setattr(shadow_v5, "START", str(tmp_path / "start.txt"))
    monkeypatch.setattr(shadow_v5, "RULE", str(tmp_path / "rule.txt"))
    monkeypatch.setattr(shadow_v5, "V5_R2", str(tmp_path / "v5_r2.csv"))


RANK = {"edges": [0.0, 0.01, 0.02, 0.03]}


def _res(fc, trend=1, q=5, events=(), sma=90.0, price=100.0):
    return pd.DataFrame([{"ticker": "AAA", "name": "A", "price": price, "stop": 80.0, "quintile": q, "trend": trend,
                          "fc": fc, "events": list(events), "sma200": sma, "signal": "Halten", "currency": "USD"}])


def test_entry_hold_exit_lock(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    idx = pd.bdate_range("2026-01-01", periods=10)
    close = pd.DataFrame({"AAA": np.full(10, 100.0)}, index=idx)
    ind = {"sma200": pd.DataFrame({"AAA": np.full(10, 90.0)}, index=idx)}
    # Tag 1: Einstieg (F5, Aufwärtstrend)
    out = shadow_v5.update(_res(0.05), close, ind, idx[-1], RANK, 40)
    assert out["n_open"] == 1
    # R4: Fünftel fällt auf F4 -> Signal beendet, ohne Sperre (Kurs über GD 200) -> Wiedereinstieg möglich
    close2 = close.copy(); close2.iloc[-1] = 110.0
    out = shadow_v5.update(_res(0.0, q=4, price=110.0), close2, ind, idx[-1], RANK, 40)
    assert out["n_open"] == 0 and not out["locks"]
    out = shadow_v5.update(_res(0.05), close, ind, idx[-1], RANK, 40)
    assert out["n_open"] == 1
    # Ausstiegssignal -> Ausbuchung und Sperre; Kurs unter GD 200 -> kein Wiedereinstieg
    close3 = close.copy(); close3.iloc[-2:] = 85.0
    out = shadow_v5.update(_res(0.05, events=["GD-200-Bruch"], price=85.0), close3, ind, idx[-1], RANK, 40)
    assert out["n_open"] == 0 and "AAA" in out["locks"]
    out = shadow_v5.update(_res(0.05, price=85.0), close3, ind, idx[-1], RANK, 40)
    assert out["n_open"] == 0
    # 5 Schlusskurse über GD 200 -> Sperre gelöst, Wiedereinstieg
    out = shadow_v5.update(_res(0.05), close, ind, idx[-1], RANK, 40)
    assert out["n_open"] == 1 and not out["locks"]


def test_rule_switch_archives_r2(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    idx = pd.bdate_range("2026-01-01", periods=10)
    close = pd.DataFrame({"AAA": np.full(10, 100.0)}, index=idx)
    ind = {"sma200": pd.DataFrame({"AAA": np.full(10, 90.0)}, index=idx)}
    # Altbestand nach Regel R2 (ohne Regeldatei)
    book = pd.DataFrame([["AAA", "A", "2026-01-02", 95.0, 80.0, np.nan, np.nan, np.nan]], columns=depot.COLS)
    book.to_csv(shadow_v5.V5, index=False)
    out = shadow_v5.update(_res(0.05), close, ind, idx[-1], RANK, 40)
    arch = pd.read_csv(shadow_v5.V5_R2)
    assert arch["exit_reason"].iloc[0] == "Regelwechsel auf R4"
    assert out["rule"] == "R4" and out["n_open"] == 1 and out["r2_archive"]["n_closed"] == 1
    # zweiter Lauf: kein erneutes Archivieren
    out = shadow_v5.update(_res(0.05), close, ind, idx[-1], RANK, 40)
    assert len(pd.read_csv(shadow_v5.V5_R2)) == 1 and out["n_open"] == 1
