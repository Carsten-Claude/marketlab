"""Hysterese der Einstiegssignale."""
import pandas as pd
import yaml

from marketlab import signals

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))["signals"]


def _r(ex, cr, trend=1, events=()):
    return pd.Series({"excess": ex, "cr": cr, "trend": trend, "events": list(events)})


def test_entry_needs_full_threshold():
    assert signals.decide(_r(0.0071, 1.36), CFG)["signal"] == "Einstieg"
    assert signals.decide(_r(0.0069, 1.36), CFG)["signal"] == "Halten"


def test_active_signal_holds_above_hold_threshold():
    assert signals.decide(_r(0.0069, 1.36), CFG, "Einstieg")["signal"] == "Einstieg"
    assert signals.decide(_r(0.0040, 1.10), CFG, "Einstieg")["signal"] == "Halten"
    assert signals.decide(_r(0.0060, 0.70), CFG, "Einstieg")["signal"] == "Halten"


def test_exit_rules_override_hysteresis():
    r = _r(0.0069, 1.36, events=["Schlusskurs 2 Tage in Folge unter GD 200"])
    assert signals.decide(r, CFG, "Einstieg")["signal"] == "Ausstieg"
