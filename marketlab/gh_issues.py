"""Klick-Buchungen über GitHub-Issues: offene Anfragen einsammeln und nach dem Speichern schließen.

Jeder Lauf (täglich und bei jedem Klick) sammelt ALLE offenen Issues „Depot: …“ des Repository-
Inhabers ein. Geht ein Lauf verloren (von GitHub verworfen, Speichern fehlgeschlagen), bleibt das
Issue offen und wird beim nächsten Lauf verarbeitet. Geschlossen wird erst nach erfolgreichem Speichern.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import requests

from . import depot

API = "https://api.github.com"
PENDING = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "depot_pending_issues.json")


def _hdr():
    return {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}


def sweep() -> list[int]:
    repo, owner = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_REPOSITORY_OWNER"]
    r = requests.get(f"{API}/repos/{repo}/issues", headers=_hdr(), params={"state": "open", "per_page": 100}, timeout=30)
    r.raise_for_status()
    nums = []
    for it in sorted(r.json(), key=lambda x: x["number"]):
        if "pull_request" in it or it["user"]["login"] != owner or not it["title"].startswith("Depot:"):
            continue
        try:
            depot.add_request(it["title"], it["number"])
        except ValueError:
            depot.reject_request(it["title"], it["number"], "abgelehnt: Titel nicht erkannt")
        nums.append(it["number"])
    with open(PENDING, "w") as f:
        json.dump(nums, f)
    return nums


def close_processed() -> None:
    if not os.path.exists(PENDING):
        return
    repo = os.environ["GITHUB_REPOSITORY"]
    nums = json.load(open(PENDING))
    req = pd.read_csv(depot.REQUESTS, dtype={"issue": str})
    req["issue"] = req["issue"].astype(str).str.replace(r"\.0$", "", regex=True)
    for n in nums:
        rows = req[req["issue"] == str(n)]
        status = rows["status"].iloc[-1] if len(rows) else "nicht verarbeitet"
        if status == "offen":
            continue   # noch nicht ausgeführt – bleibt offen für den nächsten Lauf
        requests.post(f"{API}/repos/{repo}/issues/{n}/comments", headers=_hdr(), timeout=30,
                      json={"body": f"Musterdepot: {status}. Das Dashboard ist in wenigen Minuten aktualisiert."})
        requests.patch(f"{API}/repos/{repo}/issues/{n}", headers=_hdr(), json={"state": "closed"}, timeout=30)
    os.remove(PENDING)
