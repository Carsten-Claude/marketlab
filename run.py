"""Einstiegspunkt: python run.py monthly | daily"""
import logging
import sys

from marketlab import dashboard, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "monthly":
        pipeline.run_monthly(use_web="--no-web" not in sys.argv)
        mode = "daily"   # direkt danach Dashboard aktualisieren
    if mode == "request":   # Klick-Buchung: Titel kommt über die Umgebungsvariable DEPOT_TITLE
        import os
        from marketlab import depot
        print("Vorgemerkt:", depot.add_request(os.environ.get("DEPOT_TITLE", ""), os.environ.get("DEPOT_ISSUE")))
        mode = "daily"
    if mode == "daily":
        run = pipeline.run_daily()
        print("Dashboard:", dashboard.render(run))
