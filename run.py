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
    if mode == "close-issues":   # nach erfolgreichem Speichern: bearbeitete Klick-Buchungen schließen
        from marketlab import gh_issues
        gh_issues.close_processed()
        sys.exit(0)
    if mode in ("daily", "sweep") and "GITHUB_TOKEN" in __import__("os").environ:
        from marketlab import gh_issues
        print("Offene Depot-Anfragen:", gh_issues.sweep())
        mode = "daily"
    if mode == "daily":
        run = pipeline.run_daily()
        print("Dashboard:", dashboard.render(run))
