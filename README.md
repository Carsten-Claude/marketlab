# MarketLab

Privater Agent: prüft börsentäglich Qualitätsaktien in Europa, USA und Asien und fusioniert
Analysten- und Chartinformation nach dem Kalman-Prinzip zu einer Gesamtaussage mit minimaler Varianz.

## Ablauf

| Lauf | Wann | Was |
|---|---|---|
| Monatlich | 1. des Monats, 03:00 UTC | Universum (Indexlisten + Seed-Liste), Größenfilter > 10 Mrd. USD, Qualitätswert, Watchlist (120), Kalibrierung Technik + Analysten, Rauschmodell, Rückblick-Test |
| Täglich | Mo–Fr 05:05 UTC (≈ 7 Uhr Sommer / 6 Uhr Winter) | Kurse, Messungen r₁ (Analysten) und r₂ (Technik), Kalman-Update, Signale, Dashboard |

Der erste Tageslauf startet automatisch den Monatslauf, wenn noch keine Kalibrierung existiert.

## Methodik in Kürze

- **Horizont:** 8 Wochen (40 Handelstage).
- **Technik (r₂):** Signalzustand aus Trend (GD 50/200), RSI(14), Bollinger %B und relativem 3-Monats-Momentum.
  Jeder Zustand bekommt aus 5 Jahren Kurshistorie die mittlere 8-Wochen-Rendite und deren Varianz.
- **Analysten (r₁):** US-Titel: Top-5-Häuser mit kleinster historischer Fehlervarianz, Gewichte als
  long-only Minimum-Varianz-Portfolio über die Fehler-Kovarianzmatrix, Verzerrung je Haus korrigiert.
  Sonst: Konsens-Median mit Unsicherheitsaufschlag (im Dashboard als „Konsens“ gekennzeichnet).
- **Rauschmodell:** Realisierte Rendite = wahre erwartete Rendite + gemeinsames Kursrauschen ε.
  σε² wird aus der Kovarianz der Prognosefehler beider Quellen geschätzt; in die Fusion geht nur das Messrauschen ein.
- **Kalman:** Zustand = wahre erwartete Rendite (Random Walk). Tägliches Update mit korrelierten Messfehlern
  (ρ_v als Annahme). Rang = r̂ / σ (Schätzfehler der heutigen Fusion).
- **Signale:** Einstieg bei Chance/Risiko > 1, r̂ > 1 % und intaktem Aufwärtstrend. Ausstieg bei Chance/Risiko < −1,
  frischem Bruch des GD 200 (2 Schlusskurse) oder GD 50 kreuzt GD 200 nach unten. Stopp = Kurs − 1,1 Monats-Standardabweichungen.
- **Erfolgskontrolle:** `state/signal_log.csv` protokolliert jedes Signal; das Dashboard wertet abgeschlossene Signale aus.

Alle Schwellen stehen in `config.yaml`.

## Lokal testen (ohne Netz)

```
MARKETLAB_SYNTHETIC=1 python run.py monthly --no-web
```
