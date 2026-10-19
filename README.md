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
- **Technik (r₂):** Signalzustand aus Trend (GD 50/200), RSI(14), Bollinger %B und relativem Momentum 12-1 (12 Monate ohne den jüngsten Monat).
  Jeder Zustand bekommt aus 5 Jahren Kurshistorie die mittlere 8-Wochen-Rendite und deren Varianz.
- **Analysten (r₁):** US-Titel: Top-5-Häuser mit kleinster historischer Fehlervarianz, Gewichte als
  long-only Minimum-Varianz-Portfolio über die Fehler-Kovarianzmatrix, Verzerrung je Haus korrigiert.
  Sonst: Konsens-Median mit Unsicherheitsaufschlag (im Dashboard als „Konsens“ gekennzeichnet).
- **Treffsicherheit:** Beide Quellen werden im Testzeitraum (jüngste 30 % der Historie, nicht zur Kalibrierung verwendet)
  geprüft: y = a + b·r. Verwendet wird r_kal = a + b·r; das Messrauschen folgt aus dem Testfehler. Eine Quelle ohne
  Treffsicherheit (b ≈ 0) liefert nur noch den Mittelwert und wirkt als Anker.
- **Qualitätstor:** Watchlist nur mit belegten (≥ 3 Jahre) und stabilen Gewinnen; Fair Value nur bei stabilen Gewinnen.
- **Rauschmodell:** Realisierte Rendite = wahre erwartete Rendite + gemeinsames Kursrauschen ε.
  σε² wird aus der Kovarianz der Prognosefehler beider Quellen geschätzt; in die Fusion geht nur das Messrauschen ein.
- **Kalman:** Zustand = wahre erwartete Rendite (Random Walk). Tägliches Update mit korrelierten Messfehlern
  (ρ_v als Annahme). Rang = Vorsprung / σ, mit Vorsprung = r̂ − Watchlist-Median (die gemeinsame Marktdrift zählt nicht als Signal); Top 10 mit höchstens 5 US-Titeln.
- **Signale:** Einstieg bei Chance/Risiko > 1, Vorsprung > 1 % und intaktem Aufwärtstrend. Ausstieg bei Chance/Risiko < −1,
  frischem Bruch des GD 200 (2 Schlusskurse) oder GD 50 kreuzt GD 200 nach unten. Stopp = Kurs − 1,1 Monats-Standardabweichungen.
- **Erfolgskontrolle:** `state/signal_log.csv` protokolliert jedes Signal; das Dashboard vergleicht abgeschlossene Einstiegssignale mit der Ø Watchlist im selben Zeitraum.
- **Horizonttest:** Der Monatslauf misst die Treffsicherheit beider Quellen zusätzlich für 8, 13 und 26 Wochen.

Alle Schwellen stehen in `config.yaml`.

## Lokal testen (ohne Netz)

```
MARKETLAB_SYNTHETIC=1 python run.py monthly --no-web
```
