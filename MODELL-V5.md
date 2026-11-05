# Vorgemerkt für Modellversion 5 (Entscheidung 30.10.2026, Umsetzung frühestens nach Erfolgskontrolle Mitte Dezember)

## B – Einstieg über Analysten-Stufe statt Vorsprungsschwelle
Befund (Daten 22.–30.10.): Die Rang-Kalibrierung der Analysten liefert nur fünf Stufen (r1). Rund 30 Titel teilen
die oberste Stufe; ihr Vorsprung liegt einheitlich bei ≈ 0,74 %, die Einstiegsschwelle bei 0,70 %. Die Reihenfolge
innerhalb der Stufe entsteht allein aus kleinen Unterschieden der Technik-Komponente (ohne nachgewiesene Treffsicherheit).
Folge: Ein Stufenwechsel (z. B. DXCM 30.10.: Vorsprung 0,74 % → 0,32 %) beendet ein Signal nach wenigen Tagen.

Vorschlag: Einstieg = oberstes Analysten-Fünftel UND Aufwärtstrend; Ende nur bei Abstieg um ≥ 2 Stufen,
Ausstiegssignal oder Stopp. Vorher prüfen: Nachlauf „seit Ausstieg“ im Signal-Depot (seit 30.10. gemessen).

## Bis dahin (Variante D)
Signalregeln unverändert (Version 4). Zusätzlich gemessen: Sektoranteil der aktiven Signale, Nachlauf geschlossener
Positionen, im Signal-Protokoll Trend, Analysten-Fünftel und Sektor je Titel.

## Entscheidung 30.10.2026: A + B werden umgesetzt
A – Binäre Einstiegsregel: oberstes Analysten-Fünftel UND Aufwärtstrend (keine Vorsprungs-/C/R-Schwelle).
B – Für gehaltene Titel wird das Fünftel mit aktuellen Kurszielen, aber dem Einstiegskurs bestimmt
    („Kursziel-Falle“: steigender Kurs verringert das Kurspotenzial und warf CSCO/DXCM aus F5).
    Ausstiegssignal und Stopp bleiben unverändert.

## Befund P2: Varianzen der beiden Quellen nicht vergleichbar (Fehler, mit V5 beheben)
Analysten: var1 = Standardfehler² des Fünftel-Mittels (σ ≈ 0,2–0,5 %, schrumpft mit der Stichprobe).
Technik:   var2 = Messrauschen einer Einzelprognose (σ_vT ≈ 6,5 %).
Gleich definiert wären beide ähnlich groß (Analysten linear σ_vA ≈ 6,6 %). Folge heute: Analystengewicht ≈ 99 %,
Chance/Risiko (Vorsprung/σ) systematisch überhöht. Mit einheitlicher Definition sinkt C/R auf ≈ 0,1–0,2;
die Schwelle C/R > 1 wäre nie erfüllt – deshalb gemeinsam mit A umstellen.

## P2-Test (Variante V2, Entscheidung 30.10.2026)
Läuft ab dem Monatslauf 01.11.2026 mit (marketlab/p2test.py), Ergebnis in state/p2_test.md.
Regel vorab: Prognosequelle nur, wenn Test 1 (Technik-Beitrag t ≥ 2 in Lern- und Testzeitraum) und
Test 2 (Fusion trennt bestes Fünftel besser als Analysten allein) bestanden; sonst Filter, sofern Test 3 dafür spricht.
Rohdaten der Analystenprognosen werden in state/analyst_obs.csv.gz gespeichert.

## Ergebnis P2-Test (01.11.2026): Chartanalyse = nur Filter
Technik ohne Zusatzinformation im Testzeitraum (t = 0,02); korrigierte Fusion verschlechtert die Auswahl
(Spreizung −0,53 % statt +0,46 %); Trendfilter in F5 +1,96 % Vorsprung (t = 2,32). Folge: Technik als
Prognosequelle raus, Trend bleibt Filter; Varianzkorrektur der Fusion entfällt.

## Ergänzung 05.11.2026: Sperre für Wiedereinstieg
Arista 03.11. Ausstieg (GD-200-Bruch) zu 164,14, 04.11. Wiedereinstieg zu 169,91. Vorschlag: nach Ausstiegssignal
Wiedereinstieg erst nach 5 Schlusskursen über GD 200. Rückrechnung: marketlab/backtest_v5.py → state/backtest_v5.md.

## Entscheidung 05.11.2026: Version 5 = R2 (A + B + Sperre)
Rückrechnung (state/backtest_v5.md), Testzeitraum nach Kosten: R2 +32,4 % p. a., R4 +32,6 %, R0 (V4-nah) +31,1 %;
R2 mit halb so vielen Trades (64/Jahr), Haltedauer 54 Tage, Hin-und-Her −75 %. Kursziel-Falle in der Historie
nicht renditewirksam; B wegen Robustheit und Horizont-Passung gewählt. Absolute Vorsprünge durch
Überlebenden-Verzerrung überhöht, nur der Regelvergleich ist belastbar.
Regeln V5: Einstieg = oberstes Analysten-Fünftel (aktuelle Kursziele, aktueller Kurs) + Aufwärtstrend;
Ende = F5 mit Einstiegskurs verloren, Ausstiegssignal (GD-200-Bruch 2 Tage, Todeskreuz) oder Stopp;
nach Ausstiegssignal oder Ausstieg unter GD 200 Wiedereinstieg erst nach 5 Schlusskursen über GD 200.
Technik nur als Filter, keine Kalman-Gewichtung nach Messrauschen.
