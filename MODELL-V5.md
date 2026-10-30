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
