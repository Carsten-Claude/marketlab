"""Statisches HTML-Dashboard (GitHub Pages)."""
from __future__ import annotations

import html
import math
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")

WEEKDAY = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def _pct(x, digits=1, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x * 100:+.{digits}f}" if sign else f"{x * 100:.{digits}f}"
    return s.replace(".", ",") + " %"


def _num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x:,.{digits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _sg(x, digits=2):
    return f"{x:+.{digits}f}".replace(".", ",")


def _e(s) -> str:
    return html.escape(str(s))


BADGE = {
    "Einstieg": ('<span class="badge good"><svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1 9 8H1z"/></svg>'
                 'Einstieg</span>'),
    "Halten": ('<span class="badge hold"><svg viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="3.2"/>'
               '</svg>Halten</span>'),
    "Ausstieg": ('<span class="badge crit"><svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 9 1 2h8z"/></svg>'
                 'Ausstieg</span>'),
}


def _cr_bar(cr: float, lim: float = 1.0) -> str:
    v = max(-lim, min(lim, cr))
    w = abs(v) / lim * 50
    left = 50 if v >= 0 else 50 - w
    return (f'<span class="crbar" title="Chance/Risiko {_sg(cr)}"><span class="mid"></span>'
            f'<span class="fill" style="left:{left:.1f}%;width:{w:.1f}%"></span></span>')


def _row(r: pd.Series) -> str:
    detail = "".join(
        f"<tr><td>{_e(d['firm'])}</td><td class='num'>{_num(d['target'])}</td>"
        f"<td class='num'>{_pct(d['weight'], 0, False)}</td></tr>" for d in r["analyst_detail"])
    method_cls = "mv" if r["analyst_method"] == "Minimum-Varianz" else "fb"
    method_lbl = ("MV" if method_cls == "mv" else "Konsens") + (f" · F{int(r['quintile'])}" if r.get("quintile") == r.get("quintile") and r.get("quintile") else "")
    reasons = "".join(f"<li>{_e(x)}</li>" for x in r["reasons"])
    trend = {1: "auf", 0: "gemischt", -1: "ab"}.get(r["trend"], "–")
    return f"""
<tr class="main">
  <td class="num rank opt">{int(r['rank'])}</td>
  <td class="title"><span class="nm">{_e(r['name'])}</span><span class="tk">{_e(r['ticker'])} · {_e(r['region'])}</span></td>
  <td>{BADGE[r['signal']]}</td>
  <td class="num"><span class="crval">{_sg(r['cr'])}</span>{_cr_bar(r['cr'])}</td>
  <td class="num strong">{_pct(r['excess'])}</td>
  <td class="num opt">{_pct(r['x'])}</td>
  <td class="num opt">± {_pct(r['sd'], 1, False)}</td>
  <td class="num opt">{_pct(r['r1'])} <span class="tag {method_cls}">{method_lbl}</span></td>
  <td class="num opt">{_pct(r['r2'])}</td>
  <td class="num opt">{_pct(r['w_analyst'], 0, False)} / {_pct(r['w_tech'], 0, False)}</td>
  <td class="num opt">{_num(r['price'])}</td>
  <td class="num opt">{_num(r['stop'])}</td>
  <td class="num opt">{_pct(r['fv_gap'], 0) if r['fv_ok'] else '<span class="muted" title="Gewinne zu instabil für eine KGV-basierte Bewertung">n. b.</span>'}</td>
  <td class="num opt">{_num(r['quality'], 0)}</td>
</tr>
<tr class="det"><td colspan="14">
  <details><summary>Begründung und Details</summary>
  <div class="detgrid">
    <div><h4>Signal</h4><ul>{reasons}</ul>
      <h4>Technik</h4><p>Kurs {_num(r['price'])} {_e(r['currency'] or '')} · Stopp {_num(r['stop'])} · Qualität {_num(r['quality'], 0)}<br>{_e(r['tech_state'])} · Stichprobe {int(r['tech_n']):,} Fälle<br>
      RSI {_num(r['rsi'], 0)} · %B {_num(r['pct_b'], 2)} · GD 50 {_num(r['sma50'])} · GD 200 {_num(r['sma200'])} ·
      Trend {trend} · rel. Momentum 12-1 {_pct(r['rel_mom'])}</p></div>
    <div><h4>Analysten ({_e(r['analyst_method'])})</h4>
      <table class="mini"><thead><tr><th>Haus</th><th class="num">Kursziel</th><th class="num">Gewicht</th></tr></thead>
      <tbody>{detail}</tbody></table>
      <p>Messfehler: r₁ {_pct(r['r1'])} ± {_pct(r['sd1'], 1, False)} · r₂ {_pct(r['r2'])} ± {_pct(r['sd2'], 1, False)}<br>
      Fusion: r̂ {_pct(r['x'])} ± {_pct(r['sd'], 1, False)} · inkl. Kursrauschen ± {_pct(r['sd_total'], 1, False)}</p></div>
    <div><h4>Bewertung und Erwartung</h4>
      <p>{f"Fair Value {_num(r['fair_value'])} {_e(r['currency'] or '')} ({_pct(r['fv_gap'], 0)} zum Kurs)" if r['fv_ok'] else "Fair Value nicht belastbar (Gewinne zu instabil)"}<br>
      Gewinnwachstum 12M {_pct(r['growth_12m'])} · Revision 30T {_pct(r['revision_30d'])} · 90T {_pct(r['revision_90d'])}</p>
      <p class="muted">Bewertung und Gewinnerwartung werden angezeigt, fließen aber noch nicht ins Filter
      (geplant als dritte und vierte Messquelle).</p></div>
  </div></details>
</td></tr>"""


def render(run: dict, path: str | None = None) -> str:
    res: pd.DataFrame = run["results"]
    calib, cfg, asof, track = run["calib"], run["cfg"], run["asof"], run.get("track", {"n": 0})
    cand = res[res["signal"] != "Ausstieg"].copy()
    cand["prio"] = (cand["signal"] != "Einstieg").astype(int)
    cand = cand.sort_values(["prio", "cr", "quality"], ascending=[True, False, False])
    picked, n_us = [], 0
    for idx, r in cand.iterrows():          # Quote: höchstens max_us_in_top US-Titel
        is_us = r["region"] == "USA"
        if is_us and n_us >= cfg["signals"]["max_us_in_top"]:
            continue
        picked.append(idx)
        n_us += int(is_us)
        if len(picked) == cfg["signals"]["top_n"]:
            break
    top = cand.loc[picked].copy()
    top["rank"] = range(1, len(top) + 1)
    exits = res[res["signal"] == "Ausstieg"]
    n_in, n_out = int((res["signal"] == "Einstieg").sum()), len(exits)
    bt = calib.get("backtest", {})
    synth = calib.get("provider") == "synthetic"
    ca = calib["analysts"]
    firms = sorted(ca["firms"].items(), key=lambda kv: kv[1]["var"])[:12]
    n_mv = int((res["analyst_method"] == "Minimum-Varianz").sum())
    mu = float(res["mu"].iloc[0]) if len(res) else 0.0
    hz = calib.get("horizon_test") or []
    rk = ca.get("rank") or {}
    if rk.get("ok"):
        rk_rows = "".join(
            f"<tr><td>F{q + 1}{' (höchste Ziele)' if q == len(rk['m']) - 1 else (' (niedrigste)' if q == 0 else '')}</td>"
            f"<td class='num'>{_pct(rk['m_learn'][q])}</td><td class='num'>{_pct(rk['m_test'][q])}</td>"
            f"<td class='num'>{_pct(rk['m'][q])} ± {_pct(rk['se'][q], 1, False)}</td><td class='num'>{_num(rk['n'][q], 0)}</td></tr>"
            for q in range(len(rk["m"])))
        rk_txt = (f"<table class='mini'><thead><tr><th>Fünftel</th><th class='num'>Lernzeitraum</th><th class='num'>Testzeitraum</th>"
                  f"<th class='num'>verwendet</th><th class='num'>Fälle</th></tr></thead><tbody>{rk_rows}</tbody></table>"
                  f"<span class='muted'>Vorsprung ggü. Ø aller Titel über 8 Wochen. Bestätigung im Testzeitraum b = "
                  f"{_num(rk.get('b_raw'), 2)} → Stufen mit Faktor {_num(rk['b'], 2)} verwendet. Unsicherheit aus "
                  f"{rk['n_blocks']} nicht überlappenden 8-Wochen-Blöcken.</span>")
    else:
        rk_txt = "Zu wenige Daten für eine Rang-Kalibrierung – Analystenquelle neutral."

    def _c(x):
        return "–" if x is None else f"{x:+.3f}".replace(".", ",")
    hz_rows = "".join(
        f"<tr><td>{h['weeks']} Wochen</td><td class='num'>{_c(h['tech_corr'])}</td><td class='num'>{_pct(h['tech_spread_pa'])}</td>"
        f"<td class='num'>{_c(h['ana_corr'])}</td><td class='num'>{_pct(h['ana_spread_pa'])}</td></tr>" for h in hz)
    hz_txt = (f"<table class='mini'><thead><tr><th>Horizont</th><th class='num'>Technik Korr.</th>"
              f"<th class='num'>Spreizung p. a.</th><th class='num'>Analysten Korr.</th><th class='num'>Spreizung p. a.</th>"
              f"</tr></thead><tbody>{hz_rows}</tbody></table><span class='muted'>Spreizung = Ø Rendite der besten 20 % "
              f"der Prognosen minus Ø aller Fälle, auf ein Jahr hochgerechnet.</span>") if hz else \
             "Wird beim nächsten Monatslauf erstellt." 
    noise = calib["noise"]

    exit_rows = "".join(
        f"<tr><td class='title'><span class='nm'>{_e(r['name'])}</span><span class='tk'>{_e(r['ticker'])} · {_e(r['region'])}</span></td>"
        f"<td>{_e('; '.join(r['reasons']))}</td><td class='num'>{_num(r['price'])}</td>"
        f"<td class='num'>{_sg(r['cr'])}</td></tr>" for _, r in exits.iterrows()) or \
        "<tr><td colspan='4' class='muted'>Keine Ausstiegssignale in der Watchlist.</td></tr>"

    firm_rows = "".join(
        f"<tr><td>{_e(f)}</td><td class='num'>{v['n']}</td><td class='num'>± {_pct(v['var'] ** 0.5, 1, False)}</td>"
        f"<td class='num'>{_pct(v['bias'])}</td></tr>" for f, v in firms) or \
        "<tr><td colspan='4' class='muted'>Keine Häuser-Historie verfügbar – überall Konsens-Rückfall.</td></tr>"

    if track.get("n", 0) > 0:
        track_txt = (f"{track['n']} abgeschlossene Einstiegssignale: Ø Rendite nach 8 Wochen {_pct(track['mean'])} "
                     f"gegenüber {_pct(track['bench'])} der Watchlist im selben Zeitraum, Vorsprung {_pct(track['excess'])}; "
                     f"besser als die Watchlist in {_pct(track['hit_excess'], 0, False)} der Fälle. "
                     f"{track.get('pending', 0)} noch offen.")
    else:
        track_txt = (f"Noch keine abgeschlossenen Signale (Auswertung jeweils 8 Wochen nach Signal). "
                     f"{track.get('pending', 0)} offen.")

    skt, ska = calib["technical"].get("skill", {}), ca.get("skill", {})

    def _sk(sk):
        if not sk or sk.get("corr") is None:
            return "nicht messbar (zu wenige Daten) – Quelle liefert nur den Mittelwert"
        corr = f"{sk['corr']:+.3f}".replace(".", ",")
        return (f"Korrelation {corr}, Treffsicherheit b = {_num(sk['b'], 2)} "
                f"(roh {_num(sk.get('b_raw'), 2)}), Messrauschen ± {_pct(sk['var_v'] ** 0.5, 1, False)}, "
                f"{_num(sk['n'], 0)} Testfälle")
    skill_txt = (f"<b>Technik:</b> {_sk(skt)}<br><b>Analysten (linear, nur zur Information):</b> {_sk(ska)}<br>"
                 f"<span class='muted'>b = 1: Prognose voll gültig, b = 0: keine Information. Die Kalman-Gewichte "
                 f"folgen daraus automatisch.</span>")
    bt_txt = "Noch kein Rückblick-Test verfügbar."
    if bt.get("corr_pred_real") is not None:
        bt_txt = (f"Testzeitraum {bt['period'][0]} bis {bt['period'][1]} (nicht zur Kalibrierung verwendet), "
                  f"{_num(bt['n'], 0)} Beobachtungen. Korrelation Prognose/Ergebnis {_num(bt['corr_pred_real'], 3)}. "
                  f"Beste 20 % der Signalzustände: Ø 8-Wochen-Rendite {_pct(bt['mean_top'])} "
                  f"(Trefferquote {_pct(bt['hit_top'], 0, False)}) gegenüber {_pct(bt['mean_all'])} "
                  f"({_pct(bt['hit_all'], 0, False)}) im Durchschnitt.")

    stamp = f"{WEEKDAY[asof.weekday()]}, {asof.strftime('%d.%m.%Y')}"
    rows = "".join(_row(r) for _, r in top.iterrows())
    banner = ('<div class="warn">Testlauf mit synthetischen Daten – keine echten Kurse.</div>' if synth else "")

    page = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>MarketLab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10); --series:#2a78d6;
  --good:#0ca30c; --goodtext:#006300; --crit:#d03b3b; --warnbg:#fff4dc;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10); --series:#3987e5;
  --goodtext:#0ca30c; --warnbg:#3a2e10; }} }}
:root[data-theme="dark"] {{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10); --series:#3987e5;
  --goodtext:#0ca30c; --warnbg:#3a2e10; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.5 Inter, system-ui, sans-serif; }}
.wrap {{ max-width:1320px; margin:0 auto; padding:32px 16px 64px; }}
header h1 {{ font:600 30px/1.1 "Source Serif 4", Georgia, serif; margin:0 0 4px; letter-spacing:-.01em; }}
header .sub {{ color:var(--ink2); margin:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0 32px; }}
.kpi {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px 16px; }}
.kpi .v {{ font:600 26px/1.1 "Source Serif 4", Georgia, serif; }}
.kpi .l {{ color:var(--ink2); font-size:12.5px; margin-top:2px; }}
h2 {{ font:600 20px/1.2 "Source Serif 4", Georgia, serif; margin:36px 0 6px; }}
h2 + p {{ color:var(--ink2); margin:0 0 12px; max-width:860px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-weight:500; color:var(--muted); font-size:12px; padding:10px 10px; border-bottom:1px solid var(--axis); white-space:nowrap; }}
td {{ padding:9px 10px; border-bottom:1px solid var(--grid); vertical-align:middle; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.strong {{ font-weight:600; }}
.rank {{ color:var(--muted); }}
.title .nm {{ display:block; font-weight:500; }}
.title .tk {{ display:block; color:var(--muted); font-size:12px; }}
tr.main td {{ border-bottom:none; }}
tr.det td {{ padding-top:0; }}
details summary {{ cursor:pointer; color:var(--ink2); font-size:12.5px; padding:2px 0 6px; }}
.detgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:20px; padding:6px 0 12px; }}
.detgrid h4 {{ margin:6px 0 4px; font-size:12.5px; color:var(--ink2); font-weight:600; }}
.detgrid p, .detgrid ul {{ margin:0 0 8px; color:var(--ink); font-size:13px; }}
.detgrid ul {{ padding-left:18px; }}
.mini th, .mini td {{ padding:4px 6px; font-size:12.5px; }}
.muted {{ color:var(--muted); }}
.badge {{ display:inline-flex; align-items:center; gap:6px; font-weight:500; font-size:13px; white-space:nowrap; }}
.badge svg {{ width:10px; height:10px; }}
.badge.good {{ color:var(--goodtext); }} .badge.good svg {{ fill:var(--good); }}
.badge.crit {{ color:var(--crit); }} .badge.crit svg {{ fill:var(--crit); }}
.badge.hold {{ color:var(--ink2); }} .badge.hold svg {{ fill:var(--muted); }}
.tag {{ font-size:10.5px; padding:1px 5px; border-radius:4px; border:1px solid var(--ring); color:var(--ink2); margin-left:4px; }}
.tag.mv {{ border-color:var(--series); color:var(--series); }}
.crval {{ display:inline-block; min-width:44px; }}
.crbar {{ position:relative; display:inline-block; width:70px; height:8px; vertical-align:middle; margin-left:6px; background:var(--grid); border-radius:4px; }}
.crbar .mid {{ position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--axis); }}
.crbar .fill {{ position:absolute; top:0; bottom:0; background:var(--series); border-radius:4px; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
.two .card {{ padding:4px 0; }}
.prose {{ padding:14px 16px; color:var(--ink); }}
.warn {{ background:var(--warnbg); border-radius:8px; padding:10px 14px; margin:16px 0 0; font-weight:500; }}
@media (max-width:760px) {{ .opt {{ display:none; }} .crbar {{ width:40px; }} td, th {{ padding-left:6px; padding-right:6px; }} .detgrid {{ grid-template-columns:1fr; }} .wrap {{ padding-top:20px; }} }}
footer {{ margin-top:40px; color:var(--muted); font-size:12px; max-width:860px; }}
</style></head><body><div class="wrap">
<header>
  <h1>MarketLab</h1>
  <p class="sub">Qualitätsaktien Europa · USA · Asien — Stand {stamp} · Horizont 8 Wochen</p>
  {banner}
</header>

<section class="kpis">
  <div class="kpi"><div class="v">{len(res)}</div><div class="l">Titel in der Watchlist</div></div>
  <div class="kpi"><div class="v">{n_in}</div><div class="l">Einstiegssignale</div></div>
  <div class="kpi"><div class="v">{n_out}</div><div class="l">Ausstiegssignale</div></div>
  <div class="kpi"><div class="v">{n_mv}</div><div class="l">Titel mit Minimum-Varianz-Analystenziel</div></div>
  <div class="kpi"><div class="v">{_pct(mu)}</div><div class="l">Watchlist-Median r̂ (8 Wochen, Bezugswert)</div></div>
  <div class="kpi"><div class="v">{_pct(noise['sigma_eps2'] ** 0.5, 1, False)}</div><div class="l">Kursrauschen σε (8 Wochen, {'geschätzt' if noise['identified'] else 'Annahme'})</div></div>
</section>

<h2>Top {len(top)} nach Chance/Risiko</h2>
<p>Rangkriterium ist der Vorsprung je Schätzfehler: Kalman-fusionierte erwartete 8-Wochen-Rendite r̂ abzüglich des Watchlist-Medians ({_pct(mu)}), geteilt durch σ. Die gemeinsame Marktdrift zählt damit nicht als Signal. Einstiegssignale stehen vorn, danach die übrigen Titel nach Chance/Risiko; höchstens {cfg["signals"]["max_us_in_top"]} US-Titel. Titel mit aktivem Ausstiegssignal stehen in der Liste darunter.
r₁ = Analystenquelle (MV = Minimum-Varianz der Top-5-Häuser, Konsens = Rückfall), r₂ = Chartquelle.
Gewichte zeigen den Anteil Analyst / Technik an der heutigen Fusion.</p>
<div class="card"><table>
<thead><tr><th class="num opt">#</th><th>Titel</th><th>Signal</th><th class="num">Chance/Risiko</th>
<th class="num">Vorsprung</th><th class="num opt">r̂ (Kalman)</th><th class="num opt">Schätzfehler σ</th><th class="num opt">r₁ Analysten</th><th class="num opt">r₂ Technik</th>
<th class="num opt">Gewicht A / T</th><th class="num opt">Kurs</th><th class="num opt">Stopp</th><th class="num opt">Fair Value</th>
<th class="num opt">Qualität</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>Ausstiegssignale in der Watchlist</h2>
<p>Alle Titel der Watchlist, bei denen eine Ausstiegsregel greift.</p>
<div class="card"><table>
<thead><tr><th>Titel</th><th>Grund</th><th class="num">Kurs</th><th class="num">Chance/Risiko</th></tr></thead>
<tbody>{exit_rows}</tbody></table></div>

<h2>Güte und Kalibrierung</h2>
<p>Kalibrierung vom {_e(str(calib['created'])[:10])} · Universum {calib['universe_size']} Titel,
nach Größenfilter {calib['size_filter']}.</p>
<div class="two">
  <div class="card prose"><h4 style="margin:0 0 6px">Treffsicherheit im Testzeitraum</h4>{skill_txt}
    <h4 style="margin:14px 0 6px">Analysten nach Rang (Fünftel)</h4>{rk_txt}
    <h4 style="margin:14px 0 6px">Horizonttest (Testzeitraum)</h4>{hz_txt}
    <h4 style="margin:14px 0 6px">Rückblick-Test Technik</h4>{bt_txt}
    <h4 style="margin:14px 0 6px">Erfolgskontrolle des Agenten</h4>{track_txt}</div>
  <div class="card"><table>
    <thead><tr><th>Analystenhaus (geringste Fehlervarianz)</th><th class="num">Prognosen</th>
    <th class="num">σ (8 Wo.)</th><th class="num">Verzerrung</th></tr></thead>
    <tbody>{firm_rows}</tbody></table>
    <p class="muted" style="padding:8px 10px 6px;margin:0">Fehlerkorrelation zwischen Häusern ρ_ff = {ca['rho_ff']:.2f};
    Gewichte als long-only Minimum-Varianz-Portfolio über die Fehler-Kovarianzmatrix.
    σ und Korrelation beziehen sich auf das Messrauschen nach Abzug des gemeinsamen Kursrauschens.
    Restkorrelation Analyst/Technik ρ_v = {noise['rho_v']:.2f} (Annahme, mit zwei Quellen nicht schätzbar).</p></div>
</div>

<footer>Automatisch erzeugt aus öffentlichen Marktdaten (Yahoo Finance). Ausstiegsregeln: Chance/Risiko unter
{_sg(cfg['signals']['exit_cr'])}, Schlusskurs {cfg['signals']['gd200_break_days']} Tage unter GD 200 oder GD 50 kreuzt GD 200 nach unten.
Stopp = Kurs − {_num(cfg['signals']['stop_monthly_std'], 1)} Monats-Standardabweichungen.</footer>
</div></body></html>"""
    path = path or os.path.join(ROOT, "docs", "index.html")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
