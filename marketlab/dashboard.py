"""Statisches HTML-Dashboard (GitHub Pages)."""
from __future__ import annotations

import html
import math
import os
import re

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
    "Beobachten": ('<span class="badge hold"><svg viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="3.2"/>'
                   '</svg>Beobachten</span>'),
}


def _cr_bar(cr: float, lim: float = 1.0) -> str:
    v = max(-lim, min(lim, cr))
    w = abs(v) / lim * 50
    left = 50 if v >= 0 else 50 - w
    return (f'<span class="crbar" title="Chance/Risiko {_sg(cr)}"><span class="mid"></span>'
            f'<span class="fill" style="left:{left:.1f}%;width:{w:.1f}%"></span></span>')


KEEP_UPPER = {"ASML", "HSBC", "BYD", "TSMC", "ICICI", "HDFC", "SBI", "ITC", "NTPC", "TCS", "BSE", "AIA", "CK",
              "HKEX", "LG", "SK", "AB", "SA", "SE", "AG", "NV", "N.V", "N.V.", "ADR", "PLC", "UBS", "BNP", "AXA", "SAP",
              "LVMH", "ABB", "DSV", "RELX", "BAE", "GSK", "BP", "CRH", "ASM", "KLA", "DBS", "OCBC", "JD", "NIO"}
NAME_FIX = {"NTES": "NetEase"}


def _nm(name: str) -> str:
    """Anzeigename: Kürzel-Namen ersetzen, GROSSSCHRIFT asiatischer Börsen vereinheitlichen."""
    n = NAME_FIX.get(str(name).strip(), str(name).strip())
    if n.isupper() and len(n) > 4:
        n = " ".join(w if w in KEEP_UPPER else w.capitalize() for w in n.split())
    return n


_SUFFIX = re.compile(r"(,?\s+(Inc\.?|Incorporated|Corporation|Corp\.?|Company|Co\.?|Holdings?|Group|plc|PLC|"
                     r"N\.?V\.?|S\.?A\.?|SE|AG|Ltd\.?|Limited|LLC|AB|ser\.\s*[AB]|Class\s+[A-C]))+\.?$")


def _short(name: str) -> str:
    """Kurzname für den Fließtext: Rechtsform- und Gattungszusätze entfernen."""
    n = _nm(name)
    for _ in range(3):
        n2 = _SUFFIX.sub("", n).strip().rstrip(",")
        if n2 == n or not n2:
            break
        n = n2
    return n


def _q(r) -> str:
    q = r.get("quintile")
    return f" · F{int(q)}" if q == q and q else ""


def _details(r: pd.Series) -> str:
    detail = "".join(
        f"<tr><td>{_e(d['firm'])}</td><td class='num'>{_num(d['target'])}</td>"
        f"<td class='num'>{_pct(d['weight'], 0, False)}</td></tr>" for d in r["analyst_detail"])
    reasons = "".join(f"<li>{_e(x)}</li>" for x in r["reasons"])
    trend = {1: "auf", 0: "gemischt", -1: "ab"}.get(r["trend"], "–")
    fv = (f"Fair Value {_num(r['fair_value'])} {_e(r['currency'] or '')} ({_pct(r['fv_gap'], 0)} zum Kurs)"
          if r['fv_ok'] else "Fair Value nicht belastbar (Gewinne zu instabil)")
    return f"""<div class="detgrid">
    <div><h4>Signal</h4><ul>{reasons}</ul>
      <h4>Kennzahlen</h4><p>Vorsprung {_pct(r['excess'])} ± {_pct(r['sd'], 1, False)} · Chance/Risiko {_sg(r['cr'])}<br>
      Kurs {_num(r['price'])} {_e(r['currency'] or '')} · Stopp {_num(r['stop'])} · Qualität {_num(r['quality'], 0)}</p>
      <h4>Technik</h4><p>{_e(r['tech_state'])}<br>RSI {_num(r['rsi'], 0)} · GD 50 {_num(r['sma50'])} · GD 200 {_num(r['sma200'])} ·
      Trend {trend} · rel. Momentum 12-1 {_pct(r['rel_mom'])}</p></div>
    <div><h4>Analysten ({_e(r['analyst_method'])}{_q(r)})</h4>
      <table class="mini"><thead><tr><th>Haus</th><th class="num">Kursziel</th><th class="num">Gewicht</th></tr></thead>
      <tbody>{detail}</tbody></table>
      <p>r₁ Analysten {_pct(r['r1'])} · r₂ Technik {_pct(r['r2'])} · Gewicht {_pct(r['w_analyst'], 0, False)} / {_pct(r['w_tech'], 0, False)}<br>
      Fusion r̂ {_pct(r['x'])} ± {_pct(r['sd'], 1, False)} · inkl. Kursrauschen ± {_pct(r['sd_total'], 1, False)}</p></div>
    <div><h4>Bewertung und Erwartung</h4>
      <p>{fv}<br>Gewinnwachstum 12M {_pct(r['growth_12m'])} · Revision 30T {_pct(r['revision_30d'])} · 90T {_pct(r['revision_90d'])}</p>
      <p class="muted">Information, fließt nicht ins Signal.</p></div>
  </div>"""


def _card(r: pd.Series, kind: str) -> str:
    """kind: new_in | active | new_out | ended"""
    sig = {"new_in": "Einstieg", "active": "Einstieg", "new_out": "Ausstieg", "ended": "Halten"}[kind]
    cls = {"Einstieg": "g", "Ausstieg": "r", "Halten": "n"}[sig]
    tag = {"new_in": "<span class='new'>neu</span>", "ended": "<span class='new grey'>Signal beendet</span>",
           "new_out": "<span class='new'>neu</span>", "active": ""}[kind]
    if sig == "Einstieg":
        kv = (f"<span>Vorsprung <b>{_pct(r['excess'])}</b></span><span>Stopp <b>{_num(r['stop'])}</b></span>"
              f"<span>Kurs <b>{_num(r['price'])}</b> <span class='muted'>{_e(r['currency'] or '')}</span></span>")
    else:
        kv = f"<span>Kurs <b>{_num(r['price'])}</b> <span class='muted'>{_e(r['currency'] or '')}</span></span>"
    why = "; ".join(r["reasons"])
    return (f"<div class='bcard {cls}'><div class='top'><div><b>{_e(_nm(r['name']))}</b>{tag}<br>"
            f"<span class='muted small'>{_e(r['ticker'])} · {_e(r['region'])}</span></div>{BADGE[sig]}</div>"
            f"<div class='kv'>{kv}</div><div class='why'>{_e(why)}</div>"
            f"<details class='more'><summary>Details</summary>{_details(r)}</details></div>")


def _listrow(r: pd.Series, new: bool) -> str:
    lbl = BADGE[r["signal"]] if r["signal"] == "Einstieg" else BADGE["Beobachten"]
    return (f"<details class='lrow'><summary><span class='t'><b>{_e(_nm(r['name']))}</b>{'<span class=new>neu</span>' if new else ''}"
            f"<br><span class='muted small'>{_e(r['ticker'])} · {_e(r['region'])}</span></span>{lbl}"
            f"<span class='num'>{_pct(r['excess'])}</span><span class='num hide-s'>{_num(r['stop'])}</span></summary>"
            f"<div class='lin'>{_details(r)}</div></details>")


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
    banner = ('<div class="warn">Testlauf mit synthetischen Daten – keine echten Kurse.</div>' if synth else "")
    if calib.get("analysts_reused"):
        banner += (f'<div class="warn">Analystendaten beim letzten Monatslauf nicht abrufbar – '
                   f'Analysten-Kalibrierung vom {_e(calib["analysts_reused"])} weiterverwendet.</div>')
    elif not ca.get("n"):
        banner += '<div class="warn">Keine Analysten-Kalibrierung vorhanden – Analystenquelle eingeschränkt.</div>'

    # ---- Tagesbrief: Veränderungen gegenüber dem Vortag
    prev = run.get("prev") or {}
    has_prev = bool(prev)
    by_sig = lambda s_: res[res["signal"] == s_]
    ent = by_sig("Einstieg")
    new_in = ent[[prev.get(t) != "Einstieg" for t in ent["ticker"]]] if has_prev else ent
    active = ent[[prev.get(t) == "Einstieg" for t in ent["ticker"]]] if has_prev else ent.iloc[0:0]
    new_out = exits[[prev.get(t) != "Ausstieg" for t in exits["ticker"]]] if has_prev else exits.iloc[0:0]
    old_out = exits.drop(new_out.index)
    ended = res[[prev.get(t) == "Einstieg" and sig != "Einstieg" and sig != "Ausstieg"
                 for t, sig in zip(res["ticker"], res["signal"])]] if has_prev else res.iloc[0:0]

    def names(df, cap=4):
        n_ = [_short(x) for x in df["name"]]
        if len(n_) > cap:
            return ", ".join(n_[:cap]) + f" und {len(n_) - cap} weitere"
        return n_[0] if len(n_) == 1 else (", ".join(n_[:-1]) + " und " + n_[-1] if n_ else "")
    ZW = {1: "ein", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf"}
    word = lambda n_: ZW.get(n_, str(n_))
    parts = []
    if len(new_in):
        parts.append(f"<b>{word(len(new_in)).capitalize() if len(new_in) > 1 else 'Ein'} neue{'s' if len(new_in) == 1 else ''} "
                     f"Einstiegssignal{'e' if len(new_in) > 1 else ''}</b>: {_e(names(new_in))}.")
    if len(ended):
        parts.append(f"<b>Einstiegssignal beendet</b>: {_e(names(ended))}.")
    if len(new_out):
        parts.append(f"<b>{word(len(new_out)).capitalize()} Titel neu im Ausstieg</b>: {_e(names(new_out))}.")
    if not parts:
        parts.append("<b>Keine neuen Signale heute.</b>")
    if len(active):
        parts.append(f"Weiterhin aktiv: {_e(names(active, 3))}.")
    if not has_prev:
        parts = [f"Erster Lauf ohne Vortagesvergleich: <b>{len(ent)} Einstiegssignal{'e' if len(ent) != 1 else ''}</b>"
                 f"{(': ' + _e(names(ent))) if len(ent) else ''}, {len(exits)} Ausstiegssignale."]
    brief = " ".join(parts)

    action_cards = ("".join(_card(r, "new_in") for _, r in new_in.iterrows())
                    + "".join(_card(r, "ended") for _, r in ended.iterrows())
                    + "".join(_card(r, "new_out") for _, r in new_out.iterrows()))
    action_html = (f'<div class="label">Heute neu</div><div class="cards">{action_cards}</div>' if action_cards else "")
    act = [_card(r, "active") for _, r in active.iterrows()]
    active_html = ""
    if act:
        active_html = f'<div class="label">Aktive Einstiegssignale ({len(act)})</div><div class="cards">{"".join(act[:6])}</div>'
        if len(act) > 6:
            active_html += (f'<details class="sec"><summary>Weitere {len(act) - 6} aktive Einstiegssignale</summary>'
                            f'<div class="secin"><div class="cards">{"".join(act[6:])}</div></div></details>')

    new_set = set(new_in["ticker"])
    watch_rows = "".join(_listrow(r, r["ticker"] in new_set) for _, r in top.iterrows())
    exit_rows = "".join(
        f"<tr><td class='title'><span class='nm'>{_e(_nm(r['name']))}</span><span class='tk'>{_e(r['ticker'])} · {_e(r['region'])}</span></td>"
        f"<td>{_e('; '.join(r['reasons']))}</td><td class='num'>{_num(r['price'])}</td></tr>" for _, r in old_out.iterrows()) or \
        "<tr><td colspan='3' class='muted'>Keine weiteren Ausstiegssignale.</td></tr>"

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
body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.5 Inter, system-ui, sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding:28px 16px 56px; }}
h1 {{ font:600 28px/1.1 "Source Serif 4", Georgia, serif; margin:0; }}
.sub {{ color:var(--ink2); margin:4px 0 0; font-size:14px; }}
.brief {{ font:500 19px/1.45 "Source Serif 4", Georgia, serif; margin:18px 0 4px; max-width:780px; }}
.label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin:22px 0 8px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(290px,1fr)); gap:12px; align-items:start; }}
.bcard {{ background:var(--surface); border:1px solid var(--ring); border-left:4px solid var(--good); border-radius:10px; padding:14px 16px; }}
.bcard.r {{ border-left-color:var(--crit); }} .bcard.n {{ border-left-color:var(--axis); }}
.bcard .top {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }}
.kv {{ display:flex; gap:16px; flex-wrap:wrap; font-size:14px; color:var(--ink2); margin:10px 0 4px; }}
.kv b {{ color:var(--ink); }}
.why {{ font-size:13px; color:var(--muted); }}
.small {{ font-size:12px; }}
.new {{ font-size:11px; font-weight:600; color:var(--series); border:1px solid var(--series); border-radius:4px; padding:0 5px; margin-left:6px; white-space:nowrap; }}
.new.grey {{ color:var(--ink2); border-color:var(--axis); }}
.badge {{ display:inline-flex; align-items:center; gap:6px; font-weight:600; font-size:13px; white-space:nowrap; }}
.badge svg {{ width:10px; height:10px; }}
.badge.good {{ color:var(--goodtext); }} .badge.good svg {{ fill:var(--good); }}
.badge.crit {{ color:var(--crit); }} .badge.crit svg {{ fill:var(--crit); }}
.badge.hold {{ color:var(--ink2); font-weight:500; }} .badge.hold svg {{ fill:var(--muted); }}
details.more > summary {{ cursor:pointer; color:var(--ink2); font-size:12.5px; margin-top:8px; }}
details.sec {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; margin:12px 0; }}
details.sec > summary {{ cursor:pointer; padding:13px 16px; font-weight:600; list-style:none; display:flex; justify-content:space-between; }}
details.sec > summary::-webkit-details-marker {{ display:none; }}
details.sec > summary::after {{ content:"＋"; color:var(--muted); }} details.sec[open] > summary::after {{ content:"－"; }}
.secin {{ padding:0 16px 14px; }}
.lhdr, details.lrow > summary {{ display:grid; grid-template-columns:1fr 130px 90px 90px; gap:8px; align-items:center; }}
.lhdr {{ font-size:12px; color:var(--muted); padding:6px 0; border-bottom:1px solid var(--axis); }}
details.lrow {{ border-bottom:1px solid var(--grid); }}
details.lrow > summary {{ cursor:pointer; list-style:none; padding:10px 0; }}
details.lrow > summary::-webkit-details-marker {{ display:none; }}
.lin {{ padding:0 0 10px; }}
.detgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; padding:8px 0 4px; }}
.detgrid h4 {{ margin:6px 0 4px; font-size:12.5px; color:var(--ink2); font-weight:600; }}
.detgrid p, .detgrid ul {{ margin:0 0 8px; font-size:13px; }} .detgrid ul {{ padding-left:18px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; font-weight:500; color:var(--muted); font-size:12px; padding:8px; border-bottom:1px solid var(--axis); }}
td {{ padding:8px; border-bottom:1px solid var(--grid); vertical-align:top; font-size:14px; }}
.mini th, .mini td {{ padding:4px 6px; font-size:12.5px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.muted {{ color:var(--muted); }}
.title .nm {{ display:block; font-weight:500; }} .title .tk {{ display:block; color:var(--muted); font-size:12px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:4px 0 16px; }}
.kpi {{ border:1px solid var(--ring); border-radius:8px; padding:10px 12px; }}
.kpi .v {{ font:600 20px/1.1 "Source Serif 4", Georgia, serif; }} .kpi .l {{ color:var(--ink2); font-size:12px; }}
.gq h4 {{ margin:16px 0 6px; font-size:13px; }} .gq p {{ font-size:13.5px; color:var(--ink2); }}
.warn {{ background:var(--warnbg); border-radius:8px; padding:10px 14px; margin:14px 0 0; font-weight:500; }}
@media (max-width:620px) {{ .lhdr, details.lrow > summary {{ grid-template-columns:1fr 100px 64px; }} .hide-s {{ display:none; }}
  .brief {{ font-size:17px; }} }}
footer {{ margin-top:32px; color:var(--muted); font-size:12px; }}
</style></head><body><div class="wrap">
<h1>MarketLab</h1>
<p class="sub">Tagesbrief {stamp} · Horizont 8 Wochen</p>
{banner}
<p class="brief">{brief}</p>
{action_html}
{active_html}

<div class="label">Mehr</div>
<details class="sec"><summary>Beobachtungsliste – Top {len(top)}</summary><div class="secin">
  <div class="lhdr"><span>Titel · antippen für Details</span><span>Signal</span><span class="num">Vorsprung</span><span class="num hide-s">Stopp</span></div>
  {watch_rows}
  <p class="muted small">Sortiert nach Chance/Risiko (Vorsprung je Schätzfehler), höchstens {cfg["signals"]["max_us_in_top"]} US-Titel.</p>
</div></details>

<details class="sec"><summary>Bestehende Ausstiegssignale ({len(old_out)})</summary><div class="secin">
  <table><thead><tr><th>Titel</th><th>Grund</th><th class="num">Kurs</th></tr></thead><tbody>{exit_rows}</tbody></table>
</div></details>

<details class="sec"><summary>Kennzahlen, Güte und Methode</summary><div class="secin gq">
  <div class="kpis">
    <div class="kpi"><div class="v">{len(res)}</div><div class="l">Titel in der Watchlist</div></div>
    <div class="kpi"><div class="v">{n_in} / {n_out}</div><div class="l">Einstiege / Ausstiege</div></div>
    <div class="kpi"><div class="v">{n_mv}</div><div class="l">mit Minimum-Varianz-Ziel</div></div>
    <div class="kpi"><div class="v">{_pct(mu)}</div><div class="l">Watchlist-Median r̂ (Bezugswert)</div></div>
  </div>
  <p>Vorsprung = Kalman-fusionierte erwartete 8-Wochen-Rendite r̂ minus Watchlist-Median. Chance/Risiko = Vorsprung ÷ Schätzfehler.
  Einstieg ab Vorsprung {_pct(cfg["signals"]["entry_min_excess"])} und Chance/Risiko über {_sg(cfg["signals"]["entry_cr"])} bei intaktem Aufwärtstrend;
  ein aktives Signal bleibt bis Vorsprung {_pct(cfg["signals"]["hold_min_excess"])}.
  Kalibrierung vom {_e(str(calib['created'])[:10])} · Universum {calib['universe_size']} Titel, nach Größenfilter {calib['size_filter']}.</p>
  <h4>Erfolgskontrolle des Agenten</h4><p>{track_txt}</p>
  <h4>Analysten nach Rang (Fünftel)</h4>{rk_txt}
  <h4>Treffsicherheit im Testzeitraum</h4><p>{skill_txt}</p>
  <h4>Horizonttest</h4>{hz_txt}
  <h4>Rückblick-Test Technik</h4><p>{bt_txt}</p>
  <h4>Analystenhäuser mit geringster Fehlervarianz</h4>
  <table class="mini"><thead><tr><th>Haus</th><th class="num">Prognosen</th><th class="num">σ (8 Wo.)</th><th class="num">Verzerrung</th></tr></thead>
  <tbody>{firm_rows}</tbody></table>
  <p class="muted small">Fehlerkorrelation zwischen Häusern ρ_ff = {ca['rho_ff']:.2f}; Minimum-Varianz-Gewichte zur Hälfte Richtung Gleichgewichtung geschrumpft.
  Restkorrelation Analyst/Technik ρ_v = {noise['rho_v']:.2f} (Annahme). Kursrauschen σε = {_pct(noise['sigma_eps2'] ** 0.5, 1, False)}.</p>
</div></details>

<footer>Automatisch erzeugt aus öffentlichen Marktdaten (Yahoo Finance). Ausstiegsregeln: Chance/Risiko unter
{_sg(cfg['signals']['exit_cr'])}, Schlusskurs {cfg['signals']['gd200_break_days']} Tage unter GD 200 oder GD 50 kreuzt GD 200 nach unten.
Stopp = Kurs − {_num(cfg['signals']['stop_monthly_std'], 1)} Monats-Standardabweichungen.</footer>
</div></body></html>"""
    path = path or os.path.join(ROOT, "docs", "index.html")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
