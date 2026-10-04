"""A browser view of the whole fleet's training, served from the shared volume.

`poker-loop --status --watch` answers the same questions for *one* machine in a
terminal. This serves every machine at once, in a page that refreshes itself, so
a long training day can be watched from anywhere on the network instead of from
seven separate ssh sessions.

**Standard library only.** No Flask, no framework: `http.server` is enough for a
page a handful of people look at, and the project's one firm rule about
dependencies is not to add them for convenience (the GUI is Tkinter for exactly
this reason). The page is a single self-contained HTML string with inline CSS and
a small fetch loop; there are no static assets to serve and nothing to build.

It only ever **reads**, and reads the same files `--status` does: each machine's
`loop_state.json`, the current generation's worker logs, the models directory,
the frozen benchmark set, and the global ranking's snapshot. Nothing it does can
disturb a run, so it is safe to leave open.

    poker-dashboard                      # http://127.0.0.1:8770
    poker-dashboard --host 0.0.0.0       # reachable from the other machines

It binds to localhost by default on purpose: the volume lives on a private
network, but a monitoring page that appears on every interface without being
asked is not a default worth having.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Everything here comes from torch-free modules, and that is a hard requirement,
# not tidiness: importing torch on the NFS server under load was measured at over
# five minutes. `rl/monitor.py` exists so this file never has to import
# `rl/loop.py`, which pulls torch in through `rl/ppo.py`.
from pokerlab.rl.global_store import DEFAULT_MODELS_DIR
from pokerlab.rl.monitor import (
    DEFAULT_MAX_POINTS,
    STAGE_LABELS,
    LoopState,
    benchmark_series,
    stage_label,
    worker_history,
    worker_progress,
)
from pokerlab.rl.training_pool import available_labels

DEFAULT_BENCHMARK_DIR = Path("checkpoints/benchmark")

DEFAULT_PORT = 8770
DEFAULT_MACHINES_DIR = Path("checkpoints/machines")
# The fleet-wide numbers change slowly and cost a directory walk over ~9,000
# files, so they are recomputed at most this often however fast the page polls.
FLEET_CACHE_SECONDS = 30.0


def read_machines(machines_dir: Path, iterations: int) -> list[dict]:
    """One entry per machine, newest generation first, with its workers."""
    out: list[dict] = []
    if not machines_dir.is_dir():
        return out
    for directory in sorted(p for p in machines_dir.iterdir() if p.is_dir()):
        state = LoopState.load(directory / "loop_state.json")
        rows = worker_progress(directory / "logs" / "loop", state.generation)
        workers = []
        for row in rows:
            entry = asdict(row)
            entry["stage_label"] = stage_label(row, iterations)
            workers.append(entry)
        done = sum(1 for r in rows if r.iterations >= iterations)
        # A generation ends when its slowest worker does, so the machine's own
        # "how much longer" is the longest ETA reported, not the mean. Absent
        # (None) until at least one worker has reached a stage that reports.
        etas = [
            r.progress.eta_seconds
            for r in rows
            if r.progress is not None and r.progress.eta_seconds is not None
        ]
        out.append(
            {
                "machine": directory.name,
                "started": state.started,
                "generation": state.generation,
                "phase": state.phase,
                "declared_workers": state.workers,
                "workers": workers,
                "done": done,
                "iterations_target": iterations,
                # Progress of the generation as a whole, which is what tells you
                # whether a machine is moving without reading every row.
                "progress": (
                    sum(r.iterations for r in rows) / (len(rows) * iterations)
                    if rows and iterations
                    else 0.0
                ),
                "history": state.history[-15:],
                "finish_eta": max(etas) if etas else None,
            }
        )
    return out


# A worker is `w01`. Anything else is refused outright rather than sanitised:
# the name reaches the filesystem, and a whitelist is the only check that cannot
# be argued around.
_WORKER_NAME = re.compile(r"^w\d{1,4}$")


def read_history(
    machines_dir: Path,
    machine: str,
    worker: str,
    generation: int | None = None,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict | None:
    """One worker's per-iteration curves, or None if it names nothing real.

    Served on demand rather than folded into `/api/status`, because the status
    is polled every few seconds by every open tab: a fleet of 140 workers at 400
    points x 6 metrics would be megabytes per poll for curves nobody is looking
    at. One worker's history is ~30 KB and is fetched only while its panel is
    open.

    `machine` is resolved and required to be a direct child of the machines
    directory, so a query string cannot walk out of it with `..` or an absolute
    path, and `worker` must match `_WORKER_NAME`.
    """
    if not _WORKER_NAME.match(worker):
        return None
    try:
        base = machines_dir.resolve(strict=True)
        target = (machines_dir / machine).resolve(strict=True)
    except OSError:
        return None
    if target.parent != base or not target.is_dir():
        return None
    if generation is None:
        generation = LoopState.load(target / "loop_state.json").generation
    history = worker_history(
        target / "logs" / "loop", generation, worker, max_points=max_points
    )
    if history is None:
        return None
    out = asdict(history)
    out["machine"] = machine
    out["generation"] = generation
    return out


def store_ratings(global_dir: Path) -> dict:
    """Mean, median, best and top-1% mean of every rating in the store.

    Read straight out of the `registry.json` snapshot rather than through
    `load_ranking`, because nothing here needs `PoolMember` objects and parsing
    ~9,200 of them is wasted work for four numbers (measured: 0.21 s either way,
    so this is about not carrying the objects rather than about speed). The
    snapshot may be a few minutes stale, which is exactly what a monitoring page
    wants -- and it is never used to decide anything.

    **The top-1% mean is the number to watch**: the population's best models have
    been flat at ~1578-1610 across every age cohort, which is the evidence that the
    loop generates variety without accumulating strength (see "The ranking is wrong
    at the top" in CLAUDE.md).
    """
    snapshot = Path(global_dir) / "registry.json"
    try:
        members = json.loads(snapshot.read_text(encoding="utf-8")).get("members", [])
    except (OSError, ValueError):
        return {}
    ratings = sorted(
        (float(m["rating"]) for m in members if isinstance(m.get("rating"), (int, float))),
        reverse=True,
    )
    if not ratings:
        return {}
    top = ratings[: max(1, len(ratings) // 100)]
    return {
        "rated": len(ratings),
        "mean": sum(ratings) / len(ratings),
        "median": ratings[len(ratings) // 2],
        "best": ratings[0],
        "top1_mean": sum(top) / len(top),
        "top1_count": len(top),
    }


def read_fleet(models_dir: Path, benchmark_dir: Path, global_dir: Path) -> dict:
    """The numbers shared by every machine: the store, the frozen set, the scale."""
    series = [
        {"name": name, "models": len(paths)} for name, paths in benchmark_series(benchmark_dir)
    ]
    return {
        "models": len(available_labels(models_dir)),
        "benchmark_models": sum(s["models"] for s in series),
        "benchmark_series": series,
        "ratings": store_ratings(global_dir),
    }


class State:
    """Holds the fleet numbers between requests, so polling stays cheap."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._fleet: dict = {}
        self._fleet_at = 0.0

    def snapshot(self) -> dict:
        now = time.time()
        if now - self._fleet_at > FLEET_CACHE_SECONDS:
            self._fleet = read_fleet(
                self.args.models_dir, self.args.benchmark_dir, self.args.global_dir
            )
            self._fleet_at = now
        machines = read_machines(self.args.machines_dir, self.args.iterations)
        active = [m for m in machines if any(w["age"] is not None and w["age"] < 600
                                             for w in m["workers"])]
        return {
            "generated": time.strftime("%H:%M:%S"),
            "fleet": self._fleet,
            "machines": machines,
            "active_machines": len(active),
            "stage_labels": STAGE_LABELS,
        }


PAGE = """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Poker RL</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#15181d; --muted:#666e7a; --line:#e2e5ea;
  --good:#1a7f4b; --bad:#b4342a; --warn:#9a6700; --accent:#2b5fd9; --bar:#dfe3e9;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12151a; --panel:#1a1e25; --ink:#e8eaee; --muted:#98a1ad; --line:#2a303a;
  --good:#4ac57e; --bad:#f07167; --warn:#e0a83c; --accent:#7aa2f7; --bar:#2a303a;
}}
:root[data-theme="dark"]{
  --bg:#12151a; --panel:#1a1e25; --ink:#e8eaee; --muted:#98a1ad; --line:#2a303a;
  --good:#4ac57e; --bad:#f07167; --warn:#e0a83c; --accent:#7aa2f7; --bar:#2a303a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;margin-bottom:18px}
h1{font-size:19px;margin:0;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:22px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:24px;font-weight:620;font-variant-numeric:tabular-nums;margin-top:3px}
.tile .n{color:var(--muted);font-size:12px}
.machine{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  margin-bottom:14px;overflow:hidden}
.mhead{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;
  padding:11px 14px;border-bottom:1px solid var(--line)}
.mname{font-weight:620}
.pill{font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);
  color:var(--muted);white-space:nowrap}
.pill.on{color:var(--good);border-color:currentColor}
.pill.off{color:var(--bad);border-color:currentColor}
.grow{flex:1}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:right;font-weight:550;color:var(--muted);font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;padding:8px 10px;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:7px 10px;text-align:right;border-top:1px solid var(--line);white-space:nowrap}
tr:hover td{background:color-mix(in srgb,var(--accent) 6%,transparent)}
.bar{display:inline-block;width:110px;height:7px;border-radius:4px;background:var(--bar);
  overflow:hidden;vertical-align:middle;margin-right:8px}
.bar>i{display:block;height:100%;background:var(--accent)}
.bar.mini{width:44px;margin-right:5px;vertical-align:middle}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.dim{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.empty{padding:16px 14px;color:var(--muted)}
.legend{color:var(--muted);font-size:12px;margin:8px 0 0;padding:0 14px 12px}
button{font:inherit;color:var(--muted);background:var(--panel);cursor:pointer;
  border:1px solid var(--line);border-radius:7px;padding:4px 10px}
tr.worker{cursor:pointer}
tr.worker td:first-child::before{content:"\u25B8 ";color:var(--muted)}
tr.worker.open td:first-child::before{content:"\u25BE "}
tr.panelrow td{padding:0;background:color-mix(in srgb,var(--accent) 4%,transparent)}
.charts{display:grid;gap:10px;padding:12px 14px;
  grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px 6px}
.hp{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:12px 14px 0}
.hp .chip{border:1px solid var(--line);border-radius:6px;padding:2px 7px;font-size:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel)}
.hp .chip b{font-weight:400;color:var(--muted)}
.hp .arm{border-color:var(--accent);color:var(--accent)}
.hp{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:12px 14px 0}
.hp .chip{border:1px solid var(--line);border-radius:6px;padding:2px 7px;font-size:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel)}
.hp .chip b{font-weight:600;color:var(--muted);font-weight:400}
.hp .arm{border-color:var(--accent);color:var(--accent)}
.chart h4{margin:0 0 2px;font-size:11.5px;font-weight:600;letter-spacing:.02em;
  display:flex;gap:8px;align-items:baseline}
.chart h4 .last{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:650}
.chart .note{color:var(--muted);font-size:10.5px;margin:0 0 4px}
.chart canvas{width:100%;height:120px;display:block}
.swatch{display:inline-block;width:8px;height:8px;border-radius:2px;vertical-align:baseline}
@media(max-width:700px){.bar{width:60px}th,td{padding:6px 6px}
  .charts{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header>
  <h1>Poker RL &mdash; addestramento</h1>
  <span class="sub" id="stamp"></span>
  <span class="grow"></span>
  <button id="theme">tema</button>
</header>
<div class="tiles" id="tiles"></div>
<div id="machines"></div>
</div>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const num = (v, d = 1, plus = true) => v === null || v === undefined ? '<span class="dim">&ndash;</span>'
  : `<span class="${v > 0 ? 'good' : v < 0 ? 'bad' : ''}">${plus && v > 0 ? '+' : ''}${v.toFixed(d)}</span>`;
const sig = s => s === null || s === undefined || s === '-' ? '<span class="dim">&ndash;</span>'
  : num(parseFloat(s), 1);

// A duration in the shortest readable form: how long ago, and how much is left.
function dur(a){
  return a < 60 ? Math.round(a) + 's' : a < 3600 ? Math.round(a/60) + 'm'
       : Math.floor(a/3600) + 'h' + String(Math.floor(a%3600/60)).padStart(2,'0');
}

// The two stages that take the best part of an hour each -- the benchmark
// round and the population round -- report how far in they are. Without it the
// cell reads "elo: gioco" for ninety minutes and says nothing about whether the
// worker is moving.
function stage(w){
  const p = w.progress;
  if (!p || !p.total) return esc(w.stage_label);
  const pct = Math.min(100, Math.round(100 * p.done / p.total));
  const left = p.eta_seconds === null || p.eta_seconds === undefined ? '' : ` ~${dur(p.eta_seconds)}`;
  const tip = `${p.done}/${p.total}${p.detail ? ' - ' + p.detail : ''}`;
  return `<span title="${esc(tip)}"><span class="bar mini"><i style="width:${pct}%"></i></span>`
       + `${esc(w.stage_label)} <span class="dim">${pct}%${esc(left)}</span></span>`;
}

function age(a){
  if (a === null || a === undefined) return '<span class="dim">&ndash;</span>';
  const t = a < 60 ? Math.round(a) + 's' : a < 3600 ? Math.round(a/60) + 'm'
          : Math.floor(a/3600) + 'h' + String(Math.floor(a%3600/60)).padStart(2,'0');
  return a > 600 ? `<span class="warn">${t} !</span>` : `<span class="dim">${t}</span>`;
}

function tiles(d){
  const f = d.fleet, series = (f.benchmark_series || []).length;
  const r = f.ratings || {};
  const items = [
    ['macchine attive', `${d.active_machines}/${d.machines.length}`, 'che scrivono log da meno di 10 min'],
    ['modelli nello store', (f.models || 0).toLocaleString('it'), 'checkpoints/models'],
    ['rating medio', r.mean === undefined ? '&ndash;' : r.mean.toFixed(0),
     r.rated === undefined ? 'registry.json non leggibile'
       : `mediana ${r.median.toFixed(0)} su ${r.rated.toLocaleString('it')} valutati`],
    ['rating del 1% migliore', r.top1_mean === undefined ? '&ndash;' : r.top1_mean.toFixed(0),
     r.best === undefined ? '' : `migliore ${r.best.toFixed(0)}, su ${r.top1_count} modelli`],
    ['ancore congelate', f.benchmark_models || 0, `${series} serie`],
    ['worker in corsa', d.machines.reduce((a,m) => a + m.workers.filter(w => w.stage !== 'done' && w.stage !== 'error').length, 0), 'su tutte le macchine'],
  ];
  $('#tiles').innerHTML = items.map(([k,v,n]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${esc(n)}</div></div>`).join('');
}

/* How this machine's generation splits between the two arms, so the sweep is
   visible without opening a panel. Absent when no worker recorded any, which is
   every log written before the sweep existed. */
function sweepPill(m){
  const counts = new Map();
  for (const w of m.workers){
    const hp = w.hyperparameters;
    if (!hp || !Object.keys(hp).length) continue;
    /* An absent hp_arm is counted, not skipped: it is how a machine still
       running an old supervisor identifies itself. */
    const arm = hp.hp_arm || '';
    counts.set(arm, (counts.get(arm) || 0) + 1);
  }
  if (!counts.size) return '';
  const parts = [...counts.entries()].sort((a, b) => b[1] - a[1])
    .map(([arm, n]) => n + ' ' + (HP_ARM_LABELS[arm] || arm));
  return '<span class="pill" title="ogni worker allena con i propri iperparametri: '
    + 'clicca una riga per vederli">' + esc(parts.join(', ')) + '</span>';
}

function machine(m){
  const alive = m.workers.some(w => w.age !== null && w.age < 600);
  const pct = Math.round((m.progress || 0) * 100);
  const head = `<div class="mhead">
    <span class="mname">${esc(m.machine)}</span>
    <span class="pill ${alive ? 'on' : 'off'}">${alive ? 'attiva' : 'ferma'}</span>
    <span class="pill">gen ${m.generation}</span>
    <span class="pill">${esc(m.phase)}</span>
    <span class="pill">${m.done}/${m.workers.length} worker completi</span>
    ${m.finish_eta === null || m.finish_eta === undefined ? ''
      : `<span class="pill" title="il worker piu' lento nei giri finali di elo e ancore">fine ~${dur(m.finish_eta)}</span>`}
    ${sweepPill(m)}
    <span class="grow"></span>
    <span class="bar"><i style="width:${pct}%"></i></span><span class="dim">${pct}%</span>
  </div>`;
  if (!m.workers.length) return `<div class="machine">${head}<div class="empty">nessun log per la generazione ${m.generation}</div></div>`;
  const rows = m.workers.map(w => {
    const p = Math.min(100, Math.round(w.iterations / Math.max(1, m.iterations_target) * 100));
    const key = `${m.machine}|${w.name}|${m.generation}`;
    hpByKey.set(key, w.hyperparameters || null);
    parentByKey.set(key, w.parent_rating);
    return `<tr class="worker${openPanels.has(key) ? ' open' : ''}" data-key="${esc(key)}">
      <td class="mono">${esc(w.name)}<span class="dim" title="${w.inherited ? 'eredita i pesi' : 'parte da zero'}"> ${w.inherited ? '^' : '.'}</span></td>
      <td><span class="bar"><i style="width:${p}%"></i></span><span class="dim">${w.iterations}/${m.iterations_target}</span></td>
      <td>${stage(w)}</td>
      <td title="${w.train_hands ? 'media su ' + w.train_hands.toLocaleString('it') + ' mani'
        : 'nessuna iterazione ancora'}">${num(w.train_bb100, 1)}${
        w.train_hands && w.train_hands < 100000 ? '<span class="dim" title="finestra ancora incompleta">*</span>' : ''}</td>
      <td>${sig(w.eval)}</td>
      <td class="mono">${w.rating === '-' ? '<span class="dim">&ndash;</span>' : esc(w.rating)}</td>
      <td class="mono dim">${w.pool_rating === '-' ? '<span class="dim">&ndash;</span>' : esc(w.pool_rating)}</td>
      <td${w.benchmark_live ? ' class="dim" title="provvisorio: giro contro le ancore in corso (' + w.benchmark_sessions + ' sessioni)"' : ''}>${num(w.benchmark_bb100, 1)}</td>
      <td class="mono${w.benchmark_live ? ' dim' : ''}">${w.benchmark_rating === '-' ? '<span class="dim">&ndash;</span>' : esc(w.benchmark_rating) + (w.benchmark_live ? '~' : '')}</td>
      <td class="dim">${w.entropy}</td>
      <td>${age(w.age)}</td>
    </tr>`;
  }).join('');
  return `<div class="machine">${head}
    <table><thead><tr>
      <th>worker</th><th>progresso</th><th>fase</th><th>train/100</th><th>eval/100</th>
      <th>rating</th>
      <th title="rating medio del campo che questo worker ha pescato: contro chi valgono eval e rating">pool</th>
      <th>ancore/100</th><th>elo ancore</th><th>entropia</th><th>agg.</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="legend">train/100 = bb ogni 100 mani sulle ultime 100.000 mani di training,
      contro pool e propri snapshot (~&plusmn;3 bb/100 a finestra piena) &middot;
      eval = ultima valutazione contro il pool estratto &middot;
      pool = rating medio del campo pescato da quel worker, cio&egrave; contro chi valgono eval e rating &middot;
      serie = giro finale per ogni cartella benchmark &middot;
      entropia = quanto la policy sta ancora mischiando (tetto ln 11 = 2.40) &middot;
      <b>!</b> = log fermo da oltre 10 minuti &middot;
      la fase mostra l'avanzamento e il tempo stimato per i due giri finali (serie ed elo), che durano circa un'ora ciascuno &middot;
      <b>clicca una riga</b> per gli iperparametri di quel worker e le curve di addestramento</p>
  </div>`;
}

/* ---- training curves -------------------------------------------------------
   Drawn by hand on a canvas rather than with a chart library: the page has no
   build step and no static assets, and the machines are on a private network
   with no reason to reach a CDN. A line, an axis and a last-value badge is all
   these need, and that is ~60 lines. */
const openPanels = new Set();    /* machine|worker|gen currently expanded */
/* Filled as the rows render, read when a panel opens. A Map rather than a data
   attribute because `esc` does not escape quotes, and settings do not belong in
   an HTML attribute just to be read back out of it two lines later. */
const hpByKey = new Map();
const parentByKey = new Map();
const drawn = new Map();         /* last history fetched, so a redraw is free */

const ITER_CHARTS = [
  /* Two series: the per-iteration value, which at 512 hands has a standard
     deviation of 42 bb/100 and is unreadable on its own, and the trailing
     100,000-hand mean, which is the same number the table's cell shows. The
     raw line is kept, faint, because the outliers are what a chart of a noisy
     metric is read for -- the mean would hide the single collapsed iteration
     the curve exists to reveal. */
  {k:'train_bb100', t:'training', u:'bb/100', d:1, zero:true,
   mean:'train_bb100_mean', meanT:'media 100k mani', rawT:'per iterazione',
   note:'linea spessa = media mobile sulle ultime 100.000 mani (come la colonna train/100); '
      + 'linea tenue = valore della singola iterazione'},
  {k:'entropy', t:'entropia',     u:'', d:3,
   note:"esplorazione: non c'\u00e8 una std, lo spazio azioni \u00e8 discreto (11 bin, tetto ln 11 \u2248 2.4)"},
  {k:'kl',      t:'kl',           u:'', d:4, ref:0.05, refT:'0.05 = passo troppo grande'},
  {k:'clip',    t:'clip fraction',u:'', d:3},
  {k:'value',   t:'value loss',   u:'', d:3, note:'se sale invece di restare piatta, il passo \u00e8 troppo grande'},
  {k:'policy',  t:'policy loss',  u:'', d:4, zero:true},
];

function fmtN(v, d){ return (v >= 0 ? '' : '') + v.toFixed(d); }

function draw(cv, sets, o){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 300, h = cv.clientHeight || 120;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const C = k => cs.getPropertyValue(k).trim();
  const all = sets.flatMap(s => s.pts);
  g.font = '10px ui-monospace,SFMono-Regular,Menlo,monospace';
  if (!all.length){
    g.fillStyle = C('--muted'); g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('nessun dato', w / 2, h / 2); return;
  }
  let x0 = Math.min(...all.map(p => p[0])), x1 = Math.max(...all.map(p => p[0]));
  let y0 = Math.min(...all.map(p => p[1])), y1 = Math.max(...all.map(p => p[1]));
  if (o.zero){ y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  if (o.ref !== undefined){ y0 = Math.min(y0, o.ref); y1 = Math.max(y1, o.ref); }
  if (y1 - y0 < 1e-12){ y1 = y0 + 1; y0 -= 1; }
  const pad = (y1 - y0) * 0.1; y0 -= pad; y1 += pad;
  if (x1 === x0) x1 = x0 + 1;
  const L = 44, R = 8, T = 10, B = 15;
  const X = v => L + (v - x0) / (x1 - x0) * (w - L - R);
  const Y = v => T + (1 - (v - y0) / (y1 - y0)) * (h - T - B);

  g.strokeStyle = C('--line'); g.lineWidth = 1;
  g.fillStyle = C('--muted'); g.textBaseline = 'middle'; g.textAlign = 'right';
  for (const v of [y1 - pad, (y0 + y1) / 2, y0 + pad]){
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
    g.fillText(fmtN(v, o.d), L - 5, y);
  }
  if (y0 < 0 && y1 > 0){                       /* the zero line, for signed metrics */
    const y = Math.round(Y(0)) + 0.5;
    g.strokeStyle = C('--muted'); g.globalAlpha = .5;
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.globalAlpha = 1;
  }
  if (o.ref !== undefined){                    /* the one real threshold: kl */
    const y = Math.round(Y(o.ref)) + 0.5;
    g.strokeStyle = C('--warn'); g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.setLineDash([]);
  }
  g.textBaseline = 'alphabetic'; g.fillStyle = C('--muted');
  g.textAlign = 'left';  g.fillText(x0, L, h - 3);
  g.textAlign = 'right'; g.fillText('iter ' + x1, w - R, h - 3);

  for (const s of sets){
    g.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
    g.strokeStyle = s.color; g.lineWidth = s.width || (s.dots ? 1.4 : 1.2);
    g.lineJoin = 'round'; g.beginPath();
    s.pts.forEach((pt, i) => i ? g.lineTo(X(pt[0]), Y(pt[1])) : g.moveTo(X(pt[0]), Y(pt[1])));
    g.stroke();
    if (s.dots){
      g.fillStyle = s.color;
      for (const pt of s.pts){ g.beginPath(); g.arc(X(pt[0]), Y(pt[1]), 2.4, 0, 7); g.fill(); }
    }
    g.globalAlpha = 1;
  }
}

function chartBox(title, note, last, sets, o){
  const legend = sets.length > 1
    ? sets.map(s => `<span class="dim"><span class="swatch" style="background:${s.color}"></span> ${esc(s.name)}</span>`).join(' ')
    : '';
  return `<div class="chart">
    <h4>${esc(title)} ${legend}<span class="last">${last}</span></h4>
    ${note ? `<p class="note">${note}</p>` : ''}
    <canvas></canvas></div>`;
}

function renderCharts(host, hist){
  const cs = getComputedStyle(document.documentElement);
  const C = k => cs.getPropertyValue(k).trim();
  const xs = hist.iterations || [];
  const specs = [];
  for (const c of ITER_CHARTS){
    const ys = hist[c.k] || [];
    const pts = xs.map((x, i) => [x, ys[i]]).filter(p => p[1] !== undefined);
    let last = pts.length ? pts[pts.length - 1][1] : null;
    const sets = [{name: c.rawT || c.t, color: C('--accent'), pts, alpha: c.mean ? .28 : 1}];
    const ms = c.mean ? (hist[c.mean] || []) : [];
    if (ms.length){
      const mpts = xs.map((x, i) => [x, ms[i]]).filter(p => p[1] !== undefined);
      /* The badge shows the mean, not the last raw point: it is the number the
         table shows and the only one of the two worth reading as a value. */
      if (mpts.length) last = mpts[mpts.length - 1][1];
      sets.push({name: c.meanT, color: C('--accent'), pts: mpts, width: 1.9});
    }
    specs.push({
      o: c, sets,
      title: c.t + (c.u ? ' (' + c.u + ')' : ''),
      note: c.refT ? `tratteggio: ${c.refT}` : (c.note || ''),
      last: last === null ? '&ndash;' : fmtN(last, c.d),
    });
  }
  const evalPts = hist.eval_bb100 || [];
  specs.push({
    o: {d: 1, zero: true},
    sets: [{name: 'eval vs pool', color: C('--accent'), pts: evalPts, dots: true}],
    title: 'eval vs pool (bb/100)',
    note: 'contro il pool che questo worker ha pescato, quindi non confrontabile fra worker',
    last: evalPts.length ? fmtN(evalPts[evalPts.length - 1][1], 1) : '&ndash;',
  });
  const rating = hist.eval_rating || [];
  specs.push({
    o: {d: 0},
    sets: [{name: 'rating', color: C('--warn'), pts: rating, dots: true}],
    title: 'rating in-run',
    note: 'cumulativo da 1500, 10 sessioni da 1000 mani per punto; K 24 le prime 10 sessioni, poi 8 e 3: si muove presto e poi si assesta',
    last: rating.length ? fmtN(rating[rating.length - 1][1], 0) : '&ndash;',
  });
  host.innerHTML = specs.map(s => chartBox(s.title, s.note, s.last, s.sets, s.o)).join('');
  const canvases = host.querySelectorAll('canvas');
  specs.forEach((s, i) => draw(canvases[i], s.sets, s.o));
}

/* The axes, in the order a reader wants them, with the labels the rest of the
   page uses. Anything the worker recorded that is not listed still shows, under
   its own name, so an axis added to the sweep appears here without a change. */
const HP_LABELS = {
  lr: 'lr', hands: 'mani/iter', ppo_epochs: 'epoche PPO',
  clip_epsilon: 'clip', minibatch_size: 'minibatch', gae_lambda: 'lambda GAE',
  value_coef: 'coef. value', max_grad_norm: 'max grad norm', entropy_coef: 'coef. entropia', opponent_probability: 'prob. avversario',
  /* Together these two are the strength of the field the worker trains against:
     what share of its seats come from the best-rated band, and how deep that
     band is. The `pool` column shows the strength they actually produced. */
  pool_top_share: 'quota top', pool_top_n: 'fascia top',
};
const HP_ARM_LABELS = {
  sampled: 'campionati', inherited: 'ereditati e perturbati',
  'sampled-fallback': 'campionati: genitore senza metadati',
  /* Launched by a supervisor that predates the sweep: it runs fine, it was just
     never given an arm. A supervisor lives for weeks and only picks up new
     behaviour when restarted, so this is the fleet saying which hosts still
     need it. */
  '': 'senza sweep: supervisor da riavviare',
};

function hpHtml(key){
  const hp = hpByKey.get(key);
  const parent = parentByKey.get(key);
  const parentChip = parent == null ? '' : '<span class="chip"><b>Elo genitore</b> ' + esc(parent) + '</span>';
  if (!hp || !Object.keys(hp).length){
    return '<div class="hp"><span class="dim">iperparametri non registrati: log di un worker '
      + 'avviato prima dello sweep</span>' + parentChip + '</div>';
  }
  const arm = hp.hp_arm || '';
  const chips = ['<span class="chip arm" title="solo un braccio campionato in modo indipendente si legge come curva di risposta">'
    + esc(HP_ARM_LABELS[arm] || arm) + '</span>'];
  if (parentChip) chips.push(parentChip);
  const known = Object.keys(HP_LABELS).filter(k => k in hp);
  const extra = Object.keys(hp).filter(k => k !== 'hp_arm' && !(k in HP_LABELS));
  for (const k of known.concat(extra)){
    chips.push('<span class="chip"><b>' + esc(HP_LABELS[k] || k) + '</b> ' + esc(hp[k]) + '</span>');
  }
  return '<div class="hp">' + chips.join('') + '</div>';
}

/* Built in one place because three call sites used to repeat the same string,
   and the settings block has to appear in all of them. */
function panelHtml(key){
  return '<td colspan="12">' + hpHtml(key)
    + '<div class="charts"><div class="empty">carico...</div></div></td>';
}

async function loadPanel(key){
  const host = document.querySelector(`[data-panel="${CSS.escape(key)}"] .charts`);
  if (!host) return;
  const [m, w, gen] = key.split('|');
  if (drawn.has(key)) renderCharts(host, drawn.get(key));
  try{
    const url = `api/history?machine=${encodeURIComponent(m)}&worker=${encodeURIComponent(w)}&gen=${encodeURIComponent(gen)}`;
    const hist = await (await fetch(url, {cache:'no-store'})).json();
    drawn.set(key, hist);
    renderCharts(host, hist);
  }catch(e){
    if (!drawn.has(key)) host.innerHTML = '<div class="empty">storia non disponibile per questo worker</div>';
  }
}

/* The machine tables are re-rendered whole on every poll, so an expanded panel
   has to be put back afterwards rather than left in the DOM. */
function restorePanels(){
  for (const row of document.querySelectorAll('tr.worker')){
    const key = row.dataset.key;
    if (!openPanels.has(key)) continue;
    const panel = document.createElement('tr');
    panel.className = 'panelrow';
    panel.setAttribute('data-panel', key);
    panel.innerHTML = panelHtml(key);
    row.after(panel);
    loadPanel(key);
  }
}

document.addEventListener('click', ev => {
  const row = ev.target.closest('tr.worker');
  if (!row) return;
  const key = row.dataset.key;
  if (openPanels.has(key)){
    openPanels.delete(key); drawn.delete(key);
    row.classList.remove('open');
    document.querySelector(`[data-panel="${CSS.escape(key)}"]`)?.remove();
  } else {
    openPanels.add(key); row.classList.add('open');
    const panel = document.createElement('tr');
    panel.className = 'panelrow';
    panel.setAttribute('data-panel', key);
    panel.innerHTML = panelHtml(key);
    row.after(panel);
    loadPanel(key);
  }
});

window.addEventListener('resize', () => { for (const k of openPanels) loadPanel(k); });

async function tick(){
  try{
    const d = await (await fetch('api/status', {cache:'no-store'})).json();
    tiles(d);
    $('#machines').innerHTML = d.machines.map(machine).join('');
    restorePanels();
    $('#stamp').textContent = 'aggiornato alle ' + d.generated;
  }catch(e){ $('#stamp').textContent = 'server non raggiungibile'; }
}
$('#theme').onclick = () => {
  const now = document.documentElement.getAttribute('data-theme');
  const next = now === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try{ localStorage.setItem('theme', next); }catch(e){}
};
try{ const t = localStorage.getItem('theme'); if (t) document.documentElement.setAttribute('data-theme', t); }catch(e){}
tick(); setInterval(tick, REFRESH_MS);
</script></body></html>
"""


class QuietServer(ThreadingHTTPServer):
    """`ThreadingHTTPServer` that does not shout about clients hanging up.

    `socketserver` prints a traceback from `handle_error` for any exception a
    handler lets escape, and a client disconnecting mid-response is not an error
    worth reporting -- it happens whenever a tab is closed while its poll is in
    flight. Only the connection-reset family is swallowed; anything else still
    gets the default traceback, because a real bug here must stay visible.
    """

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exception = sys.exc_info()[1]
        if isinstance(exception, BrokenPipeError | ConnectionResetError):
            return
        super().handle_error(request, client_address)


def make_handler(state: State, refresh: int):
    page = PAGE.replace("REFRESH_MS", str(max(2, refresh) * 1000)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, kind: str) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The client went away mid-response: a tab closed, a reload, a
                # laptop lid. Every open page polls every few seconds, so this is
                # routine rather than exceptional, and there is nothing to do
                # about it -- the response has nowhere to go. Swallowed because
                # the alternative is a 25-line traceback in the terminal the
                # dashboard is running in, which is where the operator is reading
                # everything else.
                pass

        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if route in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif route == "/api/status":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self._send(body, "application/json; charset=utf-8")
            elif route == "/api/history":
                query = parse_qs(urlparse(self.path).query)
                generation = query.get("gen", [None])[0]
                history = read_history(
                    state.args.machines_dir,
                    query.get("machine", [""])[0],
                    query.get("worker", [""])[0],
                    int(generation) if generation and generation.isdigit() else None,
                )
                if history is None:
                    self.send_error(404)
                    return
                self._send(
                    json.dumps(history).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            else:
                self.send_error(404)

        def log_message(self, *_args) -> None:
            """Silent: one poll per client every few seconds would bury the console."""

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a browser view of every machine's training, read-only."
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from the other machines")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--machines-dir", type=Path, default=DEFAULT_MACHINES_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--global-dir", type=Path, default=Path("checkpoints/global"),
        help="the shared ranking, read only for its registry.json snapshot",
    )
    parser.add_argument("--iterations", type=int, default=1000,
                        help="the loop's --iterations, which every progress bar is "
                        "scaled against; a wrong value draws every worker at the "
                        "wrong fraction")
    parser.add_argument("--refresh", type=int, default=15, help="seconds between polls")
    args = parser.parse_args(argv)

    state = State(args)
    server = QuietServer((args.host, args.port), make_handler(state, args.refresh))
    shown = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
    print(f"dashboard su http://{shown}:{args.port}  (ogni {args.refresh}s, Ctrl-C per uscire)")
    print(f"  macchine: {args.machines_dir}\n  modelli : {args.models_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nchiuso")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
