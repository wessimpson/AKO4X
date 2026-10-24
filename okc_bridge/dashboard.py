#!/usr/bin/env python3
"""Live performance dashboard for the okc campaigns: one chart per problem, speedup vs SGLang per run.

  python3 okc_bridge/dashboard.py [--out DIR] [--watch SECONDS]

Writes a self-contained index.html (data inlined, Plotly.js from cdnjs) and data.json to
<state>/dashboard/ ($AKO_OKC_STATE, default ~/.ako4x-okc). With --watch it regenerates every N
seconds and the page reloads itself on the same interval, keeping its scroll position.

Read-only on the campaign state: it only reads evals.jsonl, local.jsonl, code/ and local/, and
writes nothing but the two output files.

What is plotted, per problem, in time order (#1, #2, ...): every run that produced a score.
  official check / official scored run: evals.jsonl finished rows, status done, passed, with a
      speedup, at the judge's speedup (the judge's free duplicates are not new runs; skipped).
  local: local.jsonl finished rows that passed, at their PROJECTED competition score
      local speedup x ratio(problem), never the raw sidecar number. Calibration and e2e-test rows
      re-time judged files: they feed the ratio, they are not runs.
ratio(problem), in order of preference:
  1. <state>/sidecar/calibration.json[problem].ratio_median (`local.py calibrate`)
  2. the median judge/sidecar ratio over files that have both results ("pairs"), matched on the
     code with its `# okc:` header line removed (a round's workspace rewrites that line)
  3. the median of the other problems' ratios, marked "projection uncalibrated"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import broker  # noqa: E402

PROBLEMS = broker.PROBLEMS
NOT_RUNS = ("calibration", "e2e-test")  # local children that re-time old files
# The sidecar's GPU lock covered setup from 2026-10-23 15:50 CDT on; before that a file that
# times itself during setup could be measured while another eval held the GPU.
LOCK_FIX_TS = dt.datetime(2026, 10, 23, 20, 50, tzinfo=dt.timezone.utc).timestamp()
SELF_TIMING = re.compile(r"torch\.cuda\.Event|perf_counter|^\s*(import time\b|from time import)", re.M)
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"


# ── reading (never writes, never creates directories) ────────────────────


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file; a line still being appended (not valid JSON yet) is skipped."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def strip_header(code: str) -> str:
    """The code without its first line when that's the `# okc: problem=...` header."""
    first, _, rest = code.partition("\n")
    return rest if first.startswith("# okc:") else code


_hash_cache: dict[str, tuple[str, bool]] = {}


def code_info(path: Path, sha: str) -> tuple[str, bool]:
    """(hash of the code minus its header, whether it times itself). Files are written once, so cache."""
    key = str(path)
    if key in _hash_cache:
        return _hash_cache[key]
    if not path.is_file():
        return f"sha:{sha}", False
    code = path.read_text(errors="replace")
    info = (hashlib.sha256(strip_header(code).encode()).hexdigest(), bool(SELF_TIMING.search(code)))
    _hash_cache[key] = info
    return info


def calibration_file(root: Path) -> dict:
    p = root / "sidecar" / "calibration.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except json.JSONDecodeError:
        return {}


def round_of(child: str | None) -> str:
    m = re.search(r"-r(\d+)$", child or "")
    return f"r{m.group(1)}" if m else (child or "?")


def is_local_run(r: dict) -> bool:
    return r.get("child") not in NOT_RUNS and not str(r.get("label") or "").startswith("calibrate:")


def official_rows(problem: str, root: Path) -> tuple[list[dict], dict[str, str]]:
    """(scored official rows, {withdrawn job: reason})."""
    rows = read_jsonl(root / problem / "evals.jsonl")
    withdrawn = {r["job"]: r.get("reason", "") for r in rows if r.get("event") == "withdrawn" and r.get("job")}
    keep = [r for r in rows if r.get("event") == "finished" and r.get("status") == "done" and r.get("passed")
            and r.get("speedup") and not r.get("cached") and r.get("kind") in ("check", "run")]
    return keep, withdrawn


def local_rows(problem: str, root: Path) -> list[dict]:
    """Every passing local result with a speedup, calibration rows included."""
    return [r for r in read_jsonl(root / problem / "local.jsonl")
            if r.get("event") == "finished" and r.get("passed") and r.get("speedup") and not r.get("cached")]


def pre_lock_fix(r: dict, root: Path, problem: str) -> bool:
    if r["ts"] >= LOCK_FIX_TS:
        return False
    flagged = any("timing" in str(f) for f in r.get("flags") or [])
    return flagged or code_info(root / problem / "local" / f"{r['sha256']}.py", r["sha256"])[1]


# ── the projection ───────────────────────────────────────────────────────


def pairs(problem: str, root: Path) -> list[dict]:
    """Files with both an official and a local result, matched on header-stripped code."""
    official, _ = official_rows(problem, root)
    judged: dict[str, list[dict]] = {}
    for r in official:
        if not r.get("flags"):
            judged.setdefault(code_info(root / problem / "code" / f"{r['sha256']}.py", r["sha256"])[0], []).append(r)
    timed: dict[str, list[dict]] = {}
    for r in local_rows(problem, root):
        timed.setdefault(code_info(root / problem / "local" / f"{r['sha256']}.py", r["sha256"])[0], []).append(r)
    out = []
    for h in sorted(set(judged) & set(timed), key=lambda h: min(r["ts"] for r in timed[h])):
        j = statistics.median(r["speedup"] for r in judged[h])
        s = statistics.median(r["speedup"] for r in timed[h])
        out.append({"code": h[:12], "judge": j, "local": s, "ratio": j / s,
                    "jobs": [r["job"] for r in judged[h]], "local_jobs": [r["job"] for r in timed[h]],
                    "pre_lock_fix": any(pre_lock_fix(r, root, problem) for r in timed[h])})
    return out


def ratios(root: Path) -> dict[str, dict]:
    cal = calibration_file(root)
    out = {}
    for p in PROBLEMS:
        pr = pairs(p, root)
        c = cal.get(p) or {}
        if c.get("ratio_median"):
            out[p] = {"ratio": c["ratio_median"], "n": c.get("n", 0), "source": "sidecar calibration",
                      "uncalibrated": False, "pairs": pr}
        elif pr:
            out[p] = {"ratio": statistics.median(x["ratio"] for x in pr), "n": len(pr), "source": "matched pairs",
                      "uncalibrated": False, "pairs": pr}
        else:
            out[p] = {"ratio": None, "n": 0, "source": None, "uncalibrated": True, "pairs": pr}
    own = {p: v["ratio"] for p, v in out.items() if not v["uncalibrated"]}
    for p, v in out.items():
        if v["uncalibrated"]:
            others = [r for q, r in own.items() if q != p]
            v["ratio"] = statistics.median(others) if others else 1.0
            v["n"] = 0
            v["source"] = (f"median of {', '.join(q for q in own if q != p)}" if others
                           else "none available (1.0)")
    return out


# ── the dashboard data ───────────────────────────────────────────────────


BOARD_EVERY_S = 300
TEAM = "wessimpson"


def board_snapshots(problem: str, root: Path, fetch: bool = True) -> list[dict]:
    """The problem's public leaderboard over time: <state>/leaderboard/<problem>.jsonl, one row per
    snapshot ({ts, leader, leader_team, others: every other team's best speedup}). Appends a fresh
    snapshot when the last one is older than BOARD_EVERY_S (okc leaderboard needs no token)."""
    path = root / "leaderboard" / f"{problem}.jsonl"
    snaps = read_jsonl(path)
    if fetch and (not snaps or time.time() - snaps[-1]["ts"] > BOARD_EVERY_S):
        try:
            import okc_client
            rows = okc_client.leaderboard(problem).get("rows") or []
            if rows:
                snap = {"ts": time.time(), "leader": rows[0]["speedup"], "leader_team": rows[0].get("team"),
                        "others": sorted((r["speedup"] for r in rows if r.get("team") != TEAM), reverse=True)}
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as f:
                    f.write(json.dumps(snap) + "\n")
                snaps.append(snap)
        except Exception as e:  # noqa: BLE001 -- offline: keep the last snapshot
            print(f"leaderboard {problem}: {e}", file=sys.stderr)
    return snaps


def rank_on(snap: dict | None, y: float) -> dict | None:
    """Where a score would place us on a board snapshot (ties go our way)."""
    if not snap:
        return None
    others = snap["others"]
    rank = 1 + sum(1 for v in others if v > y)
    top10 = others[9] if len(others) >= 10 else None
    return {"rank": rank, "of": len(others) + 1, "leader": snap["leader"], "leader_team": snap.get("leader_team"),
            "top10": top10, "as_of": snap["ts"]}


def fmt_time(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def hover(pt: dict) -> str:
    e = html.escape
    kind = {"check": "official check", "run": "official scored run", "local": "local (sidecar)"}[pt["kind"]]
    lines = [f"<b>#{pt['n']} {e(pt['label'])}</b>", f"round {e(pt['round'])} · {kind}"]
    if pt["kind"] == "local":
        lines.append(f"sidecar score: {pt['local_speedup']:.4f}x")
        r = pt["ratio"]
        basis = f"{r['source']}, {r['n']} pair{'s' if r['n'] != 1 else ''}" if r["n"] else r["source"]
        lines.append(f"<b>projected competition score: {pt['y']:.4f}x</b>")
        lines.append(f"ratio ×{r['ratio']:.4f} ({e(basis)})")
        if r["uncalibrated"]:
            lines.append("<i>projection uncalibrated</i>")
        if pt["pre_lock_fix"]:
            lines.append("<i>pre lock fix: self-timing file timed before 15:50 CDT, may be noisy</i>")
    else:
        ci = pt.get("ci")
        ci_s = f"  [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        lines.append(f"<b>competition score: {pt['y']:.4f}x</b>{ci_s}")
        if pt.get("flags"):
            lines.append(f"judge flags: {e(', '.join(map(str, pt['flags'])))[:160]}")
    rk = pt.get("rank_now")
    if rk:
        gap = rk["leader"] / pt["y"] - 1
        lines.append(f"<b>would rank #{rk['rank']} of {rk['of']}</b> on the current board"
                     + (f" · top 10 needs {rk['top10']:.4f}x" if rk.get("top10") else "")
                     + f" · #1 {e(str(rk['leader_team']))} {rk['leader']:.4f}x"
                     + (f" ({gap:+.0%} from here)" if gap > 0 else " (we'd lead)"))
    if pt.get("withdrawn"):
        lines.append(f"<b>WITHDRAWN</b>: {e(pt['withdraw_reason'] or 'no reason recorded')}")
    lines.append(f"job {e(pt['job'])} · {fmt_time(pt['ts'])}")
    return "<br>".join(lines)


def problem_data(problem: str, root: Path, ratio: dict, fetch_board: bool = False) -> dict:
    snaps = board_snapshots(problem, root, fetch_board)
    now_snap = snaps[-1] if snaps else None
    official, withdrawn = official_rows(problem, root)
    pts = []
    for r in official:
        pts.append({"ts": r["ts"], "kind": r["kind"], "y": r["speedup"], "ci": r.get("ci"),
                    "flags": r.get("flags") or [], "withdrawn": r["job"] in withdrawn,
                    "withdraw_reason": withdrawn.get(r["job"], "")} | _common(r))
    for r in local_rows(problem, root):
        if is_local_run(r):
            pts.append({"ts": r["ts"], "kind": "local", "y": r["speedup"] * ratio["ratio"],
                        "local_speedup": r["speedup"], "local_ci": r.get("ci"),
                        "ratio": {k: ratio[k] for k in ("ratio", "n", "source", "uncalibrated")},
                        "pre_lock_fix": pre_lock_fix(r, root, problem)} | _common(r))
    pts.sort(key=lambda p: p["ts"])
    for i, p in enumerate(pts, 1):
        p["n"] = i
        p["time"] = fmt_time(p["ts"])
        p["rank_now"] = rank_on(now_snap, p["y"])
        then = [sn for sn in snaps if sn["ts"] <= p["ts"]]  # the board as it stood when this run finished
        p["leader_then"] = then[-1]["leader"] if then else None
        p["hover"] = hover(p)
    scored = [p for p in pts if p["kind"] == "run" and not p["flags"] and not p["withdrawn"]]
    board = max(scored, key=lambda p: p["y"], default=None)
    latest = max(scored, key=lambda p: p["ts"], default=None)
    pick = lambda p: p and {k: p.get(k) for k in ("n", "y", "ci", "job", "label", "time")}
    counts = {k: sum(p["kind"] == k for p in pts) for k in ("check", "run", "local")}
    leader = now_snap and {"y": now_snap["leader"], "team": now_snap.get("leader_team"), "time": fmt_time(now_snap["ts"]),
                           "our_rank": rank_on(now_snap, board["y"]) if board else None}
    return {"problem": problem, "points": pts, "board": pick(board), "latest_run": pick(latest), "leader": leader,
            "counts": dict(counts, total=len(pts)),
            "ratio": {k: v for k, v in ratio.items()},
            "calibration_rows": [{"label": r.get("label"), "child": r.get("child"), "job": r["job"],
                                  "local": r["speedup"], "time": fmt_time(r["ts"])}
                                 for r in local_rows(problem, root) if not is_local_run(r)],
            "withdrawn": withdrawn}


def _common(r: dict) -> dict:
    return {"label": str(r.get("label") or ""), "child": r.get("child"), "round": round_of(r.get("child")),
            "job": r.get("job", ""), "sha256": r.get("sha256", "")}


def build(root: Path | None = None, fetch_board: bool = False) -> dict:
    root = root or broker.state_root()
    rs = ratios(root)
    return {"generated": time.time(), "generated_str": fmt_time(time.time()), "state": str(root),
            "lock_fix": fmt_time(LOCK_FIX_TS),
            "problems": [problem_data(p, root, rs[p], fetch_board) for p in PROBLEMS]}


# ── the page ─────────────────────────────────────────────────────────────

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OKC Campaign Speedups</title>
<script src="__PLOTLY__"></script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --line: #b4b2a9;
  --check: #2a78d6; --run: #eb6834; --local: #1baf7a; --withdrawn: #d03b3b; --stale: #b3261e; --leader: #8d5bd6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --line: #5c5b57;
    --check: #3987e5; --run: #d95926; --local: #199e70; --withdrawn: #e66767; --stale: #ff8a80; --leader: #a77ef0;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --line: #5c5b57;
  --check: #3987e5; --run: #d95926; --local: #199e70; --withdrawn: #e66767; --stale: #ff8a80; --leader: #a77ef0;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1500px; margin: 0 auto; padding: 16px; }
header { display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: baseline; margin-bottom: 12px; }
h1 { font-size: 18px; margin: 0; font-weight: 600; }
.updated { color: var(--ink-2); font-variant-numeric: tabular-nums; }
.updated.stale { color: var(--stale); font-weight: 600; }
.note { color: var(--muted); font-size: 12.5px; width: 100%; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 12px 4px; margin-bottom: 14px; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0 4px 2px; }
.card h2 .board { font-variant-numeric: tabular-nums; }
.card .sub { color: var(--ink-2); font-size: 12.5px; margin: 0 4px; }
.card .sub.warn { color: var(--stale); }
.chart { width: 100%; height: 360px; }
.empty { color: var(--muted); padding: 40px 4px; }
</style>
</head>
<body>
<main>
<header>
  <h1>OKC campaigns: speedup vs SGLang per run</h1>
  <span class="updated" id="updated"></span>
  <div class="note" id="note"></div>
</header>
<div id="charts"></div>
</main>
<script>
const DATA = __DATA__;
const REFRESH_S = __REFRESH__;
const KINDS = [
  {key: "check", name: "official check", symbol: "circle", size: 9, color: "--check"},
  {key: "run", name: "official scored run", symbol: "diamond", size: 12, color: "--run"},
  {key: "local", name: "local, projected", symbol: "triangle-up", size: 12, color: "--local"},
  {key: "local-pre", name: "local, projected, pre lock fix", symbol: "triangle-up-open", size: 11, color: "--local"},
];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fx = v => v == null ? "–" : v.toFixed(4) + "x";
let hovering = false;

function header() {
  const age = Math.round(Date.now() / 1000 - DATA.generated);
  const el = document.getElementById("updated");
  const stale = REFRESH_S > 0 && age > 3 * REFRESH_S + 30;
  const ago = age < 90 ? age + " s ago" : Math.round(age / 60) + " min ago";
  el.textContent = "Last updated " + DATA.generated_str + " (" + ago + ")" +
    (REFRESH_S > 0 ? " · refreshes every " + REFRESH_S + " s" : " · static snapshot") +
    (stale ? " · STALE: is the watcher still running?" : "");
  el.classList.toggle("stale", stale);
}

function render() {
  const surface = css("--surface"), ink = css("--ink"), ink2 = css("--ink-2"), muted = css("--muted");
  const root = document.getElementById("charts");
  if (!root.childElementCount) {
    for (const p of DATA.problems) {
      const card = document.createElement("section");
      card.className = "card";
      const c = p.counts, b = p.board, r = p.ratio;
      const basis = r.n ? r.source + ", " + r.n + " pair" + (r.n === 1 ? "" : "s") : r.source;
      card.innerHTML =
        "<h2>" + p.problem + " · board <span class='board'>" + fx(b && b.y) + "</span>" +
        (p.leader && p.leader.our_rank ? " (#" + p.leader.our_rank.rank + " of " + p.leader.our_rank.of + ")" : "") +
        (p.leader ? " · #1 " + fx(p.leader.y) : "") + " · " + c.total + " runs</h2>" +
        "<div class='sub" + (r.uncalibrated ? " warn" : "") + "'>" + c.check + " checks, " + c.run + " scored runs, " +
        c.local + " local · local points projected ×" + r.ratio.toFixed(4) + " (" + basis + ")" +
        (r.uncalibrated ? " · projection uncalibrated" : "") + "</div>";
      const div = document.createElement("div");
      div.className = "chart";
      div.id = "chart-" + p.problem;
      card.appendChild(div);
      root.appendChild(card);
    }
  }
  for (const p of DATA.problems) {
    const div = document.getElementById("chart-" + p.problem);
    if (!p.points.length) { div.innerHTML = "<div class='empty'>No scored runs yet.</div>"; continue; }
    const pts = p.points;
    const traces = [{
      x: pts.map(q => q.n), y: pts.map(q => q.y), mode: "lines", type: "scatter",
      line: {color: css("--line"), width: 2}, hoverinfo: "skip", showlegend: false,
    }];
    for (const k of KINDS) {
      const sel = pts.filter(q => k.key === "local-pre" ? q.kind === "local" && q.pre_lock_fix
        : q.kind === k.key && !(k.key === "local" && q.pre_lock_fix));
      if (!sel.length) continue;
      const open = k.symbol.endsWith("-open");
      traces.push({
        x: sel.map(q => q.n), y: sel.map(q => q.y), name: k.name, mode: "markers", type: "scatter",
        hovertext: sel.map(q => q.hover), hovertemplate: "%{hovertext}<extra></extra>",
        marker: {symbol: k.symbol, size: k.size, color: css(k.color),
                 line: open ? {color: css(k.color), width: 2} : {color: surface, width: 1.5}},
      });
    }
    const wd = pts.filter(q => q.withdrawn);
    if (wd.length) traces.push({
      x: wd.map(q => q.n), y: wd.map(q => q.y), name: "withdrawn", mode: "markers", type: "scatter",
      hovertext: wd.map(q => q.hover), hovertemplate: "%{hovertext}<extra></extra>",
      marker: {symbol: "x-thin-open", size: 16, line: {color: css("--withdrawn"), width: 2}},
    });
    const lt = pts.filter(q => q.leader_then != null);
    if (lt.length) traces.push({
      x: lt.map(q => q.n), y: lt.map(q => q.leader_then), name: "#1 on the board at the time", mode: "lines",
      type: "scatter", line: {color: css("--leader"), width: 2, shape: "hv"}, hoverinfo: "skip",
    });
    const shapes = [], annotations = [];
    if (p.leader) {
      shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, y0: p.leader.y, y1: p.leader.y,
                   line: {color: css("--leader"), width: 1.5, dash: "dash"}});
      annotations.push({xref: "paper", x: 1, xanchor: "right", y: p.leader.y, yanchor: "bottom",
                        text: "#1 now: " + p.leader.team + " " + fx(p.leader.y), showarrow: false,
                        font: {size: 11, color: css("--leader")}});
    }
    if (p.board) {
      shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, y0: p.board.y, y1: p.board.y,
                   line: {color: css("--run"), width: 1, dash: "dash"}});
      annotations.push({xref: "paper", x: 0, xanchor: "left", y: p.board.y, yanchor: "bottom",
                        text: "board " + fx(p.board.y), showarrow: false, font: {size: 11, color: ink2}});
    }
    const layout = {
      paper_bgcolor: surface, plot_bgcolor: surface, font: {family: "system-ui, -apple-system, sans-serif", color: ink2, size: 12},
      margin: {l: 64, r: 16, t: 8, b: 44}, hovermode: "closest", shapes, annotations,
      hoverlabel: {align: "left", bgcolor: surface, bordercolor: css("--axis"), font: {color: ink, size: 12}},
      legend: {orientation: "h", x: 1, xanchor: "right", y: 1.02, yanchor: "bottom", font: {color: ink2}},
      xaxis: {title: {text: "run (time order)", font: {color: muted}}, tickprefix: "#", tickformat: "d",
              tick0: 1, dtick: Math.max(1, Math.ceil(pts.length / 16)), range: [0.5, pts.length + 0.5], gridcolor: css("--grid"),
              linecolor: css("--axis"), zeroline: false, tickfont: {color: muted}},
      yaxis: {title: {text: "speedup vs SGLang", font: {color: muted}}, ticksuffix: "x", tickformat: ".2f",
              gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: {color: muted}},
    };
    Plotly.react(div, traces, layout, {displaylogo: false, responsive: true,
                                       modeBarButtonsToRemove: ["select2d", "lasso2d"]});
    if (!div._hooked) {
      div.on("plotly_hover", () => { hovering = true; });
      div.on("plotly_unhover", () => { hovering = false; });
      div._hooked = true;
    }
  }
}

document.getElementById("note").textContent =
  "Official points are the judge's score; local points are plotted at their projected competition score " +
  "(sidecar speedup × the problem's judge/sidecar ratio), never the raw sidecar number. " +
  "Hollow triangles: self-timing files timed on the sidecar before its GPU lock covered setup (" + DATA.lock_fix + "), may be noisy.";

const KEY = "okc-dashboard-scroll";
function restoreScroll() {
  let y = null;
  try { y = sessionStorage.getItem(KEY); sessionStorage.removeItem(KEY); } catch (e) {}
  if (y == null) { const m = location.hash.match(/^#y=(\d+)$/); if (m) y = m[1]; }
  if (y != null) window.scrollTo(0, +y);
}
function reloadSoon() {
  if (hovering) { setTimeout(reloadSoon, 5000); return; }
  const y = Math.round(window.scrollY);
  try { sessionStorage.setItem(KEY, String(y)); } catch (e) {}
  try { history.replaceState(null, "", "#y=" + y); } catch (e) {}
  location.reload();
}

if (typeof Plotly === "undefined") {
  document.getElementById("charts").innerHTML = "<p class='card'>Plotly.js didn't load (offline?). The data is in data.json.</p>";
} else {
  render();
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render); } catch (e) {}
}
header();
setInterval(header, 5000);
restoreScroll();
window.addEventListener("load", restoreScroll);
if (REFRESH_S > 0) setTimeout(reloadSoon, REFRESH_S * 1000);
</script>
</body>
</html>
"""


def render_html(data: dict, refresh_s: int = 0) -> str:
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (PAGE.replace("__PLOTLY__", PLOTLY).replace("__REFRESH__", str(int(refresh_s)))
            .replace("__DATA__", blob))


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def generate(out: Path, refresh_s: int = 0, root: Path | None = None, fetch_board: bool = False) -> dict:
    data = build(root, fetch_board)
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "data.json", json.dumps(data, indent=1))
    write_atomic(out / "index.html", render_html(data, refresh_s))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output directory (default <state>/dashboard)")
    p.add_argument("--watch", type=int, metavar="SECONDS", help="regenerate every SECONDS; the page reloads as often")
    a = p.parse_args()
    out = Path(a.out).expanduser() if a.out else broker.state_root() / "dashboard"
    while True:
        try:
            data = generate(out, a.watch or 0, fetch_board=True)
            summary = ", ".join(f"{d['problem']} {d['counts']['total']}" for d in data["problems"])
            print(f"{data['generated_str']}: wrote {out / 'index.html'} ({summary} runs)", flush=True)
        except Exception as e:  # the campaigns write these files as we read them: try again next time
            if not a.watch:
                raise
            print(f"{fmt_time(time.time())}: dashboard failed: {e!r}", file=sys.stderr, flush=True)
        if not a.watch:
            return
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
