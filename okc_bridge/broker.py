#!/usr/bin/env python3
"""okc broker: the only path from an AKO4X campaign to the official judge.

Parent-side (never copied into a child env). Every official evaluation goes through
`eval`, which, in order:

  1. lints the file with okc_bridge/rules.py (static competition rules): errors spend nothing
  2. returns the stored result if this exact file was already judged: spends nothing
  3. refuses past the per-round cap ([round] max_evals) or after the kernel track closes
  4. under a global lock, reads today's live limits from `okc problems`, picks the slot
     (check, check, run, check, check, run, ...; see choose_kind), spaces submissions to
     stay under okc's rate limit, and starts the job
  5. waits for the job, stores the raw JSON, the code and the judge's per-kernel profile,
     and prints a summary

State lives in $AKO_OKC_STATE (default ~/.ako4x-okc), shared by every campaign worktree:
  <state>/okc.toml                   settings (copied from okc_bridge/okc.toml by launch.sh)
  <state>/<problem>/evals.jsonl      one line per started job and per finished evaluation
  <state>/<problem>/code/<sha>.py    every file that reached the judge
  <state>/<problem>/jobs/<job>.json  okc's raw job JSON
  <state>/<problem>/profiles/<job>.txt the judge's per-kernel profile, one section per shape

Usage:
  broker.py eval --problem P --file F --child C --label L [--json] [--out result.json]
  broker.py lint --problem P --file F
  broker.py status --problem P [--child C] [--json]
  broker.py results --problem P --child C            (JSON: one row per evaluation)
  broker.py best --problem P                         (JSON: best passing check and run)
  broker.py profile --problem P (--job J | --sha S)
  broker.py promote-best --problem P [--dry-run]     (scored run of the best unscored check)
  broker.py withdraw --problem P --job J --reason R
  broker.py wait-reset --problem P                   (blocks until a slot is free)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import okc_client as okc  # noqa: E402
import rules  # noqa: E402

PROBLEMS = ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16",
            "muse_decode_layer", "muse_prefill_layer")


# ── config + state ───────────────────────────────────────────────────────


def config_path() -> Path:
    """$AKO_OKC_STATE/okc.toml when present (one config for every campaign worktree), else
    okc_bridge/okc.toml (the defaults)."""
    shared = state_root() / "okc.toml"
    return shared if shared.is_file() else HERE / "okc.toml"


def load_config() -> dict:
    with open(config_path(), "rb") as f:
        return tomllib.load(f)


def state_root() -> Path:
    root = Path(os.environ.get("AKO_OKC_STATE") or Path.home() / ".ako4x-okc")
    root.mkdir(parents=True, exist_ok=True)
    return root


def pdir(problem: str) -> Path:
    if problem not in PROBLEMS:
        sys.exit(f"unknown problem {problem!r}; one of {', '.join(PROBLEMS)}")
    d = state_root() / problem
    for sub in ("code", "jobs", "profiles", "stage"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def utc_day(ts: float | None = None) -> str:
    return dt.datetime.fromtimestamp(ts or time.time(), dt.timezone.utc).strftime("%Y-%m-%d")


def seconds_until_reset() -> float:
    now = dt.datetime.now(dt.timezone.utc)
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - now).total_seconds()


def track_closed(cfg: dict) -> bool:
    return dt.datetime.now(dt.timezone.utc) > dt.datetime.fromisoformat(cfg["track"]["closes"])


def read_evals(problem: str) -> list[dict]:
    path = pdir(problem) / "evals.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_eval(problem: str, row: dict) -> None:
    with open(pdir(problem) / "evals.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


@contextlib.contextmanager
def global_lock():
    with open(state_root() / "lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── the schedule ─────────────────────────────────────────────────────────


def choose_kind(lim: dict, cfg: dict) -> str | None:
    """Next official evaluation for one problem today: "check", "run" or None (none left).

    The pattern is checks_per_run checks, then one scored run, repeating. It's derived from
    today's live used counts, so it holds across restarts and runs spent by hand: with 2,
    (checks used, runs used) = (0,0) check, (1,0) check, (2,0) run, (2,1) check, (4,1) run...
    When one kind is used up the other one is spent."""
    s = cfg["schedule"]
    k = s["checks_per_run"]
    checks_left = lim["checks"]["left"] - s["reserve_checks"]
    runs_left = lim["scored"]["left"] - s["reserve_runs"]
    want = "run" if lim["checks"]["used"] >= k * (lim["scored"]["used"] + 1) else "check"
    if want == "run" and runs_left <= 0:
        want = "check" if checks_left > 0 else None
    elif want == "check" and checks_left <= 0:
        want = "run" if runs_left > 0 else None
    return want


def budget(problem: str, cfg: dict) -> dict:
    lim = okc.problems()[problem]
    return {
        "checks_left": lim["checks"]["left"], "checks_used": lim["checks"]["used"],
        "runs_left": lim["scored"]["left"], "runs_used": lim["scored"]["used"],
        "next_kind": None if track_closed(cfg) else choose_kind(lim, cfg),
        "resets_in_s": int(seconds_until_reset()),
        "reserve": [cfg["schedule"]["reserve_checks"], cfg["schedule"]["reserve_runs"]],
    }


# ── results ──────────────────────────────────────────────────────────────


def summarize(raw: dict) -> dict:
    """Flatten okc's job JSON into what the agents read."""
    res = raw.get("result") or {}
    records = res.get("records") or []
    shapes, loads = [], {}
    for r in records:
        if r.get("event") != "benchmark" or r.get("status") != "pass":
            continue
        case = r.get("case") or {}
        w = case.get("weight", 1)
        shapes.append({
            "case": {k: v for k, v in case.items() if k not in ("seed", "weight")},
            "weight": w, "yours_us": round(r["median_ns"] / 1e3, 3),
            "sglang_us": round(r["baseline_ns"] / 1e3, 3), "speedup": round(r["speedup"], 4),
            "sol_us": round(r.get("speed_of_light_ns", 0) / 1e3, 3),
            "flags": r.get("flags") or [],
        })
        ld = loads.setdefault(case.get("load", "?"), [0.0, 0.0])
        ld[0] += r["baseline_ns"] * w
        ld[1] += r["median_ns"] * w
    tests = [r for r in records if r.get("event") == "test"]
    failing = [r for r in records if r.get("status") not in (None, "pass")]
    checks = res.get("checks") or {}
    if res.get("error"):
        message = str(res["error"])
    elif checks.get("rejected"):
        message = f"judge static checks rejected: {checks['rejected']}"
    elif failing:
        r = failing[0]
        message = (f"{r.get('event')} {json.dumps(r.get('case', {}))}: "
                   f"{r.get('message') or r.get('error') or r.get('status')}")
    else:
        message = res.get("status") or raw.get("status", "")
    return {
        "job": okc.job_id(raw), "status": raw.get("status"),
        "passed": bool(res.get("passed")) and raw.get("status") == "done",
        "speedup": res.get("speedup"), "ci": res.get("speedup_ci"),
        "score_us": res.get("score_us"), "baseline_us": res.get("baseline_us"),
        "message": message,
        "tests": {"passed": sum(r.get("status") == "pass" for r in tests), "total": len(tests)},
        "loads": {k: round(v[0] / v[1], 4) for k, v in loads.items() if v[1]},
        "shapes": shapes,
        "flags": (res.get("flags") or []) + (checks.get("flagged") or []),
        "judge_rejected": checks.get("rejected") or [],
        "duplicate_of": raw.get("duplicate_of"),
        "log_tail": (res.get("log_tail") or "")[-4000:] if not res.get("passed") else "",
    }


def write_profile(problem: str, job: str, raw: dict) -> Path | None:
    parts = []
    for r in (raw.get("result") or {}).get("records") or []:
        if r.get("event") == "profile" and r.get("report"):
            parts.append(f"## {json.dumps(r.get('case', {}))}\n{r['report']}\n")
    log = (raw.get("result") or {}).get("log_tail")
    if log:
        (pdir(problem) / "profiles" / f"{job}.log").write_text(log)
    if not parts:
        return None
    path = pdir(problem) / "profiles" / f"{job}.txt"
    path.write_text("\n".join(parts))
    return path


def finished(problem: str) -> list[dict]:
    """Evaluations the judge finished (one row per job; cached and gate rows excluded)."""
    return [e for e in read_evals(problem) if e.get("event") == "finished"]


def by_sha(problem: str, sha: str) -> dict | None:
    for e in reversed(finished(problem)):
        if e["sha256"] == sha and e["status"] in ("done", "failed", "rejected", "duplicate"):
            return e
    return None


def round_used(problem: str, child: str) -> int:
    """Official evaluations a child spent: jobs it started, minus the judge's free duplicates."""
    evals = read_evals(problem)
    dups = {e["job"] for e in evals if e.get("event") == "finished" and e["status"] == "duplicate"}
    return len({e["job"] for e in evals
                if e.get("event") == "started" and e.get("child") == child} - dups)


# ── commands ─────────────────────────────────────────────────────────────


def gate_row(problem, child, label, sha, status, message, **extra) -> dict:
    row = {"event": "gate", "ts": time.time(), "utc_day": utc_day(), "problem": problem,
           "child": child, "label": label, "sha256": sha, "status": status, "message": message}
    row.update(extra)
    append_eval(problem, row)
    return row


def submit(problem: str, code: str, kind: str, cfg: dict, *, child: str, label: str) -> tuple[str, Path]:
    """Start one official job (the caller holds the lock). Returns (job id, staging dir)."""
    sha = hashlib.sha256(code.encode()).hexdigest()
    d = pdir(problem)
    (d / "code" / f"{sha}.py").write_text(code)
    stage = d / "stage" / sha[:16]
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "submission.py").write_text(code)

    last = state_root() / "last_submit"
    wait_s = cfg["rate"]["min_spacing_s"] - (time.time() - (float(last.read_text()) if last.is_file() else 0))
    if wait_s > 0:
        time.sleep(wait_s)
    out = okc.start(problem, stage / "submission.py", "submit" if kind == "run" else "check")
    last.write_text(str(time.time()))
    job = okc.job_id(out)
    append_eval(problem, {"event": "started", "ts": time.time(), "utc_day": utc_day(),
                          "problem": problem, "child": child, "label": label, "sha256": sha,
                          "kind": kind, "job": job})
    return job, stage


def finish(problem: str, job: str, stage: Path, *, child: str, label: str, sha: str, kind: str) -> dict:
    raw = okc.wait(job, stage)
    d = pdir(problem)
    (d / "jobs" / f"{job}.json").write_text(json.dumps(raw, indent=1))
    judged = raw
    if raw.get("status") == "duplicate" and raw.get("duplicate_of"):
        # The judge already had this file: it returns the earlier job, free. Show that job's
        # result, but keep status "duplicate" so it never counts as a new check or scored run.
        judged = okc.wait(raw["duplicate_of"], stage)
        (d / "jobs" / f"{raw['duplicate_of']}.json").write_text(json.dumps(judged, indent=1))
    prof = write_profile(problem, job, judged)
    summary = summarize(judged)
    if judged is not raw:
        summary.update(status="duplicate", duplicate_of=raw["duplicate_of"],
                       message=f"the judge already had this file (job {raw['duplicate_of']}, "
                               f"{judged.get('mode')}); its result is shown, nothing was spent")
    row = {"event": "finished", "ts": time.time(), "utc_day": utc_day(), "problem": problem,
           "child": child, "label": label, "sha256": sha, "kind": kind,
           "code_path": str(d / "code" / f"{sha}.py"),
           "profile_path": str(prof) if prof else None,
           "log_path": str(d / "profiles" / f"{job}.log") if (d / "profiles" / f"{job}.log").is_file() else None,
           **summary}
    row["job"] = job
    append_eval(problem, row)
    return row


def cmd_eval(a, cfg) -> dict:
    code = Path(a.file).read_text()
    sha = hashlib.sha256(code.encode()).hexdigest()
    findings = rules.lint(code, a.problem)
    errs = rules.errors(findings)
    warns = [str(f) for f in findings if f.level == rules.WARN]
    if errs:
        return gate_row(a.problem, a.child, a.label, sha, "LINT_REJECTED",
                        "static rules check failed; nothing was spent",
                        lint=[str(f) for f in findings])
    prev = by_sha(a.problem, sha)
    if prev:
        row = dict(prev, cached=True, label=a.label, lint=warns)
        gate_row(a.problem, a.child, a.label, sha, "CACHED", f"same file as job {prev['job']}",
                 job=prev["job"])
        return row
    if track_closed(cfg):
        return gate_row(a.problem, a.child, a.label, sha, "TRACK_CLOSED", "the kernel track is closed")
    used, cap = round_used(a.problem, a.child), cfg["round"]["max_evals"]
    if used >= cap:
        return gate_row(a.problem, a.child, a.label, sha, "ROUND_CAP",
                        f"this run's {cap} official evaluations are spent; stop and write your synthesis")
    with global_lock():
        b = budget(a.problem, cfg)
        kind = b["next_kind"]
        if kind is None:
            return gate_row(a.problem, a.child, a.label, sha, "BUDGET_EXHAUSTED",
                            f"no official evaluations left today; resets in {b['resets_in_s'] // 3600}h"
                            f"{b['resets_in_s'] % 3600 // 60:02d}m", budget=b)
        job, stage = submit(a.problem, code, kind, cfg, child=a.child, label=a.label)
    row = finish(a.problem, job, stage, child=a.child, label=a.label, sha=sha, kind=kind)
    row["lint"] = warns
    return row


def with_context(row: dict, problem: str, child: str | None, cfg: dict) -> dict:
    try:
        row["budget_after"] = budget(problem, cfg)
    except okc.OkcError as e:
        row["budget_after"] = {"error": str(e)}
    if child:
        row["round"] = {"used": round_used(problem, child), "cap": cfg["round"]["max_evals"]}
    return row


def best(problem: str) -> dict:
    """Best passing, unflagged official result of each kind (by speedup)."""
    withdrawn = {e["job"] for e in read_evals(problem) if e.get("event") == "withdrawn"}
    out = {}
    for kind in ("check", "run"):
        rows = [e for e in finished(problem) if e["kind"] == kind and e["passed"] and not e["flags"]
                and e["status"] == "done" and e["job"] not in withdrawn]
        out[kind] = max(rows, key=lambda e: e["speedup"] or 0, default=None)
    return out


def cmd_promote_best(a, cfg) -> dict:
    """Spend the next slot on the best checked file, if the slot is a scored run and that file
    beats the best scored run and was never scored itself."""
    b = best(a.problem)
    chk, run = b["check"], b["run"]
    if not chk:
        return {"status": "NOOP", "message": "no passing, unflagged check yet"}
    scored = {e["sha256"] for e in finished(a.problem) if e["kind"] == "run" and e["status"] == "done"}
    if chk["sha256"] in scored:
        return {"status": "NOOP", "message": f"best check {chk['job']} is already scored"}
    if any(e["sha256"] == chk["sha256"] and e["kind"] == "run" and e["status"] == "duplicate"
           for e in finished(a.problem)):
        return {"status": "NOOP", "message": f"the judge answered a scored run of {chk['job']}'s file as a "
                                             f"duplicate of the check: it can't be promoted"}
    if run and (run["speedup"] or 0) >= (chk["speedup"] or 0):
        return {"status": "NOOP", "message": f"best run {run['speedup']:.4f}x >= best check {chk['speedup']:.4f}x"}
    code = Path(chk["code_path"]).read_text()
    if rules.errors(rules.lint(code, a.problem)):
        return {"status": "NOOP", "message": "best check no longer lints clean"}
    if track_closed(cfg):
        return {"status": "TRACK_CLOSED", "message": "the kernel track is closed"}
    with global_lock():
        bud = budget(a.problem, cfg)
        if bud["next_kind"] != "run":
            return {"status": "NOOP", "message": f"next slot is {bud['next_kind']}, not a run", "budget": bud}
        if a.dry_run:
            return {"status": "DRY_RUN", "message": f"would submit {chk['job']} ({chk['speedup']:.4f}x)"}
        job, stage = submit(a.problem, code, "run", cfg, child="promote", label=f"promote:{chk['job']}")
    return finish(a.problem, job, stage, child="promote", label=f"promote:{chk['job']}",
                  sha=chk["sha256"], kind="run")


def cmd_wait_reset(a, cfg) -> dict:
    while True:
        if track_closed(cfg):
            return {"status": "TRACK_CLOSED"}
        try:
            b = budget(a.problem, cfg)
            if b["next_kind"]:
                return {"status": "READY", "budget": b}
            nap = min(b["resets_in_s"] + 90, 1800)
        except okc.OkcError as e:
            print(f"wait-reset: {e}", file=sys.stderr)
            nap = 300
        time.sleep(nap)


def unaccounted(problem: str, b: dict) -> int:
    """Official runs today the broker didn't start (made by hand or by another tool)."""
    ours = {e["job"] for e in read_evals(problem) if e.get("event") == "started" and e["utc_day"] == utc_day()}
    return b["checks_used"] + b["runs_used"] - len(ours)


def render(row: dict) -> str:
    """Human-readable summary of an eval row, for the child's bench output."""
    st = row.get("status")
    lines = []
    if row.get("event") == "gate" and not row.get("cached"):
        lines.append(f"{st}: {row.get('message')}")
        for f in row.get("lint") or []:
            lines.append(f"  {f}")
        return "\n".join(lines)
    kind = "SCORED RUN (on the public board)" if row.get("kind") == "run" else "check (off the board)"
    tag = " [cached: same file as an earlier job; nothing spent]" if row.get("cached") else ""
    if st == "duplicate":
        tag += f" [{row.get('message')}]"
    lines.append(f"okc {kind} job {row.get('job')}: {st}{tag}")
    if row.get("passed"):
        ci = row.get("ci")
        ci_s = f" [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        lines.append(f"SCORE: {row['speedup']:.4f}x vs SGLang{ci_s}   "
                     f"tests {row['tests']['passed']}/{row['tests']['total']}")
        lines.append("loads: " + ", ".join(f"{k} {v:.3f}x" for k, v in row["loads"].items()))
        lines.append(f"{'shape':<52} {'yours_us':>9} {'sglang_us':>9} {'speedup':>8} {'floor_us':>8}")
        for s in row["shapes"]:
            name = " ".join(f"{k}={v}" for k, v in s["case"].items())[:52]
            lines.append(f"{name:<52} {s['yours_us']:>9.2f} {s['sglang_us']:>9.2f} "
                         f"{s['speedup']:>7.3f}x {s['sol_us']:>8.2f}" + (f"  flags={s['flags']}" if s['flags'] else ""))
    else:
        lines.append(f"FAILED: {row.get('message')}")
        if row.get("log_tail"):
            lines.append("log tail:\n" + row["log_tail"])
    if row.get("flags"):
        lines.append(f"JUDGE FLAGS: {row['flags']}  (flagged code is never promoted or archived)")
    for f in row.get("lint") or []:
        lines.append(f"lint: {f}")
    b = row.get("budget_after") or {}
    if "next_kind" in b:
        lines.append(f"today: {b['checks_left']} checks + {b['runs_left']} runs left; next official evaluation: "
                     f"{b['next_kind'] or 'none until the 00:00 UTC reset'}")
    if row.get("round"):
        lines.append(f"this run: {row['round']['used']}/{row['round']['cap']} official evaluations used")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--problem", required=True)
    e.add_argument("--file", required=True)
    e.add_argument("--child", required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--json", action="store_true")
    e.add_argument("--out", help="also write the result row as JSON here")
    li = sub.add_parser("lint")
    li.add_argument("--problem", required=True)
    li.add_argument("--file", required=True)
    s = sub.add_parser("status")
    s.add_argument("--problem", required=True)
    s.add_argument("--child")
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("results")
    r.add_argument("--problem", required=True)
    r.add_argument("--child", required=True)
    bst = sub.add_parser("best")
    bst.add_argument("--problem", required=True)
    pr = sub.add_parser("profile")
    pr.add_argument("--problem", required=True)
    pr.add_argument("--job")
    pr.add_argument("--sha")
    pb = sub.add_parser("promote-best")
    pb.add_argument("--problem", required=True)
    pb.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("withdraw")
    w.add_argument("--problem", required=True)
    w.add_argument("--job", required=True)
    w.add_argument("--reason", required=True)
    wr = sub.add_parser("wait-reset")
    wr.add_argument("--problem", required=True)
    a = p.parse_args()
    cfg = load_config()
    if hasattr(a, "problem"):
        pdir(a.problem)

    if a.cmd == "eval":
        try:
            row = with_context(cmd_eval(a, cfg), a.problem, a.child, cfg)
        except okc.OkcError as ex:
            row = gate_row(a.problem, a.child, a.label, "", "ERROR", str(ex))
        if a.out:
            Path(a.out).write_text(json.dumps(row, indent=1))
        print(json.dumps(row) if a.json else render(row))
        sys.exit(0 if row.get("event") == "finished" or row.get("cached") else 3)
    elif a.cmd == "lint":
        findings = rules.lint(Path(a.file).read_text(), a.problem)
        print(rules.report(findings))
        sys.exit(1 if rules.errors(findings) else 0)
    elif a.cmd == "status":
        b = budget(a.problem, cfg)
        bb = best(a.problem)
        out = {"problem": a.problem, "budget": b, "unaccounted_today": unaccounted(a.problem, b),
               "best_check": bb["check"] and {k: bb["check"][k] for k in ("job", "speedup", "ci", "label", "child")},
               "best_run": bb["run"] and {k: bb["run"][k] for k in ("job", "speedup", "ci", "label", "child")},
               "pattern": f"{cfg['schedule']['checks_per_run']} checks then 1 scored run, repeating"}
        if a.child:
            out["round"] = {"used": round_used(a.problem, a.child), "cap": cfg["round"]["max_evals"]}
        print(json.dumps(out, indent=1) if a.json else "\n".join(f"{k}: {v}" for k, v in out.items()))
    elif a.cmd == "results":
        rows = [e for e in finished(a.problem) if e["child"] == a.child]
        gates = [e for e in read_evals(a.problem) if e.get("event") == "gate" and e["child"] == a.child]
        print(json.dumps({"finished": rows, "gates": gates}, indent=1))
    elif a.cmd == "best":
        print(json.dumps(best(a.problem), indent=1))
    elif a.cmd == "profile":
        rows = [e for e in finished(a.problem)
                if (a.job and e["job"] == a.job) or (a.sha and e["sha256"].startswith(a.sha))]
        if not rows or not rows[-1].get("profile_path"):
            sys.exit("no profile for that job (it may have failed before timing)")
        print(Path(rows[-1]["profile_path"]).read_text())
    elif a.cmd == "promote-best":
        row = cmd_promote_best(a, cfg)
        print(json.dumps(row, indent=1))
    elif a.cmd == "withdraw":
        msg = okc.withdraw(a.job)
        append_eval(a.problem, {"event": "withdrawn", "ts": time.time(), "utc_day": utc_day(),
                                "problem": a.problem, "job": a.job, "reason": a.reason})
        print(msg)
    elif a.cmd == "wait-reset":
        print(json.dumps(cmd_wait_reset(a, cfg)))


if __name__ == "__main__":
    main()
