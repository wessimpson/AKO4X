#!/usr/bin/env python3
"""Local evaluations on the sidecar: an H200 SXM on RunPod running a copy of the judge.

  local.py eval --problem P --file F --child C --label L [--out O]   (free; spends no okc slot)
  local.py results --problem P [--child C]                           (JSON: local evaluations)
  local.py calibrate [--problem P] [--per-problem N]                 (re-time judged files on the sidecar)
  local.py status                                                    (pod, spend, calibration)

The sidecar (sidecar/ in the sidecar-judge worktree, `[sidecar] repo` in okc.toml) returns the
judge's job schema, so results render exactly like official ones, marked LOCAL. It is not the
judge: its numbers are only as good as `calibrate` shows. The pod is run by pod_guard.py, the only
process that holds the RunPod key; this file asks for the pod through <state>/sidecar/want and
reads its SSH address from <state>/sidecar/ssh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import broker  # noqa: E402
import rules  # noqa: E402

WAIT_FOR_POD_S = 25 * 60
EVAL_TIMEOUT_S = 45 * 60


def sdir() -> Path:
    d = broker.state_root() / "sidecar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def scfg(cfg: dict) -> dict:
    s = cfg.get("sidecar", {})
    return {"enabled": s.get("enabled", True), "repo": os.path.expanduser(s.get("repo", "")),
            "budget_usd": s.get("budget_usd", 225.0), "idle_stop_min": s.get("idle_stop_min", 60)}


def local_dir(problem: str) -> Path:
    d = broker.pdir(problem) / "local"
    d.mkdir(exist_ok=True)
    return d


def append(problem: str, row: dict) -> None:
    with open(broker.pdir(problem) / "local.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def read_rows(problem: str) -> list[dict]:
    p = broker.pdir(problem) / "local.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def unavailable(why: str) -> dict:
    return {"event": "gate", "status": "LOCAL_UNAVAILABLE", "message": why}


def pod_ssh(cfg: dict) -> str | dict:
    """The pod's SSH address, asking pod_guard to start it and waiting if it's down."""
    s = scfg(cfg)
    if not s["enabled"] or (sdir() / "disabled").exists():
        reason = (sdir() / "disabled").read_text().strip() if (sdir() / "disabled").exists() else "disabled in okc.toml"
        return unavailable(f"the local H200 is off ({reason}); use careful review and the official bench")
    if not s["repo"] or not (Path(s["repo"]) / "sidecar").is_dir():
        return unavailable("no sidecar checkout configured ([sidecar] repo in okc.toml)")
    if not (sdir() / "pod.json").is_file():
        return unavailable("no local H200 has been set up yet")
    beat = sdir() / "guard_heartbeat"
    if not beat.is_file() or time.time() - float(beat.read_text() or 0) > 180:
        return unavailable("the pod guard isn't running, so nothing can start the local H200")
    (sdir() / "last_used").write_text(str(time.time()))
    deadline = time.time() + WAIT_FOR_POD_S
    while not (sdir() / "ssh").is_file():
        (sdir() / "want").write_text(str(time.time()))
        if time.time() > deadline:
            return unavailable("the local H200 didn't come up within 25 minutes; carry on without it")
        time.sleep(20)
    return (sdir() / "ssh").read_text().strip()


def run_sidecar(problem: str, file: Path, cfg: dict) -> dict:
    ssh = pod_ssh(cfg)
    if isinstance(ssh, dict):
        return ssh
    repo = scfg(cfg)["repo"]
    out = local_dir(problem) / f".{os.getpid()}-{time.time_ns()}.json"
    env = dict(os.environ, SIDECAR_SSH=ssh)
    busy = sdir() / "busy"
    busy.mkdir(exist_ok=True)
    marker = busy / f"{os.getpid()}"
    marker.write_text(problem)
    try:
        p = subprocess.run([sys.executable, "-m", "sidecar", "evaluate", problem, str(file), "--out", str(out)],
                           cwd=repo, env=env, capture_output=True, text=True, timeout=EVAL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return unavailable(f"the local evaluation took over {EVAL_TIMEOUT_S // 60} minutes")
    finally:
        marker.unlink(missing_ok=True)
        (sdir() / "last_used").write_text(str(time.time()))
    if not out.is_file():
        return unavailable(f"the sidecar returned no result (rc {p.returncode}): {(p.stderr or p.stdout)[-1500:]}")
    raw = json.loads(out.read_text())
    out.unlink()
    return raw


def calibration(problem: str) -> dict | None:
    p = sdir() / "calibration.json"
    return json.loads(p.read_text()).get(problem) if p.is_file() else None


def cmd_eval(a, cfg) -> dict:
    code = Path(a.file).read_text()
    sha = hashlib.sha256(code.encode()).hexdigest()
    base = {"event": "gate", "ts": time.time(), "problem": a.problem, "child": a.child, "label": a.label,
            "sha256": sha, "source": "local"}
    findings = rules.lint(code, a.problem)
    if rules.errors(findings):
        return dict(base, status="LINT_REJECTED", message="static rules check failed",
                    lint=[str(f) for f in findings])
    cached = local_dir(a.problem) / f"{sha}.json"
    if cached.is_file():
        raw = json.loads(cached.read_text())
        tag = {"cached": True}
    else:
        src = local_dir(a.problem) / f"{sha}.py"
        src.write_text(code)
        raw = run_sidecar(a.problem, src, cfg)
        if raw.get("status") == "LOCAL_UNAVAILABLE":
            return dict(base, **raw)
        cached.write_text(json.dumps(raw))
        tag = {}
    row = {"event": "finished", "ts": time.time(), "problem": a.problem, "child": a.child, "label": a.label,
           "sha256": sha, **broker.summarize(raw), "kind": "local", "job": f"local-{sha[:12]}", "source": "local",
           "code_path": str(local_dir(a.problem) / f"{sha}.py"), **tag}
    if not tag:
        append(a.problem, {k: v for k, v in row.items() if k not in ("shapes",)})
    prof = local_dir(a.problem) / f"{sha}.profile.txt"
    reports = [(r.get("case") or {}, r.get("report", "")) for r in (raw.get("result") or {}).get("records", [])
               if r.get("event") == "profile"]
    if reports:
        prof.write_text("\n\n".join(f"== {c.get('load')} · {c.get('layer') or ''} m={c.get('m', '')}\n{t}"
                                    for c, t in reports))
        row["profile_path"] = str(prof)
    row["lint"] = [str(f) for f in findings if f.level == rules.WARN]
    row["calibration"] = calibration(a.problem)
    return row


def render(row: dict) -> str:
    if row.get("event") == "gate":
        out = [f"LOCAL {row['status']}: {row.get('message', '')}"]
        out += [f"  {f}" for f in row.get("lint") or []]
        return "\n".join(out)
    text = broker.render(dict(row, kind="check"))
    text = text.replace("okc check (off the board)", "LOCAL sidecar H200 (free; NOT the judge, not on any board)", 1)
    cal = row.get("calibration")
    note = ("uncalibrated: no judged file has been re-timed on the sidecar for this problem yet"
            if not cal else
            f"calibration on {cal['n']} judged files: judge/sidecar speedup ratio {cal['ratio_median']:.3f} "
            f"(range {cal['ratio_min']:.3f}-{cal['ratio_max']:.3f}); rank agreement {cal['rank_agreement']}")
    return text + f"\nLocal result. {note}. Only an official bench counts."


def cmd_calibrate(a, cfg) -> dict:
    """Re-time already-judged files (no okc slot spent) and compare with the judge's speedups."""
    path = sdir() / "calibration.json"
    allcal = json.loads(path.read_text()) if path.is_file() else {}
    for problem in ([a.problem] if a.problem else broker.PROBLEMS):
        judged = {}
        for e in broker.finished(problem):
            if e.get("status") == "done" and e.get("passed") and e.get("speedup") and not e.get("flags"):
                judged.setdefault(e["sha256"], e)
        picks = sorted(judged.values(), key=lambda e: -e["speedup"])
        if len(picks) > a.per_problem:  # the best, the worst and evenly in between
            step = (len(picks) - 1) / (a.per_problem - 1)
            picks = [picks[round(i * step)] for i in range(a.per_problem)]
        pairs = []
        for e in picks:
            ns = argparse.Namespace(problem=problem, file=e["code_path"], child="calibration",
                                    label=f"calibrate:{e['job']}")
            r = cmd_eval(ns, cfg)
            if r.get("event") == "finished" and r.get("speedup"):
                pairs.append({"job": e["job"], "judge": e["speedup"], "sidecar": r["speedup"]})
            print(f"{problem} {e['job']}: judge {e['speedup']:.3f} sidecar {r.get('speedup')}", flush=True)
        if pairs:
            ratios = [p["judge"] / p["sidecar"] for p in pairs]
            by_j = [p["job"] for p in sorted(pairs, key=lambda p: -p["judge"])]
            by_s = [p["job"] for p in sorted(pairs, key=lambda p: -p["sidecar"])]
            concord = sum(1 for i in range(len(pairs)) for j in range(i + 1, len(pairs))
                          if (by_j.index(pairs[i]["job"]) < by_j.index(pairs[j]["job"]))
                          == (by_s.index(pairs[i]["job"]) < by_s.index(pairs[j]["job"])))
            total = len(pairs) * (len(pairs) - 1) // 2
            allcal[problem] = {"n": len(pairs), "ratio_median": statistics.median(ratios),
                               "ratio_min": min(ratios), "ratio_max": max(ratios),
                               "rank_agreement": f"{concord}/{total} pairs" if total else "n/a",
                               "pairs": pairs, "ts": time.time()}
            path.write_text(json.dumps(allcal, indent=1))
    return allcal


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--problem", required=True)
    e.add_argument("--file", required=True)
    e.add_argument("--child", default="manual")
    e.add_argument("--label", default="unlabeled")
    e.add_argument("--out")
    r = sub.add_parser("results")
    r.add_argument("--problem", required=True)
    r.add_argument("--child")
    c = sub.add_parser("calibrate")
    c.add_argument("--problem")
    c.add_argument("--per-problem", type=int, default=5)
    sub.add_parser("status")
    a = p.parse_args()
    cfg = broker.load_config()
    if a.cmd == "eval":
        row = cmd_eval(a, cfg)
        if a.out:
            Path(a.out).write_text(json.dumps(row, indent=1))
        print(render(row))
        sys.exit(0 if row.get("event") == "finished" else 3)
    elif a.cmd == "results":
        print(json.dumps([r for r in read_rows(a.problem) if not a.child or r.get("child") == a.child], indent=1))
    elif a.cmd == "calibrate":
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in cmd_calibrate(a, cfg).items()},
                         indent=1))
    elif a.cmd == "status":
        d = sdir()
        info = {f: (d / f).read_text().strip() for f in ("ssh", "pod", "disabled") if (d / f).is_file()}
        spend = d / "spend.json"
        info["spend"] = json.loads(spend.read_text()) if spend.is_file() else None
        info["busy"] = [x.read_text() for x in (d / "busy").glob("*")] if (d / "busy").is_dir() else []
        info["calibration"] = {k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                               for k, v in (json.loads((d / "calibration.json").read_text()).items()
                                            if (d / "calibration.json").is_file() else [])}
        print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
