"""The okc CLI, driven with --json. Parent-side only: never copied into a child env.

A job's JSON has `status` (done, failed, rejected or duplicate) and `result`: passed,
speedup, speedup_ci, score_us, error, checks (the judge's static checks: flagged,
rejected), flags, and records: one per setup / test / benchmark / profile event.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TERMINAL = {"done", "failed", "rejected", "duplicate"}


class OkcError(RuntimeError):
    pass


def binary() -> str:
    found = shutil.which("okc") or str(Path.home() / ".okc/bin/okc")
    if not Path(found).exists():
        raise OkcError("okc not found: curl -fsSL https://openkernelchallenge.dev/install.sh | sh")
    return found


def _run(args: list[str], cwd: Path | None = None, timeout: float = 120) -> object:
    proc = subprocess.run([binary(), *args], cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=os.environ.copy())
    if proc.returncode != 0 and not proc.stdout.strip():
        raise OkcError(f"okc {' '.join(args)}: {proc.stderr.strip() or proc.returncode}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise OkcError(f"okc {' '.join(args)}: not JSON: {proc.stdout[:400]} {proc.stderr[:400]}") from e


def problems() -> dict[str, dict]:
    """name -> limits: {"scored": {per_day, used, left}, "checks": {per_day, used, left}}."""
    return {p["name"]: p["limits"] for p in _run(["problems", "--json"])["problems"]}


def leaderboard(problem: str) -> dict:
    return _run(["leaderboard", problem, "--json"])


def start(problem: str, file: Path, mode: str) -> dict:
    """Queue a run ("check" or "submit"); returns okc's JSON, which carries the job id."""
    assert mode in ("check", "submit")
    out = _run([mode, problem, file.name, "--no-wait", "--json"], cwd=file.parent, timeout=300)
    if not job_id(out):
        raise OkcError(f"okc {mode}: no job id in {json.dumps(out)[:400]}")
    return out


def job_id(out: dict) -> str | None:
    return out.get("id") or out.get("job_id") or (out.get("job") or {}).get("id")


def wait(job: str, cwd: Path, timeout: float = 5400, outage: float = 12 * 3600) -> dict:
    """Follow a job to the end, in the submitted file's directory so okc's profile lands there.
    Rides out network outages and sleeps (the job keeps running at the judge) for up to `outage`."""
    give_up = time.time() + outage
    while True:
        try:
            out = _run(["status", job, "--wait", "--json"], cwd=cwd, timeout=timeout)
        except (OkcError, subprocess.TimeoutExpired) as e:
            if time.time() > give_up:
                raise OkcError(f"job {job}: gave up waiting: {e}") from None
            print(f"okc wait {job}: {e}; retrying in 60 s", file=sys.stderr, flush=True)
            time.sleep(60)
            continue
        if out.get("status") in TERMINAL:
            return out
        if time.time() > give_up:
            raise OkcError(f"job {job} still {out.get('status')}")
        time.sleep(30)


def withdraw(job: str) -> str:
    proc = subprocess.run([binary(), "withdraw", job, "--yes"], capture_output=True, text=True,
                          timeout=120, env=os.environ.copy())
    if proc.returncode != 0:
        raise OkcError(f"okc withdraw {job}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()
