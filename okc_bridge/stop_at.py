#!/usr/bin/env python3
"""Stop every okc campaign cleanly at a set local time.

  python3 okc_bridge/stop_at.py 00:00        # next local midnight
  python3 okc_bridge/stop_at.py now          # right away

At the time: writes $AKO_OKC_STATE/STOP, so the broker starts no new official evaluations
(benches answer STOPPED); waits for jobs already at the judge to finish and be recorded (up
to --grace minutes); then ends the campaign: the okc-* tmux sessions (masters), every process
still working under the campaign worktrees (subs, benches), and the caffeinate session.
STOP stays in place until launch.sh is run again. Log: $AKO_OKC_STATE/stop.log.
"""

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import broker  # noqa: E402

WT_ROOT = Path(os.environ.get("AKO_OKC_WORKTREES") or Path.home() / "ako4x-okc").resolve()


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(broker.state_root() / "stop.log", "a") as f:
        f.write(line + "\n")


def target_time(spec: str) -> dt.datetime:
    now = dt.datetime.now()
    if spec == "now":
        return now
    hh, mm = (int(x) for x in spec.split(":"))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=1)


def campaign_pids() -> list[int]:
    """Processes whose working directory is under the campaign worktrees (masters' tools,
    subs, benches), excluding this one."""
    out = subprocess.run(["lsof", "-a", "-d", "cwd", "-Fpn"], capture_output=True, text=True).stdout
    pids, pid = [], None
    for line in out.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid and pid != os.getpid():
            if Path(line[1:]).resolve().is_relative_to(WT_ROOT):
                pids.append(pid)
    return sorted(set(pids))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("at", help="HH:MM local time (next occurrence), or 'now'")
    p.add_argument("--grace", type=int, default=30, help="minutes to wait for in-flight judge jobs")
    a = p.parse_args()

    when = target_time(a.at)
    log(f"armed: stopping the okc campaigns at {when:%Y-%m-%d %H:%M} local")
    while (left := (when - dt.datetime.now()).total_seconds()) > 0:
        time.sleep(min(left, 60))

    (broker.state_root() / "STOP").write_text(f"stopped by stop_at.py at {dt.datetime.now().isoformat()}\n")
    log("STOP written: no new official evaluations")

    deadline = time.time() + a.grace * 60
    while (jobs := broker.inflight()) and time.time() < deadline:
        log(f"waiting for {len(jobs)} job(s) at the judge: "
            + ", ".join(f"{j['problem']} {j['job']}" for j in jobs))
        time.sleep(30)
    if jobs:
        log(f"grace period over; {len(jobs)} job(s) unrecorded (results stay on okc: "
            + ", ".join(j["job"] for j in jobs) + ")")
    time.sleep(20)  # let the last bench write its snapshot

    pg = subprocess.run([sys.executable, str(HERE / "pod_guard.py"), "stop"], capture_output=True, text=True)
    log(f"local H200: {(pg.stdout or pg.stderr).strip()[-300:] or 'stopped'}")
    sessions = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True).stdout.split()
    for s in sessions:
        if s.startswith("okc-") and s != "okc-stop":
            subprocess.run(["tmux", "kill-session", "-t", s])
            log(f"ended tmux session {s}")
    time.sleep(5)
    for pid in campaign_pids():
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"terminated pid {pid}")
        except ProcessLookupError:
            pass
    time.sleep(10)
    for pid in campaign_pids():
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"killed pid {pid}")
        except ProcessLookupError:
            pass
    if "caffeinate" in sessions:
        subprocess.run(["tmux", "kill-session", "-t", "caffeinate"])
        log("ended caffeinate; the Mac can sleep again")
    for problem in broker.PROBLEMS:
        b = broker.best(problem)
        if b["check"] or b["run"]:
            log(f"{problem}: best check {b['check'] and b['check']['speedup']}, "
                f"best scored run {b['run'] and b['run']['speedup']}")
    log("done")
    if "okc-stop" in sessions:
        subprocess.run(["tmux", "kill-session", "-t", "okc-stop"])


if __name__ == "__main__":
    main()
