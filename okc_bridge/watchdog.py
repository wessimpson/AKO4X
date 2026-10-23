#!/usr/bin/env python3
"""Keep the okc campaigns moving across laptop sleeps and network outages.

  python3 okc_bridge/watchdog.py            # run in its own tmux session (launch.sh starts one)

Every minute, while the network is up:
- records any job still at the judge whose bench died (a dropped connection, a sleep), so its
  result reaches the broker's store instead of being lost (finish() records a job once);
- nudges a master whose screen has sat unchanged on an API/network error, or idle with no
  background task, for a while: it types a short "continue" message into its tmux pane.
Masters that are waiting on a background task (a sub, a bench, the daily reset) are left alone.
Log: $AKO_OKC_STATE/watchdog.log.
"""

import datetime as dt
import hashlib
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import broker  # noqa: E402
import okc_client as okc  # noqa: E402

TICK_S = 60
ORPHAN_AGE_S = 15 * 60       # a started job this old with no finished row gets followed here too
ERROR_IDLE_S = 10 * 60       # unchanged this long on an error screen -> nudge
IDLE_S = 45 * 60             # unchanged this long, nothing running in the background -> nudge
NUDGE_EVERY_S = 30 * 60      # at most one nudge per master per this long

ERROR_RE = re.compile(r"API Error|Connection error|ECONNRESET|ETIMEDOUT|ENOTFOUND|fetch failed|"
                      r"Request timed out|Unable to connect|network error|socket hang up|"
                      r"Overloaded|overloaded_error|hit your limit|usage limit", re.I)
BUSY_RE = re.compile(r"esc to interrupt|still running|shells? running|running in background|"
                     r"\d+ shells?\b|\bbackground task", re.I)
NO_SUB_S = 10 * 60           # a campaign with no sub running this long, local H200 available -> start a round
LOCAL_NUDGE = ("The local H200 is available and this campaign has had no sub running for over 10 minutes. "
               "Don't wait idle: start the next round now (a local-only round if today's official slots are "
               "spent; the sub iterates on scripts/local_bench.sh), and keep any reset wait running.")
NUDGE = ("The laptop slept and the network dropped for a while, so a step may have failed. "
         "Check your background tasks and the broker's results, redo whatever failed, and carry on "
         "with the campaign loop.")

reaping: set[str] = set()


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(broker.state_root() / "watchdog.log", "a") as f:
        f.write(line + "\n")


def online() -> bool:
    try:
        okc.problems()
        return True
    except Exception:
        return False


def reap(row: dict) -> None:
    try:
        r = broker.finish(row["problem"], row["job"], broker.pdir(row["problem"]) / "stage" / row["sha256"][:16],
                          child=row["child"], label=row["label"], sha=row["sha256"], kind=row["kind"])
        log(f"recorded {row['problem']} {row['job']} ({row['kind']}): {r.get('status')} {r.get('speedup')}")
    except Exception as e:
        log(f"could not record {row['problem']} {row['job']}: {e}")
    finally:
        reaping.discard(row["job"])


def subs_running() -> set[str]:
    """Campaign families (muse-bf16-proj, ...) with a sub session (claude --print) working in a round child."""
    out = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True).stdout
    return {m.group(1) for line in out.splitlines() if "--print" in line
            for m in re.finditer(r"ako4x-run-(muse-[a-z0-9-]+?)-r\d+", line)}


def local_available() -> bool:
    d = broker.state_root() / "sidecar"
    beat = d / "guard_heartbeat"
    return ((d / "pod.json").is_file() and not (d / "disabled").exists() and beat.is_file()
            and time.time() - float(beat.read_text() or 0) < 180)


def masters() -> list[str]:
    out = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True).stdout.split()
    return [s for s in out if s.startswith("okc-muse-")]


def main():
    seen: dict[str, tuple[str, float]] = {}   # session -> (screen hash, since)
    nudged: dict[str, float] = {}
    no_sub: dict[str, float] = {}        # session -> since when no sub has been running
    last = time.time()
    log("watchdog up")
    while True:
        time.sleep(TICK_S)
        now = time.time()
        if now - last > 5 * 60:
            log(f"woke after {(now - last) / 60:.0f} min (sleep or stall)")
            seen.clear()  # time asleep isn't idle time
        last = now
        if broker.stopped():
            continue
        if not online():
            continue

        for row in broker.inflight():
            if row["job"] not in reaping and now - row["ts"] > ORPHAN_AGE_S:
                reaping.add(row["job"])
                threading.Thread(target=reap, args=(row,), daemon=True).start()

        running, avail = subs_running(), local_available()
        for s in masters():
            fam = s[len("okc-"):]
            if fam in running or not avail:
                no_sub.pop(s, None)
            else:
                no_sub.setdefault(s, now)
                if now - no_sub[s] > NO_SUB_S and now - nudged.get(s, 0) > NUDGE_EVERY_S:
                    subprocess.run(["tmux", "send-keys", "-t", s, "C-u"])
                    subprocess.run(["tmux", "send-keys", "-t", s, "-l", LOCAL_NUDGE])
                    time.sleep(1)
                    subprocess.run(["tmux", "send-keys", "-t", s, "Enter"])
                    nudged[s] = now
                    log(f"nudged {s}: no sub running for {(now - no_sub[s]) / 60:.0f} min with the local H200 up")
                    continue
            screen = subprocess.run(["tmux", "capture-pane", "-p", "-t", s], capture_output=True, text=True).stdout
            h = hashlib.sha256(screen.encode()).hexdigest()
            if s not in seen or seen[s][0] != h:
                seen[s] = (h, now)
                continue
            still = now - seen[s][1]
            tail = "\n".join(l for l in screen.splitlines() if l.strip())[-3000:]
            error = bool(ERROR_RE.search(tail[-1500:]))
            busy = bool(BUSY_RE.search(tail))
            due = (error and still > ERROR_IDLE_S) or (not busy and still > IDLE_S)
            if due and now - nudged.get(s, 0) > NUDGE_EVERY_S:
                subprocess.run(["tmux", "send-keys", "-t", s, "-l", NUDGE])
                time.sleep(1)
                subprocess.run(["tmux", "send-keys", "-t", s, "Enter"])
                nudged[s] = now
                log(f"nudged {s} ({'error screen' if error else 'idle'} for {still / 60:.0f} min)")


if __name__ == "__main__":
    main()
