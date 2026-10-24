#!/usr/bin/env python3
"""Run the sidecar's RunPod H200 for the campaigns: on demand, idle-stopped, under a spend cap.

  pod_guard.py run                 # the loop (its own tmux session, okc-podguard)
  pod_guard.py up                  # start the pod now and wait for SSH
  pod_guard.py stop                # stop the pod (GPU billing stops; the network volume stays)
  pod_guard.py adopt POD [--volume V]   # take over a pod made with sidecar.runpod.pod create
  pod_guard.py spend               # what the guard has counted so far

This is the only process that reads the RunPod key: $RUNPOD_API_KEY, else ~/.config/sidecar/runpod_key.
Local benches (local.py) ask for the pod by touching <state>/sidecar/want and read its address from
<state>/sidecar/ssh, which exists only while the pod is up.

Every 30 s the loop:
- every 5 minutes reads the account's real balance (GraphQL) into <state>/sidecar/balance.json, and stops the pod
  for good when it reaches [sidecar] min_balance_usd;
- counts spend as a fallback: costPerHr × running time, into <state>/sidecar/spend.json;
- at [sidecar] budget_usd stops the pod for good: writes <state>/sidecar/disabled, and local benches answer
  LOCAL_UNAVAILABLE (delete the file and raise the budget to resume);
- starts the pod when a bench asked for it, and stops it after [sidecar] idle_stop_min minutes with no
  local bench running or asked for.
A pod that comes back without its GPU (someone else took the machine) is replaced on the same volume.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import broker  # noqa: E402
import local  # noqa: E402

KEY_FILE = Path.home() / ".config" / "sidecar" / "runpod_key"
TICK_S = 30


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(local.sdir() / "pod_guard.log", "a") as f:
        f.write(line + "\n")


def rp(cfg):
    """The sidecar's RunPod module, with the key in this process's environment only."""
    if not os.environ.get("RUNPOD_API_KEY"):
        if not KEY_FILE.is_file():
            raise SystemExit(f"no RunPod key: set RUNPOD_API_KEY or write it to {KEY_FILE} (chmod 600)")
        os.environ["RUNPOD_API_KEY"] = KEY_FILE.read_text().strip()
    repo = local.scfg(cfg)["repo"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sidecar.runpod import pod  # noqa: E402
    return pod


def state() -> dict:
    p = local.sdir() / "pod.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def save_state(st: dict) -> None:
    (local.sdir() / "pod.json").write_text(json.dumps(st, indent=1))
    (local.sdir() / "pod").write_text(st.get("id", ""))


def spend() -> dict:
    p = local.sdir() / "spend.json"
    return json.loads(p.read_text()) if p.is_file() else {"usd": 0.0, "running_s": 0.0}


def add_spend(dt_s: float, per_hr: float) -> dict:
    s = spend()
    s["usd"] = round(s["usd"] + per_hr * dt_s / 3600, 4)
    s["running_s"] = round(s["running_s"] + dt_s, 1)
    s["per_hr"] = per_hr
    (local.sdir() / "spend.json").write_text(json.dumps(s))
    return s


def balance() -> dict | None:
    """The account's real balance and burn rate (RunPod GraphQL, read-only)."""
    import urllib.request
    req = urllib.request.Request("https://api.runpod.io/graphql", method="POST",
                                 data=json.dumps({"query": "query { myself { clientBalance currentSpendPerHr } }"}).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                          "Content-Type": "application/json", "User-Agent": "okc-sidecar/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            me = json.loads(r.read())["data"]["myself"]
        out = {"balance_usd": float(me["clientBalance"]), "spend_per_hr": float(me["currentSpendPerHr"]), "ts": time.time()}
        (local.sdir() / "balance.json").write_text(json.dumps(out))
        return out
    except Exception as e:  # noqa: BLE001 -- the estimate is the fallback
        log(f"balance query failed: {e}")
        return None


def get_pod(pod, pid: str) -> dict | None:
    try:
        return pod.call("GET", f"/pods/{pid}")
    except (Exception, SystemExit) as e:
        log(f"status of {pid} failed: {e}")
        return None


def running(p: dict | None) -> bool:
    return bool(p) and p.get("desiredStatus") == "RUNNING"


def drop_ssh() -> None:
    (local.sdir() / "ssh").unlink(missing_ok=True)


def ssh_ok(addr: str) -> bool:
    host, *rest = addr.split()
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                        "-o", "ConnectTimeout=10", *rest, host,
                        "test -d /workspace/meta-muse-challenge/sidecar && nvidia-smi -L"],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0 and "H200" in r.stdout


# SSH sessions don't inherit the container's environment, so the image's Python (/opt/sglang/bin, with
# torch) isn't on PATH. Save PID 1's environment and source it first thing in .bashrc, which bash reads
# for ssh commands too (before Ubuntu's non-interactive early return). /root resets on a restart: rerun.
ENV_FIX = (r"""tr '\0' '\n' < /proc/1/environ | grep -E '^(PATH|LD_LIBRARY_PATH|CUDA_HOME|SIDECAR_[A-Z_]+)=' """
           r"""| sed 's/^/export /; s/=\(.*\)$/="\1"/' > /root/.sidecar_env && """
           r"""echo 'export SIDECAR_LOCK_DIR=/workspace/.sidecar-locks' >> /root/.sidecar_env && """
           r"""(grep -q sidecar_env /root/.bashrc || sed -i '1i . /root/.sidecar_env' /root/.bashrc)""")


def fix_env(addr: str) -> bool:
    host, *rest = addr.split()
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", *rest, host, ENV_FIX + " && . /root/.sidecar_env && python3 -c 'import torch'"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        log(f"environment fix failed: {r.stderr[-500:]}")
    return r.returncode == 0


def sync_repo(cfg, addr: str) -> bool:
    r = subprocess.run(["sh", "sidecar/runpod/sync.sh", "push"], cwd=local.scfg(cfg)["repo"],
                       env=dict(os.environ, SIDECAR_SSH=addr), capture_output=True, text=True, timeout=900)
    if r.returncode:
        log(f"sync failed: {r.stderr[-800:]}")
    return r.returncode == 0


def bring_up(cfg, pod) -> bool:
    st = state()
    if not st.get("id"):
        log("no pod adopted yet (pod_guard.py adopt POD)")
        return False
    p = get_pod(pod, st["id"])
    if p is None:
        return False
    if not running(p):
        log(f"starting pod {st['id']}")
        try:
            pod.call("POST", f"/pods/{st['id']}/start")
        except (Exception, SystemExit) as e:
            log(f"start failed: {e}")
            return False
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        p = get_pod(pod, st["id"])
        addr = pod.ssh_string(p) if p else None
        if addr and ssh_ok(addr) and fix_env(addr):
            if not st.get("synced") or st.get("synced_addr") != addr:
                if not sync_repo(cfg, addr):
                    return False
                st.update(synced=time.time(), synced_addr=addr)
                save_state(st)
            (local.sdir() / "ssh").write_text(addr)
            log(f"pod {st['id']} up: {addr} (${(p or {}).get('costPerHr')}/h)")
            return True
        if p and running(p) and (p.get("gpu") or {}).get("count", 1) == 0:
            log("pod came back without a GPU; it needs replacing on the same volume (pod_guard.py adopt)")
            return False
        time.sleep(15)
    log("pod didn't come up within 20 minutes")
    return False


def stop_pod(cfg, pod, why: str) -> None:
    drop_ssh()
    st = state()
    if st.get("id"):
        try:
            pod.call("POST", f"/pods/{st['id']}/stop")
            log(f"stopped pod {st['id']}: {why}")
        except (Exception, SystemExit) as e:
            log(f"stop failed ({why}): {e}")


def loop(cfg) -> None:
    pod = rp(cfg)
    s = local.scfg(cfg)
    log(f"pod guard up: budget ${s['budget_usd']}, idle stop {s['idle_stop_min']} min")
    last, last_bal = time.time(), 0.0
    while True:
        (local.sdir() / "guard_heartbeat").write_text(str(time.time()))
        time.sleep(TICK_S)
        (local.sdir() / "guard_heartbeat").write_text(str(time.time()))
        now, dt_s = time.time(), time.time() - last
        last = now
        cfg = broker.load_config()
        s = local.scfg(cfg)
        st = state()
        p = get_pod(pod, st["id"]) if st.get("id") else None
        if now - last_bal > 300:
            last_bal = now
            bal = balance()
            floor = float(s.get("min_balance_usd", 25.0))
            if bal and bal["balance_usd"] <= floor:
                stop_pod(cfg, pod, f"RunPod balance ${bal['balance_usd']:.2f} reached the ${floor:.0f} floor")
                (local.sdir() / "disabled").write_text(f"RunPod balance ${bal['balance_usd']:.2f} reached the ${floor:.0f} floor")
                continue
        if running(p):
            # The pod bills in the cloud while this laptop sleeps, and nothing here can stop it then, so a pod
            # that is running now has been running the whole gap: count all of it (a cap would undercount).
            sp = add_spend(dt_s, float(p.get("costPerHr") or 0))
            if sp["usd"] >= s["budget_usd"]:
                stop_pod(cfg, pod, f"spend ${sp['usd']:.2f} reached the ${s['budget_usd']} budget")
                (local.sdir() / "disabled").write_text(f"spend ${sp['usd']:.2f} reached the ${s['budget_usd']} budget")
                continue
        elif p is not None and (local.sdir() / "ssh").exists():  # None = the status call failed (network): keep the address
            drop_ssh()
            log("pod isn't running; local benches will ask for it again")
        if (local.sdir() / "disabled").exists():
            if running(p):
                stop_pod(cfg, pod, "disabled")
            continue
        busy = list((local.sdir() / "busy").glob("*")) if (local.sdir() / "busy").is_dir() else []
        want = local.sdir() / "want"
        asked = want.is_file() and now - float(want.read_text() or 0) < 120
        used = local.sdir() / "last_used"
        idle = now - (float(used.read_text()) if used.is_file() else 0)
        if asked and not (local.sdir() / "ssh").exists():
            bring_up(cfg, pod)
        elif running(p) and not busy and not asked and idle > s["idle_stop_min"] * 60:
            stop_pod(cfg, pod, f"idle {idle / 60:.0f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("up")
    sub.add_parser("stop")
    sub.add_parser("spend")
    ad = sub.add_parser("adopt")
    ad.add_argument("pod")
    ad.add_argument("--volume")
    a = ap.parse_args()
    cfg = broker.load_config()
    if a.cmd == "spend":
        print(json.dumps(spend(), indent=1))
        return
    pod = rp(cfg)
    if a.cmd == "adopt":
        save_state({"id": a.pod, "volume": a.volume})
        print(f"adopted pod {a.pod}")
    elif a.cmd == "up":
        sys.exit(0 if bring_up(cfg, pod) else 1)
    elif a.cmd == "stop":
        stop_pod(cfg, pod, "requested")
    elif a.cmd == "run":
        loop(cfg)


if __name__ == "__main__":
    main()
