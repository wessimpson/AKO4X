"""The judge's per-kernel GPU-time profile of an evaluated submission. Free: it reads saved results.

`bash scripts/profile.sh`               the latest evaluated snapshot
`bash scripts/profile.sh --label iter-3` the latest snapshot whose label contains "iter-3"
`bash scripts/profile.sh --list`         every snapshot with its score

Each section is one scored shape: the CUDA kernels your kernel_fn launched under the judge's
profiler, their GPU time, share and call count. The judge profiles every passing run.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def snapshots() -> list[Path]:
    traj = PROJECT_ROOT / "trajectory"
    return sorted(p for p in traj.iterdir() if (p / "results.json").is_file()) if traj.is_dir() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    snaps = snapshots()
    if args.list:
        for s in snaps:
            r = json.loads((s / "results.json").read_text())
            score = f"{r['speedup']:.4f}x" if r.get("passed") else f"FAILED ({r.get('message', '')[:60]})"
            print(f"{s.name:<60} {r.get('kind', '?'):<5} {r.get('job')}  {score}")
        return
    picks = [s for s in snaps if args.label in s.name]
    if not picks:
        sys.exit("no evaluated snapshot" + (f" matching {args.label!r}" if args.label else "") + " under trajectory/")
    snap = picks[-1]
    prof = snap / "profile.txt"
    print(f"# {snap.name}")
    if not prof.is_file():
        sys.exit("no profile for this snapshot (the run failed before timing, so the judge didn't profile it)")
    print(prof.read_text())


if __name__ == "__main__":
    main()
