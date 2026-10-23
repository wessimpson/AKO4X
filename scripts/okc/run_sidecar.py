"""Local evaluation of solution/submission.py on the sidecar H200 (free: spends no okc slot).

`bash scripts/local_bench.sh --label "local-N description"`

The sidecar is a copy of the judge on a rented H200 SXM: the same tests, the same CUDA-graph timing
against SGLang with a cold L2, the same per-shape table. It is NOT the judge: its numbers carry a
calibration note (how far its speedups sat from the judge's on files both have timed), and only an
official `scripts/bench.sh` counts. It runs one timing at a time across every campaign, so a call can
queue; if the pod is off it starts it (up to ~25 minutes the first time). A finished evaluation is
snapshotted under trajectory/<timestamp>_LOCAL_<label>/.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:80] or "unlabeled"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)["okc"]
    submission = PROJECT_ROOT / "solution" / "submission.py"
    if not submission.is_file():
        sys.exit("solution/submission.py not found")
    out = PROJECT_ROOT / ".ako" / "last-local.json"
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    print(f"Sending solution/submission.py to the local H200 ({cfg['problem']}, label {args.label!r}); "
          "no okc slot is spent. Typically 2-10 minutes, more if other runs are queued.")
    proc = subprocess.run([cfg.get("python", "python3"), cfg["local"], "eval", "--problem", cfg["problem"],
                           "--file", str(submission), "--child", PROJECT_ROOT.name, "--label", args.label,
                           "--out", str(out)], text=True)
    if out.is_file():
        row = json.loads(out.read_text())
        if row.get("event") == "finished":
            snap = PROJECT_ROOT / "trajectory" / f"{datetime.now():%Y%m%d_%H%M%S}_LOCAL_{sanitize_label(args.label)}"
            snap.mkdir(parents=True)
            shutil.copy2(submission, snap / "submission.py")
            (snap / "results.json").write_text(json.dumps(row, indent=1))
            if row.get("profile_path") and Path(row["profile_path"]).is_file():
                shutil.copy2(row["profile_path"], snap / "profile.txt")
            print(f"Snapshot: {snap.relative_to(PROJECT_ROOT)}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
