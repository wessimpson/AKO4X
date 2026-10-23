"""Official evaluation of solution/submission.py on the Open Kernel Challenge judge.

`bash scripts/bench.sh --label "iter-N description"`

Every call that reaches the judge spends one of today's official evaluations for this
problem (an `okc check`, or a scored `okc submit` that goes on the public board; the
broker picks which). Files that fail the static rules check, and files the judge has
already seen, spend nothing. A finished or cached evaluation is snapshotted under
trajectory/<timestamp>_<label>/ (submission.py, results.json, profile.txt, judge.log,
config.toml).
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


def load_okc_config() -> dict:
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)["okc"]


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:80] or "unlabeled"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True,
                        help="Iteration label; every official evaluation is logged under it")
    args = parser.parse_args()

    cfg = load_okc_config()
    submission = PROJECT_ROOT / "solution" / "submission.py"
    if not submission.is_file():
        sys.exit("solution/submission.py not found")

    ako = PROJECT_ROOT / ".ako"
    ako.mkdir(exist_ok=True)
    out = ako / "last-eval.json"
    out.unlink(missing_ok=True)
    print(f"Sending solution/submission.py to the okc broker ({cfg['problem']}, label {args.label!r})...")
    print("Waiting for the judge (typically 1-5 minutes; longer if its queue is busy).")
    proc = subprocess.run(
        [cfg.get("python", "python3"), cfg["broker"], "eval",
         "--problem", cfg["problem"], "--file", str(submission),
         "--child", PROJECT_ROOT.name, "--label", args.label, "--out", str(out)],
        text=True,
    )
    if not out.is_file():
        sys.exit(proc.returncode or 1)
    row = json.loads(out.read_text())
    if row.get("event") == "finished":
        snap = PROJECT_ROOT / "trajectory" / f"{datetime.now():%Y%m%d_%H%M%S}_{sanitize_label(args.label)}"
        snap.mkdir(parents=True)
        shutil.copy2(submission, snap / "submission.py")
        shutil.copy2(PROJECT_ROOT / "config.toml", snap / "config.toml")
        (snap / "results.json").write_text(json.dumps(row, indent=1))
        for key, name in (("profile_path", "profile.txt"), ("log_path", "judge.log")):
            if row.get(key) and Path(row[key]).is_file():
                shutil.copy2(row[key], snap / name)
        print(f"Snapshot: {snap.relative_to(PROJECT_ROOT)}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
