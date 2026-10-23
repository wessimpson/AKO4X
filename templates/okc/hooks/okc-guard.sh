#!/bin/bash
# PreToolUse hook (Bash): official evaluations go through scripts/bench.sh only.
# Blocks running the okc CLI, reading its token, or calling the challenge's HTTP API
# directly, so the daily budget, the check/check/run schedule and the rules lint can't be
# bypassed. Exit 2 blocks the call and shows stderr to the agent.
#
# SMOKE TEST:
#   echo '{"tool_input":{"command":"okc check solution/submission.py"}}' | $0; echo "exit=$?"   # 2
#   echo '{"tool_input":{"command":"bash scripts/bench.sh --label x"}}' | $0; echo "exit=$?"  # 0
set -u

COMMAND=$(python3 -c "
import json, sys
try:
    print((json.loads(sys.stdin.read()).get('tool_input') or {}).get('command', ''))
except Exception:
    print('')
" 2>/dev/null)

if python3 - "$COMMAND" <<'EOF'
import re, sys
cmd = sys.argv[1]
patterns = [
    r"(^|[;&|(`]|\$\(|\b(xargs|exec|env|nohup|time|sudo|then|do)\s)\s*(\S*/)?okc(\s|$)",  # the okc binary as a command
    r"\.okc/",                                   # its install dir
    r"openkernelchallenge\.dev/api",             # the HTTP API
    r"OKC_TOKEN",                                # its token
    r"\bapi\s+token\b",                          # okc api token
    r"/okc/config\.json",                        # its saved token
    r"runpod_key|RUNPOD_API_KEY|pod_guard\.py",   # the RunPod credit: only the parent's pod guard runs the pod
    r"broker\.py",                               # the parent-side broker, other than via bench.sh
]
sys.exit(0 if any(re.search(p, cmd) for p in patterns) else 1)
EOF
then
    echo "Blocked: official okc evaluations go through 'bash scripts/bench.sh --label ...' only (it applies the daily budget, the check/run schedule and the rules lint). Use scripts/lint.sh, scripts/budget.sh and scripts/profile.sh for the free tools." >&2
    exit 2
fi
exit 0
