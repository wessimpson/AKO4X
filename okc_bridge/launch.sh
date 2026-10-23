#!/bin/bash
# Start one closed-loop AKO4X campaign per Open Kernel Challenge problem, each a master
# Claude Code session in its own tmux session and git worktree.
#
#   bash okc_bridge/launch.sh muse_moe_w4a16 muse_bf16_proj ...    # start (or re-attach) these
#   bash okc_bridge/launch.sh --all                                # all five problems
#   tmux attach -t okc-muse-moe-w4a16                              # watch one (detach: C-b d)
#
# Per problem: a worktree at $AKO_OKC_WORKTREES/<problem> (default ~/ako4x-okc/<problem>),
# detached at the okc branch, turned into a campaign branch by scripts/campaign_start.py;
# round child envs land beside the worktrees. The broker's state and settings
# ($AKO_OKC_STATE, default ~/.ako4x-okc; okc.toml there) are shared by every campaign.
# Idempotent: an existing worktree or tmux session is left as it is.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WT_ROOT="${AKO_OKC_WORKTREES:-$HOME/ako4x-okc}"
STATE="${AKO_OKC_STATE:-$HOME/.ako4x-okc}"
ALL=(muse_bf16_proj muse_attention_decode muse_moe_w4a16 muse_decode_layer muse_prefill_layer)

if [ "${1:-}" = "--all" ]; then set -- "${ALL[@]}"; fi
if [ $# -eq 0 ]; then
    echo "usage: $0 <problem>... | --all   (problems: ${ALL[*]})" >&2
    exit 1
fi
command -v tmux >/dev/null || { echo "tmux not found" >&2; exit 1; }
command -v claude >/dev/null || { echo "claude not found" >&2; exit 1; }
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY is set: claude would bill the API instead of your subscription. Unset it first." >&2
    exit 1
fi

mkdir -p "$STATE" "$WT_ROOT"
if [ -f "$STATE/STOP" ]; then
    rm "$STATE/STOP"
    echo "removed $STATE/STOP (left by stop_at.py): official evaluations are allowed again"
fi
[ -f "$STATE/okc.toml" ] || cp "$REPO/okc_bridge/okc.toml" "$STATE/okc.toml"

for P in "$@"; do
    [ -d "$REPO/okc_problems/$P" ] || { echo "unknown problem $P" >&2; exit 1; }
    FAM="${P//_/-}"
    WT="$WT_ROOT/$P"
    if [ ! -d "$WT" ]; then
        git -C "$REPO" worktree add --detach "$WT" okc >/dev/null
        (cd "$WT" && python3 scripts/campaign_start.py --operator "$P" --gpu h200 --backend okc \
            --mode 2 --base okc | sed -n '/Branch:/p')
    fi

    # Master permissions: run unattended, but never touch okc directly.
    mkdir -p "$WT/.claude/hooks"
    cp "$REPO/templates/okc/hooks/okc-guard.sh" "$WT/.claude/hooks/okc-guard.sh"
    chmod +x "$WT/.claude/hooks/okc-guard.sh"
    cat > "$WT/.claude/settings.local.json" <<'JSON'
{
  "permissions": {
    "allow": ["Bash(*)", "Edit(*)", "Read(*)", "Write(*)", "Glob(*)", "Grep(*)", "Agent(*)",
              "Skill(*)", "WebFetch(*)", "WebSearch(*)"],
    "deny": ["Bash(okc:*)", "Bash(okc *)", "Bash(~/.okc/bin/okc:*)", "Read(~/.okc/**)",
             "Read(~/.config/okc/**)", "Read(~/Library/Application Support/okc/**)"]
  },
  "hooks": {
    "PreToolUse": [{"matcher": "Bash",
                    "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/okc-guard.sh"}]}]
  }
}
JSON

    SESSION="okc-$FAM"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "$SESSION already running (tmux attach -t $SESSION)"
        continue
    fi
    PROMPT="Run a Mode-2 campaign on family=$FAM (operator/problem $P), gpu=h200, backend=okc, mode=2. Follow CLAUDE.md: MASTER.md plus the OKC campaign overrides at its end, which win where they conflict. Keep running rounds back to back until the kernel track closes or I stop you. When today's official evaluations for $P are spent, wait for the 00:00 UTC reset in the background and carry on. Never run okc yourself."
    # Don't let a parent Claude Code session's env (when launched from one) leak into the master.
    UNSETS=$(env | grep -oE '^(CLAUDE[A-Z0-9_]*|CLAUDECODE|AI_AGENT)=' | sed 's/=$//; s/^/-u /' | tr '\n' ' ')
    tmux new-session -d -s "$SESSION" -c "$WT" -x 220 -y 50
    tmux send-keys -t "$SESSION" "env $UNSETS claude $(printf '%q' "$PROMPT")" Enter
    echo "started $SESSION in $WT (tmux attach -t $SESSION)"
done

# Helpers shared by every campaign: the watchdog (records orphaned judge jobs, nudges stalled masters)
# and the pod guard (runs the local H200 on demand, idle-stops it, enforces [sidecar] budget_usd).
for helper in watchdog pod_guard; do
    SESSION="okc-${helper/_/}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "$SESSION already running"
    else
        tmux new-session -d -s "$SESSION" "cd '$REPO' && python3 okc_bridge/$helper.py run 2>&1 | tee -a '$STATE/$helper.out'"
        echo "started $SESSION"
    fi
done
