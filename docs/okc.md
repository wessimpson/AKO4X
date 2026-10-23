# AKO4X on the Open Kernel Challenge (okc backend)

This fork points AKO4X's closed loop at the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/): five kernel problems, each judged on one H200 against SGLang's own kernel. The benchmark is the official judge, reached through `okc` — so there's no dataset, no local GPU, and the scarce resource is the daily cap of official evaluations (per problem: 20 checks + 10 scored runs, reset at 00:00 UTC).

## Quick start

```bash
# once: okc installed and `okc login` done by a person (it accepts the rules)
git checkout okc
bash okc_bridge/launch.sh muse_moe_w4a16          # or several problems, or --all
tmux attach -t okc-muse-moe-w4a16                 # watch the master (detach: C-b d)
```

Each problem gets a git worktree (`~/ako4x-okc/<problem>`, a `campaign/<problem>/<ts>` branch made by `scripts/campaign_start.py --backend okc --base okc`) and a master Claude Code session in tmux. The master spawns a child env per round (`~/ako4x-okc/ako4x-run-<family>-r<N>`), drives a sub agent through it, archives the best officially judged kernel under `reference/<family>/`, and loops. When the day's evaluations are spent it waits for the reset in the background.

## Where the rules are enforced

| Rule | Where |
|---|---|
| Daily caps (10 runs + 20 checks per problem) | `okc_bridge/broker.py` reads live limits from `okc problems` before every evaluation |
| check, check, scored run, repeating | `broker.choose_kind`, derived from today's used counts (survives restarts and manual runs) |
| Rate limits (6/min per token) | the broker's global lock + `[rate] min_spacing_s` |
| Static rules (CUDA graphs, precision, caching, harness access, ...) | `okc_bridge/rules.py` lint before anything is spent; the sub's `CLAUDE.md` (`templates/okc/task.md`); the master's step-8 compliance gate (`master/MASTER-OKC.md`) |
| Withdraw reward-hacky runs | MASTER-OKC step 8: the master withdraws scored runs of code that breaks a rule (`master.okc_withdraw`) |
| Agents don't accept rules / sign in | nothing in the harness logs in; the sub can't run okc at all |
| Track closes Nov 4, 11:59 pm PT | `[track] closes`; the broker answers `TRACK_CLOSED` after it |

Only the broker talks to okc. In child envs a PreToolUse hook (`templates/okc/hooks/okc-guard.sh`) and permission deny rules block the okc CLI, its token and the HTTP API, and the sub's only path to the judge is `bash scripts/bench.sh --label ...`. That's soft isolation (same user account), backed up by `unaccounted_today` in `okc_budget()`, which counts official runs today that the broker didn't start.

## Settings

`~/.ako4x-okc/okc.toml` (copied from `okc_bridge/okc.toml` on first launch; shared by every campaign): `checks_per_run` (2 → C,C,R), reserves, `[round] max_evals` (official evaluations per round, default 6), rate spacing, track close time.

## What the sub gets

`solution/submission.py` (one file), `docs/` (TASK.md, reference.py, checks.py, baseline.py, nvfp4.py, problem.json, rules.md), `scripts/bench.sh` (one official evaluation; snapshots to `trajectory/`), and free tools: `lint.sh`, `budget.sh`, `profile.sh` (the judge's per-kernel profile — there's no NCU without a GPU). Skills: AKO4X's DSL skills (each prefixed with an okc/Hopper note), with `benchmark`, `bench`, `profiler-ncu` and `sanitizer` replaced by okc versions and a new `okc-submission` skill.

## State

`~/.ako4x-okc/<problem>/`: `evals.jsonl` (every started job, finished evaluation and gate decision), `code/<sha>.py` (every file judged), `jobs/<job>.json` (okc's raw JSON), `profiles/<job>.{txt,log}`.

## Known limits

- A scored run of a file that was already checked may come back from the judge as a free `duplicate` (not scored). `okc_promote_best` detects that and stops trying for that file.
- The DSL skills' empirical notes are from B200; the okc note at the top of each says so.
- Nested `claude --print` sessions use whatever model and subscription the local `claude` is logged in with. `launch.sh` refuses to run with `ANTHROPIC_API_KEY` set.
