---
name: benchmark
description: Reference for the active benchmark — the Open Kernel Challenge judge (okc) on an H200. Covers how a run is tested and timed (CUDA-graph replays, cold L2, fresh inputs, paired ratios vs SGLang), the scoring formula (geomean over loads of weighted shape times), the 95% CI, result statuses (done / failed / rejected / duplicate, plus the broker's LINT_REJECTED / CACHED / ROUND_CAP / BUDGET_EXHAUSTED), judge flags, correctness bounds, and what's frozen. Invoke whenever you decode a bench result, wonder whether a gain is real, or reason about why a run failed — do NOT guess. How to drive bench.sh lives in the `bench` skill.
---

# Benchmark — the Open Kernel Challenge judge

What `bash scripts/bench.sh` is a frontend for. Detailed body: `benchmark.md`. `docs/TASK.md` is the per-problem authority (shapes, weights, tests, tolerances); this skill is the cross-problem mechanics.

When this skill applies:
- Decoding a status, a failure message or a judge flag.
- Deciding whether a speedup is real or inside the noise (the CI).
- Reasoning about why correctness passed but a gain looks implausible (cached outputs, skipped kernels).
- Checking what's frozen before proposing a harness edit.
