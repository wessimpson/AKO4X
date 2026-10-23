# Benchmark reference — okc judge

## One evaluation

`bash scripts/bench.sh --label L` sends `solution/submission.py` (one file) through a broker to the official judge, which runs it on one H200 SXM in the `lmsysorg/sglang:dev-muse-spark-open` image:

1. **Static checks.** The judge rejects files that patch timing, read harness frames, touch harness I/O or secrets, edit loaded modules, import the reference, start processes or use the network (status `rejected`, or `judge static checks rejected: ...`). It may also *flag* suspicious code without rejecting it (`JUDGE FLAGS`); flagged results are never archived or promoted.
2. **Setup.** `prepare(weights)` once, untimed, up to 300 s (problems with weights).
3. **Tests.** The test cases listed in `docs/TASK.md` (some harder than the scored shapes: odd m, outlier distributions, larger sizes). The first failure ends the run: `FAILED: test {...}: <why>`.
4. **Timing.** Each scored shape is captured as a CUDA graph and replayed, alternating with SGLang's kernel on the same fresh inputs, each replay after an L2 flush (cold L2). Every timed call is also checked against the reference.
5. **Profile.** A passing run is profiled per shape: kernel names, GPU time, share, calls (`bash scripts/profile.sh`).

A whole run has 10 minutes, compiling included (`load_inline`/Triton JIT time counts toward it, not toward the score). Past that it's charged as a timeout.

## Scoring

- A shape's speedup is the **median of paired ratios** (SGLang time / yours, repetitions taken in swapped-order pairs). Your time on the shape = SGLang's median / that speedup.
- A load's time is the sum over its shapes of time × weight. The problem speedup is the **geometric mean over loads** of SGLang's load time / yours, so every load counts equally — a 2x win on one load and a 0.8x loss on another is roughly a wash.
- The judge subtracts a graph replay's fixed cost (~4 µs) from both sides.
- Each run reports a **95% CI** (at least ±1.5%). A change whose gain sits inside the CI of its parent's result may be noise. There is no A/B or variance mode here: the CI is the noise estimate, and two runs of the same file return the cached result.
- The bench output's `floor_us` per shape is the speed-of-light time (data at 4.8 TB/s or math at 989 TFLOP/s). `yours_us / floor_us` shows how much headroom a shape has; launch- and latency-bound small shapes sit far above it.
- The leaderboard row is your **best scored run**; slower scored runs don't replace it.

## Correctness

`docs/checks.py` `compare`: relative L2 error of the whole output against an FP64 reference within the problem's bound (see `docs/TASK.md`), every row within 5× the bound (near-zero rows measured against a tenth of the average row), no NaN/Inf, exact shape and dtype. Some problems add special rows (e.g. single-nonzero-entry rows that must be exact to 0.1%). Fresh inputs every call: a kernel that returns stale data fails the checks on the next call.

## Statuses

Judge: `done` (finished; `passed` says whether tests and timing passed), `failed` (an error, a failing test, a timeout), `rejected` (static checks), `duplicate` (the judge has this exact file already).
Broker (nothing was spent): `LINT_REJECTED` (the static rules check in `scripts/lint.sh` found an error), `CACHED` (same file as an earlier job — its stored result is shown), `ROUND_CAP` (this session's evaluations are spent), `BUDGET_EXHAUSTED` (today's are), `TRACK_CLOSED`, `ERROR` (okc itself failed: say so in your notes and stop if it repeats).

## Silent-skip / stale-output cascade

Under CUDA-graph replay, a kernel launched on the wrong stream (e.g. a raw `<<<>>>` launch without the current stream, a DSL launch that doesn't enter capture) is simply not in the graph. With fresh inputs every call the judge's per-call checks should catch the stale output, but a partially skipped pipeline can still pass on some shapes and post an implausible speedup. If a result looks too good for the change you made, check the profile: the kernels you expect must appear with plausible GPU time. Launch on `torch.cuda.current_stream()` always.

## Frozen for bench comparability

The judge, its scoring, its tests and tolerances, the broker (`check, check, run` schedule, budget, round cap) and `scripts/run_okc.py`'s path to it are outside your workspace's control and outside any proposal's scope. Everything else (skills, the free tools' output format) is mutable.
