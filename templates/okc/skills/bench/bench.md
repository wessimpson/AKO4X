# Bench reference

## Spending evaluations well

- **Budget.** This session has a hard cap on official evaluations (the task file says how many; `scripts/budget.sh` shows what's left). The day's slots per problem run check, check, scored run, check, check, scored run, … and are shared with every other session on this problem.
- **Scored runs are public.** The slot after two checks is an `okc submit`: its result goes on the leaderboard under the user's name (the board keeps the best one). Bench only code you'd be comfortable having reviewed by the organizers.
- **Batch your hypotheses into one careful change.** A failed run teaches you only its first failing case. Before benching: lint, re-read against `docs/reference.py`, walk the test shapes in `docs/TASK.md` (odd and large m, outlier distributions), check graph-capturability and dtypes, and make sure compile time stays well inside the 10-minute run limit.
- **Cheap diagnostics ride along.** `print()` outside `kernel_fn` (e.g. in `prepare`, or at import: chosen tile configs, compile times) comes back in the snapshot's `judge.log`. Never print tensor values inside `kernel_fn` (host sync).
- **Duplicates are free.** Re-benching an unchanged file returns the stored result marked `cached`. A trivial edit (a comment) makes it a new file and spends a slot — don't.

## Reading one result

- `SCORE` is the problem speedup vs SGLang with its 95% CI. Compare to the parent's result (the prior variant's `results.json`, or your previous iteration): a delta inside both CIs is not established.
- `loads:` is the per-load speedup; the score is their geometric mean. Find the load dragging the mean down before tuning one that already wins.
- The per-shape table gives yours vs SGLang's µs and the speed-of-light `floor_us`. Shapes far above the floor are where the headroom is; shapes near it are done.
- Then `bash scripts/profile.sh --label iter-N`: which kernels ran per shape, their GPU time and share. Gaps between the sum of kernel time and `yours_us` are launch overhead and idle time (fusion and fewer launches help); one dominant kernel far above the floor is a kernel-quality problem (tiling, occupancy, memory access).

## Snapshots

Each finished (or cached) evaluation saves `trajectory/<timestamp>_<label>/` with `submission.py`, `results.json` (everything the judge returned, flattened), `profile.txt`, `judge.log` and `config.toml`. `git log` plus `ITERATIONS.md` are your history; the master reads all three.
