
---

# OKC campaign overrides — Muse Spark Open Kernel Challenge

This campaign runs on the **okc backend**: the benchmark is the official Open Kernel Challenge judge (one H200 SXM, `lmsysorg/sglang:dev-muse-spark-open`), reached only through `okc_bridge/broker.py`. Everything above still holds **except where this section overrides it**; where they conflict, this section wins.

## What's different

- **Operator = problem.** One of `muse_bf16_proj`, `muse_attention_decode`, `muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer` (specs under `okc_problems/<problem>/`). `family` = the problem kebab-cased (`muse-moe-w4a16`). Lock `gpu=h200`, `backend=okc`, `mode=2` unless the user's prompt says otherwise: `master.init_campaign(family, gpu="h200", backend="okc", mode=2)`.
- **No dataset, no baseline file.** The score is measured against SGLang's kernel inside every judge run, so `reference/<family>/baseline.json` stays the Round-0 environment shell for the whole campaign. Ignore every baseline auto-promote / freshness instruction above.
- **The kernel is one file.** A child's kernel is `solution/submission.py` (`SubResult.final_kernel_path` points there). `spawn_child(problem, parent_kernel_path, label, family=family)` takes a variant's `kernel.py` as the parent; `parent_kernel_path=None` starts from the problem's starter.
- **Official evaluations are the scarce resource.** Each problem gets 20 checks + 10 scored runs per UTC day (reset 00:00 UTC = 5 pm PDT / 4 pm PST). The broker spends them in a fixed pattern — **check, check, scored run, repeating** — and caps each round at `[round] max_evals` in `okc_bridge/okc.toml`. A scored run goes on the public leaderboard; the board keeps the best one.
- **The record of truth is the broker's store**, not the sub's notes: `master.okc_round_results(problem, child_dir)` returns every evaluation the round spent (the exact code judged, kind, job id, speedup + 95% CI, per-load and per-shape results, judge flags, profile and log paths).
- **Library calls are allowed.** The competition lets submissions build on anything in the image (FlashInfer, DeepGEMM, CUTLASS, Triton, TileLang, CuTe DSL, cuBLAS via torch). **Skip the step-8 library-delegation check entirely**, and never put a "hand-written only" constraint into a phase-1 prompt. Calling SGLang's own kernel unchanged just scores 1.00x.
- **Never run `okc` yourself**, never call the challenge's HTTP API, never read or print the okc token, never log in or accept rules for anyone. Only `master.okc_*` helpers touch the judge.

## Round loop overrides

**Before step 1 (every round): budget.** `b = master.okc_budget(problem)`.
- `b["budget"]["next_kind"] is None` → no slots left today (or the track closed). Run `master.okc_wait_for_budget(problem)` **in the background** (it blocks until the reset, up to ~24 h) and start the round when it returns `READY`. `TRACK_CLOSED` → the kernel track is over (Nov 4, 11:59 pm PT): stop the campaign.
- `b["unaccounted_today"] > 0` → someone spent official runs outside the broker today. Note it in the round's ledger line; the schedule adapts by itself (it reads live counts).
- Don't spawn a round with fewer than 2 slots left today unless it's the last round before the track closes; wait for the reset instead (a round that can't evaluate is wasted).

**Step 3 (phase-1).** `master.run_sub_phase1(child_dir, prompt, timeout=14400)`, in the background. In the prompt, frame the direction (see "Phase-1 prompt construction"); the sub's `CLAUDE.md` already carries the competition rules, its evaluation cap and the slot pattern — don't restate them, but do carry forward any rules violation found in a previous round verbatim.

**Step 8 (archive) — replaces the artifact-source lookup, the library-delegation check and the baseline auto-promote:**

1. `r = master.okc_round_results(problem, child_dir)`. Candidates are `r["finished"]` rows with `passed` true, `status == "done"` and empty `flags`. None → no archive (ledger: `archived=F (no passing evaluation)`).
2. Pick the round's best by `speedup` (prefer the higher CI lower bound `ci[0]` when two are within each other's CI).
3. **Compliance gate (your judgment — the competition's rules, not a grep).** Read the chosen row's `code_path` in full against `okc_problems/RULES.md` and the problem's `okc_problems/<problem>/TASK.md`: CUDA-graph-safe (no host syncs, no value-dependent allocation/branching on the timed path); nothing cached or replayed from inputs (no `data_ptr()`-keyed caches, no output buffers reused as results across calls); weights only repacked or losslessly compressed, never re-quantized (NVFP4 stays NVFP4, BF16 stays BF16; router hi+lo split is the one exception); activations BF16 where SGLang keeps them, FP8 KV cache stays FP8, attention queries no coarser than FP8 E4M3; attention over every key (no top-k / sparse / block skipping); no timing or harness tampering, frames, env/secrets, module edits, processes, network, reference imports; no work moved outside the timed calls; correct and not wildly slower on nearby shapes (no shape tables that only cover the scored shapes with a slow or wrong fallback). Also read the sub's `ITERATIONS.md` for `RULES:` notes. Plus the silent-skip check from step 8 above: a speedup implausible for the change made, or a profile missing kernels you'd expect, is suspect.
   - **Violation** → no archive; ledger reason `rules-violation: <rule>`; append a one-line note to `reference/<family>/README.md` history; carry the constraint into the next phase-1 prompt verbatim. **Then withdraw every scored run of the violating code**: for each `r["finished"]` row with `kind == "run"` whose code breaks the rule (check each scored row's own `code_path`, not just the chosen one), `master.okc_withdraw(problem, row["job"], "<rule>")`, and record `withdrawn <job>` in the ledger line. The rules require this; don't wait for the user.
4. **Archive.** `master.archive_variant(family, name, Path(row["code_path"]), parent, spawn_meta, header_text, config_path=child_dir / "config.toml", result_json_path=<the matching trajectory snapshot's results.json, or write row to <child>/.ako/archive-result.json and pass that>)`. Compose the 5-section header per `templates/agent/lessons-convention.md` from the sub's `ITERATIONS.md` + git log (start Identity with `Mode-2 archive — header composed by master from ITERATIONS.md.`), and include the official result in Identity: job id, kind, speedup, CI, per-load speedups. Sub-1.00x variants are archived as forensic records, as above.
5. **Anchor rotation** by official speedup (rotate only if the new variant's speedup beats the anchor's; a brand-new family's anchor is SGLang at 1.00x).
6. **Promote.** `master.okc_promote_best(problem)` (in the background; it spends the next slot only when that slot is a scored run and the best checked file beats the best scored run and was never scored). Record what it did (`promoted <job> <speedup>` or its NOOP reason) in the ledger line.

**Step 9 ledger line (Mode 2):** `YYYY-MM-DD round-N <variant-or-no-change> (mode-2; okc; evals=<n> checks=<c> runs=<r>; best=<speedup>x <job>; archived=<T|F>; promote=<...>[; withdrawn <job>...][; unaccounted=<n>])`.

**Step 10:** commit as above. Never commit anything under `okc_bridge/` or the broker's state dir (`~/.ako4x-okc/`, outside the repo).

## Phase-1 prompt construction — okc notes

- Tell the sub which loads are dragging the geometric mean (from the parent's `result.json` `loads`), and which shapes sit far above `floor_us` — that's where the headroom is. Frame the algorithmic direction (fusion, split-K/stream-K for small-m decode, a different kernel family, weight repacking for coalesced/TMA loads, wgmma on Hopper), not micro-steps.
- The sub has only `max_evals` evaluations and no local GPU, so ask for one well-reviewed structural change per evaluation, not sweeps.
- The SKILL docs carry some Blackwell/B200 (tcgen05/tmem) material; the target is Hopper sm_90a. Each DSL skill now opens with a note saying so; don't steer subs toward Blackwell-only features.

## Cross-problem composition (overrides "Per-operator isolation is deliberate")

The five okc problems are nested, so per-operator isolation is **off** for okc campaigns. `muse_decode_layer` is the projections (`muse_bf16_proj`'s shapes) + `muse_attention_decode` + router + `muse_moe_w4a16` + norms and residuals; `muse_prefill_layer` is the same layer at prefill shapes (experts as one large grouped NVFP4 GEMM). A faster expert or projection kernel pays off on three boards.

- **Layer campaigns (`muse_decode_layer`, `muse_prefill_layer`) are integrators.** Every round's child gets `docs/components/<problem>.py`: the fastest officially passing, unflagged file of each component problem across all campaigns (from the broker's shared store; `spawn.py` `OKC_COMPONENTS`), with its per-load results in the `.json` beside it. Its `CLAUDE.md` tells the sub to compose first. Your phase-1 prompt should frame the round around (a) lifting in the components that beat what the layer runs for that piece at the layer's shapes, then (b) **fusing the glue** — the layer's profile is ~28 launches, with norms, residuals, casts, routing softmax/top-k and gating as separate small kernels costing ~2–5 µs each, on the order of the whole speed-of-light floor at batch 1 — into the big kernels' prologues/epilogues, toward a persistent per-layer kernel at small batch. Don't let a round settle for tweaking SGLang's pipeline at the edges.
- **Component campaigns (`muse_bf16_proj`, `muse_attention_decode`, `muse_moe_w4a16`) are kernel factories.** Their kernels get lifted into the layer problems, so ask subs to keep each kernel self-contained (one function per piece, shape-general within TASK.md's bounds). The structural direction for every shape with m ≤ 32 (all decode/verify loads): **weight-streaming kernels that read each weight byte once at ~80–90% of 4.8 TB/s across all 132 SMs** (split-K / stream-K, many CTAs per weight matrix), not cuBLAS/Marlin tiles built for large M. For MoE, also: routing is skewed (a few experts take most tokens) — schedule work per expert by its token count instead of uniformly; fuse gate/up with SiLU; fold the post-expert norm and weighted sum into GEMM2. For attention: fuse QK-norm/RoPE/FP8-query prologue and the gate/norm epilogue into the attention kernel; split-KV to fill the SMs at small batch.
- When a component campaign's anchor jumps, the next layer round picks it up automatically (spawn-time copy). Note sibling anchors you relied on in the ledger line (`components=<problem>:<job>,...`).

## The local H200 (sidecar)

Subs now have `bash scripts/local_bench.sh` (okc_bridge/local.py): a copy of the judge on a rented H200 SXM (sidecar/ in the sidecar-judge worktree), free of okc slots, run on demand by `okc_bridge/pod_guard.py` under a spend cap. Their `CLAUDE.md` tells them to iterate there and send only locally proven files to the judge.

- In phase-1 prompts, expect and ask for local iteration: sweeps of tile and pipeline configurations, edge-shape tests, A/B comparisons between variants, measurement before choosing a direction. One official evaluation per *locally proven* structural change, not per guess.
- The record of truth is still the judge. Step 8 archives, ranks and promotes from `okc_round_results` only; local results (`python3 okc_bridge/local.py results --problem P --child C`) are context, and their output carries a calibration note (judge/sidecar speedup ratio on files both have timed). Never archive, promote or argue compliance from a local number.
- `LOCAL_UNAVAILABLE` (pod off, budget spent, guard down) means the round runs as before.
