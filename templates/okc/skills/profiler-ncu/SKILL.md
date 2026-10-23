---
name: profiler-ncu
description: Profiling in this workspace — there is no local GPU and no Nsight Compute; the profiler is the judge's own per-shape kernel profile (kernel names, GPU time, share, call count) saved with every passing evaluation, read with `bash scripts/profile.sh`. Covers how to turn it into a bottleneck diagnosis (launch overhead vs kernel time vs speed-of-light floor), what NCU-style metrics you can and can't infer, and how to get extra signal through judge.log prints. Use after every passing bench and before choosing the next change.
---

# Profiling — the judge's kernel profile

No `ncu` here: nothing in this workspace can run on a GPU. Each passing official run comes back with a per-shape profile, and `bash scripts/profile.sh [--label L]` prints it:

```
## {"layer": "qkv_gate", "load": "decode 1", "m": 1}
kernel                                     GPU time  share  calls
nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT   9.3 µs  84.6%      1
void cublasLt::splitKreduce_kernel<...>          1.7 µs  15.4%      1
total                                           11.0 µs
```

Reading it:

- **Kernel total vs `yours_us`** (bench table): a large gap is launch latency / idle between kernels — fuse, cut launches, use PDL (programmatic dependent launch) or a persistent kernel.
- **Kernel time vs `floor_us`**: a kernel well above the speed-of-light floor at a memory-bound shape is under-using bandwidth (too few CTAs for 132 SMs → split-K / stream-K; uncoalesced or narrow loads; no TMA / cp.async pipelining). At compute-bound shapes, check you're on tensor cores (wgmma kernels are named `*sm90*`/`wgmma`; `*ampere*`/`mma.sync` paths leave Hopper performance behind).
- **SGLang's kernel names** in `docs/baseline.py`'s path (cuBLAS `nvjet_*`, FlashInfer, DeepGEMM, Triton) tell you what you're racing. Profile a run of the unmodified baseline only if you truly need it — it costs an evaluation.
- **Call counts** above 1 per shape mean extra kernels (casts, copies, `contiguous()`, reductions) you may be able to fold away.

You can't get achieved occupancy, cache hit rates or stall reasons. Reason them out from the code and the shape (tiles, registers, shared memory per CTA vs 228 KB/SM, CTAs vs 132 SMs), and use `print()` in `prepare()` or at import to report compile-time facts (Triton's chosen config, `kernel.n_regs`, `kernel.metadata.shared`) into `judge.log`.
