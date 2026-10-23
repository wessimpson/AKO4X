---
name: sanitizer
description: Memory-error and race debugging without compute-sanitizer — there is no local GPU, so diagnose from the judge's failure message, judge.log, and careful bounds reasoning. Use when a run fails with an illegal memory access, a CUDA error, NaN/Inf, or a correctness failure on only some shapes or rows.
---

# Debugging failures without a sanitizer

`compute-sanitizer` isn't available (no local GPU). A failing official run names its first failing case and reason; `trajectory/<snapshot>/judge.log` has the tail of the run's output (Python traceback, CUDA error text, your own prints outside `kernel_fn`).

- **Illegal memory access / misaligned address**: an out-of-bounds index on the tail tile (m, n or k not a multiple of the block), a wrong stride assumption (non-contiguous input, a transposed weight after repacking), or vector loads on unaligned addresses. Walk the failing test shape by hand through every index computation; guard tails with masks.
- **Fails only on odd/large m, or on outlier distributions**: tail masking, a reduction order or accumulation dtype that loses precision (accumulate in FP32), overflow in an intermediate.
- **Row errors with a fine global error**: one tile/row range computed wrongly (split-K partials not all summed, a race on an atomic or a shared-memory buffer missing a barrier).
- **NaN/Inf**: uninitialized accumulators or shared memory read before written, unmasked loads past the end feeding a max/exp.
- **Passes once, fails later / flaky**: races (missing `__syncthreads()`/barrier, workspace reused across overlapping kernels), or state carried between calls.

You can test pure index arithmetic and weight-repacking logic on CPU with local PyTorch on tiny tensors before spending an evaluation.
