---
name: okc-submission
description: How a submission file is built for this benchmark — one self-contained Python file defining `kernel_fn(data, state)` (timed, CUDA-graph replayed) and `prepare(weights)` (untimed, ≤300 s); embedding Triton kernels and CUDA C++ via `torch.utils.cpp_extension.load_inline`; compiling and autotuning within the 10-minute run limit; launching on the current stream; Hopper (sm_90a) specifics. Use when restructuring the file, adding a CUDA/CUTLASS kernel, or hitting a compile/import error on the judge. Overrides the TVM-FFI / config.toml / binding.py instructions in the DSL skills.
---

# The submission file

`solution/submission.py` is the whole submission. Detailed reference: `submission.md`.
