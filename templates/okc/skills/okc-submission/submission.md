# Building a submission

## Shape of the file

- Line 1 is `# okc: problem=<problem>` — keep it; bench and the judge read it.
- `Inputs`, `Config` (and `Weights` where the problem has them) describe every field's dtype and shape; keep their names and return exactly the shape and dtype the starter returns.
- `prepare(weights)` runs once, untimed, up to 300 s: repack weights (lossless only), build lookup tables from **weights and config** (never from `data`), compile kernels, allocate workspaces sized from config bounds, and warm up any lazily-compiled kernel so compilation doesn't land inside graph capture.
- `kernel_fn(data, state)` is the timed call, captured into a CUDA graph and replayed with fresh inputs. Allocation of fixed-size outputs/workspaces with torch is fine under capture (the caching allocator handles it); value-dependent sizes, host syncs and Python branches on tensor values are not.

## Triton

Triton kernels live in the file as `@triton.jit` functions. `@triton.autotune` benchmarks configs on first call — inside graph capture that's wrong, so trigger it in `prepare()` for every shape class, or pick configs per shape statically (a dict keyed by `(m, n, k)` bucket). Keep the config list short: every config compiles inside the 10-minute run.

## CUDA C++ / CUTLASS

```python
from torch.utils.cpp_extension import load_inline
_ext = None
def _build():
    global _ext
    if _ext is None:
        _ext = load_inline(name="my_kern", cpp_sources=CPP, cuda_sources=CUDA, functions=["launch"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                                              "--use_fast_math", "-std=c++17"],
                           verbose=False)
    return _ext
```

Build in `prepare()` (or at import), never on the timed path. Launch on PyTorch's current stream: `at::cuda::getCurrentCUDAStream()` in C++ (`#include <ATen/cuda/CUDAContext.h>`) — a `<<<grid, block>>>` launch on the default stream is not captured into the graph (stale outputs, implausible speedups). CUTLASS / CuTe headers ship in the image (check `flashinfer`'s or `deep_gemm`'s bundled `include/` dirs, or `cutlass` in site-packages, for the path; print it into `judge.log` from `prepare()` if unsure). `global` here is a module-level build cache, set in `prepare()` — never write module state from `kernel_fn`.

## Hopper (sm_90a) notes

- Tensor cores: `wgmma` (warpgroup MMA, async, operands from shared memory via descriptors) + TMA bulk copies + mbarriers is the fast path for GEMM-like work; `mma.sync` (Ampere style) runs at a fraction of peak. Triton emits wgmma on sm_90 when tiles allow (BLOCK_M ≥ 64, num_warps 4/8); CuTe DSL and CUTLASS 3.x expose it directly.
- FP8 E4M3 conversions (`cvt.rn.bf16x2.e4m3x2`, `__nv_cvt_fp8x2_to_halfraw2`) exist on sm_90. There are no FP4 tensor cores: NVFP4 weights are dequantized in-kernel (to BF16, or used through FP8-scaled paths only if values stay exact — check the rules first).
- 132 SMs: small-m decode shapes leave most SMs idle under a naive tiling — split-K / stream-K, or several independent problems in one launch.
- Thread-block clusters + distributed shared memory are available (cluster dims up to 8 portable).
- Blackwell-only material in the DSL skills (tcgen05, tensor memory/tmem, 2-CTA MMA, `sm_100`) does **not** apply here.

## Libraries

Everything in the image is fair game to build on: `flashinfer`, `deep_gemm`, `sgl_kernel`, CUTLASS, TileLang, CuTe DSL. Read `docs/baseline.py` to see exactly what SGLang calls; beating it often means fusing around it, picking a better-suited kernel for the shape, or replacing the piece the profile says dominates.
