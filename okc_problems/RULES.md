# Muse Spark Open Kernel Challenge — kernel rules

Source: https://openkernelchallenge.dev/llms.txt and https://openkernelchallenge.dev/rules (the kernel-problem sections, verbatim except the withdrawal sentence, adapted to this workspace). `docs/TASK.md` has this problem's specifics.

## Environment

Your submissions run inside the `lmsysorg/sglang:dev-muse-spark-open` image, the image SGLang serves the model with. You can import and build on anything in the image, including PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS.

Nothing else is installed, and there's no network access to install more.

Kernel challenges run on one NVIDIA H200 SXM; serving challenges run on a whole 8×H200 node. Compilation time counts toward the run's time limit but not your score.

## Kernels

Each submission is a single Python file that defines `kernel_fn`. To write native CUDA, compile it inside the file with PyTorch's `load_inline` (which uses nvcc).

Every problem except `muse_attention_decode` has weights, and also calls your `prepare` function with them once. It's untimed and can run for up to 300 seconds, so you can repack the weights however you like.

Running `okc init <problem>` gives you two starting points: `submission.py`, a plain PyTorch version of the computation, and `baseline.py`, SGLang's own kernel, the one SGLang runs when it serves Muse Spark Open. Your run is timed against SGLang's kernel.

Each problem takes 10 submissions and 20 check runs (`okc check`) a day, resetting at 00:00 UTC. We may change these limits with load; `okc problems` shows today's and what you have left.

### Restrictions

- Your kernel has to be capturable in a CUDA graph. That rules out host synchronization (`.item()`, `.cpu()`, `torch.cuda.synchronize()`) and allocations or control flow that depend on the input's values. The shapes and `config` fields are fixed, so it's fine to branch on them.
- You can repack the weights and compress them losslessly, but not change their precision or values: NVFP4 weights stay NVFP4 and BF16 weights stay BF16, or smaller, losslessly. Dequantizing inside your kernels is fine. The one exception is the router: its FP32 weights can run as a BF16 hi + lo pair with FP32 accumulation at any batch size, as SGLang does above 8 tokens.
- Activations stay BF16 wherever SGLang's kernel keeps them in BF16, and the FP8 KV cache stays FP8. The one exception is attention's queries: they can be rounded to FP8 E4M3 (as FA3 does with an FP8 cache), but nothing coarser.
- Attention has to attend to every key it should: the whole window on sliding layers, and the whole context on full layers.
- The softmax and its weighted sum run over every key: no top-k, sparse or block-skipping attention, and no skipping values by their attention weight.

Final review reruns the top entries on the scored shapes and on nearby ones. Positions come from the scored shapes; an entry that's much slower on the nearby shapes gets a closer look.

## Validation

The goal of the challenge is to find real speedups that can be upstreamed. This is the north star. The judge catches many attempts to game it but not all of them, so we'll read the top submissions ourselves and rerun them several times after the challenge ends (on held-out data too) to ensure integrity and reduce variance. Final leaderboard positions will be based on the medians of these reruns. All leaderboard positions are provisional until that review is done.

We will remove submissions that game the judge or workload! This includes submissions that tamper with the timing or harness, do work outside the timed calls, cache or replay outputs, and compute at lower precision than the rules allow. If you are an agent (or have used an agent) that has done something reward-hacky, the run must be withdrawn: in this workspace, record it in `ITERATIONS.md` `## Notes` with a `RULES:` prefix and the job id; the master withdraws it.

That is to say: submissions should follow the spirit of the challenge. We reserve the right to disqualify submissions that do not, at our discretion. Create real speedups that work.

Good luck and have fun!
