Take 30 seconds to zoom out:

1. **Trend** — read back the last few iters in `ITERATIONS.md`. Progress, plateau, regression? Which load is dragging the geometric mean down?
2. **Direction** — given the trend and the profile (`bash scripts/profile.sh --label <last>`), is the current line of experiments still the highest-leverage one for the evaluations you have left (`bash scripts/budget.sh`)?
3. **Rules** — re-read `## Rules` in `CLAUDE.md` against your current kernel: graph-safe, nothing cached from `data`, weights and activations at their required precision, every key attended, correct on nearby shapes. Anything doubtful goes in `## Notes` with a `RULES:` prefix.
4. **Prior variants** (`docs/prior/variants/*/kernel.py` headers) — any constraint or dead-end you may have drifted from?
5. **Logging** — last labeled bench has a Summary row in `ITERATIONS.md`? If you wrote an `Expected:` beforehand, capture the actual delta in `## Notes` now.
6. **Tempo** — the judge takes minutes; background `bench.sh` and keep reasoning about the next change while it runs.

Skip bullets that aren't relevant right now.
