---
name: bench
description: How to drive the official judge from this workspace — `bash scripts/bench.sh --label`, the free tools (lint.sh, budget.sh, profile.sh), the per-session evaluation cap, the check/check/scored-run slot pattern, cached duplicates, and how to judge whether a change helped from a single run's per-shape table and 95% CI. Use before every bench and when reading its output. Judge mechanics and statuses live in the `benchmark` skill.
---

# Bench — driving the judge

There is no local GPU: the judge is the only performance and correctness oracle, and every evaluation is scarce. Detailed reference: `bench.md`.

```bash
bash scripts/lint.sh                              # free: static rules check (bench runs it first)
bash scripts/budget.sh                            # free: today's slots, next slot kind, this session's count
bash scripts/bench.sh --label "iter-N desc"       # ONE official evaluation (check or scored run)
bash scripts/profile.sh --list                    # free: every evaluated snapshot and its score
bash scripts/profile.sh --label iter-N            # free: the judge's per-kernel profile for one
```
