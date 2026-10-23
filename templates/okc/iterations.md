# Iteration Log

Append a Summary row for every labeled bench (including failed and cached ones). That's the only required action.

When a change is worth pre-committing a hypothesis to, write `Expected: ...` in `## Notes` BEFORE running the bench — a mechanism, the shapes it affects, and a predicted delta (e.g. "split-K 4 on m=1 pooler fills 132 SMs — expect decode-1 load +15%, verify-32 unchanged").

At session end, write a brief synthesis in `## Notes`: kernel state, remaining bottlenecks (cite the profile), dead ends to skip ("needs tuning" ≠ dead end), what's worth trying next. Note anything rules-related with a `RULES:` prefix.

## Summary

| Iter | Title | Kind | Job | Score | CI | Tests | Notes |
|------|-------|------|-----|-------|----|-------|-------|

## Notes

