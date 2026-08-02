# Status — milestone log

Rolling intermediate reports, one entry per milestone (defined in
[plan.md](./plan.md#milestones-and-intermediate-reporting)). Updated and committed at
every milestone boundary so anyone can resume the work from the last entry without
redoing finished stages. Newest entry first.

## Resume procedure

1. Read the newest entry below: it states what is done, artifact paths, deviations, and
   the exact next step.
2. Never redo a milestone marked done — its artifacts are committed.
3. If a rental was lost mid-window (M2–M4), re-rent per plan.md ("The rental" spec in
   decision.md §8), rerun `setup_vast.sh`, and continue from the newest entry's next
   step; committed results JSONs in `perf/results/` mark which runs are already banked.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Planning docs approved | in review (PR #5) |
| M1 | Harness ready | not started |
| M2 | Box bootstrapped | not started |
| M3 | Infra calibrated | not started |
| M4 | JAX runs complete | not started |
| M5 | Wrapped up | not started |

## Entries

### 2026-08-02 — M0: planning docs

- **Achieved**: full doc set under `perf/docs/` (plan, decision log, glossary, report
  skeleton, learnings charter, this file) on branch `worktree-perf`, PR #5.
- **Deviations**: none — planning phase.
- **Blockers**: awaiting PR review/approval.
- **Next step**: M1 — build `perf/bench_native.py`, `perf/bench_vllm.sh`,
  `perf/setup_vast.sh`; CPU smoke test via
  `gemma/diffusion/_sampler_test.py::_SMALL_CONFIG`; verify `gs://gemma-data`
  checkpoint accessibility from a non-Google box (fallback: HF/Kaggle copy).
