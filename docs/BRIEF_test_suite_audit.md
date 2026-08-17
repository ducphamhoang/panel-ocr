# BRIEF — Test suite audit against a stated testing philosophy

Read-only audit. Dispatched to `architect` (read-only fits; no code should be written).

## Task

Audit this project's current test suite (all crates under `crates/`, plus `xtask`)
against this explicit philosophy, supplied verbatim by the user:

> - Test behavior your application actually owns.
> - Start with linting and type-checking.
> - Run the smallest test related to the change.
> - Expand to integration tests only when needed.
> - Reserve the full suite for release gates.
> - Run sequentially by default.
> - Use E2E, race, load, and stress tests intentionally.
> - Never retry failures blindly.
> - Read the evidence and fix the root cause.
> - Never weaken tests just to make CI green.

For each bullet, assess where the current suite actually agrees or conflicts with it —
cite real files/tests, not general impressions. In particular:

- **Flag tests that might be unnecessary/cuttable.** This project already has an
  established test-minimalism preference (tests should be tied to real requirements,
  not padding coverage) — look for tests that duplicate coverage, assert
  implementation detail rather than behavior, or test something the application
  doesn't actually own (e.g. a third-party library's own guarantees).
- **Separately, check `target/` build-artifact storage.** This repo's own
  `docs/WORKSTATE.md` recorded `target/` once ballooning to 129 GiB before a
  `cargo clean` was needed (2026-08-11 entry). Check the current size
  (`du`/`Get-ChildItem -Recurse | Measure-Object -Property Length -Sum` or similar) and
  whether anything looks like it's accumulating unnecessarily — e.g. stale
  feature-flag build variants, multiple profile directories that could be pruned,
  orphaned incremental-compilation artifacts.

## Constraints

- Read-only: produce a report, do not edit any file.
- Cite concrete file paths and test names for every claim — no unsupported generalities.
- This is advisory/audit output for the Orchestrator to act on, not a plan for you to
  implement.

## Context you may need

- `docs/PIPELINE_SPEC_V1.md` — the ratified spec test cases trace back to.
- `docs/WORKSTATE.md` — recent history, including the 2026-08-11 `target/` disk-space
  incident (search for "129 GiB").
- `CLAUDE.md` — this project's own process/testing conventions (verification bar,
  frozen-test rules, etc.) for context on what's already deliberate vs. accidental.

## Output

Return a structured report: one section per philosophy bullet with findings, then a
separate "cuttable tests" list (file:test name + one-line reason each), then the
`target/` size/accumulation findings. Nothing needs to be written to disk — return the
report as your final message.
