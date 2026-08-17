# Changelog

All notable user-facing changes to `panel-ocr` are recorded here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions before 1.6.0 are not
backfilled here — see `README.md`'s [Roadmap](README.md#roadmap) section for that
history; this file starts from the last tagged release (`v1.5.0`) forward.

Internal-only work (spec ratifications, test-suite cleanups, and inconclusive
investigations with no shipped code) is not listed here even when it produced commits —
see `docs/WORKSTATE.md` and `docs/PIPELINE_SPEC_V1.md` for that record.

## [1.6.0] — 2026-08-17

### Added
- **`panel-ocr inpaint`** — a standalone subcommand that inpaints a user-supplied
  RGBA mask directly onto an image with LaMa, with no cache/uuid state and no
  dependency on the detector or OCR. Useful for touch-up work outside the normal
  `clean` pipeline (paint your own mask, get the region filled).
- **`panel-ocr residual-check`** — a non-gating diagnostic that re-runs the text
  detector over an already-cleaned image and reports any text it still finds, as an
  objective replacement for eyeballing whether a cleaned page "looks done." Never
  fails the process on a detection; it reports numbers.

### Improved
- **OCR is faster on CPU.** The beam-search decode loop now batches all 4 beams into
  one decoder call per step instead of one call per beam per step — real, measured
  speedup of **~1.1×–1.6×** (degrading with sequence length; shorter text sees the
  smaller end of that range). Output is bit-identical to the unbatched path; this is a
  performance change only.

### Fixed
- **A real batch-mode CPU regression is fixed: 1.66×–5.3× slower than it should have
  been.** Root-caused to a denormal-float flag (`flush_denormals`) whose effect could
  silently leak or fail to reapply depending on which ONNX session initialized first —
  an order-dependent bug, not a flat regression, which is why it was inconsistent
  across runs. Fixed with a scoped guard that reapplies the flag on every inference
  call rather than relying on a process-wide once-flag. Verified bit-identical output
  across every thread count and both flush states against real cached weights.
- **Fixed a build failure on every non-x86_64 target** (macOS Apple Silicon, aarch64
  Linux) — a `#[cfg(target_arch = "x86_64")]` mismatch between a function's signature
  and one of its call sites, introduced alongside the denormal-flush fix above. Caught
  by this release's own build matrix before publishing; x86_64 builds (including the
  ones most people run) were never affected.

### Internal
- Build config: `[profile.dev] debug = "line-tables-only"` to curb `target/` disk
  growth during development (no effect on release builds or shipped behavior).
- Right-sized the CI verification bar for lower-risk changes (see `CLAUDE.md`) — no
  user-visible effect.
- A staged-pipeline redesign for overlapping OCR and LaMa-inpaint execution was
  investigated (joint design + a real hardware measurement) and **parked, not
  shipped** — the measured benefit couldn't be distinguished from this machine's own
  run-to-run noise and hardware topology at the depth this pass measured. No code
  changed as a result; see `docs/PERFORMANCE_BACKLOG.md` if picking this back up.

[1.6.0]: https://github.com/ducphamhoang/panel-ocr/compare/v1.5.0...v1.6.0
