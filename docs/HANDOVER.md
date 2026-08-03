# Handover — TEMPORARY, delete when consumed

Written 2026-08-03, replacing the previous version (2026-07-30, which described F1 Phase 2 as
code-done-but-not-recorded and P7/P8 as the other open v1.0 task — both are now done, see
below). **This file is scaffolding, not a record.** Anything in it worth keeping permanently
belongs in `PIPELINE_SPEC_V1.md`, `COOKBOOK.md`, or `RULINGS.md` instead.

## Repo state

- Branch `claude/codex-plugin-install-jxirxa`, `HEAD` = `f6212eb`, working tree clean, **pushed**
  (origin is up to date with HEAD as of this writing).
- Verification bar, actually run at HEAD: `cargo test --workspace` (920 passed, 0 failed, 3
  ignored), the onnx tier (`cargo test --workspace --all-targets --features pc-cli/onnx`, 934
  passed, 0 failed, 5 ignored), `cargo clippy --workspace --all-targets --all-features -- -D
  warnings` clean, `cargo fmt --all --check` clean. **Do not cite older figures** — re-run the
  count yourself; this repo's own docs have gone stale on this point more than once (cookbook
  rule 6).
- `export PATH="$HOME/.cargo/bin:$PATH"` or cargo is not found (`COOKBOOK.md` rule 11).
- Git has no configured identity; commits used `GIT_AUTHOR_NAME=Claude
  GIT_AUTHOR_EMAIL=noreply@anthropic.com` (+ `GIT_COMMITTER_*`) as env vars.
- The upstream PanelCleaner checkout/venv/weights used this session are session-scoped
  scratchpad state (`/tmp/claude-*/.../scratchpad/oracle_rebuild/`) and are gone in a new
  session. Rebuild fresh via `COOKBOOK.md` rule 3's recipe if the next session needs the real
  oracle again (e.g. for the CPU-EP investigation's `cv2.dnn` speed comparison, or a future
  DBNet-lines F1 re-record). **Do not let the installer touch shell rc files** — `uv`'s
  installer auto-appends a `source .../uvbin/env` line to `.bashrc`/`.zshrc`/`.profile`
  pointing at the ephemeral scratchpad path, which breaks on the next session with a `bash:
  No such file or directory` warning; either pass `INSTALLER_NO_MODIFY_PATH=1` to the curl
  installer or just export `PATH` yourself and strip any line it adds afterward. This has
  already happened and been cleaned up twice this session — don't reintroduce it a third time.
- Downloaded model weights are cached at `~/.cache/panel-ocr/models/` (`comictextdetector.pt.onnx`,
  `encoder_model.onnx`, `decoder_model.onnx`) — already sha256-verified by `panel-ocr models
  download`; no need to re-download for a same-machine continuation.

## v1.0: DONE

Every task in §13's consolidated table (F0 through P8, all 32 rows) is implemented and green,
**including F1**, which was the last open item as of the previous handover. Commit `f6212eb`
landed the real detector-oracle recording: the pinned PanelCleaner checkout
(`0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`) run for real against the real
`comictextdetector.pt.onnx` on the ratified page (`ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg`,
§16.24 item 17), producing:

- `tests/fixtures/recorded/detector/` (7 artifacts + `PROVENANCE.json`), the oracle page moved
  in (`git mv`, no second copy) per §16.29 item 2.
- The real-page gate test (`crates/pc-detect/tests/f1_real_page_gate.rs`) with its hand-derived
  `Expectations` literal — derived independently twice (two subagents, blind to each other,
  converged exactly) — plus two adversarial mutation controls proving it actually gates.
- Four previously-`#[ignore]`d tests un-gated against this real page: `d7_run.rs`'s `a6`/`b9`,
  `p5_run.rs`'s `b11`. (`n4_run.rs`'s `b13` stays `#[ignore]`d — see "What's still open" below.)
- `docs/DETECTOR_ORACLE.md`: the §16.20 item 3(e)/§16.24 item 11 completeness partition, the
  pairing verdict, and all three required signatures (two independent agent re-derivations +
  human maintainer `ducph`, 2026-08-02) — reviewed by an independent `fresh-reader` pass before
  signing, which caught and required fixing three real defects (a wrong `NO-ORACLE`/`EXPLAINED`
  verdict, an unsourced page attribution, and four `§16.24`/`§16.20` mis-citations — all fixed
  before the signature was recorded, not after).
- `crates/pc-testkit/tests/detector_oracle_doc.rs`: the four §16.24 item 11 doc-shape gates,
  live (not dormant) since the fixture already exists — including a hardened
  "all 3 signatures present, none self-referential" check that reads the producing-agent name
  off the doc itself rather than matching a hardcoded string.

**Nothing else is open for v1.0.** Do not start a new v1.0 task; the next work is v1.5-scoped
(F2, below, plus everything after it).

## What's still open, in the order the maintainer chose to do them

### 1. F2 — `calibrate-goldens` (§7.3), in progress, needs real implementation

`cargo xtask calibrate-goldens` **runs today** and correctly measures its two already-wired
sections (NLM vs `cv2.fastNlMeansDenoising`, INTER_AREA), but **section 3 of its output is
stale placeholder prose** written back when the detector didn't exist — it unconditionally
says "BLOCKED on D1+D4" regardless of what's actually recorded now. Three things are needed:

1. **Update the status table.** `a6`/`b9`/`b11` are no longer blocked (done, see above); only
   `b13` remains blocked, and for a *different* reason now — missing golden PNGs, not the
   detector (see "What's still open" #3 below). The hardcoded "BLOCKED on D1+D4" line in
   `write_verdict` and the `section_blocked` table need to reflect this split.
2. **Implement the `demo_bubbles` masking calibration report (§10.7(B)15).** This is genuinely
   new code, not a re-run. Per §16.20 item 12 (a **ratified, binding** decision, not a design
   choice open to reconsideration): run the detector on the 7 `demo_bubbles` crops
   (`tests/fixtures/upstream/demo_bubbles/*_bubble_raw.png`) **to scratch only — commit
   nothing** — then run masking and measure IoU / exact-% / max-Δ / SSIM against each crop's
   `_clean.png` reference. Non-gating report only, no test. The same decision confirms the four
   un-ignored tests above are correctly scoped ("un-ignore only against the signed maintainer
   page, never against `demo_bubbles`") — nothing to reconcile there.
3. **Implement the detector's upstream-vs-ours box-count comparison (§15.1)** on the recorded
   page. Cheap — the numbers already exist in `f1_real_page_gate.rs`'s report (3 matched pairs;
   upstream 4 vs ours 4, both extra blocks explained by a named `Mechanism`). This is mostly
   formatting already-derived facts into the doc, not new measurement.

Scope note from the session that reached this point: item 2 is real implementation work
(new detector + masker + metrics wiring in `xtask/src/calibrate.rs`) with a ratified
scratch-only constraint that's easy to get wrong (accidentally committing a `demo_bubbles`
detector artifact would violate §16.20 item 12 directly) — worth a deliberate check before
landing, not necessarily a full joint-architect planning pass given how much of the underlying
plumbing (detector, masker, `pc_testkit::metrics`) already exists and is well-understood from
F1.

### 2. CPU-EP investigation (§16.21 item 6), not started, schedule-independent of GPU

Already ratified as part of v1.5, and explicitly **may run any time after PERF-1** (which
already landed) — it does not need to wait for GPU-1/GPU-2. Scope, per the ratifying entry:
profile which ONNX ops dominate, check session/execution-mode options, try an `ort`/ONNX
Runtime version bump as a *candidate* (not a given).

**Measured this session** (informal, not a committed benchmark — see below): our detector at
default config (`intra_threads=0`) takes ~17.3s median per inference on this machine; upstream's
`cv2.dnn` takes ~1.9s median on the identical page — roughly a **9-12x gap**, consistent with
§16.21 item 6's own prior measurement of an 11x gap at matched thread count (`cv2.dnn` 1.20s vs
ours 13.6s at 16 threads). **A thread-count sweep this session ruled out threading as the
cause**: intra_threads 0/1/4/8/20 gave medians of 17.3s/142.9s/37.9s/20.4s/16.1s — so `0`
(default) is already near-optimal, and even pinning to all cores only shaves ~7% off. The
static `libonnxruntime.a` linked here has no OpenMP symbols, so the "with_intra_threads is a
documented no-op under OpenMP prebuilt binaries" scenario doesn't apply either. **The gap is a
backend/model-execution difference, not a config bug** — likely ORT's default CPU EP not
hitting an equivalently-optimized conv path (Winograd/im2col) that OpenCV's DNN module has had
years to tune for this kind of model. Nothing was implemented from this finding; it's a
starting point for whoever picks up item 6, not a conclusion.

**Hard constraint, already ratified, do not skip**: the model bytes are immutable
(sha256-pinned, §16.16); **no runtime replacement is permitted** — swapping to `cv2.dnn`/OpenCV
was explicitly considered and explicitly rejected for this investigation, because `cv2.dnn` *is*
the independent oracle F1 just finished comparing against, and running our detector on the same
backend as the oracle would collapse the whole point of the comparison. And **any fix that
perturbs even one recorded float — including a plain version bump — is fixture-affecting**: it
would invalidate the F1 fixture just signed and committed in `f6212eb`, requiring a full
re-record + re-sign (the same 3-signature process this session just went through). So: measure
before adopting, always; nothing here lands quietly.

### 3. `n4_run.rs::b13_pending_recorded_page_end_to_end_golden` — still blocked, not by F1

§11.7(B)13 requires committed reference golden PNGs (`_noise_mask.png`, `_clean_denoised.png`)
for SSIM/alpha-histogram assertions. Neither exists anywhere in the tree — the F1 detector
recording never produced them (denoise is a separate stage). Generating them by running our own
denoiser and accepting the result would be cookbook rule 7's exact violation (no independent
oracle for this stage on this page). Needs a joint-architect decision on how the golden is
derived and signed before anyone touches this — not a unilateral generation. Its `#[ignore]`
reason string already names this correctly; nothing to fix there, just nothing to do yet either.

### 4. v1.5 sequence after the above, per §16.24 item 5 (ratified order)

GPU-1 (device config / policy resolver / fatal-refusal wiring, **no CUDA linkage**) → GPU-2 (the
`cuda` feature itself) → legacy INI import + Lab-space NLM (batched) → LaMa inpainting → PSD/
layered export → DBNet line synthesis (flagged as the highest-risk item in the whole v1.5 scope,
and forces a second F1 re-record + re-sign when it lands, budgeted up front) → `Mask
RefineMode::Annotation` last. The CPU-EP investigation (#2 above) is order-flexible within this
sequence per §16.24 item 5's own text ("may run any time after PERF-1"); GPU-1 is scoped as
config-only and shouldn't conflict with CPU-EP work in the same files, but **GPU-2 would** —
both touch `crates/pc-detect/src/onnx.rs`'s `Session::builder()` construction directly (GPU-2
needs to pin `ConvAlgorithmSearch::Heuristic`/`Default` there per §16.22 item 3). If running
these as parallel worktrees, don't pair CPU-EP investigation with GPU-2; CPU-EP + GPU-1 or
CPU-EP + F2 are both low-conflict pairings.

## Known discrepancies flagged this session, not yet corrected (recorded in `docs/DETECTOR_ORACLE.md`'s own "Known discrepancies" section too)

Three small spec/artifact mismatches surfaced while deriving F1's real-page pairing, all
**flagged, not fixed** per this project's convention (a ratified transcription is corrected by
a new entry, not a silent edit):

1. §16.27's transcription of upstream blocks 1/2's served `xyxy` has its `y2` values
   transposed against the real recorded artifact (spec ~line 5433). The entry's conclusion
   still holds against the real vectors; needs a spec erratum.
2. A previously-quoted `mask_score` of `0.0359` (spec ~lines 3430/4012/5494,
   `crates/pc-detect/src/oracle.rs:85`, `f1_oracle_comparator.rs:1058`) reads
   `0.03446455505279035` in this real recording's `PROVENANCE.json`. Both are `< 0.1` so no
   conclusion changes, but five sites now quote a stale number (one inside a frozen test file).
3. §14 item 17's "measured across two real manga pages the filter has never fired" is
   contradicted by this page — our own coverage filter *does* fire here (measured 0.0868 on
   block index 3). §16.20 item 9 already records this page's correct outcome; item 17's summary
   sentence is the stale one.

Also found: `DEVIATION(17)`, which §14 item 17's own text says "the implementation site
carries", does not exist anywhere in `crates/`/`xtask/` (`DEVIATION(12)` does, at
`crates/pc-detect/src/mask.rs:150`). A register/code desync the `mask_coverage` row's
`EXPLAINED-§14.17` close rests on but doesn't create — worth landing alongside whichever of the
three erratum entries above gets written first, since all four are in the same neighborhood.

None of these block anything currently green; they're follow-up erratum entries for whoever is
next in `docs/PIPELINE_SPEC_V1.md`'s §16.x sequence.
