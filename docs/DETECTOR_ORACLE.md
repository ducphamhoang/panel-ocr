# Detector oracle verdicts — F1 real-page recording

Spec §16.20 item 3(d)/3(e)/3(f), §16.24 items 6/10/11, §16.29. This document records the verdicts for
the one recorded detector page, produced by the atomic F1 recording commit. It is the
`docs/DETECTOR_ORACLE.md` named throughout §16.2x.

## The recording

| | |
|---|---|
| Page | `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg` |
| Page sha256 | `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e` (§16.24 item 17's ratified literal) |
| Upstream commit | `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` (pinned, PanelCleaner) |
| Model | `comictextdetector.pt.onnx`, sha256 `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f` |
| Upstream backend | `cv2.dnn` (§16.20 item 6) |
| Ours backend | `ort`, execution provider `cpu`, `intra_threads=0`, `inter_threads=0` |
| Recorded | 2026-08-02 |
| Producing agent | Claude (orchestrator session, `claude/codex-plugin-install-jxirxa`) — mechanical act only, per item (f) below |

Both `scale == 1.0` on both sides, so `base_image` is a byte-for-byte decode of the source
JPEG with no resize in play — the completeness partition below notes this scope explicitly
wherever it matters.

## Completeness partition (§16.20 item 3(e), §16.24 items 10/11)

Every scalar serialized field of `PageDataRaw` + `DetectedBlock` (§2.4) and `RawBlock`'s three
serialized fields (the pre-coverage-filter artifact `OursSide` reads, §16.24 item 8) lands
in exactly one bucket: `ORACLE-EXACT` / `EXPLAINED-§14.x` / `DIAGNOSTIC` / `NO-ORACLE` /
`OUT-OF-SCOPE-F1`. No row is `OPEN`. (`blocks` itself is a serialized field of `PageDataRaw`
but is a container, not a scalar — it is decomposed into `DetectedBlock`'s own rows below
rather than carrying an independent verdict of its own.)

### `PageDataRaw` (upstream `#raw.json` / ours `#raw.json`, top-level keys)

| Field | Verdict | Evidence |
|---|---|---|
| `schema_version` | `NO-ORACLE` | Ours-only bookkeeping; upstream's `#raw.json` has no equivalent concept. |
| `original_path` | `NO-ORACLE` | Machine-dependent absolute path on both sides (§16.20 item 3, trailing ¶ on machine-dependent paths); normalised out of both artifacts before commit — verified: `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_oracle.json` and our own `#raw.json` carry no absolute path (checked by the recorder's own `_assert_no_absolute_paths`, `xtask/scripts/record_detector_oracle.py:786-815`, and by `original_path` in our committed `#raw.json` being the repo-relative literal). |
| `base_image` | `ORACLE-EXACT` (scoped: `scale == 1.0` only) | `PROVENANCE.json`'s `detector.ours.decoded_rgb_digest` and `detector.upstream.decoded_rgb_digest` are the **same** literal, `82c9b93e3bd95420df7c8bf9f5ffb5b322ec04c7d8576e8cf87ad254adf385e4`. `xtask/tests/provenance_digests.rs::our_recorded_decoded_rgb_digest_equals_the_decoded_committed_page` independently grounds **our** side by decoding the committed page itself (not taking either side's own claim); two-sided equality — that upstream's digest is the *same* value, not merely self-consistent — comes from `recorded_provenance.rs`'s R23 walker (`crates/pc-testkit/src/provenance.rs:243`). This recording never exercises a resize (`scale == 1.0`), so nothing here says anything about `base_image` identity when `scale != 1.0` — that case is untested by this fixture. |
| `raw_mask` | `NO-ORACLE` | **Ratified, not a judgment call made here**: §16.20 item 3(d) states plainly "`confidence`, `language` and `raw_mask` are DIAGNOSTIC or NO-ORACLE rows only, never gated" and settles `raw_mask`'s bucket as `NO-ORACLE` "unconditionally", reaffirmed unchanged at §16.22 item 7 and restated at §16.25. Evidence for *why*, **as it stood when this page was recorded on 2026-08-02**: §14 item 12 — v1 then shipped the `Simple` refinement by default, not upstream's `refine_mask`/`refine_undetected_mask` (`refine_mode=REFINEMASK_ANNOTATION`), ratified out of v1 scope by §15 item 2. **That default moved on 2026-08-10 (§16.46 item 1(a)): `Annotation` is now the shipped default and `DEVIATION(12)` is retired.** The verdict does not move with it — this recording was made under `Simple` and stays so, because §16.46 item 8 pins the recorders to `MaskRefineMode::Simple` explicitly rather than letting them inherit the default, and §16.20 item 3(d) settles `raw_mask` as `NO-ORACLE` *"unconditionally"* regardless of which refinement produced it. §16.20 item 3's own text quotes a measured IoU 0.258 (upstream 2,950 non-zero px, ours 11,103) without naming which page it was measured on; I could not independently verify those two figures against any committed artifact — this page's own `_raw_mask.png` decodes to 11,102 non-zero px, close enough to the quoted "ours 11,103" that the two may be the same measurement, but the spec source does not say so and this doc does not claim it either way. Gated instead by the synthetic primaries §8.7(A)4/5. |
| `scale` | `ORACLE-EXACT` | Both sides record `1.0`; `#raw.json` and `..._upstream_oracle.json` agree exactly (§16.24 item 8 names this a genuine oracle field). |
| `image_size` | `ORACLE-EXACT` | Both sides record `(1200, 1660)`; agree exactly. |
| `blocks` (container) | — | Decomposed into `DetectedBlock`'s own fields below; the container itself carries no independent verdict beyond its element count, which the real-page gate test asserts (`ours_total == 4` pre-filter, 3 pairs + 1 `CoverageFilteredUpstream` = 4; `upstream_total == 4`, 3 pairs + 1 `ClassDuplicateOf` = 4). |

### `DetectedBlock` (per-element fields of `blocks`)

| Field | Verdict | Evidence |
|---|---|---|
| `rect` | `ORACLE-EXACT` | The geometry identity itself (§16.20 item 3(b)): `crates/pc-detect/tests/f1_real_page_gate.rs::the_recorded_page_matches_the_upstream_oracle_under_the_signed_pairing` asserts `report.gating().is_empty()` over all 3 paired blocks, with `LineInformed` on every pair (verified by independent hand-derivation twice — see Signatures below) and a mutation control (`dropping_the_pre_expansion_lines_breaks_the_identity_on_the_real_page`) proving the identity is not vacuously satisfied. |
| `language` | `DIAGNOSTIC` | §16.20 item 3(d): never gated on value. Measured: all 3 pairs agree (english/english/japanese each side); zero `LanguageDelta` rows in the real-page report. Coverage is `Present` on all 4 upstream blocks (no `InconsistentOracleCoverage`). |
| `confidence` | `DIAGNOSTIC` | Never gated on value (§16.20 item 3(d), grounded in §16.25 item 5's measured 0.079–0.089 noise floor against a 0.4 gate). Measured: 3 `ConfidenceDelta` rows, `(0.713, 0.715)`, `(0.704, 0.774)`, `(0.643, 0.624)` — the real-page gate asserts this exact ordered vector, not just its non-gating status. |
| `mask_coverage` | `EXPLAINED-§14.17` | §14 item 17: "Coverage-filter scope and operand" — our own metric (`mean(raw_mask within rect)/255`) runs ~1.7–2.5× upstream's `mask_score` on identical boxes from a ratified operand difference (refined vs unrefined mask, every-block vs line-less-only scope, §16.20 item 9). Not a comparable oracle quantity, but the divergence is a named, ratified deviation rather than an unexplained gap. **Caveat, flagged rather than silently resolved**: §14 item 17 itself states "measured across two real manga pages the filter has never fired" — but on *this* page our own filter does fire (measured coverage 0.0868 on block index 3, below the 0.1 threshold; see the pairing table below). §16.20 item 9 already records "both implementations dropped that box" for this page, so §14.17's "never fired" sentence is the one that is stale, not this row — flagged in Known discrepancies below rather than corrected here. Also note: `DEVIATION(17)`, which §14 item 17's own text says "the implementation site carries", does not currently exist anywhere in `crates/`/`xtask/` (`grep -rn DEVIATION` finds `DEVIATION(12)` on the `Simple` variant in `crates/pc-config/src/profile.rs` but no `DEVIATION(17)`) — a register/code desync this row's `EXPLAINED` close rests on but does not create. **Re-measured 2026-08-10 after §16.46 flipped the shipped defaults**: `DEVIATION(17)` is still absent from `crates/`/`xtask/`, so that desync stands; `DEVIATION(12)` is still written at the same `Simple` variant, but that variant is no longer the default one (§16.46 item 1(a)) and the entry is marked RETIRED in place there (item 2). Two scope notes that keep this row from being read wider than it is: this recording was and remains a `Simple` recording, because §16.46 item 8 pins the recorders to `MaskRefineMode::Simple` rather than letting them inherit the default — so the operand difference described above is still exactly what produced these numbers — and §16.46 item 6(b) separately records that under the new default a *fresh* run scores the unrefined mask, i.e. `DEVIATION(17)`'s **operand** half no longer applies to a default run at all, while its scope half and the `0.1` threshold are untouched. |

### `RawBlock` (`detector_blocks.json`, the pre-coverage-filter artifact `OursSide` reads)

| Field | Verdict | Evidence |
|---|---|---|
| `rect` | `ORACLE-EXACT` | This is Leg 1 (`ours.rect == rect_yolo`, `oracle.rs:838`) — the forced-pairing discriminator itself. `leg1_rows_checked == 3` in the real-page report, all three exact. |
| `class_index` | `DIAGNOSTIC` | Maps to the same `language` diagnostic above (class 0 ↔ English, class 1 ↔ Japanese); same evidence. |
| `confidence` | `DIAGNOSTIC` | The same underlying value as `DetectedBlock::confidence` above — the coverage filter drops boxes, never mutates a surviving box's confidence. Same evidence. |

### `PageData` (§2.5, preprocessor output — out of scope for a *detector* oracle)

Per §16.24 item 10's erratum, every §2.5 field is `OUT-OF-SCOPE-F1`: F1 records and compares
the **detector's** output only; the preprocessor is a separate stage with its own recorded-page
gate (`p5_run.rs::b11_pending_recorded_page_tier_arithmetic`, hand-derived separately and
already landing in this same commit).

| Field | Verdict |
|---|---|
| `schema_version`, `original_path`, `base_image`, `raw_mask`, `scale`, `image_size` | `OUT-OF-SCOPE-F1` |
| `page_language` | `OUT-OF-SCOPE-F1` |
| `text_boxes` (`TextBox::rect`, `TextBox::language`) | `OUT-OF-SCOPE-F1` |
| `extended_boxes` | `OUT-OF-SCOPE-F1` |
| `masking_regions` (`MaskingRegion::masking`, `MaskingRegion::reference`) | `OUT-OF-SCOPE-F1` |

## Pairing verdict (§16.20 item 3(b), the `Expectations` literal)

Three pairs, geometrically exact (`LineInformed` on all three, per §16.29 item 1's operand
rule — `lines_pre_expand` when recorded, else served `lines`):

| ours idx | ours rect | upstream idx | upstream xyxy | derivation | branch |
|---|---|---|---|---|---|
| 0 | (674,1397,740,1438) | 3 | [674,1397,740,1438] | `YoloSynthesizedCorners` | `LineInformed` |
| 1 | (567,74,663,123) | 0 | [567,74,663,123] | `YoloUnioned` | `LineInformed` |
| 2 | (607,631,724,703) | 2 | [607,631,724,703] | `YoloSynthesizedCorners` | `LineInformed` |

Two unmatched blocks, each with a checkable mechanism:

| Side | Index | Rect | Mechanism | Evidence |
|---|---|---|---|---|
| Ours | 3 | (438,1407,498,1446) | `CoverageFilteredUpstream { pre_filter_index: 4 }` | Upstream's line-less `mask_score` filter dropped this box (`pre_filter_blocks[4]`, recorded `mask_score = 0.03446455505279035 < 0.1`, §14.17/§16.20 item 9). Our *own* coverage filter also drops the same box (measured coverage 0.0868 < 0.1) but on a different operand — that is not the cited mechanism, since `OursSide` reads the pre-filter list where this box is still present. |
| Upstream | 1 | [607,630,723,705] | `ClassDuplicateOf { upstream_index: 2 }` | Upstream's per-class NMS (`agnostic=False`, §14.13) emitted the same balloon twice, once per language class; our class-agnostic NMS kept one (index 2, the paired Japanese block). Citation resolves in range and is itself paired. |

Accounting closes both equations: `pairs(3) + class_duplicates(1) = upstream_total(4)`;
`pairs(3) + coverage_filtered_ours(1) = ours_total(4)`.

**This pairing was derived independently twice**, by two agents blind to each other's work,
before either wrote it to disk — see Signatures below. Both derivations landed on the
identical literal (pairs, branches, derivations, mechanisms, totals) with no disagreement to
route to a tie-break.

## Known discrepancies flagged during this recording, not fixed here

- **§16.27's transcription of the served `xyxy` for upstream blocks 1/2 has its `y2` values
  transposed** against this real recording: spec line 5433 (approx.) states `[607,630,723,703]`
  (eng) / `[607,631,724,705]` (ja); the committed artifact says `[607,630,723,**705**]` (eng) /
  `[607,631,724,**703**]` (ja). `[607,630,723,703]` is in fact block 1's `rect_yolo`, which is
  likely where the transcription slipped. The entry's *conclusion* (x1 identical, y1/x2 differ
  by 1, y2 differs by 2) still holds against the real vectors. Needs a spec erratum, not a
  silent edit — flagged here, not corrected, per this project's convention that a ratified
  transcription is corrected by a new entry, not by editing the old one.
- **A previously-quoted `mask_score` of 0.0359 for `[438,1407,498,1446]`** (spec lines ~3430,
  4012, 5494; `crates/pc-detect/src/oracle.rs:85`; `f1_oracle_comparator.rs:1058`) reads
  `0.03446455505279035` in this real run's `PROVENANCE.json`. Both are `< 0.1`, so no
  conclusion changes, but five sites now quote a number the committed artifact contradicts —
  one of them (`f1_oracle_comparator.rs:1058`) inside a frozen test file, the rest prose
  (three spec lines and one library doc comment, `oracle.rs:85`). Flagged for a follow-up
  erratum, not fixed here.
- **§14 item 17's "the filter has never fired" is contradicted by this page.** Our own
  coverage filter (§8.7(A)5) *does* fire here — measured 0.0868 on block index 3
  (`[438,1407,498,1446]`), below the 0.1 threshold, matching this doc's `mask_coverage` row
  above and the pairing table's `CoverageFilteredUpstream` entry below. §16.20 item 9 already
  records this page's outcome correctly ("both implementations dropped that box"); §14 item
  17's summary sentence is the stale one. Flagged, not corrected here.
- **`pre_filter_blocks`' `lines` arrays are empty for every index in the committed oracle
  JSON**, while `PROVENANCE.json`'s own `upstream_pre_filter_mask_scores` diagnostic reports
  `len_lines` 2 and 3 for indices 0 and 2. Harmless for this pairing (the only
  `CoverageFilteredUpstream` citation is index 4, line-less under both readings), but a future
  citation against index 0 or 2 would need this reconciled first — recorder-side question, not
  a blocker for this commit.

## Signatures (§16.20 item 3(f))

Three signatures, none the producing agent's, required before this fixture commits: two
independent re-derivations of the identities from the committed artifact, plus the human
maintainer signing the completeness of the partition and every judgment-resting row.

| # | Signer | Date | Method |
|---|---|---|---|
| 1 | `architect` subagent (independent pass, read-only, no shared context with #2) | 2026-08-02 | Re-derived the pairing, branches, derivations and both mechanisms directly from the four committed JSON files, by hand arithmetic over `identity_lines()`/`rect_yolo`, before seeing #2's answer. Flagged the two discrepancies recorded above independently. Wrote no code (read-only). |
| 2 | `rust-engineer` subagent (independent pass, write access, no shared context with #1) | 2026-08-02 | Re-derived the same pairing independently (blind to #1), then **also** wrote and ran the real-page gate test plus two adversarial mutation controls, confirming the literal survives falsification (re-pointing one pair produces 7 gating rows across every check the comparator has). Flagged the same discrepancies independently. **Caveat for signature 3's reviewer**: for the pairing/mechanism judgment itself, this is a genuine independent re-derivation; for `f1_real_page_gate.rs` specifically, this signer is also its author, so it is not a re-derivation *of that test* by someone else — the test's correctness rests on it agreeing with signer #1's independently-reached literal (which it does) rather than on a third party re-checking the test file itself before this doc's own step-1a fresh-reader pass did so. |
| 3 | ducph (human maintainer) | 2026-08-02 | Reviewed the reported pairing, both `Mechanism` citations (`ClassDuplicateOf { upstream_index: 2 }`, `CoverageFilteredUpstream { pre_filter_index: 4 }`), and the completeness partition as summarized by the orchestrating session — including the three blocking corrections a `fresh-reader` pass required before this signature (the `raw_mask` verdict, the unsourced `demo_bubbles` attribution, and the §16.24/§16.20 mis-citations) — and signs off on the whole document. |

Signature 3 is the one open item before the atomic commit lands. A `fresh-reader` pass (per
this project's step-1a convention) independently re-verified this document's claims against
the primary sources before signature 3 was sought, and found three blocking defects — the
`raw_mask` verdict bucket, the unsourced `demo_bubbles` page attribution on the IoU figures,
and four `§16.24`/`§16.20` mis-citations (this document had cited `§16.24 item 5` throughout
for text that actually lives at `§16.20 item 3` and `§16.24 items 10/11`) — all corrected in
this revision. Its full report is recorded in this session's transcript, not in
`docs/RULINGS.md` (that file is reserved for Fable ratification rulings, a different kind of
record).
