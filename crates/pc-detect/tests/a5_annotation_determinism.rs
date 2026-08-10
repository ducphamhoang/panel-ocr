//! Task **A5** — spec §16.37 item 3's determinism gate, scoped by §16.46 item 9
//! (Fable tie-break, 2026-08-10). **FROZEN once written.**
//!
//! §16.46 item 9 makes this gate a **precondition** for flipping `mask_refine_mode`'s
//! shipped default to `Annotation`. It is not a precondition for the `inpainting_enabled`
//! flip, which sequences independently.
//!
//! **What this gate proves, and what it deliberately does not.** §16.37 item 3, verbatim:
//! *"The A5 gate asserts our refinement is byte-identical across runs and thread counts on
//! the recorded page. Ask what change turns it red: only nondeterminism. **A port that is
//! wrong in the same way on every run passes it forever.**"* Correctness evidence lives in
//! `a1_annotate_topk.rs`, `a2_annotate_otsu.rs`, `a3_annotate_merge.rs` and
//! `a3b_annotate_refine.rs`, which check per-function values against the upstream oracle.
//! Nothing here is a parity claim.
//!
//! **The calibration document is NOT part of this gate.** §16.37 item 3 also requires a
//! non-gating upstream comparison pinning the CPU feature set and OpenCV's IPP build/enable
//! status. §16.46 item 9(a) rules those pins attach to the *report*, not to this gate, and
//! item 9(c) records that the report stays owed. Do not read a green run here as the
//! calibration having happened.
//!
//! **Scope covers both upstream functions, by composition rather than by two gates.**
//! §16.46 item 9(a): *"One gate through `pc_detect::run` suffices — no per-function split.
//! Both `refine_mask` and `refine_undetected_mask` are covered because `run`'s Annotation
//! branch composes them (`lib.rs:118-126`)."*
//!
//! **One honest limit, transcribed rather than summarised**, from the same ruling: *"on the
//! recorded page the undetected pass invents no block (degenerate case), so the
//! recursion-over-invented-blocks branch executes its scan-and-filter but not its recursive
//! refine — this residual is inherent to A5 as ratified (item 3 scopes the gate to the
//! recorded page; a second page is quarantined to A6 by item 9) and does not widen this
//! precondition. `a3b_annotate_refine.rs`'s synthetic-input coverage remains the guard for
//! that branch."*

use image::GrayImage;
use pc_config::{MaskRefineMode, TextDetectorConfig};
use pc_core::ImageHandle;
use pc_detect::{DetectInput, ReplayDetector};
use std::path::{Path, PathBuf};

/// The recorded detector page (§7.2) and the four `DetectInput` knobs the recorder used, so
/// a test reproducing its output reproduces its inputs too. Same four values as
/// `d7_run.rs` and `a4_run_annotation.rs`; duplicated rather than shared because each frozen
/// suite states its own inputs.
const RECORDED_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const RECORDED_ORIGINAL_PATH: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg";
const RECORDED_HEIGHT_LOWER: u32 = 1000;
const RECORDED_HEIGHT_UPPER: u32 = 4000;

/// **This file's anti-vacuity anchor.** Determinism assertions over an all-zero mask hold
/// forever, so before comparing runs to each other this suite pins what a single run
/// actually produces. These three literals are the ones `a4_run_annotation.rs` already
/// freezes for the same page in the same mode; they are duplicated here **deliberately**,
/// and if the Annotation output ever legitimately changes, both files move together or the
/// two disagree about the same input — the draft-wide-contradiction failure cookbook rule 8
/// records. They are not derived from this test's own output.
const ANNOTATION_DIMENSIONS: (u32, u32) = (1200, 1660);
const ANNOTATION_NONZERO: usize = 2_942;
const ANNOTATION_FINGERPRINT: u64 = 2_790_551_277;

fn recorded_replay_detector() -> ReplayDetector {
    ReplayDetector::new(
        &pc_testkit::paths::recorded_root().join("detector"),
        RECORDED_STEM,
    )
}

/// `DetectInput` over the recorded page in `mode`. `destinations` is `None` for memory mode
/// (§4.1), or the directory the two PNGs are written into.
///
/// The mode is an explicit argument and never `TextDetectorConfig::default()`'s value: this
/// suite's subject is `Annotation`, and a test that inherited the shipped default would stop
/// testing what its name says the day that default moves. §16.46 item 13(b).
fn recorded_input(mode: MaskRefineMode, destinations: Option<&Path>) -> DetectInput {
    let page = pc_testkit::paths::recorded(format!("detector/{RECORDED_STEM}.jpg"));
    DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: ImageHandle::from_path(page),
        original_path: PathBuf::from(RECORDED_ORIGINAL_PATH),
        target_height_lower: RECORDED_HEIGHT_LOWER,
        target_height_upper: RECORDED_HEIGHT_UPPER,
        base_image_dest: destinations.map(|dir| dir.join(format!("{RECORDED_STEM}_base.png"))),
        raw_mask_dest: destinations.map(|dir| dir.join(format!("{RECORDED_STEM}_raw_mask.png"))),
        min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
        config: TextDetectorConfig {
            mask_refine_mode: mode,
            ..TextDetectorConfig::default()
        },
    }
}

fn output_mask(output: &pc_detect::DetectOutput) -> GrayImage {
    output
        .page
        .raw_mask
        .load()
        .expect("run output carries a loadable raw mask")
        .to_luma8()
}

/// The refined mask's raw bytes. **This is the comparison every determinism assertion in
/// this file rests on**, which is why
/// `the_byte_comparison_these_gates_rest_on_can_tell_two_real_masks_apart` drives this exact
/// function with two genuinely different masks rather than trusting it.
fn mask_bytes(output: &pc_detect::DetectOutput) -> Vec<u8> {
    output_mask(output).into_raw()
}

fn base_bytes(output: &pc_detect::DetectOutput) -> Vec<u8> {
    output
        .page
        .base_image
        .load()
        .expect("run output carries a loadable base image")
        .to_rgb8()
        .into_raw()
}

fn nonzero_count(mask: &GrayImage) -> usize {
    mask.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

fn positional_fingerprint(mask: &GrayImage) -> u64 {
    mask.pixels()
        .enumerate()
        .filter(|(_, pixel)| pixel.0[0] != 0)
        .map(|(index, _)| index as u64 + 1)
        .sum()
}

fn run_annotation(detector: &ReplayDetector) -> pc_detect::DetectOutput {
    pc_detect::run(recorded_input(MaskRefineMode::Annotation, None), detector)
        .expect("the recorded page runs in Annotation mode")
}

// ─────────────────────────────────────────────────────────────────────────────
// 0. The controls. These run first in file order on purpose: every assertion below
//    them is a comparison, and a comparison is worth nothing until something has shown
//    it can come out unequal.
// ─────────────────────────────────────────────────────────────────────────────

/// **Anti-vacuity for the whole file.** A single Annotation run on the recorded page must
/// produce a specific, non-degenerate mask. Without this, an implementation that returned an
/// all-zero mask — or refused the mode and fell through to something empty — would satisfy
/// every determinism assertion in this suite forever.
///
/// The three literals are `a4_run_annotation.rs`'s, not this run's output (see their
/// declaration above for why that matters).
///
/// Turns red if the Annotation path stops running, starts producing a different mask, or
/// produces one whose size changed.
#[test]
fn a_single_annotation_run_on_the_recorded_page_is_non_degenerate() {
    let detector = recorded_replay_detector();
    let mask = output_mask(&run_annotation(&detector));

    assert_eq!(mask.dimensions(), ANNOTATION_DIMENSIONS);
    assert_eq!(nonzero_count(&mask), ANNOTATION_NONZERO);
    assert_eq!(positional_fingerprint(&mask), ANNOTATION_FINGERPRINT);
}

/// **The negative control the determinism assertions depend on.** Every gate below compares
/// two `Vec<u8>`s produced by `mask_bytes` and asserts they are EQUAL. That is satisfiable by
/// a `mask_bytes` that returns a constant, by two runs that both produced nothing, and by a
/// `run` that ignored its `mask_refine_mode` and did the same thing either way.
///
/// So this drives the same function with the two modes on the same page and requires the
/// results to DIFFER. It is the reason a green run of this file is evidence rather than
/// decoration — cookbook rule 6a: a probe that cannot reach its target reports success.
///
/// Turns red if `Annotation` silently runs `Simple`, or if `mask_bytes` stops reading the
/// mask it claims to read.
#[test]
fn the_byte_comparison_these_gates_rest_on_can_tell_two_real_masks_apart() {
    let detector = recorded_replay_detector();
    let annotation = mask_bytes(&run_annotation(&detector));
    let simple = mask_bytes(
        &pc_detect::run(recorded_input(MaskRefineMode::Simple, None), &detector)
            .expect("the recorded page runs in Simple mode"),
    );

    assert_eq!(
        annotation.len(),
        simple.len(),
        "same page, same geometry: a length difference would make the inequality below \
         trivially true for the wrong reason"
    );
    assert_ne!(
        annotation, simple,
        "accepting Annotation by silently running Simple would satisfy every determinism \
         assertion in this file"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Determinism: repeated runs, thread counts, and the disk artifacts.
// ─────────────────────────────────────────────────────────────────────────────

/// §16.46 item 9(a): *"structural identity across repeated runs"*.
///
/// **Structural identity is asserted separately from byte identity, and neither implies the
/// other here.** `PageDataRaw: PartialEq` compares `ImageHandle` by `path` only (§2.3), and
/// in memory mode both sides are `None` — so the struct comparison says nothing at all about
/// pixels, which is exactly the defect cookbook rule 1 records against `d7_run.rs`'s old
/// helper. The struct comparison covers `blocks`, `scale`, `image_size`, `mask_coverage` and
/// the schema shape; `mask_bytes` and `base_bytes` cover the pixels. Both are asserted.
///
/// Turns red on any run-to-run divergence in the Annotation path: an iteration-order change
/// in the merge loop, a `HashMap` introduced into `candidate_mask_list`, or an uninitialised
/// buffer read.
#[test]
fn annotation_is_identical_across_ten_sequential_runs_structurally_and_bytewise() {
    let detector = recorded_replay_detector();
    let first = run_annotation(&detector);
    let expected_page = first.page.clone();
    let expected_mask = mask_bytes(&first);
    let expected_base = base_bytes(&first);

    for run in 1..10 {
        let output = run_annotation(&detector);
        assert_eq!(
            output.page, expected_page,
            "run {run} diverged structurally"
        );
        assert_eq!(
            mask_bytes(&output),
            expected_mask,
            "run {run} mask diverged"
        );
        assert_eq!(
            base_bytes(&output),
            expected_base,
            "run {run} base diverged"
        );
    }

    // The premise the struct comparison above needs stated rather than assumed: both handles
    // really are path-less, so `PartialEq` skipping them is a known gap covered by the two
    // byte comparisons, not an unknown one.
    assert_eq!(expected_page.base_image.path, None);
    assert_eq!(expected_page.raw_mask.path, None);
}

/// §16.46 item 9(a): *"and at least two rayon thread counts"*. Two counts, eight concurrent
/// runs each, all compared against the same single-threaded result.
///
/// `pc-detect` has no rayon dependency of its own — parallelism lives in `pc-pipeline`
/// (§4.5) — so what this exercises is the stage being driven concurrently from a pool, which
/// is how the pipeline drives it. A shared `&dyn TextDetector` is used deliberately: a
/// per-thread detector would not exercise the sharing.
///
/// **Both comparisons, for the reason the module header gives**: `PageDataRaw: PartialEq`
/// compares `ImageHandle` by `path` only (§2.3) and both sides are `None` in memory mode, so
/// the struct comparison covers `blocks`, `scale`, `image_size` and `mask_coverage` while
/// saying nothing about pixels — and `mask_bytes` covers the pixels while saying nothing
/// about the survivor set. Neither implies the other.
///
/// An earlier draft of this test compared **only** the mask bytes, so a thread-dependent
/// change to the coverage filter's survivor set would have passed it while the sequential
/// test caught it — the two halves of one gate disagreeing about what the gate checks. Found
/// by an independent review of D1, not by the author.
///
/// Turns red if the Annotation path acquires thread-affine state, or if `ReplayDetector`
/// stops being safe to share.
#[test]
fn annotation_is_identical_under_one_and_eight_rayon_threads_structurally_and_bytewise() {
    let detector = recorded_replay_detector();
    let single = run_annotation(&detector);
    let expected_page = single.page.clone();
    let expected_mask = mask_bytes(&single);
    assert_eq!(
        expected_mask.len(),
        (ANNOTATION_DIMENSIONS.0 * ANNOTATION_DIMENSIONS.1) as usize,
        "premise: the comparison below is over a full-page mask, not an empty buffer"
    );

    let shared: &dyn pc_detect::TextDetector = &detector;
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        let results: Vec<(pc_core::PageDataRaw, Vec<u8>)> = pool.install(|| {
            use rayon::prelude::*;
            (0..8)
                .into_par_iter()
                .map(|_| {
                    let output =
                        pc_detect::run(recorded_input(MaskRefineMode::Annotation, None), shared)
                            .expect("run");
                    (output.page.clone(), mask_bytes(&output))
                })
                .collect()
        });
        assert_eq!(results.len(), 8);
        for (index, (page, mask)) in results.iter().enumerate() {
            assert_eq!(
                *page, expected_page,
                "run {index} diverged structurally under {threads} rayon threads"
            );
            assert_eq!(
                *mask, expected_mask,
                "run {index} mask diverged under {threads} rayon threads"
            );
        }
    }
}

/// §16.46 item 9(a): *"plus disk-artifact byte-identity across runs"*.
///
/// The in-memory comparisons above cannot catch an encoder that is itself nondeterministic —
/// a PNG writer emitting a timestamp chunk, or a filter heuristic keyed on allocation order —
/// and the disk artifacts are what `xtask record-fixtures` commits and what every downstream
/// golden reads. So the two runs write into two separate directories and the **file bytes**
/// are compared, not the images they decode to.
///
/// Turns red if either written PNG stops being reproducible byte-for-byte.
#[test]
fn annotation_disk_artifacts_are_byte_identical_across_two_runs() {
    let detector = recorded_replay_detector();
    let first_dir = tempfile::TempDir::new().expect("temp dir");
    let second_dir = tempfile::TempDir::new().expect("temp dir");

    for dir in [first_dir.path(), second_dir.path()] {
        pc_detect::run(
            recorded_input(MaskRefineMode::Annotation, Some(dir)),
            &detector,
        )
        .expect("the recorded page runs in Annotation mode into a directory");
    }

    for suffix in ["_base.png", "_raw_mask.png"] {
        let name = format!("{RECORDED_STEM}{suffix}");
        let first = std::fs::read(first_dir.path().join(&name)).expect("first run wrote it");
        let second = std::fs::read(second_dir.path().join(&name)).expect("second run wrote it");
        assert!(
            !first.is_empty(),
            "{suffix}: an empty file would make the comparison below vacuous"
        );
        assert!(
            first == second,
            "{suffix} is not reproducible across runs ({} vs {} bytes)",
            first.len(),
            second.len()
        );
    }
}
