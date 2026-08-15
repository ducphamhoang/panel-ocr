//! Task B1 (§16.51) — the beam-batched decode loop. FROZEN per CLAUDE.md.
//!
//! ADDITIVE to `p7_decode.rs`, which stays byte-identical: that file pins the SCORING
//! semantics, this one pins the per-step CALL SHAPE. Cookbook rule 8 exit 1, "a
//! frozen-test addition, not an amendment" — no clause of `p7_decode.rs` is contradicted
//! and none of its 7 tests changes.
//!
//! PROVENANCE OF EVERY LITERAL BELOW, so none of them is read back off the code they
//! gate. Two independent sources:
//!
//! (1) The CALL SHAPE is upstream's, measured by RUNNING upstream (cookbook rule 3), not
//!     by reading it: real `manga_ocr` on `transformers` 5.14.1/5.15.0, model
//!     `kha-white/manga-ocr-base`, `model.decoder.forward` instrumented, on
//!     `tests/fixtures/upstream/demo_bubbles/nightmare_bubble_raw.png`, 2026-08-16:
//!         TOTAL decoder forward calls: 22
//!         DISTINCT input_ids batch dims: [4]      <- batch == num_beams
//!         DISTINCT input_ids seq dims:   [1]
//!     One call per step carrying all four beams. That is what this file pins.
//!
//! (2) The exact PREFIX VECTORS were recorded from the CURRENT, pre-batching
//!     implementation at HEAD 82ce685 with a throwaway recording `LogitsSource`, so they
//!     describe behaviour that already holds and that batching must not disturb:
//!         RESULT           = [2, 4, 4, 4, 4, 3]
//!         PER-STEP BATCHES = [1, 4, 4, 4, 4]
//!         TOTAL CALLS      = 17
//!
//! Toy vocabulary, as in `p7_decode.rs`: VOCAB=12, CLS=2, SEP=3.

use pc_core::StageError;
use pc_ocr::decode::{beam_search, BeamSearchConfig, LogitsSource};
use std::cell::RefCell;

const VOCAB: usize = 12;
const CLS: u32 = 2;
const SEP: u32 = 3;

/// The scripted row used by the call-shape tests: for the first four steps tokens
/// 4/5/6/7 are the only live continuations (so exactly four beams stay alive and SEP
/// cannot finish anything), and from step 4 SEP dominates so all four beams finish on
/// the same step and `early_stopping` breaks the loop.
fn four_live_beams_row(prefix: &[u32]) -> Vec<f32> {
    let step = prefix.len() - 1;
    let mut r = vec![-50.0_f32; VOCAB];
    if step < 4 {
        r[4] = 0.0;
        r[5] = -0.01;
        r[6] = -0.02;
        r[7] = -0.03;
    } else {
        r[SEP as usize] = 0.0;
    }
    r
}

fn four_live_beams_config() -> BeamSearchConfig {
    BeamSearchConfig {
        num_beams: 4,
        length_penalty: 2.0,
        no_repeat_ngram_size: 0,
        max_length: 10,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    }
}

/// Records every `logits_batch` call as the exact list of prefixes it carried, and
/// separately counts any single-prefix `logits` call that leaks through.
struct BatchRecorder {
    batch_calls: RefCell<Vec<Vec<Vec<u32>>>>,
    single_calls: RefCell<usize>,
}

impl BatchRecorder {
    fn new() -> Self {
        Self {
            batch_calls: RefCell::new(Vec::new()),
            single_calls: RefCell::new(0),
        }
    }
}

impl LogitsSource for BatchRecorder {
    fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
        *self.single_calls.borrow_mut() += 1;
        Ok(four_live_beams_row(prefix))
    }

    fn logits_batch(&self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>, StageError> {
        self.batch_calls
            .borrow_mut()
            .push(prefixes.iter().map(|p| p.to_vec()).collect());
        Ok(prefixes.iter().map(|p| four_live_beams_row(p)).collect())
    }
}

#[test]
fn beam_search_presents_all_four_live_beams_in_one_logits_batch_call_per_step() {
    // Requirement: match the measured upstream call shape — ONE decoder call per step
    // carrying every live beam (upstream: `input_ids` (4, 1) on all 22 steps), instead
    // of today's one call per beam per step.
    let source = BatchRecorder::new();

    let result = beam_search(&source, &four_live_beams_config()).expect("decodes");

    // The exact, ordered per-call prefix lists. This is an identity assertion, not a
    // count: a swap of two beams within a step, or a step whose call carried only the
    // first beam, changes this vector while leaving every cardinality intact
    // (cookbook rule 13, "cardinality is not identity").
    let expected: Vec<Vec<Vec<u32>>> = vec![
        vec![vec![2]],
        vec![vec![2, 4], vec![2, 5], vec![2, 6], vec![2, 7]],
        vec![vec![2, 4, 4], vec![2, 4, 5], vec![2, 5, 4], vec![2, 4, 6]],
        vec![
            vec![2, 4, 4, 4],
            vec![2, 4, 4, 5],
            vec![2, 4, 5, 4],
            vec![2, 5, 4, 4],
        ],
        vec![
            vec![2, 4, 4, 4, 4],
            vec![2, 4, 4, 4, 5],
            vec![2, 4, 4, 5, 4],
            vec![2, 4, 5, 4, 4],
        ],
    ];
    assert_eq!(*source.batch_calls.borrow(), expected);

    // Anti-vacuity literal: five calls, and the per-call batch widths are 1,4,4,4,4.
    // Neither number is computable from the tree; both come from the recorded run in
    // this file's header. A per-beam implementation would produce 17 calls of width 1.
    assert_eq!(source.batch_calls.borrow().len(), 5);
    let widths: Vec<usize> = source.batch_calls.borrow().iter().map(Vec::len).collect();
    assert_eq!(widths, vec![1, 4, 4, 4, 4]);

    // The batch path is used EXCLUSIVELY: `beam_search` must not also fall back to the
    // single-prefix method for any beam.
    assert_eq!(
        *source.single_calls.borrow(),
        0,
        "beam_search called the single-prefix `logits` {} time(s); it must route every \
         beam through `logits_batch`",
        source.single_calls.borrow()
    );

    // And the decoded sequence is unchanged by any of the above.
    assert_eq!(result, vec![2, 4, 4, 4, 4, 3]);
}

#[test]
fn the_default_logits_batch_calls_logits_once_per_prefix_in_the_given_order() {
    // Requirement: this default is the ONLY reason `p7_decode.rs` can stay frozen and
    // green. If it reordered, deduplicated, or short-circuited, that file's
    // `..._reruns_the_full_prefix_each_step` assertion would be checking a call sequence
    // the production loop never makes.
    struct DefaultOnly {
        seen: RefCell<Vec<Vec<u32>>>,
    }
    impl LogitsSource for DefaultOnly {
        fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
            self.seen.borrow_mut().push(prefix.to_vec());
            // A distinguishable row per prefix, so the returned ORDER is checkable too.
            let mut r = vec![-50.0_f32; VOCAB];
            r[prefix.len()] = prefix.len() as f32;
            Ok(r)
        }
    }
    let source = DefaultOnly {
        seen: RefCell::new(Vec::new()),
    };

    let rows = source
        .logits_batch(&[&[2][..], &[2, 5][..], &[2, 9, 9][..]])
        .expect("default fan-out");

    // Exact, ordered — not a set, not a count.
    assert_eq!(
        *source.seen.borrow(),
        vec![vec![2_u32], vec![2, 5], vec![2, 9, 9]]
    );
    assert_eq!(rows.len(), 3);
    // Row i must be the row produced for prefix i, in that order. The marker index is
    // `prefix.len()`, so rows 0/1/2 carry their signal at indices 1/2/3.
    assert_eq!(rows[0][1], 1.0);
    assert_eq!(rows[1][2], 2.0);
    assert_eq!(rows[2][3], 3.0);
}

#[test]
fn overriding_logits_batch_yields_the_same_decoded_sequence_as_the_default_fan_out() {
    // Requirement: batching is a call-shape change only. The measured decoder identity
    // that licenses this (batched row i == batch-1 run for prefix i, max abs diff 0e0
    // across 4 rows on the real pinned decoder, 2026-08-16) is pinned separately in
    // `p7f_decode_step_batch.rs`; this test pins that the LOOP does not reorder or
    // rescore anything when the batch path is taken.
    struct DefaultOnly;
    impl LogitsSource for DefaultOnly {
        fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
            Ok(four_live_beams_row(prefix))
        }
    }

    let batched = BatchRecorder::new();
    let batched_result = beam_search(&batched, &four_live_beams_config()).expect("decodes");
    let default_result = beam_search(&DefaultOnly, &four_live_beams_config()).expect("decodes");

    // Both sides are asserted against a HARD-CODED literal, not only against each other:
    // comparing two runs of the same loop would pass even if the loop were wrong for
    // both (cookbook rule 7).
    assert_eq!(batched_result, vec![2, 4, 4, 4, 4, 3]);
    assert_eq!(default_result, vec![2, 4, 4, 4, 4, 3]);
    // Anti-vacuity: the two sides really did take different paths. Without this, a
    // regression that ignored `logits_batch` entirely would satisfy everything above.
    assert_eq!(*batched.single_calls.borrow(), 0);
    assert_eq!(batched.batch_calls.borrow().len(), 5);
}

#[test]
fn a_logits_batch_failure_aborts_the_step_and_propagates_the_source_error() {
    struct Failing;
    impl LogitsSource for Failing {
        fn logits(&self, _prefix: &[u32]) -> Result<Vec<f32>, StageError> {
            unreachable!("the batch path must be used")
        }
        fn logits_batch(&self, _prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>, StageError> {
            Err(StageError::Inference("decoder exploded".into()))
        }
    }

    let error = beam_search(&Failing, &four_live_beams_config()).expect_err("propagates");

    assert!(matches!(error, StageError::Inference(_)));
    assert!(
        error.to_string().contains("decoder exploded"),
        "the source's own message must survive, got: {error}"
    );
}

#[test]
fn an_empty_logits_row_is_rejected_by_its_row_index_not_silently_skipped() {
    // Requirement: `decode.rs`'s existing empty-row check must be preserved per ROW, and
    // must say WHICH row (cookbook rule 13: diagnostics are part of the gate).
    struct EmptySecondRow;
    impl LogitsSource for EmptySecondRow {
        fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
            Ok(four_live_beams_row(prefix))
        }
        fn logits_batch(&self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>, StageError> {
            Ok(prefixes
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if i == 1 {
                        Vec::new()
                    } else {
                        four_live_beams_row(p)
                    }
                })
                .collect())
        }
    }

    let error = beam_search(&EmptySecondRow, &four_live_beams_config()).expect_err("rejects");

    assert!(matches!(error, StageError::InvalidInput(_)));
    let message = error.to_string();
    assert!(
        message.contains('1'),
        "the message must name the offending row index 1, got: {message}"
    );
}

#[test]
fn a_row_count_that_disagrees_with_the_prefix_count_is_an_error_not_a_truncation() {
    // Requirement: a source returning fewer rows than beams must fail loudly. Silently
    // consuming only the returned rows would drop beams and change the decode, which is
    // exactly the class of bug a green suite hides.
    // Construction note: this must under-return specifically at a step with 4 live
    // beams (the message contract this test checks names both the sent and returned
    // counts as "4" and "2"). Subtracting a flat 2 from every call's length would
    // instead fire on step 1's single seed beam (1 prefix -> 0 rows), which is a real
    // mismatch but not the one this test's literals describe — so this stub only
    // truncates once 4 beams are actually live, leaving step 1 untouched.
    struct ShortByTwo;
    impl LogitsSource for ShortByTwo {
        fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
            Ok(four_live_beams_row(prefix))
        }
        fn logits_batch(&self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>, StageError> {
            let keep = if prefixes.len() == 4 {
                2
            } else {
                prefixes.len()
            };
            Ok(prefixes
                .iter()
                .take(keep)
                .map(|p| four_live_beams_row(p))
                .collect())
        }
    }

    let error = beam_search(&ShortByTwo, &four_live_beams_config()).expect_err("rejects");

    assert!(matches!(error, StageError::InvalidInput(_)));
    let message = error.to_string();
    // Both numbers, so the message is actionable: 4 beams were sent, 2 rows came back.
    assert!(
        message.contains('4') && message.contains('2'),
        "the message must state both the expected and the actual row count, got: {message}"
    );
}
