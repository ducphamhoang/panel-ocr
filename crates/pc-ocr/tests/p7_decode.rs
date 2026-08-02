//! Task P7d — the beam-search decode loop (spec §16.30 item 4, the ratified decode
//! strategy). FROZEN per CLAUDE.md.
//!
//! Every expected sequence below was independently verified three ways before this file
//! was written: real HuggingFace `transformers==4.51.1`'s `generate()`, real
//! `transformers==5.14.1`'s `generate()` (a structurally different, newer internal
//! implementation — the two library versions agree exactly), and a from-scratch,
//! non-transformers Python reimplementation of the classic beam-search algorithm. All
//! three produced byte-identical token sequences for all five cases. If your
//! implementation disagrees with any of these, the bug is in the implementation.
//!
//! Toy vocabulary for these tests: VOCAB=12, CLS=2 (decoder_start_token_id),
//! SEP=3 (eos_token_id). Unrelated to the real 6144-token manga-ocr vocabulary —
//! these tests exercise the decode loop's CONTROL FLOW and SCORING, not real tokens.

use pc_core::StageError;
use pc_ocr::decode::{beam_search, BeamSearchConfig, LogitsSource};

const VOCAB: usize = 12;
const CLS: u32 = 2;
const SEP: u32 = 3;

/// A `LogitsSource` driven by a plain Rust closure, mirroring the Python probe scripts'
/// `logits_fn(seq, step)` shape exactly: `step` is `prefix.len() - 1` (the 0-indexed
/// position just reached).
struct Scripted<F: Fn(&[u32], usize) -> Vec<f32>> {
    logits_fn: F,
}

impl<F: Fn(&[u32], usize) -> Vec<f32>> LogitsSource for Scripted<F> {
    fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
        Ok((self.logits_fn)(prefix, prefix.len() - 1))
    }
}

fn row(entries: &[(u32, f32)]) -> Vec<f32> {
    let mut r = vec![-50.0_f32; VOCAB];
    for &(token, value) in entries {
        r[token as usize] = value;
    }
    r
}

#[test]
fn case1_basic_beam_mechanics_the_locally_best_first_token_does_not_win() {
    // Verified sequence: [2, 6, 7, 3]. At step 0, token 5 has a HIGHER single-step logit
    // than token 6 (0.0 vs -0.05) -- but continuing from 6 leads to a much better second
    // token than continuing from 5 does, so beam search's winning path goes through 6, not
    // the locally-better 5. A greedy decoder would have picked 5. This is the load-bearing
    // proof that the port is doing real beam search, not greedy-with-extra-steps.
    let source = Scripted {
        logits_fn: |seq: &[u32], step: usize| -> Vec<f32> {
            if step == 0 {
                row(&[(5, 0.0), (6, -0.05)])
            } else if step == 1 {
                if *seq.last().unwrap() == 5 {
                    row(&[(SEP, 0.0), (7, -3.0)])
                } else {
                    row(&[(SEP, -0.01), (7, 0.0)])
                }
            } else {
                row(&[(SEP, 0.0)])
            }
        },
    };
    let config = BeamSearchConfig {
        num_beams: 4,
        length_penalty: 2.0,
        no_repeat_ngram_size: 3,
        max_length: 10,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    };

    let result = beam_search(&source, &config).expect("decodes");

    assert_eq!(result, vec![2, 6, 7, 3]);
}

#[test]
fn case2_no_repeat_ngram_size_three_bans_a_third_repeated_trigram() {
    // Verified sequence: [2, 4, 4, 4, 8, 4, 4, 3]. Token 4 is always the single best choice
    // (logit 0.0 every step) and SEP is intentionally uncompetitive until step 6, so absent
    // ngram banning the winning beam would just repeat "4,4,4,4,4,4...". With
    // no_repeat_ngram_size=3: after "[2,4,4]" the trigram (4,4)->4 has been seen once; the
    // NEXT occurrence of prefix (4,4) as the last two tokens would repeat that trigram, so
    // token 4 is banned at that step and the beam is forced to token 8 (next best) instead.
    // The literal `892, 862` -> `923` gap this pattern exists to catch in the real 6144-vocab
    // decode is exactly this shape: a repeated-content trigram forcing an alternative token.
    let source = Scripted {
        logits_fn: |_seq: &[u32], step: usize| -> Vec<f32> {
            let sep_value = if step < 6 { -2.0 } else { 0.0 };
            row(&[(4, 0.0), (8, -1.0), (SEP, sep_value)])
        },
    };
    let config = BeamSearchConfig {
        num_beams: 4,
        length_penalty: 2.0,
        no_repeat_ngram_size: 3,
        max_length: 10,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    };

    let result = beam_search(&source, &config).expect("decodes");

    assert_eq!(result, vec![2, 4, 4, 4, 8, 4, 4, 3]);
}

#[test]
fn case3_max_length_truncates_without_appending_eos_when_sep_never_competes() {
    // Verified sequence: [2, 7, 7, 7, 7, 7, 7, 7] -- length exactly 8 == max_length, no
    // trailing SEP. SEP's logit is set so far below token 7's that it never enters the
    // top-2*num_beams candidate set at any step, so no beam ever finishes; the loop must
    // still terminate at max_length and return the best truncated (un-EOS'd) beam.
    let source = Scripted {
        logits_fn: |_seq: &[u32], _step: usize| -> Vec<f32> {
            let mut r = vec![-1.0e9_f32; VOCAB];
            r[7] = 0.0;
            r[SEP as usize] = -1.0e9;
            r
        },
    };
    let config = BeamSearchConfig {
        num_beams: 4,
        length_penalty: 2.0,
        no_repeat_ngram_size: 0,
        max_length: 8,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    };

    let result = beam_search(&source, &config).expect("decodes");

    assert_eq!(result, vec![2, 7, 7, 7, 7, 7, 7, 7]);
    assert_eq!(result.len(), 8);
    assert!(
        !result.contains(&SEP),
        "no beam ever reached EOS in this case"
    );
}

#[test]
fn case4_length_penalty_two_point_zero_favors_the_longer_higher_total_score_path() {
    // Verified sequence with length_penalty=2.0 (the ratified value): [2, 9, 6, 3]. Two
    // competing first tokens (9 and 10) lead to a short 2-token completion (10->SEP) or a
    // longer 3-token completion (9->6->SEP) with a better raw cumulative log-prob. At
    // length_penalty=2.0 the longer path's better total score, divided by its longer
    // length^2.0, still wins.
    let source = Scripted {
        logits_fn: |seq: &[u32], step: usize| -> Vec<f32> {
            if step == 0 {
                row(&[(9, -0.1), (10, -0.05)])
            } else if step == 1 {
                if *seq.last().unwrap() == 9 {
                    row(&[(SEP, -5.0), (6, -0.01)])
                } else {
                    row(&[(SEP, -0.01), (6, -5.0)])
                }
            } else if step == 2 {
                row(&[(SEP, -0.01)])
            } else {
                row(&[(SEP, 0.0)])
            }
        },
    };
    let config = BeamSearchConfig {
        num_beams: 4,
        length_penalty: 2.0,
        no_repeat_ngram_size: 0,
        max_length: 10,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    };

    let result = beam_search(&source, &config).expect("decodes");

    assert_eq!(result, vec![2, 9, 6, 3]);
}

#[test]
fn case5_length_penalty_zero_point_zero_favors_the_shorter_path_on_the_same_logits() {
    // Same exact logits function as case 4, only `length_penalty` changed to 0.0. Verified
    // sequence: [2, 10, 3] -- WITHOUT length normalization, the raw (unfavorable, more
    // negative) cumulative score of the longer path no longer wins, so the winner flips to
    // the shorter path. This isolates length_penalty's effect: it is the ONLY parameter
    // that differs between this test and case 4, and it changes the winning sequence.
    let source = Scripted {
        logits_fn: |seq: &[u32], step: usize| -> Vec<f32> {
            if step == 0 {
                row(&[(9, -0.1), (10, -0.05)])
            } else if step == 1 {
                if *seq.last().unwrap() == 9 {
                    row(&[(SEP, -5.0), (6, -0.01)])
                } else {
                    row(&[(SEP, -0.01), (6, -5.0)])
                }
            } else if step == 2 {
                row(&[(SEP, -0.01)])
            } else {
                row(&[(SEP, 0.0)])
            }
        },
    };
    let config = BeamSearchConfig {
        num_beams: 4,
        length_penalty: 0.0,
        no_repeat_ngram_size: 0,
        max_length: 10,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    };

    let result = beam_search(&source, &config).expect("decodes");

    assert_eq!(result, vec![2, 10, 3]);
}

#[test]
fn the_loop_reseeds_with_the_configured_start_token_and_reruns_the_full_prefix_each_step() {
    // The pinned decoder ONNX export has NO KV-cache input (P7c's recorded signature: only
    // `input_ids` + `encoder_hidden_states`), so every step must present the FULL prefix
    // seen so far, not just the newest token. Verified by recording every prefix this
    // source was actually called with, using num_beams=1's-worth of signal (a single
    // dominant token each step so only one beam matters) — checks call shape, not scoring.
    use std::cell::RefCell;
    let seen_prefixes: RefCell<Vec<Vec<u32>>> = RefCell::new(Vec::new());
    let source = Scripted {
        logits_fn: |seq: &[u32], _step: usize| -> Vec<f32> {
            seen_prefixes.borrow_mut().push(seq.to_vec());
            let mut r = vec![-50.0_f32; VOCAB];
            let next = if seq.len() < 3 { 5 } else { SEP };
            r[next as usize] = 0.0;
            r
        },
    };
    let config = BeamSearchConfig {
        num_beams: 4,
        length_penalty: 2.0,
        no_repeat_ngram_size: 0,
        max_length: 10,
        decoder_start_token_id: CLS,
        eos_token_id: SEP,
    };

    let result = beam_search(&source, &config).expect("decodes");

    assert_eq!(result, vec![2, 5, 5, 3]);
    // Every call for the winning beam saw the FULL accumulated prefix, not just the last
    // token -- e.g. by the third call for this beam's lineage, a length-3 prefix [2,5,5]
    // must have been presented, not a length-1 slice.
    assert!(
        seen_prefixes
            .borrow()
            .iter()
            .any(|p| p == &vec![2_u32, 5, 5]),
        "no call presented the full 3-token prefix [2,5,5]; seen: {:?}",
        seen_prefixes.borrow()
    );
}

#[test]
fn manga_ocr_beam_config_matches_the_ratified_spec_16_30_item_4_values() {
    use pc_ocr::decode::MANGA_OCR_BEAM_CONFIG;

    assert_eq!(MANGA_OCR_BEAM_CONFIG.num_beams, 4);
    assert_eq!(MANGA_OCR_BEAM_CONFIG.length_penalty, 2.0);
    assert_eq!(MANGA_OCR_BEAM_CONFIG.no_repeat_ngram_size, 3);
    assert_eq!(MANGA_OCR_BEAM_CONFIG.max_length, 300);
    assert_eq!(MANGA_OCR_BEAM_CONFIG.decoder_start_token_id, 2);
    assert_eq!(MANGA_OCR_BEAM_CONFIG.eos_token_id, 3);
}
