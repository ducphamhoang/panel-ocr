//! Task P7a — the vendored manga-ocr vocabulary (Fable ruling 2, 2026-08-02, spec §16.30 item 2).
//! spec §9.5 row P7. FROZEN per CLAUDE.md.
//!
//! The vocabulary is `include_str!`d from `crates/pc-ocr/assets/vocab.txt`, NOT fetched
//! and NOT copied under `tests/fixtures/`. Ruling 2(a) reason: no test-reachable code path
//! may perform network I/O, and pure vocab/decode logic must stay in the default tier.
//!
//! Provenance of every literal here: `kha-white/manga-ocr-base` @
//! `aa6573bd10b0d446cbf622e29c3e084914df9741`, measured with `curl | sha256sum | wc`,
//! independently of anything this repo computes.

use pc_ocr::vocab::{Vocab, CLS_ID, MASK_ID, PAD_ID, SEP_ID, UNK_ID, VOCAB_LEN, VOCAB_TXT};
use sha2::{Digest, Sha256};

/// Measured: `sha256sum vocab.txt` on the file fetched from the pinned commit.
const PINNED_SHA256: &str = "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d";
/// Measured: `wc -c`. Ruling 2 rejected mayocream's 30,216-byte CRLF copy; 30,216 is
/// 24,072 + 6,144, i.e. exactly one extra byte per line. If this ever reads 30,216 the
/// wrong vocabulary has been vendored.
const PINNED_SIZE_BYTES: usize = 24_072;

#[test]
fn the_embedded_vocab_is_the_pinned_khawhite_file() {
    let digest = Sha256::digest(VOCAB_TXT.as_bytes());
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    assert_eq!(VOCAB_TXT.len(), PINNED_SIZE_BYTES);
    assert_eq!(hex, PINNED_SHA256);
}

#[test]
fn the_embedded_vocab_contains_no_carriage_return() {
    let carriage_returns = VOCAB_TXT.bytes().filter(|byte| *byte == b'\r').count();

    assert_eq!(
        carriage_returns, 0,
        "the vendored vocabulary is CRLF-terminated"
    );
}

#[test]
fn the_vocab_has_6144_entries_and_the_five_special_ids_at_0_through_4() {
    let vocab = Vocab::embedded();

    assert_eq!(vocab.len(), VOCAB_LEN);
    assert_eq!(vocab.len(), 6_144);
    assert_eq!(vocab.token(PAD_ID), Some("[PAD]"));
    assert_eq!(vocab.token(UNK_ID), Some("[UNK]"));
    assert_eq!(vocab.token(CLS_ID), Some("[CLS]"));
    assert_eq!(vocab.token(SEP_ID), Some("[SEP]"));
    assert_eq!(vocab.token(MASK_ID), Some("[MASK]"));
    assert_eq!(vocab.token(VOCAB_LEN as u32), None);
}

#[test]
fn the_printable_ascii_block_is_contiguous_from_id_15() {
    // ids 5..=14 are `<unused0>`..`<unused9>`, so printable ASCII 0x21..=0x7E must occupy
    // ids 15..=108 in code-point order. Spot-measured against the real file:
    // '.' = 28, '0' = 30, 'A' = 47.
    let vocab = Vocab::embedded();

    for (offset, code_point) in (0x21_u32..=0x7E).enumerate() {
        let id = 15 + offset as u32;
        let expected = char::from_u32(code_point).expect("ASCII is a valid scalar value");
        assert_eq!(
            vocab.token(id),
            Some(expected.to_string().as_str()),
            "id {id} should be U+{code_point:04X}"
        );
    }
    assert_eq!(vocab.token(28), Some("."));
    assert_eq!(vocab.token(30), Some("0"));
    assert_eq!(vocab.token(47), Some("A"));
    assert_eq!(vocab.token(1_025), Some("・"));
}

#[test]
fn decode_skip_special_omits_the_five_bracket_tokens_wherever_they_appear() {
    // Additive strengthening (cookbook rule 8): the original array omitted UNK_ID, so
    // deleting it from the skip-set kept every test green (found during the P7/P8
    // post-implementation review). All five special ids are now exercised.
    let vocab = Vocab::embedded();

    let decoded =
        vocab.decode_skip_special(&[CLS_ID, CLS_ID, 47, MASK_ID, 30, UNK_ID, SEP_ID, PAD_ID]);

    assert_eq!(decoded, "A0");
}

#[test]
fn decode_skip_special_joins_tokens_with_no_separator() {
    let vocab = Vocab::embedded();

    assert_eq!(vocab.decode_skip_special(&[47, 47, 47]), "AAA");
    assert_eq!(vocab.decode_skip_special(&[]), "");
}
