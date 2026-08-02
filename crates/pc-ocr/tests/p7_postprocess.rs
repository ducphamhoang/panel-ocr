//! Task P7a — `manga_ocr.ocr.post_process` (spec §9.5 row P7). FROZEN per CLAUDE.md.
//!
//! Upstream, verbatim:
//!     text = "".join(text.split())
//!     text = text.replace("…", "...")
//!     text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
//!     text = jaconv.h2z(text, ascii=True, digit=True)
//!
//! Deliberate non-coverage, stated so a later reader does not mistake it for an
//! omission: `jaconv.h2z`'s default `kana=True` also folds half-width katakana. The
//! pinned vocabulary contains ZERO half-width katakana tokens, so that branch is
//! unreachable from model output and is out of scope here.

use pc_ocr::post::post_process;

#[test]
fn post_process_removes_every_whitespace_character_including_u3000() {
    assert_eq!(post_process("\u{3000}x\u{3000}"), "ｘ");
    assert_eq!(post_process("a\tb\nc"), "ａｂｃ");
    assert_eq!(post_process("1 2"), "１２");
    assert_eq!(post_process(""), "");
}

#[test]
fn post_process_strips_the_c0_control_chars_python_str_split_treats_as_whitespace() {
    // Additive strengthening (cookbook rule 8): U+001C-U+001F are real Python `str.split()`
    // whitespace (verified: `("a\x1cb").split() == ['a', 'b']`) but were unasserted here —
    // found during the P7/P8 post-implementation review. Unreachable from real model
    // output (the vocabulary contains no C0 controls), same status as the horizontal-
    // ellipsis and half-width-katakana branches above, but ported for upstream parity.
    assert_eq!(post_process("a\u{1c}b\u{1d}c\u{1e}d\u{1f}e"), "ａｂｃｄｅ");
}

#[test]
fn post_process_maps_printable_ascii_to_the_fullwidth_block() {
    for code_point in 0x21_u32..=0x7E {
        let source = char::from_u32(code_point).unwrap().to_string();
        let expected = char::from_u32(code_point + 0xFEE0).unwrap().to_string();
        assert_eq!(post_process(&source), expected, "U+{code_point:04X}");
    }
    assert_eq!(post_process("A1"), "Ａ１");
    assert_eq!(post_process("~.!?0-9-"), "～．！？０－９－");
}

#[test]
fn post_process_collapses_runs_of_two_or_more_dots_or_middle_dots_into_that_many_dots() {
    assert_eq!(post_process(".."), "．．");
    assert_eq!(post_process("..."), "．．．");
    assert_eq!(post_process("...."), "．．．．");
    assert_eq!(post_process("・・"), "．．");
    assert_eq!(post_process("・."), "．．");
    assert_eq!(post_process(".・."), "．．．");
}

#[test]
fn post_process_leaves_a_lone_middle_dot_unchanged() {
    assert_eq!(post_process("・"), "・");
}

#[test]
fn post_process_rewrites_the_horizontal_ellipsis_before_collapsing() {
    assert_eq!(post_process("…"), "．．．");
}

/// The 12 token-id sequences produced by beam decode over the pinned ONNX files in the
/// original planning session, paired with the labels published in
/// `kha-white/manga-ocr`'s own `expected_results.json`. This gates vocab + decode +
/// post-process together against an independent oracle (upstream's own published labels).
const RECORDED_LABEL_ORACLE: &[(&[u32], &str)] = &[
    (
        &[2, 2, 4159, 3805, 893, 852, 918, 912, 925, 873, 861],
        "素直にあやまるしか",
    ),
    (
        &[
            2, 2, 4010, 2131, 889, 4847, 881, 848, 3987, 849, 896, 1038, 896, 2138, 1846, 892,
            3820, 897, 40,
        ],
        "立川で見た〝穴〟の下の巨大な眼は：",
    ),
    (
        &[2, 2, 1989, 2470, 1406, 4775, 916, 1031, 3229, 889, 875],
        "実戦剣術も一流です",
    ),
    (
        &[
            2, 2, 4036, 33, 30, 4908, 5361, 4515, 873, 854, 5505, 896, 1870, 889, 5624, 861, 893,
            1598, 1578, 887, 863, 892, 862, 923,
        ],
        "第３０話重苦しい闇の奥で静かに呼吸づきながら",
    ),
    (
        &[
            2, 2, 863, 896, 856, 985, 1021, 987, 1026, 1024, 890, 885, 888, 828, 958, 1003, 1021,
            15, 15, 15,
        ],
        "きのうハンパーヶとって、ゴメン！！！",
    ),
    (&[2, 2, 864, 917, 885], "ぎゃっ"),
    (&[2, 2, 990, 1021, 999, 1026, 1026, 1021], "ピンポーーン"),
    (
        &[
            2, 2, 58, 55, 60, 57, 15, 3946, 5249, 37, 1104, 896, 1422, 889, 950, 984, 1021, 896,
            1792, 896, 4176, 3701, 932, 918, 904, 924, 912, 875,
        ],
        "ＬＩＮＫ！私達７人の力でガノンの塔の結界をやぶります",
    ),
    (
        &[2, 2, 991, 939, 942, 940, 987, 1021, 971],
        "ファイアパンチ",
    ),
    (
        &[2, 2, 2044, 873, 5923, 885, 888, 854, 925],
        "少し黙っている",
    ),
    (&[2, 2, 929, 861, 925, 861, 892, 847, 45], "わかるかな〜？"),
    (
        &[
            2, 2, 4984, 2015, 893, 916, 1297, 3685, 893, 916, 3699, 1052, 896, 1104, 5249, 893, 15,
            15,
        ],
        "警察にも先生にも町中の人達に！！",
    ),
];

#[test]
fn the_twelve_published_manga_ocr_labels_are_reproduced_from_their_token_ids() {
    assert_eq!(RECORDED_LABEL_ORACLE.len(), 12);

    let vocab = pc_ocr::vocab::Vocab::embedded();
    let mut wrong = Vec::new();
    for (ids, label) in RECORDED_LABEL_ORACLE {
        let produced = post_process(&vocab.decode_skip_special(ids));
        if produced != *label {
            wrong.push(format!(
                "\n  got {produced:?}, upstream publishes {label:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of 12 rows diverge:{}",
        wrong.len(),
        wrong.join("")
    );
}
