//! Task **E4** -- the OCR report writers: spec §12.3 step 7, §12.6, §12.7(B)10,
//! §16.11 items 11 and 12. FROZEN.

mod common;

use common::{csv_fixture_analytics, read_fixture_trimmed, txt_fixture_analytics};
use pc_core::{OcrAnalytic, Rect};
use pc_export::ocr_report::{
    quote_csv_field, render, write_csv, write_txt, ReportFormat, CSV_HEADER,
};
use std::path::PathBuf;

#[test]
fn b10_csv_reproduces_the_vendored_fixture_byte_for_byte() {
    // §12.6 / §12.7(B)10: writing an `OcrAnalytic` for `img1.jpg` with boxes
    // (100,100,300,200)/"some text perhaps" and (534,275,592,414)/"or nothing at all"
    // must reproduce `ocr_output/good_detected_text.csv` -- modulo the trailing
    // newline the fixture lacks (§16.11 item 12).
    let expected = read_fixture_trimmed(&pc_testkit::paths::ocr_output("good_detected_text.csv"));
    let actual = write_csv(&csv_fixture_analytics());
    assert_eq!(
        actual.trim_end_matches('\n'),
        expected,
        "CSV report must match the vendored fixture byte-for-byte"
    );
    assert!(
        actual.ends_with('\n') && !actual.ends_with("\n\n"),
        "the writer emits exactly one trailing newline (§16.11 item 12)"
    );
    assert!(
        !actual.contains('\r'),
        "the terminator is LF, not CRLF -- the reason this is hand-written and not the \
         `csv` crate (§16.11 item 12)"
    );
}

#[test]
fn b10_txt_reproduces_the_vendored_fixture_byte_for_byte() {
    // §12.6 / §12.7(B)10: two files (page1.jpg 2 lines, page2.jpg 3 lines), with the
    // fixture's exact `"{name}: "` header form -- colon, SPACE, newline.
    let expected = read_fixture_trimmed(&pc_testkit::paths::ocr_output("good_detected_text.txt"));
    let actual = write_txt(&txt_fixture_analytics());
    assert_eq!(
        actual.trim_end_matches('\n'),
        expected,
        "TXT report must match the vendored fixture byte-for-byte"
    );
    assert!(
        actual.contains("page1.jpg: \n"),
        "the header keeps the space before the newline (§16.11 item 12): {actual:?}"
    );
    assert!(
        actual.contains("bird.\n\npage2.jpg"),
        "exactly one blank line separates two files: {actual:?}"
    );
}

#[test]
fn the_csv_header_is_the_spec_literal_and_rows_are_x1_y1_x2_y2() {
    // §12.3 step 7's header, and §16.11 item 12's coordinate rule: the row carries
    // `rect.x1,y1,x2,y2` unmodified (§2.1's convention is NOT adjusted here).
    assert_eq!(CSV_HEADER, "filename,startx,starty,endx,endy,text");
    let csv = write_csv(&[common::analytic(
        "/some/dir/img1.jpg",
        &[(Rect::new(7, 8, 9, 10), "t")],
    )]);
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some(CSV_HEADER));
    // §16.11 item 11: the filename column is the BARE file name, not the full path.
    assert_eq!(lines.next(), Some("img1.jpg,7,8,9,10,t"));
    assert_eq!(lines.next(), None);
}

#[test]
fn csv_quoting_is_rfc_4180_minimal() {
    // §16.11 item 12: quote iff the field contains `,`, `"`, CR or LF; embedded `"`
    // doubled. Everything else is emitted bare, which is why the fixture has no quotes.
    assert_eq!(quote_csv_field("plain text"), "plain text");
    assert_eq!(quote_csv_field("a,b"), "\"a,b\"");
    assert_eq!(quote_csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    assert_eq!(quote_csv_field("two\nlines"), "\"two\nlines\"");
    assert_eq!(quote_csv_field("cr\rlf"), "\"cr\rlf\"");

    let csv = write_csv(&[common::analytic(
        "img,1.jpg",
        &[(Rect::new(0, 0, 1, 1), "he said \"no\", loudly")],
    )]);
    assert_eq!(
        csv.lines().nth(1),
        Some("\"img,1.jpg\",0,0,1,1,\"he said \"\"no\"\", loudly\"")
    );
}

#[test]
fn an_empty_analytics_slice_renders_an_empty_string_in_both_formats() {
    // §16.11 item 12: NOT a bare CSV header -- "no OCR was run" must not look like
    // "OCR found nothing".
    assert_eq!(write_csv(&[]), "");
    assert_eq!(write_txt(&[]), "");
    assert_eq!(render(ReportFormat::Csv, &[]), "");
    assert_eq!(render(ReportFormat::Txt, &[]), "");
}

#[test]
fn a_file_with_no_boxes_still_gets_its_txt_header() {
    // A page where every box was filtered still appears in the report -- the run
    // processed it. Its CSV contribution is zero rows; its TXT block is the header.
    let analytics = vec![
        common::analytic("page1.jpg", &[]),
        common::analytic("page2.jpg", &[(Rect::new(0, 0, 1, 1), "x")]),
    ];
    assert_eq!(write_txt(&analytics), "page1.jpg: \n\npage2.jpg: \nx\n");
    assert_eq!(
        write_csv(&analytics),
        format!("{CSV_HEADER}\npage2.jpg,0,0,1,1,x\n")
    );
}

#[test]
fn rows_follow_analytics_order_then_removed_order() {
    // §5.7 determinism: no sorting, no grouping -- the report is a faithful dump.
    let analytics = vec![
        common::analytic("b.jpg", &[(Rect::new(1, 1, 2, 2), "second")]),
        common::analytic("a.jpg", &[(Rect::new(3, 3, 4, 4), "first")]),
    ];
    let lines = write_csv(&analytics)
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .join("|");
    assert_eq!(lines, "b.jpg,1,1,2,2,second|a.jpg,3,3,4,4,first");
}

#[test]
fn the_report_reads_removed_not_the_area_vectors() {
    // §16.11 item 11 / §15.5: `OcrAnalytic.removed` is the complete ordered box list in
    // the `run_ocr` code path (upstream forces `ocr_blacklist_pattern = ".*"` there),
    // so a populated `box_areas_ocred` with an empty `removed` yields no rows at all.
    let analytic = OcrAnalytic {
        path: PathBuf::from("img1.jpg"),
        num_boxes: 3,
        box_areas_ocred: vec![10, 20, 30],
        box_areas_removed: Vec::new(),
        removed: Vec::new(),
    };
    assert_eq!(
        write_csv(std::slice::from_ref(&analytic)),
        format!("{CSV_HEADER}\n")
    );
    assert_eq!(write_txt(&[analytic]), "img1.jpg: \n");
}
