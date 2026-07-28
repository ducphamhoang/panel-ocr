//! Task **E4** — the `panel-ocr ocr` report writers (spec §12.3 step 7, §12.6,
//! §12.7(B)10, §16.11 items 11 and 12).
//!
//! Rows come from `OcrAnalytic.removed`, and that is correct rather than a gap: §15.5
//! verified that upstream's `run_ocr` path forces `ocr_blacklist_pattern = ".*"` and
//! `ocr_max_size = 10**10` (`main.py:866-868`), so in this code path *every* box is
//! "removed" and `removed` is the complete, ordered box list.
//!
//! Both writers return a `String` and never touch the filesystem — `--output FILE` is
//! `pc-cli`'s business (§1 rule 1; §12.3 step 1's exception covers *image*
//! destinations only).

use pc_core::OcrAnalytic;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReportFormat {
    Csv,
    Txt,
}

/// §12.6's CSV header, verbatim.
pub const CSV_HEADER: &str = "filename,startx,starty,endx,endy,text";

/// Dispatch, for `pc-cli`'s `--format csv|txt`.
pub fn render(format: ReportFormat, analytics: &[OcrAnalytic]) -> String {
    match format {
        ReportFormat::Csv => write_csv(analytics),
        ReportFormat::Txt => write_txt(analytics),
    }
}

/// §12.6 / §16.11 item 12. One row per box, coordinates in original-image space
/// (`rect.x1,y1,x2,y2`, §2.1's convention unmodified), RFC 4180 **minimal** quoting,
/// `\n` terminators, single trailing newline. Empty input renders `""` — not a bare
/// header: "no OCR was run" must not look like "OCR found nothing".
pub fn write_csv(analytics: &[OcrAnalytic]) -> String {
    if analytics.is_empty() {
        return String::new();
    }
    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    for analytic in analytics {
        let filename = file_name_of(analytic);
        for removed in &analytic.removed {
            let rect = removed.rect;
            out.push_str(&quote_csv_field(&filename));
            out.push_str(&format!(
                ",{},{},{},{},",
                rect.x1, rect.y1, rect.x2, rect.y2
            ));
            out.push_str(&quote_csv_field(&removed.text));
            out.push('\n');
        }
    }
    out
}

/// §12.6 / §16.11 item 12. `"{filename}: \n"` — note the space **before** the newline,
/// which the vendored fixture has — then one line per box text, then a blank line
/// before the next file. No blank line after the last file; single trailing newline.
pub fn write_txt(analytics: &[OcrAnalytic]) -> String {
    if analytics.is_empty() {
        return String::new();
    }
    let mut blocks = Vec::with_capacity(analytics.len());
    for analytic in analytics {
        let mut block = format!("{}: \n", file_name_of(analytic));
        for removed in &analytic.removed {
            block.push_str(&removed.text);
            block.push('\n');
        }
        blocks.push(block);
    }
    blocks.join("\n")
}

/// RFC 4180 minimal quoting: quote iff the field contains `,`, `"`, `\r` or `\n`;
/// an embedded `"` is doubled. Hand-written rather than via the `csv` crate, whose
/// default `\r\n` terminator would not match the fixture (§16.11 item 12).
pub fn quote_csv_field(field: &str) -> String {
    if field.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// §16.11 item 11: the fixtures show a bare `img1.jpg`, not a full path.
fn file_name_of(analytic: &OcrAnalytic) -> String {
    analytic
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| analytic.path.to_string_lossy().into_owned())
}
