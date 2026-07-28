//! Golden comparison -- the shape of §10.7(B) item 15's calibration record and the
//! assertion helper behind §11.7(B)12/13 and §12.7(A)1/2.
//!
//! Two distinct uses, deliberately kept separate:
//!   * `GoldenReport` -- **measure and record**, never assert. `cargo xtask
//!     calibrate-goldens` (F2) writes these rows into `docs/GOLDEN_CALIBRATION.md`.
//!     Per §15.2 / ATTRIBUTION.md, no pass/fail assertion may be written against a
//!     `*_clean.png`, so the report type has no `assert` of its own by default.
//!   * `GoldenThresholds::assert_met` -- **assert**, for the gates that ARE frozen
//!     (§11.7(B)12, §12.7(A)1/2), where the tolerances came from F2 calibration.

use crate::metrics::{
    diff_set_gray, exact_equal_fraction_gray, iou, max_delta_gray, mean_abs_diff_gray, ssim_gray,
    PixelSet,
};
use image::GrayImage;

/// Every number §10.7(B) item 15 asks to be recorded, in one struct.
#[derive(Debug, Clone, PartialEq)]
pub struct GoldenReport {
    pub name: String,
    pub dims: (u32, u32),
    /// §10.7(B): "% of pixels exactly equal", as a fraction in `0.0..=1.0`.
    pub exact_fraction: f64,
    /// Max per-pixel delta among non-equal pixels; `0` when everything is equal.
    pub max_delta: u8,
    pub mean_abs_diff: f64,
    /// Global SSIM, 8x8 tiles (see `metrics` for the pinned definition).
    pub ssim: f64,
    /// `IoU(G, O)` where `G = {p : source != upstream}` and `O = {p : source != ours}`.
    /// `None` when no `source` image was supplied (the shape metric needs three images).
    pub shape_iou: Option<f64>,
    /// §10.7(B): whether `O` is a subset of `dilate(G, 2px)`.
    pub shape_subset_of_dilated: Option<bool>,
}

impl GoldenReport {
    /// Value metrics only -- for the two-image case (§11.7(B)12, §12.7(A)2).
    pub fn compare_gray(name: &str, expected: &GrayImage, actual: &GrayImage) -> Self {
        Self {
            name: name.into(),
            dims: expected.dimensions(),
            exact_fraction: exact_equal_fraction_gray(expected, actual),
            max_delta: max_delta_gray(expected, actual),
            mean_abs_diff: mean_abs_diff_gray(expected, actual),
            ssim: ssim_gray(expected, actual),
            shape_iou: None,
            shape_subset_of_dilated: None,
        }
    }

    /// Value metrics **plus** §10.7(B)'s shape metric, which needs the untouched
    /// `source` image to derive the two changed-pixel sets.
    pub fn compare_gray_with_shape(
        name: &str,
        source: &GrayImage,
        expected: &GrayImage,
        actual: &GrayImage,
        dilate_radius: u32,
    ) -> Self {
        let mut report = Self::compare_gray(name, expected, actual);
        let upstream_changes = diff_set_gray(source, expected);
        let our_changes = diff_set_gray(source, actual);
        report.shape_iou = Some(iou(&upstream_changes, &our_changes));
        report.shape_subset_of_dilated = Some(is_within_dilated(
            &our_changes,
            &upstream_changes,
            dilate_radius,
        ));
        report
    }

    /// One `docs/GOLDEN_CALIBRATION.md` table row (leading and trailing `|` included).
    pub fn to_markdown_row(&self) -> String {
        let shape_iou = self
            .shape_iou
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "—".into());
        let subset = self
            .shape_subset_of_dilated
            .map(|value| value.to_string())
            .unwrap_or_else(|| "—".into());
        format!(
            "| {} | {}×{} | {:.6} | {} | {:.6} | {:.6} | {} | {} |",
            self.name,
            self.dims.0,
            self.dims.1,
            self.exact_fraction,
            self.max_delta,
            self.mean_abs_diff,
            self.ssim,
            shape_iou,
            subset
        )
    }

    /// Column header matching `to_markdown_row`, plus the `|---|` separator line.
    pub fn markdown_header() -> String {
        concat!(
            "| Name | Dimensions | Exact fraction | Max delta | Mean abs diff | SSIM | ",
            "Shape IoU | Within dilation |\n",
            "|---|---:|---:|---:|---:|---:|---:|:---:|"
        )
        .into()
    }
}

/// The frozen side. Every field is optional so a gate can constrain only the metrics
/// its spec section actually names.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GoldenThresholds {
    pub min_ssim: Option<f64>,
    pub max_mean_abs_diff: Option<f64>,
    pub max_delta: Option<u8>,
    pub min_exact_fraction: Option<f64>,
    pub min_iou: Option<f64>,
    pub require_subset_of_dilated: bool,
}

impl GoldenThresholds {
    /// §11.7(B)12: `SSIM >= 0.98`, `mean |delta| <= 1.0`, `max delta <= 8`.
    pub fn nlm_parity() -> Self {
        Self {
            min_ssim: Some(0.98),
            max_mean_abs_diff: Some(1.0),
            max_delta: Some(8),
            ..Self::default()
        }
    }

    /// §12.7(A)2: `max delta <= 6`, `SSIM >= 0.99` (PNG -> JPEG q95).
    pub fn jpeg_q95() -> Self {
        Self {
            min_ssim: Some(0.99),
            max_delta: Some(6),
            ..Self::default()
        }
    }

    /// Panics with a message naming **every** unmet threshold and its measured value
    /// -- a parity failure must report all the numbers at once, not one per re-run.
    #[track_caller]
    pub fn assert_met(&self, report: &GoldenReport) {
        let failures = self.unmet(report);
        assert!(
            failures.is_empty(),
            "golden comparison `{}` failed:\n{}",
            report.name,
            failures.join("\n")
        );
    }

    /// Non-panicking form: the list of human-readable failure descriptions.
    pub fn unmet(&self, report: &GoldenReport) -> Vec<String> {
        let mut failures = Vec::new();
        if let Some(minimum) = self.min_ssim {
            if report.ssim < minimum {
                failures.push(format!(
                    "ssim measured {} is below minimum {minimum}",
                    report.ssim
                ));
            }
        }
        if let Some(maximum) = self.max_mean_abs_diff {
            if report.mean_abs_diff > maximum {
                failures.push(format!(
                    "mean_abs_diff measured {} exceeds maximum {maximum}",
                    report.mean_abs_diff
                ));
            }
        }
        if let Some(maximum) = self.max_delta {
            if report.max_delta > maximum {
                failures.push(format!(
                    "max_delta measured {} exceeds maximum {maximum}",
                    report.max_delta
                ));
            }
        }
        if let Some(minimum) = self.min_exact_fraction {
            if report.exact_fraction < minimum {
                failures.push(format!(
                    "exact_fraction measured {} is below minimum {minimum}",
                    report.exact_fraction
                ));
            }
        }
        if let Some(minimum) = self.min_iou {
            match report.shape_iou {
                Some(measured) if measured >= minimum => {}
                Some(measured) => failures.push(format!(
                    "shape_iou measured {measured} is below minimum {minimum}"
                )),
                None => failures.push(format!(
                    "shape_iou was not measured (minimum required: {minimum})"
                )),
            }
        }
        if self.require_subset_of_dilated && report.shape_subset_of_dilated != Some(true) {
            failures.push(format!(
                "shape_subset_of_dilated measured {:?}, required true",
                report.shape_subset_of_dilated
            ));
        }
        failures
    }
}

/// Strict pixel-identity assertion (§12.7(A)1, §8.7(A)7). On failure, reports the
/// count and the first few differing coordinates rather than dumping the buffers.
#[track_caller]
pub fn assert_images_identical_gray(expected: &GrayImage, actual: &GrayImage) {
    assert_eq!(
        expected.dimensions(),
        actual.dimensions(),
        "image dimension mismatch: expected {:?}, actual {:?}",
        expected.dimensions(),
        actual.dimensions()
    );
    let differing = diff_set_gray(expected, actual);
    if differing.is_empty() {
        return;
    }
    let examples = differing.iter().take(8).collect::<Vec<_>>();
    let noun = if differing.len() == 1 {
        "pixel"
    } else {
        "pixels"
    };
    panic!(
        "{} {noun} differ; first differing coordinates: {examples:?}",
        differing.len()
    );
}

/// `O subset dilate(G, radius)` (§10.7(B)) as a standalone helper.
pub fn is_within_dilated(ours: &PixelSet, upstream: &PixelSet, radius: u32) -> bool {
    ours.is_subset_of(&upstream.dilate(radius))
}
