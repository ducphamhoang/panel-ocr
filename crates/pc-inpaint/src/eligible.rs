//! spec §16.38 items 3(c) and 3(d) — the two eligibility sets, unioned, and where each
//! takes its fill mask from.
//!
//! Transcribed from `pcleaner/inpainting.py:75-101` at the pinned commit
//! `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, read in the fetched source.

use pc_config::InpainterConfig;
use pc_core::{MaskRegionStats, Rect};

/// Which cached mask a region's fill comes from (§16.38 item 3(d)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillSource {
    /// `_raw_mask.png`, cropped to the box and grown by `masker.min_mask_thickness`
    /// (upstream `:93-97`). Chosen for **failed** regions.
    ///
    /// **DEVIATION(12) propagates into this consumer** (§16.38 item 10). Upstream's
    /// `_raw_mask.png` is `refine_mask`'s output (`ctd_interface.py:182`); v1 ships
    /// `MaskRefineMode::Simple` instead (§14 item 12, §15 item 2), and cookbook rule 7
    /// records the measured agreement of the two artifacts as **IoU 0.258**. So for every
    /// `failed` region the filled area differs from upstream's *before any tiling happens*,
    /// and it would differ even if `DEVIATION(24)` did not exist. This gets no new register
    /// number — it is an existing deviation reaching a new reader — but a future parity
    /// investigation must not attribute the whole difference to tiling.
    RawMask,
    /// The combined fill mask, cropped to the box (upstream `:99-101`). Chosen for
    /// **poorly-fitted** regions.
    CombinedMask,
}

/// One region that will be inpainted, with everything downstream needs about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EligibleRegion {
    /// Index into the `regions` slice it came from — so a caller can trace a growth back to
    /// its `#mask_data.json` row.
    pub index: usize,
    pub rect: Rect,
    pub std_deviation: f64,
    pub source: FillSource,
}

/// spec §16.38 item 3(c): two eligibility sets, unioned.
///
///   * **failed** — every row with `failed` true (upstream `:76-80`);
///   * **poorly fitted** — `!failed && std_deviation >= inpainting_min_std_dev &&
///     thickness.is_some() && thickness <= min_inpainting_radius` (`:82-90`).
///
/// Both comparisons are **inclusive**, matching upstream's `>=` and `<=` exactly. The
/// `thickness.is_some()` clause carries upstream's own comment: *"For box masks, this is
/// none. We don't need to inpaint those, they are always good."*
///
/// **Order is part of the contract, not an accident.** Upstream builds the failed list
/// first and then appends the poorly-fitted one (`:93-101`), so the returned order is *all
/// failed regions in `regions` order, then all poorly-fitted regions in `regions` order* —
/// **not** `regions` order overall. `analytics_thicknesses` (`:119`) is emitted in exactly
/// this order.
///
/// **`inpainting_max_mask_radius` is deliberately not read here** (§16.38 items 7(a), 7(d)).
/// Upstream declares, exports, imports and clamps that key while `inpainting.py` references
/// it **zero** times, and the gate its own config comment describes is implemented at
/// `inpainting.py:89` against `min_inpainting_radius`. That half is exact parity and is not
/// a deviation; `DEVIATION(26)` is only the WARN `pc-config` emits when a profile moves it
/// off its default. A comment saying "unused" enforces nothing, which is why item 7(d)
/// requires a test that two configs differing only in that key select the identical set.
pub fn select_regions(
    regions: &[MaskRegionStats],
    config: &InpainterConfig,
) -> Vec<EligibleRegion> {
    let mut out = Vec::new();

    for (index, region) in regions.iter().enumerate() {
        if region.failed {
            out.push(EligibleRegion {
                index,
                rect: region.rect,
                std_deviation: region.std_deviation,
                source: FillSource::RawMask,
            });
        }
    }
    for (index, region) in regions.iter().enumerate() {
        let poorly_fitted = !region.failed
            && region.std_deviation >= config.inpainting_min_std_dev
            && region
                .thickness
                .is_some_and(|thickness| thickness <= config.min_inpainting_radius);
        if poorly_fitted {
            out.push(EligibleRegion {
                index,
                rect: region.rect,
                std_deviation: region.std_deviation,
                source: FillSource::CombinedMask,
            });
        }
    }

    out
}
