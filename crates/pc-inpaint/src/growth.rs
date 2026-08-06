//! spec §16.38 item 3(e) — the growth arithmetic, exactly.
//!
//! Upstream, `pcleaner/inpainting.py:106-113`:
//!
//! ```text
//! growth  = i_conf.min_inpainting_radius
//! growth += int(deviation * i_conf.inpainting_radius_multiplier)
//! growth  = min(growth, i_conf.max_inpainting_radius)
//! growth_with_isolation = growth + i_conf.inpainting_isolation_radius
//! box_padded = box.pad(growth_with_isolation, mask_image.size)
//! ```

use pc_config::InpainterConfig;
use pc_core::Rect;

/// `min(min_inpainting_radius + int(deviation * inpainting_radius_multiplier),
/// max_inpainting_radius)`.
///
/// Two things this must get right, both of which a rewrite is likely to get wrong:
///
///   * **`int(...)` truncates toward zero, it does not round** (Python semantics, matching
///     `Rect::scale`'s note in §2.1). With the default multiplier `0.2`, a deviation of
///     `9.9` gives `int(1.98) == 1` and a growth of `8`; rounding would give `9`.
///   * **the cap is applied after the addition**, so a large deviation saturates at
///     `max_inpainting_radius` rather than overflowing. `config.py:932`'s `fix()` already
///     enforces `max_inpainting_radius >= min_inpainting_radius` and §16.38 item 13(b)
///     re-enforces it as a config-load error, so the cap can never drag the result *below*
///     `min_inpainting_radius` on a validated profile.
///
/// The arithmetic runs in `i64` and saturates at `0`. Validation (§16.38 item 13(b)) already
/// rejects a negative multiplier, so on a loaded profile the sum cannot go negative — the
/// saturation is here because this function is also reachable with a hand-built config, and
/// a silent wrap into an unsigned kernel size is the failure item 13(b) was written to
/// prevent.
pub fn growth(std_deviation: f64, config: &InpainterConfig) -> u32 {
    let scaled = std_deviation * config.inpainting_radius_multiplier;
    // `as i64` on f64 saturates in Rust (it does not wrap or UB), and truncates toward zero
    // — which is exactly Python's `int()`.
    let increment = if scaled.is_nan() { 0 } else { scaled as i64 };
    let sum = i64::from(config.min_inpainting_radius).saturating_add(increment);
    let capped = sum.min(i64::from(config.max_inpainting_radius));
    capped.max(0) as u32
}

/// `growth + inpainting_isolation_radius`, then `Rect::pad` clamped to the canvas — upstream
/// `:111` and `:113`, via `structures.py:98-110`'s `max(x1-a, 0) / min(x2+a, w)`.
///
/// The isolation radius is added here, **ahead of time**, exactly as upstream's own comment
/// says: *"The isolation radius is added ahead of time here. We will grow it by this later,
/// but only after the inpainting."* The consequence is load-bearing for
/// [`crate::fill::padded_region`]: the padded box is wide enough to hold the fill mask grown
/// by `growth` **and** the isolation mask grown by a further `inpainting_isolation_radius`,
/// which is what lets both be built in the padded box's own frame.
pub fn padded_box(rect: Rect, growth: u32, config: &InpainterConfig, canvas: (u32, u32)) -> Rect {
    let with_isolation = growth.saturating_add(config.inpainting_isolation_radius);
    rect.pad(
        i32::try_from(with_isolation).unwrap_or(i32::MAX),
        (canvas.0, canvas.1),
    )
}
