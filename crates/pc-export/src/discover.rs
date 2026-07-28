//! Task **E2** — precedence resolution and `--save-only-*` narrowing
//! (spec §12.3 step 2, §12.7(A)5/6, §16.11 items 2 and 3).
//!
//! Ownership split, per §12.3 step 2: **`pc-pipeline` resolves availability** (it fills
//! in `ExportSources` from just-produced handles or its own `CachePaths`), **this module
//! resolves precedence**. Nothing here touches the filesystem.

use crate::ExportSources;
use pc_core::{ImageHandle, Output};

/// The three user-facing export categories (§16.11 item 2). `pc_core::Output` has no
/// export-side variants — `Output::step()` is deliberately non-surjective — so the
/// requested-outputs list is read through this map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Cleaned,
    Mask,
    Text,
}

impl Category {
    /// `None` for the cache-only artifacts (`BaseImage`, `RawJson`, `BoxMask`, …).
    pub fn of(output: Output) -> Option<Self> {
        match output {
            Output::MaskedOutput | Output::DenoisedOutput => Some(Self::Cleaned),
            Output::FinalMask | Output::DenoiseMask => Some(Self::Mask),
            Output::IsolatedText => Some(Self::Text),
            _ => None,
        }
    }

    /// In `files_written` order (§16.11 item 4).
    pub const ALL: &'static [Category] = &[Category::Cleaned, Category::Mask, Category::Text];
}

/// True iff `outputs` contains at least one variant mapping to `category`
/// (§16.11 item 2). An empty `outputs` requests nothing.
pub fn is_requested(outputs: &[Output], category: Category) -> bool {
    outputs
        .iter()
        .any(|output| Category::of(*output) == Some(category))
}

/// Which mask §12.3 step 4 must build. `WithDenoise` is the composite branch; it needs
/// **both** handles, because the noise mask is composited *over* the resized combined
/// mask (§16.11 item 3).
#[derive(Debug, Clone, PartialEq)]
pub enum MaskChoice {
    FinalOnly(ImageHandle),
    WithDenoise {
        final_mask: ImageHandle,
        denoise_mask: ImageHandle,
    },
}

/// Exactly one cleaned image, at most one mask, at most one text layer (§12.7(A)5).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Selection {
    pub cleaned: Option<ImageHandle>,
    pub mask: Option<MaskChoice>,
    pub text: Option<ImageHandle>,
}

impl Selection {
    /// True when nothing at all will be written — a valid outcome (§16.11 item 14).
    pub fn is_empty(&self) -> bool {
        self.cleaned.is_none() && self.mask.is_none() && self.text.is_none()
    }
}

/// spec §12.3 step 2, pinned by §16.11 item 3:
///   * `cleaned = if denoising_enabled { denoised.or(masked) } else { masked }` — a
///     stale cached denoise artifact must not resurrect itself when denoising is off;
///   * `mask` = the denoise composite when enabled and both handles exist, else the
///     combined mask alone, else nothing (a `denoise_mask` with no `final_mask` WARNs);
///   * `text = isolated_text`, independent of the other two;
///   * narrowing by `outputs` is applied **after** precedence, so the two cannot
///     interact.
pub fn resolve(sources: &ExportSources, outputs: &[Output], denoising_enabled: bool) -> Selection {
    let cleaned = if denoising_enabled {
        sources.denoised.clone().or_else(|| sources.masked.clone())
    } else {
        sources.masked.clone()
    };

    let mask = match (
        sources.final_mask.clone(),
        sources.denoise_mask.clone(),
        denoising_enabled,
    ) {
        (Some(final_mask), Some(denoise_mask), true) => Some(MaskChoice::WithDenoise {
            final_mask,
            denoise_mask,
        }),
        (Some(final_mask), _, _) => Some(MaskChoice::FinalOnly(final_mask)),
        (None, Some(_), true) => {
            tracing::warn!(
                "a denoise mask was available without a combined mask; no mask will be exported"
            );
            None
        }
        (None, _, _) => None,
    };

    Selection {
        cleaned: cleaned.filter(|_| is_requested(outputs, Category::Cleaned)),
        mask: mask.filter(|_| is_requested(outputs, Category::Mask)),
        text: sources
            .isolated_text
            .clone()
            .filter(|_| is_requested(outputs, Category::Text)),
    }
}

/// Every `Output` variant that maps to a category — the list `pc-pipeline`/`pc-cli`
/// hand to `ExportInput.outputs` for an unrestricted run (`--save-only-*` absent).
pub fn all_exportable_outputs() -> Vec<Output> {
    Output::ALL
        .iter()
        .copied()
        .filter(|output| Category::of(*output).is_some())
        .collect()
}
