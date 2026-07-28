//! Task **E1** — suffix→format mapping, per-format save options, colour-mode handling
//! and dpi carry-over (spec §12.3 steps 3 and 6, §12.7(A)1/2/3/9, §16.11 items 5–8).
//!
//! Everything here is pinned by §16.11 against the capabilities `image` 0.25.10
//! actually has; the three places where the crate cannot do what §12.3's table asks
//! are marked `// DEVIATION(§16.11 item 7)` at the call site.

use image::codecs::bmp::BmpEncoder;
use image::codecs::jpeg::{JpegEncoder, PixelDensity, PixelDensityUnit};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::codecs::pnm::{PnmEncoder, PnmSubtype, SampleEncoding};
use image::codecs::tiff::TiffEncoder;
use image::codecs::webp::WebPEncoder;
use image::{ColorType, DynamicImage, ImageDecoder, ImageEncoder, ImageError, ImageReader};
use pc_core::StageError;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// The v1 output formats — exactly `pc_config::SUPPORTED_OUTPUT_SUFFIXES`' targets
/// (§6, §12.3 step 6). `.jp2` is deliberately absent: `image` has no JPEG2000 encoder
/// and `pc-config` rejects the suffix at load time (§12.7(A)9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    Png,
    Jpeg,
    WebP,
    Tiff,
    Bmp,
    Ppm,
}

impl OutputFormat {
    /// §16.11 item 4's normalisation, then the §12.3 step 6 table. An unknown suffix is
    /// `StageError::UnsupportedFormat`, whose message reuses
    /// `pc_config::validate::supported_suffix_list()` so the runtime message and the
    /// config-validation message cannot drift (§12.7(A)9).
    pub fn from_suffix(suffix: &str) -> Result<Self, StageError> {
        match normalize_suffix(suffix).as_str() {
            ".png" => Ok(Self::Png),
            ".jpg" | ".jpeg" => Ok(Self::Jpeg),
            ".webp" => Ok(Self::WebP),
            ".tif" | ".tiff" => Ok(Self::Tiff),
            ".bmp" | ".dib" => Ok(Self::Bmp),
            ".ppm" => Ok(Self::Ppm),
            other => Err(StageError::UnsupportedFormat(format!(
                "unsupported output suffix `{other}`; supported suffixes: {}",
                pc_config::validate::supported_suffix_list()
            ))),
        }
    }

    /// True for the formats whose encoder accepts an alpha channel (§16.11 item 6).
    pub fn supports_alpha(self) -> bool {
        match self {
            Self::Png | Self::WebP | Self::Tiff | Self::Bmp => true,
            Self::Jpeg | Self::Ppm => false,
        }
    }

    /// §16.11 item 8: `image` 0.25.10 exposes pixel density on exactly one encoder.
    pub fn supports_dpi(self) -> bool {
        matches!(self, Self::Jpeg)
    }
}

/// ASCII-lowercase, leading `.` enforced — the same normalisation
/// `pc_config::validate::validate_output_suffix` applies (§16.11 item 4).
pub fn normalize_suffix(suffix: &str) -> String {
    let lowered = suffix.to_ascii_lowercase();
    if lowered.starts_with('.') {
        lowered
    } else {
        format!(".{lowered}")
    }
}

// --------------------------------------------------------------------- colour mode

/// §12.3 step 3's "the original image's colour mode", as the four modes v1 can write
/// (§16.11 item 5). 16-bit and float source variants collapse onto their 8-bit
/// counterparts, because v1 writes 8-bit only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorMode {
    L,
    La,
    Rgb,
    Rgba,
}

impl ColorMode {
    /// By `has_alpha()` / `channel_count()` rather than a variant match, so `image`'s
    /// `#[non_exhaustive]` `ColorType` needs no catch-all guess (§16.11 item 5).
    pub fn from_color_type(color: ColorType) -> Self {
        match (color.channel_count() <= 2, color.has_alpha()) {
            (true, false) => Self::L,
            (true, true) => Self::La,
            (false, false) => Self::Rgb,
            (false, true) => Self::Rgba,
        }
    }

    pub fn has_alpha(self) -> bool {
        matches!(self, Self::La | Self::Rgba)
    }
}

/// The mode of an already-decoded image.
pub fn color_mode_of(image: &DynamicImage) -> ColorMode {
    ColorMode::from_color_type(image.color())
}

/// The mode of a file, from its **header only** — `original_path` may be an 8000 px
/// strip and only its mode is wanted (§16.11 item 5).
///
// DEVIATION(§16.11 item 5): upstream's `convert(original.mode)` can re-palette a
// mode-`P` input. `image` expands palettes at decode time and has no palette encoder,
// so a palette PNG exports as Rgb8/Rgba8.
pub fn read_color_mode(path: &Path) -> Result<ColorMode, StageError> {
    let reader = ImageReader::open(path).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let decoder = reader
        .with_guessed_format()
        .map_err(|source| StageError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .into_decoder()
        .map_err(|source| StageError::Decode {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(ColorMode::from_color_type(decoder.color_type()))
}

/// §12.3 step 3's conversion. Cheap no-op when the image is already in `mode`.
pub fn convert_to_mode(image: &DynamicImage, mode: ColorMode) -> DynamicImage {
    if color_mode_of(image) == mode {
        return image.clone();
    }
    match mode {
        ColorMode::L => DynamicImage::ImageLuma8(image.to_luma8()),
        ColorMode::La => DynamicImage::ImageLumaA8(image.to_luma_alpha8()),
        ColorMode::Rgb => DynamicImage::ImageRgb8(image.to_rgb8()),
        ColorMode::Rgba => DynamicImage::ImageRgba8(image.to_rgba8()),
    }
}

/// `round(c * a + 255 * (1 - a))` per channel, alpha dropped (§16.11 item 6) — the
/// §16.9 item 15 blend with the base fixed at white. Grayscale stays grayscale.
pub fn flatten_onto_white(image: &DynamicImage) -> DynamicImage {
    match color_mode_of(image) {
        ColorMode::L | ColorMode::Rgb => image.clone(),
        ColorMode::La => {
            let source = image.to_luma_alpha8();
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(
                source.width(),
                source.height(),
                |x, y| {
                    let image::LumaA([value, alpha]) = *source.get_pixel(x, y);
                    image::Luma([crate::composite::blend_channel(
                        255,
                        value,
                        f64::from(alpha) / 255.0,
                    )])
                },
            ))
        }
        ColorMode::Rgba => {
            let source = image.to_rgba8();
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(
                source.width(),
                source.height(),
                |x, y| {
                    let image::Rgba([r, g, b, a]) = *source.get_pixel(x, y);
                    let alpha = f64::from(a) / 255.0;
                    image::Rgb([
                        crate::composite::blend_channel(255, r, alpha),
                        crate::composite::blend_channel(255, g, alpha),
                        crate::composite::blend_channel(255, b, alpha),
                    ])
                },
            ))
        }
    }
}

/// §16.11 item 6's coercion table — what each `image` 0.25.10 encoder actually accepts.
pub fn coerce_for_format(image: &DynamicImage, format: OutputFormat) -> DynamicImage {
    match format {
        // PNG, WebP and BMP accept all four v1 modes.
        OutputFormat::Png | OutputFormat::WebP | OutputFormat::Bmp => image.clone(),
        // JPEG accepts L8 and Rgb8 only.
        OutputFormat::Jpeg => flatten_onto_white(image),
        // TIFF accepts L8, Rgb8 and Rgba8 — but not La8.
        OutputFormat::Tiff => match color_mode_of(image) {
            ColorMode::La => convert_to_mode(image, ColorMode::Rgba),
            _ => image.clone(),
        },
        // P6 accepts Rgb8 only.
        OutputFormat::Ppm => convert_to_mode(&flatten_onto_white(image), ColorMode::Rgb),
    }
}

// ------------------------------------------------------------------------ encoding

/// §12.3 step 6's option table, applied to a coerced copy of `image`. `dpi` is honoured
/// only when `format.supports_dpi()` and the density fits `1..=u16::MAX`
/// (§16.11 items 7 and 8).
pub fn encode_to_vec(
    image: &DynamicImage,
    format: OutputFormat,
    dpi: Option<(u32, u32)>,
) -> Result<Vec<u8>, ImageError> {
    let image = coerce_for_format(image, format);
    let (width, height) = (image.width(), image.height());
    let color = image.color().into();
    let bytes = image.as_bytes();
    let mut buffer = Cursor::new(Vec::new());

    match format {
        OutputFormat::Png => {
            // §12.3 step 6: max compression, no interlace (`image` never interlaces).
            PngEncoder::new_with_quality(&mut buffer, CompressionType::Best, FilterType::Adaptive)
                .write_image(bytes, width, height, color)?;
        }
        OutputFormat::Jpeg => {
            // DEVIATION(§16.11 item 7): baseline, not progressive — `image`'s JPEG
            // encoder has no progressive mode. Quality 95 as specified.
            let mut encoder = JpegEncoder::new_with_quality(&mut buffer, 95);
            if let Some(density) = jfif_density(dpi) {
                encoder.set_pixel_density(density);
            }
            encoder.write_image(bytes, width, height, color)?;
        }
        OutputFormat::WebP => {
            // DEVIATION(§16.11 item 7): `image`'s WebP encoder is lossless-only, so
            // §12.3's "quality 95 for images" is written losslessly instead.
            WebPEncoder::new_lossless(&mut buffer).write_image(bytes, width, height, color)?;
        }
        OutputFormat::Tiff => {
            // DEVIATION(§16.11 item 7): no compression selector is exposed, so §12.3's
            // "LZW" is the crate default instead.
            TiffEncoder::new(&mut buffer).write_image(bytes, width, height, color)?;
        }
        OutputFormat::Bmp => {
            BmpEncoder::new(&mut buffer).write_image(bytes, width, height, color)?;
        }
        OutputFormat::Ppm => {
            // P6 binary, pinned rather than left to the crate's "dynamic header"
            // default, which is documented as arbitrary (§16.11 item 7).
            PnmEncoder::new(&mut buffer)
                .with_subtype(PnmSubtype::Pixmap(SampleEncoding::Binary))
                .write_image(bytes, width, height, color)?;
        }
    }

    Ok(buffer.into_inner())
}

/// `encode_to_vec` + write, with both failures mapped to `StageError::Io { path, .. }`
/// (§16.11 item 7 — the same idiom `pc_denoise::write_png` uses).
pub fn save(
    image: &DynamicImage,
    path: &Path,
    format: OutputFormat,
    dpi: Option<(u32, u32)>,
) -> Result<(), StageError> {
    let bytes = encode_to_vec(image, format, dpi).map_err(|error| StageError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other(error),
    })?;
    std::fs::write(path, bytes).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn jfif_density(dpi: Option<(u32, u32)>) -> Option<PixelDensity> {
    let (x, y) = dpi?;
    let fits = |value: u32| (1..=u32::from(u16::MAX)).contains(&value);
    (fits(x) && fits(y)).then_some(PixelDensity {
        density: (x as u16, y as u16),
        unit: PixelDensityUnit::Inches,
    })
}

// ----------------------------------------------------------------------- dpi (read)

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
const DPI_HEADER_BYTES: usize = 64 * 1024;

/// §16.11 item 8: dpi comes from the file header, because `image` 0.25.10 exposes
/// pixel density on no decoder at all. JPEG via the JFIF `APP0` segment, PNG via the
/// `pHYs` chunk; anything else, or an absent/truncated segment, is `None`.
pub fn read_dpi(path: &Path) -> Result<Option<(u32, u32)>, StageError> {
    let head = read_head(path, DPI_HEADER_BYTES)?;
    Ok(dpi_from_header(&head))
}

/// The pure half of `read_dpi`, so the frozen tests can pin byte layouts directly.
pub fn dpi_from_header(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.starts_with(&[0xff, 0xd8]) {
        return jfif_dpi(bytes);
    }
    if bytes.starts_with(&PNG_SIGNATURE) {
        return png_phys_dpi(bytes);
    }
    None
}

fn read_head(path: &Path, limit: usize) -> Result<Vec<u8>, StageError> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut buffer = Vec::new();
    file.take(limit as u64)
        .read_to_end(&mut buffer)
        .map_err(|source| StageError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(buffer)
}

/// `units == 1` → dpi verbatim; `units == 2` → `round(v * 2.54)` (dots/cm);
/// `units == 0` → aspect ratio only, so `None`. Scanning stops at `SOS`.
fn jfif_dpi(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut index = 2_usize;
    loop {
        while bytes.get(index) == Some(&0xff) && bytes.get(index + 1) == Some(&0xff) {
            index += 1;
        }
        if bytes.get(index) != Some(&0xff) {
            return None;
        }
        let marker = *bytes.get(index + 1)?;
        // SOS (start of scan) / EOI — no more metadata segments follow.
        if marker == 0xda || marker == 0xd9 {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([
            *bytes.get(index + 2)?,
            *bytes.get(index + 3)?,
        ]));
        if length < 2 {
            return None;
        }
        let data = bytes.get(index + 4..index + 2 + length)?;
        if marker == 0xe0 && data.starts_with(b"JFIF\0") && data.len() >= 12 {
            let units = data[7];
            let x = u32::from(u16::from_be_bytes([data[8], data[9]]));
            let y = u32::from(u16::from_be_bytes([data[10], data[11]]));
            return match units {
                1 => Some((x, y)),
                2 => Some((per_cm_to_dpi(x), per_cm_to_dpi(y))),
                _ => None,
            };
        }
        index += 2 + length;
    }
}

/// `unit == 1` (metres) → `round(ppm * 0.0254)`; `unit == 0` is a bare aspect ratio.
/// Scanning stops at `IDAT`.
fn png_phys_dpi(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut index = PNG_SIGNATURE.len();
    loop {
        let length = u32::from_be_bytes(bytes.get(index..index + 4)?.try_into().ok()?) as usize;
        let kind = bytes.get(index + 4..index + 8)?;
        if kind == b"IDAT" || kind == b"IEND" {
            return None;
        }
        if kind == b"pHYs" && length >= 9 {
            let data = bytes.get(index + 8..index + 8 + 9)?;
            let x = u32::from_be_bytes(data[0..4].try_into().ok()?);
            let y = u32::from_be_bytes(data[4..8].try_into().ok()?);
            return (data[8] == 1).then(|| (per_metre_to_dpi(x), per_metre_to_dpi(y)));
        }
        index = index.checked_add(12)?.checked_add(length)?;
    }
}

fn per_cm_to_dpi(value: u32) -> u32 {
    (f64::from(value) * 2.54).round() as u32
}

fn per_metre_to_dpi(value: u32) -> u32 {
    (f64::from(value) * 0.0254).round() as u32
}

/// Convenience for `run()`: the dpi of a source file, ignoring unreadable headers.
/// A missing/unreadable original must not fail an export whose pixels came from a
/// cache artifact (§5.6), so this downgrades an `Io` error to `None`.
pub fn read_dpi_lossy(path: &Path) -> Option<(u32, u32)> {
    match read_dpi(path) {
        Ok(dpi) => dpi,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "could not read source dpi");
            None
        }
    }
}

/// `path.extension()` normalised to a `.suffix` string (§16.11 item 4).
pub fn suffix_of(path: &Path) -> Option<String> {
    path.extension()
        .map(|extension| normalize_suffix(&extension.to_string_lossy()))
}

/// `StageError::InvalidInput` naming the field, for the two §16.11 item 4 failures.
pub fn invalid_input(message: impl Into<String>) -> StageError {
    StageError::InvalidInput(message.into())
}

/// `PathBuf` join used by `destinations`; kept here so the naming rule lives next to
/// the suffix rules it depends on (§12.3 step 1).
pub fn artifact_path(base: &Path, stem: &str, tag: &str, suffix: &str) -> PathBuf {
    base.join(format!("{stem}{tag}{suffix}"))
}
