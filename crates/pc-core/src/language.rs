//! spec §2.2

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Japanese,
    English,
}

/// spec §2.2: upstream's RTL set (`ocr/supported_languages.py:153`) restricted to the
/// codes reachable in v1. Kept as a slice so adding codes later is purely additive.
pub const RTL_BOX_ORDER_LANGUAGES: &[Language] = &[Language::Japanese];
