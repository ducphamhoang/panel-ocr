//! The vendored manga-ocr vocabulary and pure token decoding helpers.

use std::sync::OnceLock;

pub const VOCAB_TXT: &str = include_str!("../assets/vocab.txt");
pub const VOCAB_LEN: usize = 6144;
pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const CLS_ID: u32 = 2;
pub const SEP_ID: u32 = 3;
pub const MASK_ID: u32 = 4;

pub struct Vocab {
    tokens: Vec<&'static str>,
}

impl Vocab {
    pub fn embedded() -> &'static Vocab {
        static EMBEDDED: OnceLock<Vocab> = OnceLock::new();

        EMBEDDED.get_or_init(|| {
            let tokens = VOCAB_TXT.lines().collect::<Vec<_>>();
            assert_eq!(tokens.len(), VOCAB_LEN);
            Vocab { tokens }
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token(&self, id: u32) -> Option<&'static str> {
        self.tokens.get(id as usize).copied()
    }

    pub fn decode_skip_special(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter(|&&id| !matches!(id, PAD_ID | UNK_ID | CLS_ID | SEP_ID | MASK_ID))
            .filter_map(|&id| self.token(id))
            .collect()
    }
}
