//! Beam-search decoding for the manga-ocr decoder.

use pc_core::StageError;
use std::collections::{HashMap, HashSet};

/// One decoder forward pass. `prefix` is the full token sequence so far (including the
/// seed token), since the pinned decoder ONNX graph has no KV-cache input. Returns the
/// logits row for the last position only.
pub trait LogitsSource {
    fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError>;

    /// One logits row per prefix, in order. All prefixes have equal length (beam search
    /// extends every live beam by exactly one token per step). Default = today's
    /// one-call-per-prefix behaviour, so no existing impl changes semantics.
    fn logits_batch(&self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>, StageError> {
        prefixes.iter().map(|p| self.logits(p)).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BeamSearchConfig {
    pub num_beams: usize,
    pub length_penalty: f64,
    pub no_repeat_ngram_size: usize,
    pub max_length: usize,
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
}

/// spec §16.30 item 4 (ratified decode strategy).
pub const MANGA_OCR_BEAM_CONFIG: BeamSearchConfig = BeamSearchConfig {
    num_beams: 4,
    length_penalty: 2.0,
    no_repeat_ngram_size: 3,
    max_length: 300,
    decoder_start_token_id: 2,
    eos_token_id: 3,
};

struct Candidate {
    score: f64,
    beam_index: usize,
    token_id: u32,
}

/// Decode a token sequence with classic cumulative-score beam search.
pub fn beam_search(
    source: &dyn LogitsSource,
    config: &BeamSearchConfig,
) -> Result<Vec<u32>, StageError> {
    if config.num_beams == 0 {
        return Err(StageError::InvalidInput(
            "beam search requires at least one beam".into(),
        ));
    }

    let mut beams = vec![(vec![config.decoder_start_token_id], 0.0_f64)];
    let mut finished: Vec<(f64, Vec<u32>)> = Vec::new();

    loop {
        let cur_len = beams
            .first()
            .map(|(sequence, _)| sequence.len())
            .ok_or_else(|| {
                StageError::Inference("beam search produced no live beams before completion".into())
            })?;

        // Stop before generating candidates at max_length; truncated sequences do not get EOS.
        if cur_len >= config.max_length {
            for (sequence, cumulative_logprob) in beams.drain(..) {
                let generated_len = cur_len.saturating_sub(1).max(1);
                finished.push((
                    cumulative_logprob / (generated_len as f64).powf(config.length_penalty),
                    sequence,
                ));
            }
            break;
        }

        let mut candidates = Vec::new();
        let prefixes: Vec<&[u32]> = beams
            .iter()
            .map(|(sequence, _)| sequence.as_slice())
            .collect();
        let rows = source.logits_batch(&prefixes)?;
        if rows.len() != beams.len() {
            return Err(StageError::InvalidInput(format!(
                "logits_batch returned {} rows for {} live beams",
                rows.len(),
                beams.len()
            )));
        }
        for (beam_index, ((sequence, cumulative_logprob), logits)) in
            beams.iter().zip(rows.iter()).enumerate()
        {
            if logits.is_empty() {
                return Err(StageError::InvalidInput(format!(
                    "decoder logits row {beam_index} must not be empty"
                )));
            }

            let row_max = logits
                .iter()
                .map(|&logit| f64::from(logit))
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits
                .iter()
                .map(|&logit| (f64::from(logit) - row_max).exp())
                .sum();
            let log_sum_exp = exp_sum.ln();
            let mut log_probs: Vec<f64> = logits
                .iter()
                .map(|&logit| f64::from(logit) - row_max - log_sum_exp)
                .collect();

            if config.no_repeat_ngram_size > 0 && cur_len + 1 >= config.no_repeat_ngram_size {
                let ngram_size = config.no_repeat_ngram_size;
                let mut following: HashMap<Vec<u32>, HashSet<u32>> = HashMap::new();
                for window in sequence.windows(ngram_size) {
                    following
                        .entry(window[..ngram_size - 1].to_vec())
                        .or_default()
                        .insert(window[ngram_size - 1]);
                }

                let prefix_start = cur_len + 1 - ngram_size;
                let current_prefix = &sequence[prefix_start..cur_len];
                if let Some(banned) = following.get(current_prefix) {
                    for &token_id in banned {
                        if let Some(log_prob) = log_probs.get_mut(token_id as usize) {
                            *log_prob = f64::NEG_INFINITY;
                        }
                    }
                }
            }

            for (token_index, log_prob) in log_probs.into_iter().enumerate() {
                candidates.push(Candidate {
                    score: *cumulative_logprob + log_prob,
                    beam_index,
                    token_id: token_index as u32,
                });
            }
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.beam_index.cmp(&right.beam_index))
                .then_with(|| left.token_id.cmp(&right.token_id))
        });

        let candidate_count = config.num_beams.saturating_mul(2).min(candidates.len());
        let mut next_beams = Vec::with_capacity(config.num_beams);
        for (rank, candidate) in candidates.into_iter().take(candidate_count).enumerate() {
            let mut new_sequence = beams[candidate.beam_index].0.clone();
            new_sequence.push(candidate.token_id);

            if candidate.token_id == config.eos_token_id {
                if rank < config.num_beams {
                    let generated_len = new_sequence.len() - 1;
                    finished.push((
                        candidate.score / (generated_len as f64).powf(config.length_penalty),
                        new_sequence,
                    ));
                }
            } else if next_beams.len() < config.num_beams {
                next_beams.push((new_sequence, candidate.score));
            }
        }
        beams = next_beams;

        // This port is fixed to the ratified early_stopping=true behavior.
        if finished.len() >= config.num_beams {
            break;
        }
    }

    finished.sort_by(|left, right| right.0.total_cmp(&left.0));
    finished
        .into_iter()
        .next()
        .map(|(_, sequence)| sequence)
        // The max-length finalization above makes this unreachable for valid configurations.
        .ok_or_else(|| StageError::Inference("beam search produced no finished hypothesis".into()))
}
