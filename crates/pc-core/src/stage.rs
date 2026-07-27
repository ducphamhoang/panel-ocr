//! spec §3 — the stage contract.

use crate::{error::StageError, output::Step};
use serde::{de::DeserializeOwned, Serialize};

/// One pure function per stage. Model handles are NOT serializable, so stages that
/// need external resources take them via `Ctx` — keeping `Input` fully serde-derivable
/// (which is what makes disk checkpointing and v2 staleness detection possible).
///
/// Purity contract (§3): pure with respect to *global state* — no env/config reads, no
/// side effects beyond `tracing`, no writes to paths absent from the input. NOT pure
/// with respect to disk: it may read `ImageHandle` paths and write to destinations
/// named in its input.
pub trait Stage {
    type Input: Serialize + DeserializeOwned;
    type Output: Serialize + DeserializeOwned;
    /// `()` for stages with no external resources.
    type Ctx<'a>;
    const STEP: Step;

    fn run(input: Self::Input, ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError>;
}
