//! L6-4 compatibility gate for §16.40 item 2(d).
//!
//! FROZEN (CLAUDE.md).

use pc_core::StageError;
use pc_inpaint::Inpainter;
use std::sync::Arc;

struct Provider;

impl pc_pipeline::InpainterProvider for Provider {
    fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
        Err(StageError::Model("compile-only provider".into()))
    }

    fn failures_are_run_fatal(&self) -> bool {
        true
    }
}

/// §16.40 item 2(d): `pc_cli::inpainter::InpainterProvider` is a compatibility re-export
/// of the trait owned by `pc-pipeline`, not a second nominally distinct trait. Passing one
/// implementation to both typed functions is the assertion.
#[test]
fn pc_cli_reexports_the_pipeline_inpainter_provider_trait() {
    fn through_pipeline(_: &dyn pc_pipeline::InpainterProvider) {}
    fn through_cli(_: &dyn pc_cli::inpainter::InpainterProvider) {}

    let provider = Provider;
    through_pipeline(&provider);
    through_cli(&provider);
}
