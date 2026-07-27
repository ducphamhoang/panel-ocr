//! C1 tests — spec §3, the stage contract. These prove the trait is *shaped* right
//! (serde-derivable Input/Output, a borrowing Ctx, a compile-time Step), which is
//! what every stage crate depends on.

use pc_core::{Stage, StageError, Step};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct CountInput {
    schema_version: u32,
    values: Vec<i32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct CountOutput {
    total: i32,
}

/// A stage with no external resources: `Ctx<'a> = ()`.
struct SumStage;

impl Stage for SumStage {
    type Input = CountInput;
    type Output = CountOutput;
    type Ctx<'a> = ();
    const STEP: Step = Step::Mask;

    fn run(input: Self::Input, _ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        if input.values.is_empty() {
            // §2.9: only conditions that make the work impossible are errors
            return Err(StageError::InvalidInput("no values".into()));
        }
        Ok(CountOutput {
            total: input.values.iter().sum(),
        })
    }
}

trait Resource: Send + Sync {
    fn bias(&self) -> i32;
}
struct FixedResource(i32);
impl Resource for FixedResource {
    fn bias(&self) -> i32 {
        self.0
    }
}

/// A stage that needs a non-serde external resource, held behind a trait object —
/// spec §3: "model handles are not serializable, so stages that need a model take it
/// as a second, non-serde parameter", and §1 rule 4: every ML call goes through a trait.
struct BiasedStage;

impl Stage for BiasedStage {
    type Input = CountInput;
    type Output = CountOutput;
    type Ctx<'a> = &'a dyn Resource;
    const STEP: Step = Step::Detect;

    fn run(input: Self::Input, ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        Ok(CountOutput {
            total: input.values.iter().sum::<i32>() + ctx.bias(),
        })
    }
}

#[test]
// spec §3: a stage with no external resources runs with Ctx = ()
fn stage_with_unit_ctx_runs() {
    let out = SumStage::run(
        CountInput {
            schema_version: 1,
            values: vec![1, 2, 3],
        },
        (),
    )
    .unwrap();
    assert_eq!(out, CountOutput { total: 6 });
    assert_eq!(SumStage::STEP, Step::Mask);
}

#[test]
// spec §3: a stage may borrow a non-serializable resource for the duration of run()
// without that resource entering the data contract
fn stage_with_borrowed_ctx_runs() {
    let res = FixedResource(10);
    let out = BiasedStage::run(
        CountInput {
            schema_version: 1,
            values: vec![1, 2],
        },
        &res,
    )
    .unwrap();
    assert_eq!(out, CountOutput { total: 13 });
    assert_eq!(BiasedStage::STEP, Step::Detect);
}

#[test]
// spec §3: Input/Output are Serialize + DeserializeOwned, which is what makes disk
// checkpointing (§4.1) and v2 staleness detection possible
fn stage_io_is_serde_round_trippable() {
    fn round_trip<S: Stage>(input: &S::Input) -> S::Input
    where
        S::Input: PartialEq + std::fmt::Debug,
    {
        serde_json::from_str(&serde_json::to_string(input).unwrap()).unwrap()
    }
    let input = CountInput {
        schema_version: 1,
        values: vec![4, 5],
    };
    assert_eq!(round_trip::<SumStage>(&input), input);
}

#[test]
// spec §2.9 + §5.6: an unprocessable input is a StageError; a merely empty result
// is not signalled by panicking
fn stage_reports_invalid_input_as_an_error() {
    let err = SumStage::run(
        CountInput {
            schema_version: 1,
            values: vec![],
        },
        (),
    )
    .unwrap_err();
    assert!(matches!(err, StageError::InvalidInput(_)));
}
