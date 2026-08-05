//! Device policy for fixture-producing maintainer commands (§16.36 item 5).

use pc_config::TextDetectorConfig;
use pc_core::device::{Device, DevicePolicy};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingSession {
    DetectorFixture,
    CalibrateGoldens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingRefusal {
    #[allow(dead_code)]
    RequestedDevice { requested: Device },
    ResolvedPolicy {
        execution_provider: &'static str,
        session: RecordingSession,
    },
}

impl fmt::Display for RecordingRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestedDevice { requested } => write!(
                formatter,
                "fixture-producing command refuses requested device `{}`; only `cpu` is ratified for committed recordings (§16.22 item 5(b))",
                requested.as_str()
            ),
            Self::ResolvedPolicy {
                execution_provider,
                session,
            } => {
                let operation = match session {
                    RecordingSession::DetectorFixture => "detector fixture recording",
                    RecordingSession::CalibrateGoldens => "golden calibration",
                };
                write!(
                    formatter,
                    "{operation} refuses resolved execution provider `{execution_provider}`; only `cpu` is ratified for committed recordings (§16.22 item 5(b))"
                )
            }
        }
    }
}

impl std::error::Error for RecordingRefusal {}

#[allow(dead_code)]
pub fn ensure_recording_request(requested: Device) -> Result<(), RecordingRefusal> {
    if requested != Device::Cpu {
        return Err(RecordingRefusal::RequestedDevice { requested });
    }
    Ok(())
}

pub fn ensure_recording_policy(
    policy: &DevicePolicy,
    session: RecordingSession,
) -> Result<(), RecordingRefusal> {
    if policy.requested() != Device::Cpu {
        return Err(RecordingRefusal::ResolvedPolicy {
            execution_provider: policy.provenance_execution_provider(),
            session,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedSessionPins {
    pub execution_provider: &'static str,
    pub intra_threads: usize,
    pub inter_threads: usize,
}

pub fn recorded_session_pins(
    policy: &DevicePolicy,
    config: &TextDetectorConfig,
) -> RecordedSessionPins {
    RecordedSessionPins {
        execution_provider: policy.provenance_execution_provider(),
        intra_threads: config.intra_threads,
        inter_threads: config.inter_threads,
    }
}
