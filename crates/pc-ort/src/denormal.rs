#![allow(unsafe_code)] // §16.52 item 6: the workspace's single fenced unsafe site

//! Scoped denormal-flush control for the current thread's x86_64 MXCSR register
//! (§16.52 item 5's D0). Standalone: no `ort` dependency, no `onnx` feature, so it
//! compiles and its tests run in the default feature tier.
//!
//! ONNX Runtime's `session.set_denormal_as_zero` reaches the inference-*calling* thread
//! only through a process-wide `std::call_once`, consumed by whichever `InferenceSession`
//! initializes first (§16.52 items 1-2) — so the detector's confined worker thread can't
//! rely on ORT to flush its own denormals. The ratified fix is this RAII guard: save the
//! calling thread's full MXCSR word on construction, set FTZ (bit 15) and DAZ (bit 6)
//! (Intel SDM Vol.1 §10.2.3), and restore the exact saved word on drop — including across
//! an unwind. MXCSR is per-thread on x86_64, so `stmxcsr`/`ldmxcsr` on the calling thread
//! is already all the "isolation" the design needs; no `static`/`thread_local!` is used.
//!
//! This is the workspace's first `unsafe` (§16.52 item 6): the whole module is fenced
//! with a single `#[allow(unsafe_code)]`, and the two asm blocks below are its only
//! occurrences. The `_mm_getcsr`/`_mm_setcsr` intrinsics are `#[deprecated]` on the pinned
//! toolchain (1.94.1) and would be a hard error under this project's `-D warnings`
//! clippy gate; inline asm is used instead, as the deprecation message itself directs.
/// FTZ — flush-to-zero (Intel SDM Vol.1 §10.2.3).
const FTZ: u32 = 1 << 15;

/// DAZ — denormals-are-zero (Intel SDM Vol.1 §10.2.3).
const DAZ: u32 = 1 << 6;

/// Observed denormal-handling state of the CURRENT thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushState {
    pub flush_to_zero: bool,
    pub denormals_are_zero: bool,
}
/// Read the calling thread's raw 32-bit MXCSR control word via `stmxcsr`.
///
/// `stmxcsr [mem]` stores the MXCSR register to the 32-bit memory operand. There is no
/// register form of the instruction, so the value must go through a memory slot.
fn read_mxcsr() -> u32 {
    let mut word: u32 = 0;
    // SAFETY: `stmxcsr` writes exactly 4 bytes to the pointed-to memory; `word` is a
    // 4-byte-aligned local, and the instruction has no side effects on surrounding
    // program state. No SSE state is modified — only read.
    unsafe {
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut word, options(nostack));
    }
    word
}
/// Write an arbitrary 32-bit MXCSR control word on the calling thread via `ldmxcsr`.
///
/// Used by the guard's `Drop` (verbatim restore of the saved word) and by the test-only
/// `write_mxcsr_for_test` (prior-state scenario construction).
fn write_mxcsr(word: u32) {
    // SAFETY: `ldmxcsr` reads exactly 4 bytes from the pointed-to memory; `word` is a
    // 4-byte-aligned local. Per Intel SDM Vol.1 §10.2.3, loading a value with reserved
    // bits set raises `#GP` — callers pass either a value previously read from MXCSR or
    // one derived from such a value by setting/clearing only FTZ/DAZ, so the reserved
    // bits stay as the hardware left them. Modifying the FP control state is the
    // function's entire purpose.
    unsafe {
        core::arch::asm!("ldmxcsr [{}]", in(reg) &word, options(nostack));
    }
}
/// True when this build can control denormal handling: x86_64 with SSE3.
///
/// `stmxcsr`/`ldmxcsr` are SSE instructions, present on every x86_64; SSE3 is the
/// instruction-set baseline under which the DAZ/FTZ behaviour this module is about is
/// guaranteed, and is what `is_x86_feature_detected!("sse3")` checks. The guard itself
/// is safe on any x86_64 regardless (the instructions execute on all of them); this
/// exists so D1 can make the availability decision explicit before touching a worker
/// thread's MXCSR.
pub fn denormal_control_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("sse3")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
/// Read-only probe of the CURRENT thread's FTZ/DAZ bits — `None` on non-x86_64.
pub fn flush_state() -> Option<FlushState> {
    #[cfg(target_arch = "x86_64")]
    {
        let word = read_mxcsr();
        Some(FlushState {
            flush_to_zero: word & FTZ != 0,
            denormals_are_zero: word & DAZ != 0,
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}
/// RAII scope: on construction, saves the calling thread's current MXCSR word and sets
/// FTZ and DAZ; on `Drop` — including during an unwind — restores the EXACT saved word,
/// not a hard-coded default. x86_64 only.
pub struct DenormalFlushGuard {
    saved: u32,
}

impl DenormalFlushGuard {
    pub fn enable() -> Self {
        let saved = read_mxcsr();
        write_mxcsr(saved | FTZ | DAZ);
        Self { saved }
    }
}

impl Drop for DenormalFlushGuard {
    fn drop(&mut self) {
        write_mxcsr(self.saved);
    }
}
// Test-only accessors. Read the raw MXCSR control word / write an arbitrary one. These
// exist so the test file can construct precise prior-state scenarios (see
// `the_guard_restores_the_exact_prior_mxcsr_word_rather_than_a_hard_coded_default` and
// `ftz_and_daz_are_independently_verifiable_bits_neither_probe_satisfied_by_the_other`).
pub fn read_mxcsr_for_test() -> u32 {
    read_mxcsr()
}

pub fn write_mxcsr_for_test(value: u32) {
    write_mxcsr(value);
}
