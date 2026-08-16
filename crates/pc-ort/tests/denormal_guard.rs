//! FROZEN. Gate for the scoped denormal-flush guard (§16.52 item 3, item 5's D0).
//!
//! The guard exists because ONNX Runtime's `session.set_denormal_as_zero` reaches the
//! inference-CALLING thread only through a process-wide `std::call_once`, consumed by
//! whichever `InferenceSession` initializes first in the process (measured 2026-08-16,
//! `ort` 2.0.0-rc.12 + its bundled ONNX Runtime 1.24.2 -- see §16.52 items 1-2). This
//! module needs no `ort` and no `onnx` feature: it is a standalone MXCSR primitive, so
//! these tests run in the default tier, no model, no ONNX Runtime, no timing.
#![cfg(target_arch = "x86_64")]

use pc_ort::denormal::DenormalFlushGuard;

// Hard-coded from Intel SDM Vol.1 §10.2.3, deliberately NOT imported from the crate under
// test: an expectation derived from its own subject passes vacuously (cookbook rule 7/13).
const FTZ: u32 = 1 << 15;
const DAZ: u32 = 1 << 6;
const ABI_DEFAULT_MXCSR: u32 = 0x1f80;
const ROUND_TOWARD_NEG_INF: u32 = 0x2000; // MXCSR.RC = 01; neither FTZ nor DAZ

/// A denormal-flush oracle that reads no MXCSR bit at all -- it does arithmetic and looks
/// at the result. Independent of the bits the guard writes, so the bit-level and
/// behavioural assertions in this file cannot confirm each other.
fn denormal_survives_on_this_thread() -> bool {
    let smallest_normal = std::hint::black_box(f32::MIN_POSITIVE);
    let divisor = std::hint::black_box(2.0_f32);
    std::hint::black_box(smallest_normal / divisor) != 0.0
}

#[test]
fn a_subnormal_f32_flushes_to_zero_inside_the_guards_scope_and_survives_after_its_drop() {
    assert!(
        denormal_survives_on_this_thread(),
        "precondition: this test thread must start with denormals intact"
    );

    let inside = {
        let _guard = DenormalFlushGuard::enable();
        denormal_survives_on_this_thread()
    };
    let after = denormal_survives_on_this_thread();

    // Asserted as a pair, so a guard that never enables flushing (inside would stay true)
    // and a guard that never restores (after would stay false) are each rejected by a
    // DIFFERENT element. A single-sided assertion could not tell them apart.
    assert_eq!(
        (inside, after),
        (false, true),
        "(denormal survives inside guard, denormal survives after drop)"
    );
}

#[test]
fn the_guard_restores_the_exact_prior_mxcsr_word_rather_than_a_hard_coded_default() {
    let original = pc_ort::denormal::read_mxcsr_for_test();
    // A prior state that is NOT the ABI default and carries neither FTZ nor DAZ, so a
    // restore that writes 0x1f80 fails here while passing a test that started at 0x1f80.
    let prior = (original | ROUND_TOWARD_NEG_INF) & !(FTZ | DAZ);
    assert_ne!(
        prior, ABI_DEFAULT_MXCSR,
        "the probe state must differ from the ABI default or this test cannot fail"
    );

    pc_ort::denormal::write_mxcsr_for_test(prior);
    let inside = {
        let _guard = DenormalFlushGuard::enable();
        pc_ort::denormal::read_mxcsr_for_test()
    };
    let after = pc_ort::denormal::read_mxcsr_for_test();
    pc_ort::denormal::write_mxcsr_for_test(original); // leave the thread as we found it

    assert_eq!(
        (inside, after),
        (prior | FTZ | DAZ, prior),
        "(mxcsr inside guard, mxcsr after drop) -- the guard must set exactly FTZ|DAZ and \
         restore the exact saved word, touching no other field (e.g. the RC rounding mode)"
    );
}

#[test]
fn the_guard_restores_the_saved_word_even_when_its_scope_panics() {
    let original = pc_ort::denormal::read_mxcsr_for_test();

    let result = std::panic::catch_unwind(|| {
        let _guard = DenormalFlushGuard::enable();
        panic!("deliberate panic inside an active guard scope");
    });
    assert!(result.is_err(), "the panic must actually have propagated");

    let after = pc_ort::denormal::read_mxcsr_for_test();
    assert_eq!(
        after, original,
        "an unwind through the guard's scope must still restore the exact saved word"
    );
}

#[test]
fn nested_guards_restore_each_layers_own_saved_word_in_order() {
    let original = pc_ort::denormal::read_mxcsr_for_test();

    let outer_inside = {
        let _outer = DenormalFlushGuard::enable();
        let outer_state = pc_ort::denormal::read_mxcsr_for_test();
        let inner_inside = {
            let _inner = DenormalFlushGuard::enable();
            pc_ort::denormal::read_mxcsr_for_test()
        };
        let after_inner_drop = pc_ort::denormal::read_mxcsr_for_test();
        // The inner guard's drop must restore to the OUTER guard's flushing state, not to
        // the original pre-outer state -- proving each layer saves its own entry word
        // rather than sharing one global "the" saved value.
        assert_eq!(
            (inner_inside, after_inner_drop),
            (outer_state, outer_state),
            "(mxcsr inside the inner guard, mxcsr after the inner guard drops) -- nesting \
             must not collapse to a single shared saved word"
        );
        outer_state
    };
    let after_outer_drop = pc_ort::denormal::read_mxcsr_for_test();

    assert_eq!(
        (outer_inside, after_outer_drop),
        (original | FTZ | DAZ, original),
        "(mxcsr inside the outer guard, mxcsr after the outer guard drops)"
    );
}

#[test]
fn a_thread_spawned_inside_an_active_guard_scope_does_not_inherit_the_flush_state() {
    let _guard = DenormalFlushGuard::enable();
    let child_survives = std::thread::spawn(denormal_survives_on_this_thread)
        .join()
        .expect("spawned thread must not panic");
    assert!(
        child_survives,
        "MXCSR is per-thread on x86_64 -- a freshly spawned thread must start with the \
         ABI default state regardless of what the spawning thread's guard set"
    );
}

#[test]
fn ftz_and_daz_are_independently_verifiable_bits_neither_probe_satisfied_by_the_other() {
    let original = pc_ort::denormal::read_mxcsr_for_test();

    // FTZ-only: a normal-but-tiny result gets flushed to zero on store (this is what the
    // guard sets via MXCSR.FTZ). DAZ-only: a subnormal INPUT is treated as zero on load.
    // Setting one without the other must leave the other probe unaffected.
    pc_ort::denormal::write_mxcsr_for_test((original | FTZ) & !DAZ);
    let ftz_only_flushes = std::hint::black_box(f32::MIN_POSITIVE / std::hint::black_box(2.0));
    let ftz_only_treats_subnormal_input_as_zero =
        std::hint::black_box(f32::from_bits(1)) * std::hint::black_box(1e30);

    pc_ort::denormal::write_mxcsr_for_test((original | DAZ) & !FTZ);
    let daz_only_flushes = std::hint::black_box(f32::MIN_POSITIVE / std::hint::black_box(2.0));
    let daz_only_treats_subnormal_input_as_zero =
        std::hint::black_box(f32::from_bits(1)) * std::hint::black_box(1e30);

    pc_ort::denormal::write_mxcsr_for_test(original);

    assert_eq!(
        (
            ftz_only_flushes == 0.0,
            ftz_only_treats_subnormal_input_as_zero == 0.0,
            daz_only_flushes == 0.0,
            daz_only_treats_subnormal_input_as_zero == 0.0,
        ),
        (true, false, false, true),
        "(FTZ-only flushes a tiny store, FTZ-only zeroes a subnormal input, \
         DAZ-only flushes a tiny store, DAZ-only zeroes a subnormal input) -- a combined \
         probe that can't tell FTZ from DAZ would be satisfiable with either bit missing"
    );
}
