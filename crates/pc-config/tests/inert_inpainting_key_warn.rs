//! Task **L2** test -- `DEVIATION(26)` (§14 item 26, ratified by spec §16.38 item 7(e)):
//! the one-time WARN for the inert `inpainter.inpainting_max_mask_radius` key.
//!
//! DELIBERATELY IN ITS OWN INTEGRATION-TEST BINARY, for the reason
//! `colored_images_warn.rs` gives verbatim: the "one-time" requirement is implemented with
//! a process-global `Once`, so any other test in the same binary that loaded a non-default
//! `inpainting_max_mask_radius` would consume it and make this test order-dependent. One
//! test binary == one process == one `Once`. Do not add other tests to this file, and do
//! not move these into another file. (§16.5 item 12: "one-time" is once per process.)

use pc_config::{ConfigWarning, ProfileDocument};
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    fn contents(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = CapturedLog;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// `6` is `config.py:819`'s dataclass default, so this document sets the key to exactly the
/// value it already has -- the bounded case §16.38 item 7(e) requires to stay silent.
const AT_DEFAULT: &str = "[inpainter]\ninpainting_max_mask_radius = 6\n";
const TUNED: &str = "[inpainter]\ninpainting_max_mask_radius = 9\n";

#[test]
// §16.38 item 7(e) / DEVIATION(26): `pc-config` emits a ONE-TIME WARN when a loaded profile
// sets `inpainting_max_mask_radius` away from its default, naming `min_inpainting_radius` as
// the key that actually gates and stating that the key is inert upstream too. Ground: §14
// item 7's rule that "an opt-in setting must not silently behave differently" -- a user who
// tunes a knob and gets no effect is exactly that case.
//
// Everything about this behaviour is asserted in one test, because they share the single
// process-global `Once`.
fn a_tuned_inert_inpainting_radius_warns_exactly_once_per_process() {
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(log.clone())
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .without_time()
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        // 1. The shipped default profile must be completely silent -- otherwise every
        //    `profile new` warns at the user about output it did not choose. This is the
        //    bound §16.38 item 7(e) puts on the deviation.
        let shipped = ProfileDocument::parse(pc_config::DEFAULT_PROFILE_TOML).unwrap();
        assert_eq!(shipped.warnings(), &[]);
        assert!(
            log.contents().is_empty(),
            "the shipped default must not warn: {:?}",
            log.contents()
        );

        // 2. Explicitly writing the default value is also silent: the trigger is the VALUE,
        //    not the presence of the key.
        let explicit_default = ProfileDocument::parse(AT_DEFAULT).unwrap();
        assert_eq!(explicit_default.warnings(), &[]);
        assert!(
            log.contents().is_empty(),
            "the default value must not warn even when written out: {:?}",
            log.contents()
        );

        // 3. First tuned load: one WARN, and the warning is also returned as data.
        let first = ProfileDocument::parse(TUNED).unwrap();
        assert_eq!(
            first.profile().inpainter.inpainting_max_mask_radius,
            9,
            "the value is honoured, not reset -- upstream parses it too"
        );
        assert!(first
            .warnings()
            .contains(&ConfigWarning::InertInpaintingMaxMaskRadius));

        let after_first = log.contents();
        assert_eq!(
            after_first.matches("inpainting_max_mask_radius").count(),
            1,
            "expected exactly one WARN mentioning the key, got: {after_first:?}"
        );

        // §16.38 item 7(e) requires the message to name the key that ACTUALLY gates, and to
        // say the key is inert upstream as well -- so a reader does not file it as our bug.
        assert!(
            after_first.contains("min_inpainting_radius"),
            "WARN must name the key that actually gates: {after_first:?}"
        );
        assert!(
            after_first.contains("upstream"),
            "WARN must say the key is inert upstream too: {after_first:?}"
        );

        // 4. Second and third tuned loads: still exactly one WARN in the process...
        let second = ProfileDocument::parse(TUNED).unwrap();
        let _third = ProfileDocument::parse("[inpainter]\ninpainting_max_mask_radius = 0\n")
            .expect("zero is a legal radius");
        let after_third = log.contents();
        assert_eq!(
            after_third.matches("inpainting_max_mask_radius").count(),
            1,
            "the WARN is ONE-TIME per process (§16.5 item 12), got: {after_third:?}"
        );

        // ...but the structured warning is still reported on every parse, so `profile
        // validate` can always show it.
        assert!(second
            .warnings()
            .contains(&ConfigWarning::InertInpaintingMaxMaskRadius));
    });
}
