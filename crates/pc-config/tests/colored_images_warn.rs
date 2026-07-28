//! C3 test -- spec §15.7 / §14.7: the one-time `colored_images = true` WARN.
//!
//! DELIBERATELY IN ITS OWN INTEGRATION-TEST BINARY. The "one-time" requirement is
//! implemented with a process-global `Once`, so any other test in the same binary that
//! loaded a `colored_images = true` profile would consume it and make this test
//! order-dependent. One test binary == one process == one `Once`.
//! Do not add other tests to this file, and do not move these into another file.
//! (Spec §16.5 item 12: "one-time" is read as once per process.)

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

const COLORED: &str = "[denoiser]\ncolored_images = true\n";
const NOT_COLORED: &str = "[denoiser]\ncolored_images = false\n";

#[test]
// spec §15.7: `pc-config` must emit a ONE-TIME WARN when a loaded profile sets
// colored_images = true, stating that v1 uses a joint-channel approximation and that
// color_filter_strength is ignored until v1.5 -- "an opt-in setting must not silently
// behave differently". Everything about this behaviour is asserted here in one test,
// because they share the single process-global `Once`.
fn colored_images_true_warns_exactly_once_per_process() {
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(log.clone())
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .without_time()
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        // 1. colored_images = false must be completely silent (upstream's own default,
        //    §15.7 -- v1's default path is unaffected).
        let quiet = ProfileDocument::parse(NOT_COLORED).unwrap();
        assert_eq!(quiet.warnings(), &[]);
        assert!(
            log.contents().is_empty(),
            "false must not warn: {:?}",
            log.contents()
        );

        // 2. First colored load: one WARN, and the warning is also returned as data.
        let first = ProfileDocument::parse(COLORED).unwrap();
        assert!(
            first.profile().denoiser.colored_images,
            "the setting is honoured, not reset"
        );
        assert!(first
            .warnings()
            .contains(&ConfigWarning::ColoredImagesApproximation));

        let after_first = log.contents();
        assert_eq!(
            after_first.matches("colored_images").count(),
            1,
            "expected exactly one WARN mentioning the key, got: {after_first:?}"
        );

        // §15.7 requires the message to state BOTH facts.
        assert!(
            after_first.contains("color_filter_strength"),
            "WARN must say color_filter_strength is ignored: {after_first:?}"
        );
        assert!(
            after_first.contains("v1.5"),
            "WARN must say the Lab-split path lands in v1.5: {after_first:?}"
        );

        // 3. Second and third colored loads: still exactly one WARN in the process...
        let second = ProfileDocument::parse(COLORED).unwrap();
        let _third = ProfileDocument::parse(COLORED).unwrap();
        let after_third = log.contents();
        assert_eq!(
            after_third.matches("colored_images").count(),
            1,
            "the WARN is ONE-TIME per process (§15.7), got: {after_third:?}"
        );

        // ...but the structured warning is still reported on every parse, so a caller
        // (e.g. `profile validate`) can always see it.
        assert!(second
            .warnings()
            .contains(&ConfigWarning::ColoredImagesApproximation));
    });
}
