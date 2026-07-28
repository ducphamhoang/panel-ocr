//! Verbosity → `tracing` level (spec §13.1: "verbosity maps to `tracing` levels").
//!
//! Fully pinned; implemented, not stubbed.

/// `-q` → `error`; no flag → `warn`; `-v` → `info`; `-vv` → `debug`; `-vvv`+ → `trace`.
pub fn filter_directive(verbose: u8, quiet: bool) -> &'static str {
    if quiet {
        return "error";
    }
    match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    }
}

/// Install the subscriber. `RUST_LOG`, when set, wins over the flags — the usual Rust
/// convention, and the only way to filter per-crate.
pub fn init(verbose: u8, quiet: bool) {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(filter_directive(verbose, quiet)));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
