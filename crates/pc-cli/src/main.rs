//! `panel-ocr` — the binary (spec §13.1). Deliberately three lines: everything is in
//! `pc_cli` so it can be tested without spawning a process (§16.12 item 1).

use clap::Parser;

fn main() {
    let cli = pc_cli::Cli::parse();
    std::process::exit(pc_cli::run(cli));
}
