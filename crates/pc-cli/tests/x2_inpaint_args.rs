use clap::Parser;
use pc_cli::{Cli, Command};

#[test]
fn inpaint_parses_required_flags() {
    let cli = Cli::try_parse_from([
        "panel-ocr",
        "inpaint",
        "page.png",
        "--mask",
        "brush.png",
        "--output",
        "out.png",
    ])
    .expect("required flags present");
    let Command::Inpaint(args) = cli.command else {
        panic!("expected Inpaint")
    };
    assert_eq!(args.image, std::path::PathBuf::from("page.png"));
    assert_eq!(args.mask, std::path::PathBuf::from("brush.png"));
    assert_eq!(args.output, std::path::PathBuf::from("out.png"));
}

#[test]
fn inpaint_requires_mask_and_output() {
    assert!(
        Cli::try_parse_from(["panel-ocr", "inpaint", "page.png", "--output", "out.png"]).is_err()
    );
    assert!(
        Cli::try_parse_from(["panel-ocr", "inpaint", "page.png", "--mask", "brush.png"]).is_err()
    );
}
