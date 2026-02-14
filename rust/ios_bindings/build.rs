//! Build script for ios_bindings
//!
//! Generates C header using cbindgen

fn main() {
    // Only generate header in release builds or when explicitly requested
    if std::env::var("GENERATE_HEADER").is_ok() || std::env::var("PROFILE").ok() == Some("release".into()) {
        generate_header();
    }
}

fn generate_header() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_path = format!("{}/include/exo_networking.h", crate_dir);

    // Create include directory if it doesn't exist
    std::fs::create_dir_all(format!("{}/include", crate_dir)).ok();

    let config = cbindgen::Config {
        language: cbindgen::Language::C,
        include_guard: Some("EXO_NETWORKING_H".into()),
        no_includes: true,
        includes: vec![
            "stdarg.h".into(),
            "stdbool.h".into(),
            "stdint.h".into(),
            "stdlib.h".into(),
        ],
        ..Default::default()
    };

    if let Ok(bindings) = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
    {
        bindings.write_to_file(&output_path);
        println!("cargo:rerun-if-changed=src/lib.rs");
    }
}
