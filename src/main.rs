//! Impulse Compiler - Entry Point
//!
//! This is the main binary for the Impulse AI compiler tool

use impulse::ImpulseCompiler;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <input_model> [target]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let target = args.get(2).map(|s| s.as_str()).unwrap_or("cpu");

    println!("Loading model from: {}", input_path);
    println!("Target: {}", target);

    // Load the model file
    let model_bytes = std::fs::read(input_path)?;
    
    // Create and use the compiler
    let mut compiler = ImpulseCompiler::new();
    let result = compiler.compile(&model_bytes, target)?;

    println!("Compilation completed successfully!");
    println!("Compiled output size: {} bytes", result.len());
    
    Ok(())
}