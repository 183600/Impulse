//! Impulse Compiler - Entry Point
//!
//! This is the main binary for the Impulse AI compiler tool

use clap::Parser;
use impulse::ImpulseCompiler;

#[derive(Parser)]
#[command(name = "impulse-compiler")]
#[command(about = "An AI heterogeneous computing compiler", long_about = None)]
struct Args {
    /// Path to the input model file
    #[arg(value_name = "INPUT_MODEL")]
    input_model: String,

    /// Target backend (e.g., cpu, cuda)
    #[arg(short, long, default_value_t = String::from("cpu"))]
    target: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("Loading model from: {}", args.input_model);
    println!("Target: {}", args.target);

    // Load the model file
    let model_bytes = std::fs::read(&args.input_model)?;
    
    // Create and use the compiler
    let mut compiler = ImpulseCompiler::new();
    let result = compiler.compile(&model_bytes, &args.target)?;

    println!("Compilation completed successfully!");
    println!("Compiled output size: {} bytes", result.len());
    
    Ok(())
}