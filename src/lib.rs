//! Impulse - An AI Heterogeneous Computing Compiler (based on MLIR)
//! 
//! This crate implements a compiler that transforms high-level computation graphs
//! (PyTorch/ONNX/StableHLO) into high-performance executables for CPU/GPU/NPU.

pub mod compiler;
pub mod ir;
pub mod passes;
pub mod backends;
pub mod runtime;
pub mod frontend;
pub mod transforms;
pub mod utils;
pub mod autotuning;

// Only include unique test modules (without duplicates)
#[cfg(test)]
mod new_edge_case_tests_extended;

#[cfg(test)]
mod additional_edge_cases;

#[cfg(test)]
mod more_edge_case_tests_new;

#[cfg(test)]
mod attribute_edge_case_tests;

#[cfg(test)]
mod additional_boundary_tests;

#[cfg(test)]
mod additional_comprehensive_tests;

#[cfg(test)]
mod new_more_edge_cases;

#[cfg(test)]
mod final_edge_case_tests;

#[cfg(test)]
mod enhanced_edge_case_tests;

#[cfg(test)]
mod new_boundary_tests;

#[cfg(test)]
mod additional_tests;

#[cfg(test)]
mod more_comprehensive_edge_case_tests;

#[cfg(test)]
mod more_edge_case_tests;

#[cfg(test)]
mod additional_edge_case_tests_focused;

#[cfg(test)]
mod more_edge_case_tests_comprehensive;

#[cfg(test)]
mod additional_edge_case_tests_newest;

#[cfg(test)]
mod new_edge_case_tests_additional;

#[cfg(test)]
mod additional_edge_case_tests_newer;

#[cfg(test)]
mod comprehensive_edge_case_tests;

#[cfg(test)]
mod edge_case_tests_additional;

#[cfg(test)]
mod type_conversion_edge_case_tests;

#[cfg(test)]
mod complex_attribute_tests;

#[cfg(test)]
mod validation_and_module_tests;

#[cfg(test)]
mod new_advanced_edge_case_tests;

#[cfg(test)]
mod additional_edge_case_tests;

#[cfg(test)]
mod edge_case_tests_new;

#[cfg(test)]
mod additional_edge_case_tests_extended;

#[cfg(test)]
mod more_boundary_tests;

#[cfg(test)]
mod concurrent_edge_case_tests;

#[cfg(test)]
mod new_important_edge_cases;

#[cfg(test)]
mod additional_edge_test_cases;

#[cfg(test)]
mod additional_edge_case_tests_new;

#[cfg(test)]
mod additional_edge_case_tests_comprehensive;

#[cfg(test)]
mod additional_edge_case_tests_new_new;

// Re-export key types at the crate level
pub use compiler::{Compiler, CompilationResult};
pub use ir::{Module, Value, Type};
pub use runtime::{Device, ExecutionContext};
pub use utils::ir_utils;

// Additional test modules
#[cfg(test)]
mod additional_edge_case_boundary_tests;

#[cfg(test)]
mod new_edge_case_tests;

#[cfg(test)]
mod additional_edge_case_tests_final;

#[cfg(test)]
mod edge_case_tests_extended;

#[cfg(test)]
mod focused_edge_case_tests;

#[cfg(test)]
mod additional_edge_case_tests_new_new_new;

#[cfg(test)]
mod additional_ir_edge_case_tests;

#[cfg(test)]
mod rstest_additional_edge_cases;

#[cfg(test)]
mod additional_edge_case_tests_new_new_new_new;

#[cfg(test)]
mod additional_edge_case_tests_new_new_new_new_new;

#[cfg(test)]
mod additional_focused_edge_case_tests;

#[cfg(test)]
mod new_essential_edge_case_tests;

#[cfg(test)]
mod new_edge_case_tests_comprehensive;

#[cfg(test)]
mod advanced_edge_case_tests_comprehensive;

#[cfg(test)]
mod edge_case_overflows_and_limits;

#[cfg(test)]
mod additional_edge_case_tests_boundary_conditions;

#[cfg(test)]
mod boundary_edge_case_tests;

#[cfg(test)]
mod compiler_edge_case_tests;

#[cfg(test)]
mod more_edge_cases;

#[cfg(test)]
mod additional_edge_case_tests;

/// Main entry point for the Impulse compiler
pub struct ImpulseCompiler {
    /// Frontend for importing models
    pub frontend: frontend::Frontend,
    
    /// Optimization passes
    pub passes: passes::PassManager,
    
    /// Backend targets
    pub backends: backends::BackendManager,
    
    /// Runtime system
    pub runtime: runtime::Runtime,
}

impl ImpulseCompiler {
    pub fn new() -> Self {
        Self {
            frontend: frontend::Frontend::new(),
            passes: passes::PassManager::new(),
            backends: backends::BackendManager::new(),
            runtime: runtime::Runtime::new(),
        }
    }
    
    pub fn compile(&mut self, model_bytes: &[u8], target: &str) -> anyhow::Result<Vec<u8>> {
        // Import the model
        let mut module = self.frontend.import_onnx(model_bytes)?;
        
        // Apply optimization passes
        self.passes.run_passes(&mut module)?;
        
        // Lower to target backend
        let compiled_bytes = self.backends.compile(&module, target)?;
        
        Ok(compiled_bytes)
    }
}




