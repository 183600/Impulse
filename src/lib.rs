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

// Re-export key types at the crate level
pub use compiler::{Compiler, CompilationResult};
pub use ir::{Module, Operation, Value, Type};
pub use runtime::{Device, ExecutionContext};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Value, Type, Operation};

    #[test]
    fn test_compiler_creation() {
        let compiler = ImpulseCompiler::new();
        assert_eq!(compiler.passes.passes.len(), 0); // Initially empty
    }

    #[test]
    fn test_mock_compilation() {
        let mut compiler = ImpulseCompiler::new();
        let mock_model = vec![0u8; 10]; // Mock ONNX model bytes
        
        // This should fail because we haven't implemented the actual functionality yet
        // But we want to make sure the interface works
        match compiler.compile(&mock_model, "cpu") {
            Ok(_) => println!("Compilation succeeded"),
            Err(e) => println!("Expected error during compilation: {:?}", e),
        }
    }

    #[test]
    fn test_module_creation() {
        let module = Module::new("test_module");
        assert_eq!(module.name, "test_module");
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.inputs.len(), 0);
        assert_eq!(module.outputs.len(), 0);
    }

    #[test]
    fn test_module_add_operation() {
        let mut module = Module::new("test_add_op");
        let op = Operation::new("add");
        module.add_operation(op);
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].op_type, "add");
    }

    #[test]
    fn test_value_creation() {
        let value = Value {
            name: "test_value".to_string(),
            ty: Type::F32,
            shape: vec![10, 20],
        };
        assert_eq!(value.name, "test_value");
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![10, 20]);
    }

    #[test]
    fn test_zero_sized_tensors() {
        // Test tensor with zero in dimensions
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![0, 10],  // Zero-sized dimension
        };
        assert_eq!(value.shape, vec![0, 10]);

        // This would represent a zero-sized tensor which is valid in many contexts
        let size_product: usize = value.shape.iter().map(|&x| x as usize).product();
        assert_eq!(size_product, 0);
    }

    #[test]
    fn test_empty_shape_tensor() {
        // Test scalar (0-dimensional tensor)
        let value = Value {
            name: "scalar".to_string(),
            ty: Type::I32,
            shape: vec![],  // Empty shape = scalar
        };
        assert_eq!(value.shape.len(), 0);
        assert!(value.shape.is_empty());
    }

    #[test]
    fn test_operation_creation() {
        let op = Operation::new("matmul");
        assert_eq!(op.op_type, "matmul");
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
    }

    #[test]
    fn test_operation_with_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("conv2d");
        let mut attrs = HashMap::new();
        attrs.insert("padding".to_string(), crate::ir::Attribute::Int(1));
        attrs.insert("stride".to_string(), crate::ir::Attribute::Int(2));
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 2);
        assert!(op.attributes.contains_key("padding"));
        assert!(op.attributes.contains_key("stride"));
    }
}