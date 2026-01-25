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

    #[test]
    fn test_module_creation_and_operation_management() {
        // Test creating a module and managing operations
        let mut module = Module::new("test_module_ops");
        
        // Initially empty
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.name, "test_module_ops");
        
        // Add an operation
        let op1 = Operation::new("add");
        module.add_operation(op1);
        assert_eq!(module.operations.len(), 1);
        
        // Add more operations
        let op2 = Operation::new("multiply");
        module.add_operation(op2);
        
        let op3 = Operation::new("relu");
        module.add_operation(op3);
        
        assert_eq!(module.operations.len(), 3);
        
        // Verify the order
        assert_eq!(module.operations[0].op_type, "add");
        assert_eq!(module.operations[1].op_type, "multiply");
        assert_eq!(module.operations[2].op_type, "relu");

        // Test with operations that have inputs and outputs
        let mut complex_op = Operation::new("matmul");
        complex_op.inputs.push(Value {
            name: "matrix_a".to_string(),
            ty: Type::F32,
            shape: vec![10, 20],
        });
        complex_op.outputs.push(Value {
            name: "result".to_string(),
            ty: Type::F32,
            shape: vec![10, 30], // Assuming matrix multiplication A[10,20] * B[20,30]
        });
        
        module.add_operation(complex_op);
        assert_eq!(module.operations.len(), 4);
        
        // Check the complex operation
        let last_op = &module.operations[3];
        assert_eq!(last_op.op_type, "matmul");
        assert_eq!(last_op.inputs.len(), 1);
        assert_eq!(last_op.outputs.len(), 1);
        assert_eq!(last_op.inputs[0].name, "matrix_a");
        assert_eq!(last_op.outputs[0].name, "result");
    }

    #[test]
    fn test_impulse_compiler_functionality() {
        // Test the main compiler interface
        let mut compiler = ImpulseCompiler::new();
        
        // Check that components are properly initialized
        assert_eq!(compiler.passes.passes.len(), 0);
        
        // Test the compile method with empty model (expected to fail gracefully with meaningful error)
        let empty_model = vec![];
        let result = compiler.compile(&empty_model, "cpu");
        
        // The result may be an error (which is acceptable for an empty model)
        // The important thing is that the interface works without panicking
        if result.is_err() {
            let err_msg = result.unwrap_err().to_string();
            // Make sure it's a reasonable error message (not a panic or system crash)
            assert!(err_msg.len() > 0);  // Should have some error message
        }
        
        // Test compiler with mock data
        let mock_model_data = vec![1u8, 2u8, 3u8, 4u8];
        let result2 = compiler.compile(&mock_model_data, "cpu");
        
        if result2.is_err() {
            let err_msg = result2.unwrap_err().to_string();
            assert!(err_msg.len() > 0);
        }
    }
    
    #[test]
    fn test_compiler_with_large_model() {
        // Test with a large model to check memory handling
        let mut compiler = ImpulseCompiler::new();
        let large_model = vec![0u8; 10_000_000]; // 10MB model
        
        // Test that the compiler handles large input without crashing
        let _result = compiler.compile(&large_model, "cpu");
        // Note: The result may be success or failure depending on implementation,
        // but the important thing is that it doesn't panic or cause memory issues
    }
    
    #[test]
    fn test_compiler_with_empty_model() {
        let mut compiler = ImpulseCompiler::new();
        let empty_model = vec![]; // Empty model bytes
        
        // Test that the compiler handles empty input without crashing
        let _result = compiler.compile(&empty_model, "cpu");
        // Note: The result may be success or failure depending on implementation,
        // but the important thing is that it doesn't panic
    }

    #[test]
    fn test_compiler_with_special_characters() {
        let mut compiler = ImpulseCompiler::new();
        let special_bytes = vec![0xFF, 0xFE, 0xFD]; // Some special byte sequences
        
        // Test that the compiler handles special characters without crashing
        let _result = compiler.compile(&special_bytes, "cpu");
    }

    #[test]
    fn test_compiler_with_unicode_like_bytes() {
        let mut compiler = ImpulseCompiler::new();
        // Bytes that look like UTF-8 sequences but aren't necessarily valid
        let unicode_like = vec![0xC0, 0x80, 0xE0, 0x80, 0x80, 0xF0, 0x80, 0x80, 0x80];
        
        // Test that the compiler handles these bytes without crashing
        let _result = compiler.compile(&unicode_like, "cpu");
    }

    #[rstest::rstest]
    fn test_compiler_with_invalid_targets(#[values("cpu", "gpu", "invalid", "")] target: &str) {
        let mut compiler = ImpulseCompiler::new();
        let mock_model = vec![1u8, 2u8, 3u8];

        // Test compilation with different target strings
        let result = compiler.compile(&mock_model, target);
        
        // The result is expected to fail for most targets since functionality isn't implemented
        // but it shouldn't panic
        if !target.is_empty() && target != "cpu" {
            // Targets other than "cpu" might fail differently
            if result.is_err() {
                let err_msg = result.unwrap_err().to_string();
                assert!(err_msg.len() > 0);  // Should have some error message
            }
        }
    }

    #[test]
    fn test_compiler_with_very_long_strings() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with a very long target string to check for buffer issues
        let very_long_target = "v".repeat(10000);
        let mock_model = vec![1u8, 2u8, 3u8];
        
        let result = compiler.compile(&mock_model, &very_long_target);
        if result.is_err() {
            let err_msg = result.unwrap_err().to_string();
            assert!(err_msg.len() > 0);
        }
    }
}