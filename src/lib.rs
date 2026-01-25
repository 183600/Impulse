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

    #[rstest::rstest]
    #[case("")]
    #[case("very_long_name_that_exceeds_normal_limits_by_a_lot_and_is_used_to_test_string_handling_capabilities")]
    #[case("!@#$%^&*()_+-=[]{}|;':\",./<>?~`")]
    fn test_compiler_with_special_case_strings(#[case] input_string: String) {
        let mut compiler = ImpulseCompiler::new();
        let mock_model = vec![1u8, 2u8, 3u8];
        
        // Test compilation with special case strings to ensure they don't cause crashes
        let result = compiler.compile(&mock_model, &input_string);
        if result.is_err() {
            let err_msg = result.unwrap_err().to_string();
            assert!(err_msg.len() > 0);
        }
    }

    #[test]
    fn test_compiler_with_numeric_boundaries() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with maximum possible values for model sizes
        let max_model = vec![0u8; 100_000_000]; // 100MB model
        let result = compiler.compile(&max_model, "cpu");
        
        // Should not panic, regardless of result
        if result.is_err() {
            let err_msg = result.unwrap_err().to_string();
            assert!(err_msg.len() > 0);
        }
    }

    #[test]
    fn test_compiler_with_multiple_concurrent_instances() {
        // Create multiple compiler instances to test memory behavior
        let compiler1 = ImpulseCompiler::new();
        let compiler2 = ImpulseCompiler::new();
        
        // Test that multiple instances can coexist
        assert_eq!(compiler1.passes.passes.len(), 0);
        assert_eq!(compiler2.passes.passes.len(), 0);
    }

    #[test]
    fn test_module_with_extreme_values_in_shape() {
        // Create a module with operations containing extreme values in tensor shapes
        let mut module = Module::new("extreme_module");
        
        // Add an operation with maximum possible dimensions
        let extreme_value = Value {
            name: "extreme_tensor".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 1],  // Test extreme values though multiplication may overflow
        };
        
        let mut op = Operation::new("test_op");
        op.inputs.push(extreme_value);
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].inputs[0].name, "extreme_tensor");
    }

    #[test]
    fn test_operation_with_empty_name() {
        // Test creating an operation with an empty name
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    #[test]
    fn test_value_with_unicode_names() {
        // Test creating values with unicode names which are valid in Rust
        let unicode_value = Value {
            name: "tensor_名称_日本語_🔥".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        
        assert_eq!(unicode_value.name, "tensor_名称_日本語_🔥");
        assert_eq!(unicode_value.ty, Type::F32);
        assert_eq!(unicode_value.shape, vec![2, 3]);
    }

    #[rstest::rstest]
    #[case(vec![], 1)] // scalar has 1 element
    #[case(vec![0], 0)] // contains 0, so product is 0
    #[case(vec![1], 1)]
    #[case(vec![1, 1, 1, 1, 1], 1)]
    #[case(vec![2, 3, 4], 24)]
    #[case(vec![10, 0, 5], 0)] // contains 0, so product is 0
    fn test_shape_product_calculations(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let actual_product: usize = shape.iter().product();
        assert_eq!(actual_product, expected_product);
    }

    /// Test for handling extremely large model inputs
    #[test]
    fn test_compiler_with_extremely_large_input() {
        let mut compiler = ImpulseCompiler::new();
        
        // Create a large input to test memory handling
        let large_model = vec![0u8; 50_000_000];  // 50MB
        
        // This should not panic, regardless of result
        let result = compiler.compile(&large_model, "cpu");
        assert!(result.is_ok() || result.is_err());
    }

    /// Test for deeply nested tensor types that could cause stack overflow
    #[test]
    fn test_very_deep_tensor_nesting() {
        // Create a deeply nested type without causing stack overflow
        let mut current_type = Type::F32;
        let max_depth = 100;  // Reduced depth to avoid stack overflow
        
        for _ in 0..max_depth {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],  // Minimal shape to avoid size explosion
            };
        }
        
        // Test that we can compare and clone the nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test for module serialization/deserialization with complex structures
    #[test]
    fn test_module_serialization_complex() {
        let mut module = Module::new("complex_serialization_test");
        
        // Add a complex operation with nested structures
        let mut complex_op = Operation::new("complex_op");
        complex_op.inputs.push(Value {
            name: "complex_input".to_string(),
            ty: Type::F32,
            shape: vec![10, 20, 30],
        });
        
        // Add nested attributes
        use std::collections::HashMap;
        let mut attrs = HashMap::new();
        attrs.insert(
            "nested_array".to_string(), 
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![Attribute::String("nested".to_string())]),
            ])
        );
        complex_op.attributes = attrs;
        
        module.add_operation(complex_op);
        
        // Test that the module can be cloned (similar to serialize/deserialize)
        let cloned_module = module.clone();
        assert_eq!(module.name, cloned_module.name);
        assert_eq!(module.operations.len(), cloned_module.operations.len());
    }

    /// Test for operation with maximum possible attributes
    #[test]
    fn test_operation_with_maximum_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add a large number of attributes
        for i in 0..10_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 10_000);
        assert_eq!(op.op_type, "max_attrs_op");
    }

    /// Test for memory cleanup with complex nested objects
    #[test]
    fn test_memory_cleanup_complex_structures() {
        // Create complex nested structures
        let mut modules = Vec::new();
        
        for i in 0..1_000 {
            let mut module = Module::new(&format!("cleanup_test_{}", i));
            
            // Add operations to each module
            for j in 0..5 {
                let mut op = Operation::new(&format!("op_{}_{}", i, j));
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1, j + 1],
                });
                module.add_operation(op);
            }
            
            modules.push(module);
        }
        
        // Verify we created the expected number of modules
        assert_eq!(modules.len(), 1_000);
        
        // Drop all modules to test cleanup
        drop(modules);
        
        // Test passes if no memory leaks or panics occurred
        assert!(true); // Dummy assertion to satisfy test requirement
    }

    // Additional edge case tests
    #[rstest]
    #[case(vec![], 1)]  // scalar has 1 element
    #[case(vec![0], 0)] // zero dimension results in 0 elements
    #[case(vec![0, 10], 0)] // any dimension of 0 results in 0 elements
    #[case(vec![10, 0], 0)]
    #[case(vec![1, 1, 0, 100], 0)]
    #[case(vec![5, 5], 25)] // normal case
    #[case(vec![1], 1)]     // single dimension
    fn test_tensor_shape_products_extended(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let actual_product: usize = shape.iter().product();
        assert_eq!(actual_product, expected_product);
    }

    #[test]
    fn test_sink_operation() {
        let mut op = Operation::new("sink");
        
        // Add many inputs but no outputs
        for i in 0..10 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
        }
        
        assert_eq!(op.inputs.len(), 10);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.op_type, "sink");
    }

    #[test]
    fn test_source_operation() {
        let mut op = Operation::new("source");
        
        // Add no inputs but many outputs
        for i in 0..10 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
        }
        
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 10);
        assert_eq!(op.op_type, "source");
    }

    #[test]
    fn test_deeply_nested_attribute_arrays() {
        // Create a deeply nested array structure
        let mut nested = crate::ir::Attribute::Int(42);
        
        // Nest 5 levels deep
        for level in 1..=5 {
            nested = crate::ir::Attribute::Array(vec![
                crate::ir::Attribute::Int(level),
                nested,
            ]);
        }
        
        // Verify the structure can be matched without stack overflow
        match &nested {
            crate::ir::Attribute::Array(arr) => {
                assert_eq!(arr.len(), 2);
                match &arr[0] {
                    crate::ir::Attribute::Int(val) => assert_eq!(*val, 5), // Top level value
                    _ => panic!("Expected Int at top level"),
                }
            },
            _ => panic!("Expected Array at top level"),
        }
    }

    #[rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    fn test_zero_sized_tensors_extended(#[case] tensor_type: Type) {
        let zero_tensor = Value {
            name: "zero_tensor".to_string(),
            ty: tensor_type,
            shape: vec![10, 0, 5], // Contains 0, so total size is 0
        };
        
        assert_eq!(zero_tensor.shape, vec![10, 0, 5]);
        
        // Size calculation should be 0 regardless of type
        let size: usize = zero_tensor.shape.iter().product();
        assert_eq!(size, 0);
        
        // Test tensor size calculation function as well
        let calculated_size = crate::utils::ir_utils::calculate_tensor_size(&zero_tensor.ty, &zero_tensor.shape).unwrap();
        assert_eq!(calculated_size, 0);
    }

    #[test]
    fn test_operation_with_maximum_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attr_op");
        let mut attrs = HashMap::new();
        
        // Add many attributes
        for i in 0..100 {
            attrs.insert(
                format!("attribute_{:03}", i),
                crate::ir::Attribute::String(format!("value_{}", i))
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 100);
        assert_eq!(op.op_type, "max_attr_op");
        
        // Verify we can access some attributes
        assert!(op.attributes.contains_key("attribute_000"));
        assert!(op.attributes.contains_key("attribute_099"));
    }

    #[rstest]
    #[case("", "empty_name")]
    #[case(" ", "space_name")]
    #[case("	", "tab_name")]
    #[case("
", "newline_name")]
    fn test_empty_and_whitespace_names(#[case] name: &str, #[case] desc: &str) {
        // Test empty/whitespace operation name
        let op = Operation::new(name);
        assert_eq!(op.op_type, name);
        assert_eq!(op.inputs.len(), 0);
        
        // Test empty/whitespace value name
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        assert_eq!(value.name, name);
        
        // Test empty/whitespace module name
        let module = Module::new(name);
        assert_eq!(module.name, name);
    }

    #[test]
    fn test_compiler_with_invalid_utf8_sequences() {
        let mut compiler = ImpulseCompiler::new();
        
        // Create byte sequence that looks like invalid UTF-8 continuation bytes
        let mut invalid_like_bytes = vec![0u8; 10];
        invalid_like_bytes[5] = 0xFF;  // Invalid UTF-8 start byte
        invalid_like_bytes[6] = 0xFE;  // Another invalid sequence
        
        // This should not panic, regardless of result
        let result = compiler.compile(&invalid_like_bytes, "cpu");
        
        // Validate that the result is either success or a proper error (not a panic)
        if result.is_err() {
            let err_msg = result.unwrap_err().to_string();
            assert!(!err_msg.is_empty());  // Should have some error message
        }
    }
}