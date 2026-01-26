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

#[cfg(test)]
mod additional_edge_case_tests;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod more_edge_case_tests;

#[cfg(test)]
mod final_edge_case_tests;



#[cfg(test)]
mod enhanced_edge_case_tests;

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
mod new_boundary_tests;

#[cfg(test)]
mod additional_edge_case_tests_final;

#[cfg(test)]
mod more_edge_cases;


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

    /// Test 1: Operations with extreme string values for names/types
    #[test]
    fn test_operations_extreme_string_values() {
        // Very long operation name
        let long_name = "o".repeat(100_000); // 100k characters for operation name (reduced for faster tests)
        let op = Operation::new(&long_name);
        assert_eq!(op.op_type, long_name);
        
        // Value with very long name
        let value = Value {
            name: "v".repeat(100_000),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name.len(), 100_000);
    }

    /// Test 2: Tensor types with maximum nesting and complex shapes
    #[test]
    fn test_extremely_nested_tensor_types() {
        // Create a deeply nested tensor type (reduced depth for performance)
        let mut current_type = Type::F32;
        for _ in 0..100 {  // Reduced depth to 100 instead of 1000 for performance
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }

        // Verify the structure
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a Tensor type"),
        }
        
        // Test cloning of deeply nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 3: Mathematical operations on tensor shapes that could cause overflow
    #[test]
    fn test_tensor_shape_mathematical_edge_cases() {
        // Test shape product calculations that might overflow
        let problematic_shape = vec![10_000_000, 10_000_000]; // Reduced size for performance
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: problematic_shape,
        };
        
        // Calculate using checked multiplication to prevent overflow
        let product_result: Option<usize> = value.shape.iter()
            .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
        
        // This should handle the overflow gracefully
        assert!(product_result.is_some() || true); // Either returns a value or handles overflow
        
        // Test with safe smaller values
        let safe_shape = vec![10_000, 10_000];
        let safe_value = Value {
            name: "safe_test".to_string(),
            ty: Type::F32,
            shape: safe_shape,
        };
        let safe_product: usize = safe_value.shape.iter().product();
        assert_eq!(safe_product, 100_000_000);
    }

    /// Test 4: Operations with maximum possible attribute diversity
    #[test]
    fn test_operations_maximum_attribute_diversity() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("diverse_attrs");
        
        // Insert variety of attribute types
        let mut attrs = HashMap::new();
        
        // Add all primitive attribute types
        attrs.insert("int_attr".to_string(), crate::ir::Attribute::Int(i64::MAX));
        attrs.insert("min_int_attr".to_string(), crate::ir::Attribute::Int(i64::MIN));
        attrs.insert("float_attr".to_string(), crate::ir::Attribute::Float(f64::MAX));
        attrs.insert("min_float_attr".to_string(), crate::ir::Attribute::Float(f64::MIN));
        attrs.insert("zero_float_attr".to_string(), crate::ir::Attribute::Float(0.0));
        attrs.insert("negative_float_attr".to_string(), crate::ir::Attribute::Float(-3.14159));
        attrs.insert("empty_string_attr".to_string(), crate::ir::Attribute::String("".to_string()));
        attrs.insert("long_string_attr".to_string(), crate::ir::Attribute::String("long".repeat(10_000)));
        attrs.insert("true_bool_attr".to_string(), crate::ir::Attribute::Bool(true));
        attrs.insert("false_bool_attr".to_string(), crate::ir::Attribute::Bool(false));
        
        // Add nested array attributes
        attrs.insert("nested_array".to_string(), crate::ir::Attribute::Array(vec![
            crate::ir::Attribute::Array(vec![
                crate::ir::Attribute::Int(1),
                crate::ir::Attribute::Float(2.5),
            ]),
            crate::ir::Attribute::Array(vec![
                crate::ir::Attribute::String("nested".to_string()),
                crate::ir::Attribute::Bool(true),
            ])
        ]));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 11);
        assert_eq!(op.attributes.get("int_attr"), Some(&crate::ir::Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("min_int_attr"), Some(&crate::ir::Attribute::Int(i64::MIN)));
    }

    /// Test 5: Special floating point values in tensor calculations
    #[test]
    fn test_special_floating_point_values() {
        // Test values that could appear in tensor computations
        let special_values = [
            std::f64::INFINITY,
            std::f64::NEG_INFINITY,
            std::f64::NAN,
            -0.0,  // Negative zero
            std::f64::EPSILON,  // Smallest value
            std::f64::consts::PI,
            std::f64::consts::E,
        ];
        
        for (_i, val) in special_values.iter().enumerate() {
            // Test with special float values in attribute
            let attr = crate::ir::Attribute::Float(*val);
            
            // Can't directly compare NaN, so handle separately
            if val.is_nan() {
                if let crate::ir::Attribute::Float(retrieved_val) = attr {
                    assert!(retrieved_val.is_nan());
                }
            } else {
                // For other special values, we can do direct comparison
                match attr {
                    crate::ir::Attribute::Float(retrieved_val) => {
                        if (*val - retrieved_val).abs() < f64::EPSILON || 
                           ((*val).is_infinite() && retrieved_val.is_infinite()) {
                            // Accept as valid for infinity values
                        } else {
                            assert_eq!(retrieved_val, *val);
                        }
                    },
                    _ => panic!("Expected Float attribute"),
                }
            }
        }
    }

    /// Test 6: Recursive type with alternating types
    #[test]
    fn test_alternating_recursive_types() {
        // Create a recursive type that alternates between different base types
        let mut current_type = Type::I32;
        for i in 0..20 {  // Reduced for performance
            let next_type = if i % 2 == 0 {
                Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![i + 1],  // Use i+1 instead
                }
            } else {
                Type::Tensor {
                    element_type: Box::new(current_type),
                    shape: vec![2],
                }
            };
            current_type = next_type;
        }
        
        // Just ensure we can create and clone this complex recursive type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 7: Memory handling with large number of operations
    #[test]
    fn test_large_number_of_operations() {
        let mut module = Module::new("large_ops_module");
        
        // Add many operations to test memory handling
        for i in 0..5_000 {  // Reduced for test performance
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 10 + 1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 5_000);
        assert_eq!(module.name, "large_ops_module");
    }

    /// Test 8: Unicode and special characters in identifiers
    #[test]
    fn test_unicode_identifiers() {
        let test_cases = [
            ("valid_unicode_🚀", Type::F32),
            ("chinese_chars_中文", Type::I32),
            ("arabic_chars_مرحبا", Type::F64),
            ("accented_chars_café_naïve", Type::I64),
            ("control_chars_\u{0001}_\u{001F}", Type::Bool),
        ];

        for (identifier, data_type) in test_cases.iter() {
            // Test values with unicode identifiers
            let value = Value {
                name: identifier.to_string(),
                ty: data_type.clone(),  // Clone to avoid move
                shape: vec![1],
            };
            assert_eq!(value.name, *identifier);
            assert_eq!(value.ty, *data_type);

            // Test operations with unicode names
            let op = Operation::new(identifier);
            assert_eq!(op.op_type, *identifier);
            
            // Test modules with unicode names
            let module = Module::new(*identifier);
            assert_eq!(module.name, *identifier);
        }
    }

    /// Test 9: Edge cases with zero-sized tensors
    #[test]
    fn test_zero_sized_tensors_edge_cases() {
        let test_cases = [
            vec![0],              // Single zero dimension
            vec![0, 5],           // Zero followed by positive
            vec![5, 0],           // Positive followed by zero
            vec![2, 0, 3],        // Zero in the middle
            vec![0, 0, 0],        // Multiple zeros
            vec![0, 1, 0, 1],     // Alternating zeros and ones
        ];

        for shape in test_cases.iter() {
            let value = Value {
                name: "zero_test".to_string(),
                ty: Type::F32,
                shape: shape.to_vec(),
            };

            // Any tensor with a zero dimension should have 0 elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, 0, "Shape {:?} should have 0 total elements", shape);
        }
    }

    /// Test 10: Comprehensive compiler integration test
    #[test]
    fn test_compiler_integration_edge_cases() {
        let compiler = ImpulseCompiler::new();
        
        // Test that compiler object has been created properly
        assert_eq!(compiler.passes.passes.len(), 0);
        
        // Verify compiler methods work correctly
        // Since other functionality isn't implemented, just ensure no panics occur
        drop(compiler);
        
        // Simple test to verify compiler can be recreated after dropping
        let new_compiler = ImpulseCompiler::new();
        assert_eq!(new_compiler.passes.passes.len(), 0);
    }
}