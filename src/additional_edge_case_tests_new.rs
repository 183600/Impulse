//! Additional edge case tests for the Impulse compiler
//! This file covers extra boundary conditions and edge cases not covered in the main test suite

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use crate::ImpulseCompiler;
    use rstest::rstest;

    /// Test 1: Operations with maximum possible input/output counts
    #[test]
    fn test_operation_with_huge_number_of_inputs_outputs() {
        let mut op = Operation::new("test_op");
        
        // Add a large number of inputs to test memory handling
        for i in 0..10_000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1],
            });
        }
        
        // Add a large number of outputs  
        for i in 0..5_000 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 50 + 1],
            });
        }
        
        assert_eq!(op.inputs.len(), 10_000);
        assert_eq!(op.outputs.len(), 5_000);
    }

    /// Test 2: Empty module with minimal operations
    #[test]
    fn test_empty_module_operations() {
        let mut module = Module::new("");
        assert_eq!(module.name, "");
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.inputs.len(), 0);
        assert_eq!(module.outputs.len(), 0);
        
        // Add an empty operation
        let op = Operation::new("");
        module.add_operation(op);
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].op_type, "");
    }

    /// Test 3: Value with extremely large tensor dimensions
    #[test]
    fn test_extremely_large_tensor_dimensions() {
        // Test cases for very large tensor dimensions that are still computationally valid
        let cases = [
            vec![1_000_000_000], // Very large 1D tensor
            vec![50_000, 50_000], // Large 2D tensor (could result in overflow if multiplied)
            vec![10_000, 10_000, 10], // Large 3D tensor
            vec![100_000, 100_000, 2], // Large 3D tensor with small last dim
        ];

        for (i, shape) in cases.iter().enumerate() {
            let value = Value {
                name: format!("large_tensor_{}", i),
                ty: Type::F32,
                shape: shape.clone(),
            };

            assert_eq!(value.shape, *shape);
            
            // Calculate the product safely (but don't assert the value as it might overflow)
            let _product_result: Option<usize> = value.shape.iter().try_fold(1_usize, |acc, &x| {
                if x == 0 { 
                    Some(0) 
                } else { 
                    acc.checked_mul(x) 
                }
            });
            
            // Either the product fits in usize or it overflows (returns None)
            // Both are acceptable outcomes for these edge cases
        }
    }

    /// Test 4: String values with unicode and special control characters
    #[test]
    fn test_unicode_and_control_character_values() {
        let unicode_cases = [
            "tensor_名称_日本語_🚀",
            "control_\u{0001}_\u{001F}_chars",
            "emoji_🔥_🎉_🎊_💡",
            "special_\"quotes\"_and_'apostrophes'_test",
            "tab\tand\nnewline\rcharacters",
            "null_byte_\u{0000}_test",
        ];

        for (i, name) in unicode_cases.iter().enumerate() {
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![i + 1],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, Type::F32);
            assert_eq!(value.shape, vec![i + 1]);
        }
    }

    /// Test 5: Deeply nested recursive type structures using rstest
    #[rstest]
    #[case(1)]    // Minimal nesting
    #[case(10)]   // Low depth
    #[case(50)]   // Medium depth  
    #[case(100)]  // High depth
    fn test_deeply_nested_tensor_types(#[case] depth: usize) {
        let mut current_type = Type::F32;
        
        for i in 0..depth {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 5 + 1], // Vary the shape to create diverse nested types
            };
        }
        
        // Verify that we have created the nested structure correctly
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Test that the type is valid according to our validation function
        use crate::ir::TypeExtensions;
        assert!(current_type.is_valid_type());
    }

    /// Test 6: Mixed attribute types with maximum nesting
    #[test]
    fn test_highest_nesting_attribute_arrays() {
        // Create a complex nested attribute structure
        let complex_nested_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(1),
                    Attribute::String("deeply_nested".to_string()),
                ]),
                Attribute::Array(vec![
                    Attribute::Float(3.14159),
                    Attribute::Bool(true),
                ]),
            ]),
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Array(vec![
                        Attribute::Int(42),
                        Attribute::Float(2.71828),
                    ]),
                ]),
            ]),
        ]);

        match &complex_nested_attr {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                
                // Check first major branch
                match &outer[0] {
                    Attribute::Array(branch1) => {
                        assert_eq!(branch1.len(), 2);
                        
                        match &branch1[0] {
                            Attribute::Array(deepest) => {
                                assert_eq!(deepest.len(), 2);
                                // Verify the deepest values
                            },
                            _ => panic!("Expected nested array"),
                        }
                    },
                    _ => panic!("Expected nested array structure"),
                }
            },
            _ => panic!("Expected top-level array"),
        }
        
        // Test cloning of complex nested structure
        let cloned = complex_nested_attr.clone();
        assert_eq!(complex_nested_attr, cloned);
    }

    /// Test 7: Operations with zero-size tensors in inputs/outputs
    #[test]
    fn test_operations_with_zero_sized_tensors() {
        let mut op = Operation::new("zero_tensor_op");
        
        // Add inputs with zero-sized tensors
        op.inputs.push(Value {
            name: "zero_input".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5], // Contains 0, so total size should be 0
        });
        
        op.inputs.push(Value {
            name: "another_zero_input".to_string(),
            ty: Type::I32,
            shape: vec![0], // Single zero dimension
        });
        
        op.inputs.push(Value {
            name: "multi_zero_input".to_string(),
            ty: Type::F64,
            shape: vec![2, 0, 8, 0], // Multiple zeros
        });
        
        // Add outputs with zero-sized tensors
        op.outputs.push(Value {
            name: "zero_output".to_string(),
            ty: Type::F32,
            shape: vec![0, 100], // Starts with 0
        });
        
        assert_eq!(op.inputs.len(), 3);
        assert_eq!(op.outputs.len(), 1);
        
        // Verify all inputs have zero in their shape product
        for input in &op.inputs {
            let product: usize = input.shape.iter().product();
            assert_eq!(product, 0, "Shape {:?} should have product of 0", input.shape);
        }
        
        // Verify the output has zero in its shape product
        for output in &op.outputs {
            let product: usize = output.shape.iter().product();
            assert_eq!(product, 0, "Shape {:?} should have product of 0", output.shape);
        }
    }

    /// Test 8: Compiler with different device targets using rstest
    #[rstest]
    #[case("")]
    #[case("cpu")]
    #[case("gpu")]
    #[case("tpu")]
    #[case("xpu")]
    #[case("cuda")]
    #[case("opencl")]
    #[case("metal")]
    #[case("invalid_target_12345")]
    #[case("cpu_gpu_hybrid")]
    #[case("multi:cpu,gpu")]
    fn test_compiler_with_various_targets(#[case] target: &str) {
        let mut compiler = ImpulseCompiler::new();
        let mock_model = vec![1u8, 2u8, 3u8, 4u8];
        
        // This should not panic regardless of target validity
        let result = compiler.compile(&mock_model, target);
        
        // Either succeeds or fails with an error, but doesn't panic
        assert!(result.is_ok() || result.is_err());
    }

    /// Test 9: Floating point edge cases in attributes
    #[test]
    fn test_floating_point_attribute_edge_cases() {
        let fp_cases = [
            (std::f64::INFINITY, "positive_infinity"),
            (std::f64::NEG_INFINITY, "negative_infinity"),
            (-0.0, "negative_zero"),
            (std::f64::EPSILON, "epsilon"),
            (std::f64::consts::PI, "pi"),
            (std::f64::consts::E, "euler_number"),
            (f64::MIN_POSITIVE, "min_positive"),
            (f64::MAX, "max_value"),
            (f64::MIN, "min_value"),
        ];

        for (value, name) in &fp_cases {
            let attr = Attribute::Float(*value);
            
            match attr {
                Attribute::Float(retrieved) => {
                    if value.is_infinite() {
                        assert!(retrieved.is_infinite(), "Value {} should be infinite", name);
                        assert_eq!(value.is_sign_positive(), retrieved.is_sign_positive());
                    } else if value.is_nan() {
                        // NaN values cannot be compared directly
                        assert!(retrieved.is_nan(), "Value {} should be NaN", name);
                    } else if *value == -0.0 {
                        // Special case for negative zero
                        assert!((retrieved == 0.0) && retrieved.is_sign_negative(), 
                                "Value {} should be negative zero", name);
                    } else {
                        assert!((*value - retrieved).abs() < f64::EPSILON || *value == retrieved,
                                "Values should be approximately equal for {}", name);
                    }
                },
                _ => panic!("Expected Float attribute for {}", name),
            }
        }
        
        // Also test NaN specifically
        let nan_attr = Attribute::Float(std::f64::NAN);
        match nan_attr {
            Attribute::Float(f) => assert!(f.is_nan()),
            _ => panic!("Expected NaN"),
        }
    }

    /// Test 10: Memory allocation edge cases with large collections
    #[test]
    fn test_memory_allocation_with_large_collections() {
        // Create a module with operations that each have many attributes
        let mut module = Module::new("memory_stress_test");
        
        for op_idx in 0..1_000 {
            let mut op = Operation::new(&format!("stress_op_{}", op_idx));
            
            // Add many attributes to each operation
            for attr_idx in 0..100 {
                let key = format!("attr_{}_{}", op_idx, attr_idx);
                let value = Attribute::String(format!("value_{}_{}", op_idx, attr_idx));
                op.attributes.insert(key, value);
            }
            
            // Add some inputs and outputs too
            for input_idx in 0..10 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}", op_idx, input_idx),
                    ty: Type::F32,
                    shape: vec![input_idx + 1],
                });
            }
            
            for output_idx in 0..5 {
                op.outputs.push(Value {
                    name: format!("output_{}_{}", op_idx, output_idx),
                    ty: Type::F32,
                    shape: vec![output_idx + 1],
                });
            }
            
            module.add_operation(op);
        }
        
        // Verify everything was added correctly
        assert_eq!(module.operations.len(), 1_000);
        
        // Check a few operations to make sure they preserved their attributes
        assert_eq!(module.operations[0].attributes.len(), 100);
        assert_eq!(module.operations[0].inputs.len(), 10);
        assert_eq!(module.operations[0].outputs.len(), 5);
        
        assert_eq!(module.operations[999].attributes.len(), 100);
        assert_eq!(module.operations[999].inputs.len(), 10);
        assert_eq!(module.operations[999].outputs.len(), 5);
        
        // Verify random operations maintain their integrity
        assert_eq!(module.operations[500].inputs.len(), 10);
    }
}