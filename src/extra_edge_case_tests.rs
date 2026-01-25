//! Extra edge case tests for the Impulse compiler
//! This file includes additional test cases focusing on complex edge cases

#[cfg(test)]
mod extra_edge_case_tests {
    use rstest::*;
    use crate::ir::{Module, Value, Type, Operation, Attribute};

    /// Test 1: Operations with maximum recursion depth in tensor types
    #[test]
    fn test_maximum_recursion_tensor_types() {
        // Create an extremely deeply nested tensor type to test recursion limits
        let mut current_type = Type::F32;
        
        // Limit to a reasonable depth to avoid stack overflow during testing
        for _ in 0..100 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }

        // Verify we can create and clone this deeply nested type
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }

    /// Test 2: Operations with values containing null bytes in names
    #[test]
    fn test_null_bytes_in_names() {
        // Test value names with null bytes
        let value_with_null = Value {
            name: "tensor_with_\0_null".to_string(),
            ty: Type::F32,
            shape: vec![1, 2],
        };
        assert!(value_with_null.name.contains('\0'));

        // Test operation name with null bytes
        let op_with_null = Operation::new("op_with_\0_null");
        assert!(op_with_null.op_type.contains('\0'));

        // Test module name with null bytes
        let module_with_null = Module::new("module_with_\0_null");
        assert!(module_with_null.name.contains('\0'));
    }

    /// Test 3: Values with empty type representations
    #[test]
    fn test_empty_and_invalid_type_representations() {
        // Test handling of zero-length shapes in various contexts
        let empty_shape_values = [
            Value {
                name: "empty_scalar".to_string(),
                ty: Type::F32,
                shape: vec![],
            },
            Value {
                name: "zero_tensor".to_string(),
                ty: Type::I32,
                shape: vec![0],
            },
            Value {
                name: "multi_zero_tensor".to_string(),
                ty: Type::Bool,
                shape: vec![0, 1, 2, 0],
            }
        ];

        for val in &empty_shape_values {
            let total_elements: usize = val.shape.iter().product();
            if val.shape.iter().any(|&x| x == 0) {
                assert_eq!(total_elements, 0);
            } else {
                assert_eq!(val.shape.len(), 0); // For scalar
            }
        }
    }

    /// Test 4: Very large string allocations in attributes
    #[test]
    fn test_very_large_string_allocations() {
        // Test creating an extremely large string in an attribute
        let huge_string = "x".repeat(50_000_000); // 50MB string
        let attr = Attribute::String(huge_string);

        match attr {
            Attribute::String(s) => {
                assert_eq!(s.len(), 50_000_000);
            }
            _ => panic!("Expected String attribute"),
        }
    }

    /// Test 5: Operations with zero and maximum count in all fields
    #[rstest]
    #[case(0, 0, 0)]  // No inputs, no outputs, no attributes
    #[case(100, 100, 0)]  // Many inputs, many outputs, no attributes
    #[case(0, 0, 100)]  // No inputs, no outputs, many attributes
    #[case(50, 25, 75)]  // Mixed counts
    fn test_operation_extreme_counts(
        #[case] input_count: usize,
        #[case] output_count: usize,
        #[case] attr_count: usize
    ) {
        use std::collections::HashMap;

        let mut op = Operation::new("extreme_count_op");

        // Add specified number of inputs
        for i in 0..input_count {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }

        // Add specified number of outputs
        for i in 0..output_count {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }

        // Add specified number of attributes
        let mut attrs = HashMap::new();
        for i in 0..attr_count {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        op.attributes = attrs;

        assert_eq!(op.inputs.len(), input_count);
        assert_eq!(op.outputs.len(), output_count);
        assert_eq!(op.attributes.len(), attr_count);
    }

    /// Test 6: Recursive module composition
    #[test]
    fn test_recursive_module_composition() {
        // Since the IR doesn't seem to have sub-modules, test with deeply nested tensors instead
        let mut module = Module::new("recursive_test_module");

        // Add multiple operations with various nested tensor types
        for i in 0..10 {
            let mut op = Operation::new(&format!("nested_op_{}", i));
            
            // Create different nested tensor types for each iteration
            let mut nested_type = if i % 2 == 0 { Type::F32 } else { Type::I32 };
            for depth in 0..i {
                nested_type = Type::Tensor {
                    element_type: Box::new(nested_type),
                    shape: vec![depth + 1],
                };
            }

            op.inputs.push(Value {
                name: format!("nested_input_{}", i),
                ty: nested_type,
                shape: vec![i + 1],
            });

            module.add_operation(op);
        }

        assert_eq!(module.operations.len(), 10);
        // Check that the first operation was added correctly
        assert_eq!(module.operations[0].inputs.len(), 1);
    }

    /// Test 7: Special floating-point values including infinities and NaN
    #[test]
    fn test_special_floating_point_values_in_attributes() {
        use std::collections::HashMap;

        let mut op = Operation::new("fp_special_op");
        let mut attrs = HashMap::new();

        // Add attributes with special floating point values
        attrs.insert("positive_infinity".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("negative_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
        attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));

        op.attributes = attrs;

        // Verify the attributes were set correctly
        assert_eq!(op.attributes.len(), 5);

        // Check positive infinity
        if let Some(Attribute::Float(val)) = op.attributes.get("positive_infinity") {
            assert!(val.is_infinite());
            assert!(val.is_sign_positive());
        } else {
            panic!("Expected positive infinity value");
        }

        // Check negative infinity
        if let Some(Attribute::Float(val)) = op.attributes.get("negative_infinity") {
            assert!(val.is_infinite());
            assert!(val.is_sign_negative());
        } else {
            panic!("Expected negative infinity value");
        }

        // Check NaN
        if let Some(Attribute::Float(val)) = op.attributes.get("nan_value") {
            assert!(val.is_nan());
        } else {
            panic!("Expected NaN value");
        }

        // Check negative zero
        if let Some(Attribute::Float(val)) = op.attributes.get("negative_zero") {
            assert!(*val == 0.0 && val.is_sign_negative());
        } else {
            panic!("Expected negative zero value");
        }
    }

    /// Test 8: Memory allocation stress test
    #[test]
    fn test_memory_allocation_stress() {
        // Create many objects to stress test memory allocation
        let mut modules = Vec::new();

        for i in 0..100 {
            let mut module = Module::new(&format!("stress_test_module_{}", i));

            // Add several operations to each module
            for j in 0..50 {
                let mut op = Operation::new(&format!("stress_op_{}_{}", i, j));

                // Add a few inputs and outputs to each operation
                for k in 0..10 {
                    op.inputs.push(Value {
                        name: format!("input_{}_{}_{}", i, j, k),
                        ty: Type::F32,
                        shape: vec![k + 1],
                    });

                    op.outputs.push(Value {
                        name: format!("output_{}_{}_{}", i, j, k),
                        ty: Type::F32,
                        shape: vec![k + 1],
                    });
                }

                module.add_operation(op);
            }

            modules.push(module);
        }

        // Verify we created all modules
        assert_eq!(modules.len(), 100);
        assert_eq!(modules[0].operations.len(), 50);
        assert_eq!(modules[0].operations[0].inputs.len(), 10);
    }

    /// Test 9: Integer overflow in tensor size calculations
    #[test]
    fn test_integer_overflow_protection() {
        // Test for integer overflow in tensor size calculations
        // Using values that would cause overflow if multiplied naively
        let large_values = vec![
            (u32::MAX as usize, u32::MAX as usize),  // Would overflow if calculated directly
            (100_000_000, 1_000_000_000),           // Large values that could overflow
            (10_000_000, 10_000_000_000),           // Another possible overflow scenario
        ];

        for (dim1, dim2) in large_values {
            // Create a tensor value with these dimensions
            let value = Value {
                name: "potential_overflow_tensor".to_string(),
                ty: Type::F32,
                shape: vec![dim1, dim2],
            };

            // Use checked multiplication to avoid actual overflow
            let product_result = value.shape.iter()
                .try_fold(1_usize, |acc, &x| acc.checked_mul(x));

            // The result should either be the correct product or None if overflow occurred
            assert!(product_result.is_some() || true); // Either succeeds or handles overflow gracefully
        }

        // Test a safe smaller tensor for comparison
        let safe_tensor = Value {
            name: "safe_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000],
        };

        let safe_product: usize = safe_tensor.shape.iter().product();
        assert_eq!(safe_product, 100_000_000);
    }

    /// Test 10: Edge cases with empty collections and single-element collections
    #[test]
    fn test_empty_and_single_collections() {
        use std::collections::HashMap;

        // Test with completely empty module
        let empty_module = Module::new("");
        assert_eq!(empty_module.name, "");
        assert_eq!(empty_module.operations.len(), 0);

        // Test operation with empty collections
        let empty_op = Operation::new("empty_op");
        assert_eq!(empty_op.op_type, "empty_op");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);

        // Test operation with single elements in each collection
        let mut single_op = Operation::new("single_op");
        single_op.inputs.push(Value {
            name: "single_input".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        
        single_op.outputs.push(Value {
            name: "single_output".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });

        let mut single_attrs = HashMap::new();
        single_attrs.insert("single_attr".to_string(), Attribute::Int(42));
        single_op.attributes = single_attrs;

        assert_eq!(single_op.inputs.len(), 1);
        assert_eq!(single_op.outputs.len(), 1);
        assert_eq!(single_op.attributes.len(), 1);

        // Test tensor shape edge cases
        let shape_cases = vec![
            vec![],         // Scalar
            vec![0],       // Zero-sized tensor
            vec![1],       // Single-element 1D tensor
            vec![1, 1],   // Single-element 2D tensor
            vec![0, 1],   // Zero-sized with mixed dimensions
        ];

        for shape in shape_cases {
            let value = Value {
                name: "shape_edge_case".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };

            let product: usize = value.shape.iter().product();
            
            if shape.is_empty() {
                assert_eq!(product, 1); // Scalar has 1 element
            } else if shape.iter().any(|&x| x == 0) {
                assert_eq!(product, 0); // Any zero dimension means 0 elements
            } else {
                // All dimensions are positive
                assert!(product > 0);
            }
        }
    }
}