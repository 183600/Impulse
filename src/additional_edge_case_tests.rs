#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;
    use rstest::*;

    /// Test 1: Zero-sized tensors and their behavior
    #[test]
    fn test_zerosized_tensors_behavior() {
        // Test tensor with zero in dimensions
        let value_with_zero = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5],  // Contains zero
        };
        
        assert_eq!(value_with_zero.shape, vec![10, 0, 5]);
        
        // Total elements should be 0 when any dimension is 0
        let total_elements: usize = value_with_zero.shape.iter().product();
        assert_eq!(total_elements, 0);
        
        // Also test using the num_elements method if available
        match value_with_zero.num_elements() {
            Some(elements) => assert_eq!(elements, 0),
            None => panic!("num_elements should not be None for this shape"),
        }
        
        // Test with multiple zeros
        let multi_zero_value = Value {
            name: "multi_zero_tensor".to_string(),
            ty: Type::I32,
            shape: vec![0, 0, 0],
        };
        
        assert_eq!(multi_zero_value.shape, vec![0, 0, 0]);
        let multi_zero_elements: usize = multi_zero_value.shape.iter().product();
        assert_eq!(multi_zero_elements, 0);
    }

    /// Test 2: Deeply nested tensor types
    #[test]
    fn test_deeply_nested_tensor_types() {
        // Create a deeply nested tensor type to test recursion limits
        let mut current_type = Type::F32;
        
        // Use a moderate depth to avoid stack overflow during testing
        for _ in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify final type is still valid
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a nested tensor type"),
        }
        
        // Test that we can clone this deeply nested type
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }

    /// Test 3: Integer overflow in tensor size calculations
    #[test]
    fn test_tensor_size_overflow_protection() {
        // Test with large, but reasonable values
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000],  // 100M elements
        };
        
        // This should not overflow on modern systems
        let product: usize = large_value.shape.iter().product();
        assert_eq!(product, 100_000_000);
        
        // Test with safe calculation for potential overflow scenarios
        let safe_calculation = large_value.shape.iter()
            .try_fold(1usize, |acc, &x| acc.checked_mul(x));
        
        assert_eq!(safe_calculation, Some(100_000_000));
    }

    /// Test 4: Large collection handling
    #[test]
    fn test_large_collection_handling() {
        const LARGE_SIZE: usize = 10_000;
        
        // Create a module with many operations to test memory management
        let mut module = Module::new("large_collection_test");
        
        for i in 0..LARGE_SIZE {
            let op = Operation::new(&format!("operation_{}", i));
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), LARGE_SIZE);
        assert_eq!(module.name, "large_collection_test");
        
        // Test with an operation that has many attributes
        let mut large_op = Operation::new("many_attrs_op");
        let mut attrs = HashMap::new();
        
        for i in 0..5_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        large_op.attributes = attrs;
        assert_eq!(large_op.attributes.len(), 5_000);
    }

    /// Test 5: String attributes with special characters
    #[rstest]
    #[case("")]
    #[case("normal_string")]
    #[case("unicode_text_🚀_国际化")]
    #[case("special_chars_!@#$%^&*()")]
    #[case("\n\t\r\x00")]  // Control characters
    fn test_string_attributes_special_chars(#[case] input_str: &str) {
        let attr = Attribute::String(input_str.to_string());
        
        match attr {
            Attribute::String(retrieved_str) => {
                assert_eq!(retrieved_str, input_str);
            },
            _ => panic!("Expected String attribute"),
        }
    }

    /// Test 6: Floating point edge cases
    #[test]
    fn test_floating_point_edge_cases() {
        let test_cases = vec![
            (f64::INFINITY, "positive_infinity"),
            (f64::NEG_INFINITY, "negative_infinity"),
            (f64::NAN, "nan"),
            (0.0, "positive_zero"),
            (-0.0, "negative_zero"),
            (f64::EPSILON, "epsilon"),
        ];

        for (value, desc) in test_cases {
            let attr = Attribute::Float(value);
            
            match attr {
                Attribute::Float(retrieved_value) => {
                    if value.is_nan() {
                        assert!(retrieved_value.is_nan(), "Value {} should be NaN", desc);
                    } else {
                        // For infinities
                        if value.is_infinite() && retrieved_value.is_infinite() {
                            assert_eq!(value.is_sign_positive(), retrieved_value.is_sign_positive());
                        } else {
                            // For finite values
                            assert!((retrieved_value - value).abs() < f64::EPSILON, 
                                   "Values should match for {}", desc);
                        }
                    }
                },
                _ => panic!("Expected Float attribute for {}", desc),
            }
        }
    }

    /// Test 7: Deep recursion in nested types
    #[test]
    fn test_deep_recursion_nested_types() {
        // Create a deeply nested tensor type
        let mut current_type = Type::Bool;
        const DEPTH: usize = 25;

        for i in 0..DEPTH {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 5 + 1], // Vary shape to make it more interesting
            };
        }

        // Verify we can access the nested type without stack overflow
        let final_type = current_type;
        
        // Ensure it matches expected structure
        // The last iteration is i=24, so shape should be [24 % 5 + 1] = [4 + 1] = [5]
        match &final_type {
            Type::Tensor { shape, .. } => {
                // In the last iteration, i would be DEPTH-1 (24)
                // So the shape would be [(DEPTH-1) % 5 + 1]
                let expected_last_shape = ((DEPTH - 1) % 5) + 1;
                assert_eq!(shape, &vec![expected_last_shape]);
            },
            _ => panic!("Expected a tensor type after nesting"),
        }
        
        // Ensure we can clone it
        let cloned = final_type.clone();
        assert_eq!(final_type, cloned);
    }

    /// Test 8: Comprehensive operation validation with all field combinations
    #[test]
    fn test_comprehensive_operation_validation() {
        let mut op = Operation::new("comprehensive_test");
        
        // Add multiple inputs with different types
        for i in 0..10 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: if i % 4 == 0 { 
                    Type::F32 
                } else if i % 4 == 1 {
                    Type::F64
                } else if i % 4 == 2 {
                    Type::I32
                } else {
                    Type::I64
                },
                shape: vec![i + 1, i + 2],
            });
        }
        
        // Add multiple outputs
        for i in 0..5 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I64 },
                shape: vec![i + 1, i + 1],
            });
        }
        
        // Add various attributes
        let mut attrs = HashMap::new();
        attrs.insert("int_param".to_string(), Attribute::Int(42));
        attrs.insert("float_param".to_string(), Attribute::Float(3.14159));
        attrs.insert("string_param".to_string(), Attribute::String("test".to_string()));
        attrs.insert("bool_param".to_string(), Attribute::Bool(true));
        attrs.insert("array_param".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("nested".to_string()),
        ]));
        op.attributes = attrs;
        
        // Validate all aspects
        assert_eq!(op.op_type, "comprehensive_test");
        assert_eq!(op.inputs.len(), 10);
        assert_eq!(op.outputs.len(), 5);
        assert_eq!(op.attributes.len(), 5);
        
        // Check specific elements
        assert_eq!(op.inputs[0].ty, Type::F32);
        assert_eq!(op.inputs[1].ty, Type::F64);
        assert_eq!(op.inputs[2].ty, Type::I32);
        assert_eq!(op.inputs[3].ty, Type::I64);
        
        assert_eq!(op.outputs[0].ty, Type::F32);
        assert_eq!(op.outputs[1].ty, Type::I64);
        
        assert_eq!(op.attributes.get("int_param"), Some(&Attribute::Int(42)));
        assert_eq!(op.attributes.get("float_param"), Some(&Attribute::Float(3.14159)));
    }

    /// Test 9: Module with extreme number of operations
    #[test]
    fn test_module_extreme_operations_count() {
        let mut module = Module::new("extreme_ops_module");
        
        // Add a very large number of operations
        const NUM_OPS: usize = 50_000;
        
        for i in 0..NUM_OPS {
            let op = Operation::new(&format!("op_{}", i));
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), NUM_OPS);
        assert_eq!(module.name, "extreme_ops_module");
        
        // Verify some operations still have correct data
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[NUM_OPS/2].op_type, format!("op_{}", NUM_OPS/2));
        assert_eq!(module.operations[NUM_OPS-1].op_type, format!("op_{}", NUM_OPS-1));
    }

    /// Test 10: Edge cases for tensor type validation
    #[test]
    fn test_tensor_type_validation_edge_cases() {
        use crate::ir::TypeExtensions;

        // Basic types should be valid
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());

        // Valid nested types
        let nested_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 4],
        };
        assert!(nested_tensor.is_valid_type());

        // Deeply nested valid type
        let mut current_type = Type::I32;
        for _ in 0..10 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        assert!(current_type.is_valid_type());
    }
}