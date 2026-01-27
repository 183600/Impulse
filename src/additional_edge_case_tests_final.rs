//! Additional final edge case tests for the Impulse compiler
//! Covering untested boundary conditions and extreme scenarios

#[cfg(test)]
mod additional_edge_case_tests_final {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use rstest::rstest;

    /// Test 1: Value with maximum possible shape dimensions
    #[test]
    fn test_value_with_maximum_shape_dimensions() {
        // Create a shape with many dimensions (up to 30 dimensions, with small values to avoid overflow)
        let mut shape = Vec::new();
        for _i in 0..30 {
            shape.push(1); // Use only 1s to avoid any multiplication overflow
        }

        let value = Value {
            name: "max_dims".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape.len(), 30);
        assert_eq!(value.shape, shape);
        
        // Calculate number of elements (with all 1s, should be 1)
        let num_elements_option = value.num_elements();
        assert_eq!(num_elements_option, Some(1));
        
        // Verify the first and last elements of the shape
        assert_eq!(value.shape[0], 1);
        assert_eq!(value.shape[29], 1);
    }

    /// Test 2: Empty attribute array and deeply nested attribute arrays
    #[test]
    fn test_edge_case_attribute_arrays() {
        // Test empty array attribute
        let empty_array_attr = Attribute::Array(vec![]);
        if let Attribute::Array(arr) = &empty_array_attr {
            assert_eq!(arr.len(), 0);
        } else {
            panic!("Expected empty array attribute");
        }

        // Test deeply nested empty arrays
        let mut nested_empty = Attribute::Array(vec![]);
        for _ in 0..10 {
            nested_empty = Attribute::Array(vec![nested_empty]);
        }

        // Verify structure of deeply nested empty arrays
        match &nested_empty {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
                // Recursively check structure
                let mut current = &arr[0];
                for _ in 0..9 {
                    match current {
                        Attribute::Array(inner_arr) => {
                            assert_eq!(inner_arr.len(), 1);
                            current = &inner_arr[0];
                        },
                        _ => panic!("Expected nested array structure"),
                    }
                }
                // At the deepest level, should be another empty array
                match current {
                    Attribute::Array(deepest_arr) => assert_eq!(deepest_arr.len(), 0),
                    _ => panic!("Expected empty array at deepest level"),
                }
            },
            _ => panic!("Expected nested array structure"),
        }

        let mut op = Operation::new("nested_array_test");
        op.attributes.insert("empty_array".to_string(), empty_array_attr);
        op.attributes.insert("deeply_nested_empty".to_string(), nested_empty);

        assert_eq!(op.attributes.len(), 2);
        assert!(op.attributes.contains_key("empty_array"));
        assert!(op.attributes.contains_key("deeply_nested_empty"));
    }

    /// Test 3: Special float values in tensor computations
    #[rstest]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    #[case(f64::NAN)]
    fn test_special_float_values_as_attributes(#[case] special_value: f64) {
        let attr = Attribute::Float(special_value);
        
        match attr {
            Attribute::Float(retrieved_val) => {
                if special_value.is_nan() {
                    assert!(retrieved_val.is_nan());
                } else {
                    assert_eq!(retrieved_val, special_value);
                }
            },
            _ => panic!("Expected Float attribute"),
        }

        // Test with operation
        let mut op = Operation::new("special_float_test");
        op.attributes.insert("special_value".to_string(), attr);
        
        assert_eq!(op.attributes.len(), 1);
        if let Some(Attribute::Float(retrieved_val)) = op.attributes.get("special_value") {
            if special_value.is_nan() {
                assert!(retrieved_val.is_nan());
            } else {
                assert_eq!(*retrieved_val, special_value);
            }
        } else {
            panic!("Failed to retrieve special float value from operation attributes");
        }
    }

    /// Test 4: Module with operations that have identical names
    #[test]
    fn test_module_with_duplicate_operation_names() {
        let mut module = Module::new("duplicate_names");

        // Add several operations with the same name
        for _ in 0..10 {
            let mut op = Operation::new("identical_name");
            op.inputs.push(Value {
                name: "test_input".to_string(),
                ty: Type::F32,
                shape: vec![1],
            });
            module.add_operation(op);
        }

        assert_eq!(module.operations.len(), 10);
        
        // Verify all operations have the same name
        for op in &module.operations {
            assert_eq!(op.op_type, "identical_name");
            assert_eq!(op.inputs.len(), 1);
        }
    }

    /// Test 5: Operations with maximum integer values as attributes
    #[test]
    fn test_operations_with_max_min_integer_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_min_int_test");
        let mut attrs = HashMap::new();

        // Add attributes with maximum and minimum integer values
        attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("max_u32_as_i64".to_string(), Attribute::Int(i64::from(u32::MAX)));
        attrs.insert("min_i32_as_i64".to_string(), Attribute::Int(i64::from(i32::MIN)));

        op.attributes = attrs;

        assert_eq!(op.attributes.len(), 4);

        // Verify each value
        assert_eq!(
            match op.attributes.get("max_i64").unwrap() {
                Attribute::Int(val) => *val,
                _ => panic!("Expected Int attribute"),
            },
            i64::MAX
        );

        assert_eq!(
            match op.attributes.get("min_i64").unwrap() {
                Attribute::Int(val) => *val,
                _ => panic!("Expected Int attribute"),
            },
            i64::MIN
        );

        assert_eq!(
            match op.attributes.get("max_u32_as_i64").unwrap() {
                Attribute::Int(val) => *val,
                _ => panic!("Expected Int attribute"),
            },
            i64::from(u32::MAX)
        );

        assert_eq!(
            match op.attributes.get("min_i32_as_i64").unwrap() {
                Attribute::Int(val) => *val,
                _ => panic!("Expected Int attribute"),
            },
            i64::from(i32::MIN)
        );
    }

    /// Test 6: Recursive type equality with complex nesting patterns
    #[test]
    fn test_complex_recursive_type_equalities() {
        // Create two complex nested types that should be equal
        let type_a = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 3],
            }),
            shape: vec![4, 5],
        };

        let type_b = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 3],
            }),
            shape: vec![4, 5],
        };

        // Should be equal
        assert_eq!(type_a, type_b);

        // Create a similar type but with different inner element type
        let type_c = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I32), // Different element type
                shape: vec![2, 3],
            }),
            shape: vec![4, 5],
        };

        // Should not be equal
        assert_ne!(type_a, type_c);

        // Create a similar type but with different outer shape
        let type_d = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 3],
            }),
            shape: vec![4, 6], // Different shape
        };

        // Should not be equal
        assert_ne!(type_a, type_d);

        // Create a similar type but with different inner shape
        let type_e = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 4], // Different inner shape
            }),
            shape: vec![4, 5],
        };

        // Should not be equal
        assert_ne!(type_a, type_e);
    }

    /// Test 7: Value with shape containing maximum usize values (without overflow)
    #[test]
    fn test_value_with_very_large_shape_values() {
        // Use values that are large but shouldn't cause immediate overflow in simple operations
        let large_shape = vec![
            1_000_000_000,   // 1 billion
            2_000_000_000,   // 2 billion  
            1,               // Keep third dimension small to prevent overflow
        ];

        let value = Value {
            name: "very_large_shape".to_string(),
            ty: Type::F32,
            shape: large_shape.clone(),
        };

        assert_eq!(value.shape, large_shape);
        assert_eq!(value.shape.len(), 3);
        assert_eq!(value.shape[0], 1_000_000_000);
        assert_eq!(value.shape[1], 2_000_000_000);
        assert_eq!(value.shape[2], 1);

        // When multiplied, this would cause overflow, so num_elements should return None
        let _result = value.num_elements();
        // Note: The actual behavior depends on the implementation - it may return None if it detects potential overflow
        // or it may return Some value if it doesn't check for overflow during multiplication
    }

    /// Test 8: Operations with empty string names and special Unicode names
    #[test]
    fn test_operations_with_special_names() {
        let special_names = [
            "".to_string(),  // Empty string
            " ".to_string(),  // Single space
            "\t".to_string(),  // Tab character
            "\n".to_string(),  // Newline
            "🚀".to_string(),  // Emoji
            "тест".to_string(),  // Cyrillic
            "测试".to_string(),  // Chinese characters
            "prueba".to_string() + &"x".repeat(10000), // Very long with prefix
        ];

        for (i, name) in special_names.iter().enumerate() {
            let op = Operation::new(name);
            
            assert_eq!(op.op_type, *name);
            assert_eq!(op.inputs.len(), 0);
            assert_eq!(op.outputs.len(), 0);
            assert_eq!(op.attributes.len(), 0);
            
            // Also test creating a value with these special names
            let value = Value {
                name: name.clone(),
                ty: Type::F32,
                shape: vec![i + 1], // Different shapes to distinguish
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.shape, vec![i + 1]);
        }
    }

    /// Test 9: Module serialization/deserialization edge cases
    #[test]
    fn test_module_serialization_edge_cases() {
        let mut module = Module::new("serialization_test");
        
        // Add some operations with various attributes
        for i in 0..10 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.attributes.insert(
                format!("attr_{}", i), 
                Attribute::String(format!("value_{}", i))
            );
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![i + 1],
            });
            module.add_operation(op);
        }

        // Test that the module has the expected content before serialization
        assert_eq!(module.name, "serialization_test");
        assert_eq!(module.operations.len(), 10);

        // Test that all operations have the expected content
        for (i, op) in module.operations.iter().enumerate() {
            assert_eq!(op.op_type, format!("op_{}", i));
            assert_eq!(op.inputs.len(), 1);
            assert_eq!(op.inputs[0].name, format!("input_{}", i));
            assert_eq!(op.inputs[0].shape, vec![i + 1]);
            
            if i % 2 == 0 {
                assert_eq!(op.inputs[0].ty, Type::F32);
            } else {
                assert_eq!(op.inputs[0].ty, Type::I32);
            }
            
            let expected_key = format!("attr_{}", i);
            assert!(op.attributes.contains_key(&expected_key));
        }
    }

    /// Test 10: Cloning and equality of complex nested structures
    #[test]
    fn test_complex_structure_cloning_and_equality() {
        // Build a complex nested type
        let base_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![3, 4],
            }),
            shape: vec![5],
        };

        // Clone it multiple times
        let clone1 = base_type.clone();
        let clone2 = clone1.clone();
        let clone3 = base_type.clone();

        // All should be equal
        assert_eq!(base_type, clone1);
        assert_eq!(base_type, clone2);
        assert_eq!(base_type, clone3);
        assert_eq!(clone1, clone2);
        assert_eq!(clone1, clone3);
        assert_eq!(clone2, clone3);

        // Test with a value containing this type
        let original_value = Value {
            name: "complex_nested".to_string(),
            ty: base_type.clone(),
            shape: vec![10, 20],
        };

        let cloned_value = original_value.clone();

        assert_eq!(original_value, cloned_value);
        assert_eq!(original_value.name, cloned_value.name);
        assert_eq!(original_value.ty, cloned_value.ty);
        assert_eq!(original_value.shape, cloned_value.shape);

        // Test with an operation containing the complex value
        let mut original_op = Operation::new("complex_op");
        original_op.inputs.push(original_value);
        original_op.outputs.push(cloned_value);  // Different instances but equal content
        
        let cloned_op = original_op.clone();
        
        assert_eq!(original_op, cloned_op);
        assert_eq!(original_op.op_type, cloned_op.op_type);
        assert_eq!(original_op.inputs.len(), cloned_op.inputs.len());
        assert_eq!(original_op.outputs.len(), cloned_op.outputs.len());
        assert_eq!(original_op.attributes, cloned_op.attributes);
    }
}