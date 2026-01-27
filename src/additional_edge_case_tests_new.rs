//! Additional edge case tests for the Impulse compiler
//! Focus on critical edge cases that might affect system stability

#[cfg(test)]
mod additional_edge_case_tests_new {
    use crate::ir::{Value, Type, Operation, Attribute};
    use rstest::rstest;
    use std::collections::HashMap;

    /// Test 1: Very large tensor dimension that could cause overflow
    #[test]
    fn test_potential_overflow_in_num_elements() {
        // Use checked arithmetic to prevent overflow
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],  // This would overflow if multiplied naively
        };
        
        // The num_elements method uses checked arithmetic and should return None or a correct value
        let result = large_value.num_elements();
        assert!(result.is_some()); // Should handle large dimensions gracefully
        assert_eq!(result.unwrap(), 10_000_000_000);  // 10 billion
    }

    /// Test 2: Empty string as operation type
    #[test]
    fn test_empty_operation_type() {
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    /// Test 3: Value with maximum possible name length
    #[test]
    fn test_maximum_name_length_for_value() {
        // Create a string with 1 million 'a's
        let long_name = "a".repeat(1_000_000);
        let value = Value {
            name: long_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name.len(), 1_000_000);
        assert_eq!(value.name, long_name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1]);
    }

    /// Test 4: Recursive type definitions that could cause infinite loops
    #[test]
    fn test_deeply_nested_recursive_types() {
        // Create a 20-level nested tensor type
        let mut current_type = Type::Bool;
        
        for _ in 0..20 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2, 2], // Small shape to keep memory usage reasonable
            };
        }
        
        // Verify we can clone the deeply nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
        
        // Verify the structure is correct (final type should be a tensor)
        match &current_type {
            Type::Tensor { element_type: _, shape } => {
                assert_eq!(shape, &vec![2, 2]);
            },
            _ => panic!("Expected final type to be a tensor"),
        }
    }

    /// Test 5: Testing float operations with special values (NaN, Infinity)
    #[rstest]
    #[case(f64::NAN)]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    fn test_special_float_values_in_attributes(#[case] special_value: f64) {
        let attr = Attribute::Float(special_value);
        
        match attr {
            Attribute::Float(val) => {
                if special_value.is_nan() {
                    assert!(val.is_nan());
                } else {
                    assert_eq!(val, special_value);
                }
            },
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 6: Module with extremely large number of attributes
    #[test]
    fn test_operation_with_extremely_large_number_of_attributes() {
        let mut op = Operation::new("high_attr_op");
        let mut attrs = HashMap::new();
        
        // Add 100,000 attributes to test memory handling
        for i in 0..100_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::Int(i as i64)
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 100_000);
        assert_eq!(op.op_type, "high_attr_op");
        
        // Verify a few specific attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_50000"));
        assert!(op.attributes.contains_key("attr_99999"));
        
        // Verify values are correct
        if let Some(Attribute::Int(val)) = op.attributes.get("attr_0") {
            assert_eq!(*val, 0);
        } else {
            panic!("attr_0 should be Int(0)");
        }
        
        if let Some(Attribute::Int(val)) = op.attributes.get("attr_50000") {
            assert_eq!(*val, 50000);
        } else {
            panic!("attr_50000 should be Int(50000)");
        }
    }

    /// Test 7: Value with empty shape vs value with single zero dimension
    #[test]
    fn test_scalar_vs_zero_sized_differences() {
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar, has 1 element
        };
        
        let zero_sized = Value {
            name: "zero_sized".to_string(),
            ty: Type::F32,
            shape: vec![0],  // Shape with zero = zero-sized tensor, has 0 elements
        };
        
        // Verify shapes are different
        assert_ne!(scalar.shape, zero_sized.shape);
        assert_eq!(scalar.shape.len(), 0);
        assert_eq!(zero_sized.shape.len(), 1);
        assert_eq!(zero_sized.shape[0], 0);
        
        // Test element counts
        let scalar_elements = scalar.num_elements().unwrap_or(0);
        let zero_sized_elements = zero_sized.num_elements().unwrap_or(0);
        
        assert_eq!(scalar_elements, 1);      // Scalar has 1 element
        assert_eq!(zero_sized_elements, 0); // Zero-sized tensor has 0 elements
    }

    /// Test 8: Unicode handling in names and attribute values
    #[test]
    fn test_unicode_in_names_and_attributes() {
        let test_name = "tensor_🚀_测试_тест";
        let test_string = "Hello 🌍 世界 Мир";
        
        // Test operation with unicode name
        let mut op = Operation::new(test_name);
        
        // Add unicode attribute
        op.attributes.insert(
            "unicode_attr".to_string(),
            Attribute::String(test_string.to_string())
        );
        
        // Add a value with unicode name
        op.inputs.push(Value {
            name: test_name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        
        assert_eq!(op.op_type, test_name);
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.inputs[0].name, test_name);
        assert_eq!(op.attributes.len(), 1);
        
        if let Some(Attribute::String(attr_val)) = op.attributes.get("unicode_attr") {
            assert_eq!(attr_val, test_string);
        } else {
            panic!("Unicode attribute not found or wrong type");
        }
    }

    /// Test 9: Test array attributes with maximum nesting depth
    #[test]
    fn test_deeply_nested_array_attributes() {
        // Create nested arrays 15 levels deep to test recursion limits
        let mut nested_attr = Attribute::Int(123);
        
        for i in 0..15 {
            nested_attr = Attribute::Array(vec![nested_attr]);
            // Add another element at selected levels to make it more interesting
            if i % 3 == 0 {
                if let Attribute::Array(mut arr) = nested_attr {
                    arr.push(Attribute::String(format!("level_{}", i)));
                    nested_attr = Attribute::Array(arr);
                }
            }
        }
        
        // Verify the final structure
        match &nested_attr {
            Attribute::Array(arr) => {
                assert!(!arr.is_empty());
                // Should have at least 1 element, probably 2 at levels divisible by 3
            },
            _ => panic!("Expected nested array structure"),
        }
        
        // Test cloning of deeply nested structure
        let cloned = nested_attr.clone();
        assert_eq!(nested_attr, cloned);
    }

    /// Test 10: Edge cases in tensor type validation
    #[test]
    fn test_tensor_type_validation_edge_cases() {
        use crate::ir::TypeExtensions;
        
        // Test that basic types are valid
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());
        
        // Test nested valid tensors
        let valid_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            }),
            shape: vec![3, 3],
        };
        assert!(valid_nested.is_valid_type());
        
        // Test deeply nested valid tensor
        let mut current_type = Type::Bool;
        for _ in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1, 1],
            };
        }
        assert!(current_type.is_valid_type());
        
        // Test cloning of validated type
        let cloned_validated = current_type.clone();
        assert_eq!(current_type, cloned_validated);
        assert!(cloned_validated.is_valid_type());
    }
}