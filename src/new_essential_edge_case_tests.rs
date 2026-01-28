//! Essential edge case tests for the Impulse compiler
//! Covering crucial boundary conditions and potential failure points not covered in other test modules

#[cfg(test)]
mod new_essential_edge_case_tests {
    use crate::ir::{Value, Type, Operation, Attribute, Module};
    use std::collections::HashMap;

    /// Test 1: Testing tensor shapes with maximum usize values (potential overflow scenarios)
    #[test]
    fn test_tensor_shapes_with_large_usize_values() {
        // Test tensor shape that could cause overflow when calculating total elements
        // Use values that when multiplied together will exceed usize::MAX
        // In practice, multiplying by 2 will cause overflow if max_value is > usize::MAX/2
        let max_value = usize::MAX / 2 + 1;  // Use more than half of MAX to ensure overflow when multiplied by 2
        let large_shape = vec![max_value, 2];
        
        let value = Value {
            name: "large_usize_tensor".to_string(),
            ty: Type::F32,
            shape: large_shape,
        };
        
        // Verify the shape is preserved correctly
        assert_eq!(value.shape, vec![max_value, 2]);
        
        // Test that multiplication doesn't crash (it may overflow to 0 or panic depending on debug/release).
        // Using checked_mul to avoid panic in debug mode
        let result = value.num_elements();
        // Result should be None due to overflow
        assert!(result.is_none());
    }

    /// Test 2: Testing operations with empty string names and special unicode names
    #[test]
    fn test_operations_with_problematic_names() {
        // Test with empty string
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        
        // Test with unicode characters
        let unicode_op = Operation::new("μ_μαθημα_λ");
        assert_eq!(unicode_op.op_type, "μ_μαθημα_λ");
        
        // Test with emoji and special characters
        let emoji_op = Operation::new("op_🎉_🚀_✨");
        assert_eq!(emoji_op.op_type, "op_🎉_🚀_✨");
        
        // Test with control characters
        let control_op = Operation::new("control\x00\x01\x02op");
        assert_eq!(control_op.op_type, "control\x00\x01\x02op");
    }

    /// Test 3: Testing Value with invalid or problematic names
    #[test]
    fn test_values_with_problematic_names() {
        let problematic_names = [
            "",  // Empty
            "\x00",  // Null character
            "\n\r\t",  // Control characters
            "名字_姓名_名稱",  // Chinese characters
            "🚀",  // Emoji
            &"a".repeat(100_000),  // Extremely long string
        ];
        
        for name in &problematic_names {
            let value = Value {
                name: (*name).to_string(),  // Dereference the string slice
                ty: Type::F32,
                shape: vec![1],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, Type::F32);
            assert_eq!(value.shape, vec![1]);
        }
    }

    /// Test 4: Testing attribute maps with special key names
    #[test]
    fn test_operation_attributes_with_problematic_keys() {
        let mut op = Operation::new("test_op");
        let mut attrs = HashMap::new();
        
        // Add attributes with problematic keys
        attrs.insert("".to_string(), Attribute::Int(1));
        attrs.insert("\x00".to_string(), Attribute::Int(2));
        attrs.insert("key_🎉_with_emoji".to_string(), Attribute::Int(3));
        attrs.insert("κόσμε".to_string(), Attribute::Int(4));  // Greek characters
        attrs.insert("a".repeat(50_000), Attribute::Int(5));  // Very long key
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 5);
        assert!(op.attributes.contains_key(""));
        assert!(op.attributes.contains_key("\x00"));
        assert!(op.attributes.contains_key("key_🎉_with_emoji"));
        assert!(op.attributes.contains_key("κόσμε"));
    }

    /// Test 5: Testing deeply recursive clone operations that could cause stack overflow
    #[test]
    fn test_deeply_recursive_type_clone_operations() {
        // Create a moderately deep nested type to ensure clone works properly
        let mut current_type = Type::F32;
        
        // Create 500 levels of nesting, which should be safe but test recursion handling
        for i in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![(i % 10) + 1],  // Varying shapes
            };
        }
        
        // Test cloning of this deeply nested type - this is the critical test
        let cloned_type = current_type.clone();
        
        // Verify they're equal
        assert_eq!(current_type, cloned_type);
        
        // Verify the structure is preserved
        match &cloned_type {
            Type::Tensor { element_type: _, shape } => {
                assert_eq!(shape, &vec![(499 % 10) + 1]);  // Last iteration's shape
            },
            _ => panic!("Expected Tensor type after cloning"),
        }
    }

    /// Test 6: Testing values with shape containing very large numbers and zeros mixed
    #[test]
    fn test_tensor_shapes_with_zeros_and_large_numbers_mixed() {
        let test_cases = vec![
            (vec![1_000_000, 0, 500], 0),  // Contains zero, should result in 0 elements
            (vec![0, 1_000_000, 500], 0),  // Leading zero
            (vec![500, 1_000_000, 0], 0),  // Trailing zero
            (vec![1, 1, 1], 1),            // Small shape
            (vec![100_000, 100_000, 0], 0), // Large numbers with zero
        ];
        
        for (shape, expected_elements) in test_cases {
            let value = Value {
                name: "mixed_shape_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let actual_elements: usize = value.shape.iter().product();
            assert_eq!(actual_elements, expected_elements);
        }
    }

    /// Test 7: Testing modules with maximum capacity operations to test memory limits
    #[test]
    fn test_module_memory_limit_handling() {
        const NUM_OPS: usize = 100_000;  // Large but reasonable number
        
        let mut module = Module::new("memory_test_module");
        
        // Add a large number of operations to test memory handling
        for i in 0..NUM_OPS {
            let op = Operation::new(&format!("op_{:06}", i));
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), NUM_OPS);
        assert_eq!(module.name, "memory_test_module");
        
        // Access first and last operations to ensure they're intact
        assert_eq!(module.operations[0].op_type, "op_000000");
        assert_eq!(module.operations[NUM_OPS - 1].op_type, format!("op_{:06}", NUM_OPS - 1));
    }

    /// Test 8: Testing operations with maximum attribute count to test memory and performance
    #[test]
    fn test_operation_with_maximum_reasonable_attributes() {
        let mut op = Operation::new("max_attr_op");
        
        const ATTR_COUNT: usize = 50_000;  // Large number of attributes
        
        let mut attrs = HashMap::new();
        for i in 0..ATTR_COUNT {
            attrs.insert(
                format!("attribute_{:06}", i),
                Attribute::String(format!("value_for_{}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), ATTR_COUNT);
        assert_eq!(op.op_type, "max_attr_op");
        
        // Verify first and last attributes exist
        assert!(op.attributes.contains_key("attribute_000000"));
        assert!(op.attributes.contains_key(&format!("attribute_{:06}", ATTR_COUNT - 1)));
        
        // Verify values
        if let Some(Attribute::String(ref val)) = op.attributes.get("attribute_000000") {
            assert_eq!(val, "value_for_0");
        } else {
            panic!("Expected string attribute for attribute_000000");
        }
        
        if let Some(Attribute::String(ref val)) = op.attributes.get(&format!("attribute_{:06}", ATTR_COUNT - 1)) {
            assert_eq!(val, &format!("value_for_{}", ATTR_COUNT - 1));
        } else {
            panic!("Expected string attribute for last attribute");
        }
    }

    /// Test 9: Testing float values in attributes with edge cases
    #[test]
    fn test_floating_point_edge_cases_in_attributes() {
        let edge_case_floats = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            0.0,
            -0.0,
        ];
        
        for float_val in &edge_case_floats {
            let attr = Attribute::Float(*float_val);
            
            match attr {
                Attribute::Float(retrieved_val) => {
                    if float_val.is_nan() {
                        assert!(retrieved_val.is_nan(), "Original value was NaN");
                    } else if float_val.is_infinite() {
                        assert!(retrieved_val.is_infinite(), "Original value was infinite");
                        assert_eq!(float_val.is_sign_positive(), retrieved_val.is_sign_positive(), 
                                   "Sign should be preserved for infinite values");
                    } else if float_val == &0.0 {
                        // Special handling for signed zeros
                        assert!(retrieved_val == 0.0, "Zero should be preserved");
                    } else {
                        // For normal finite values
                        assert!((*float_val - retrieved_val).abs() < f64::EPSILON, 
                                "Finite values should be preserved accurately");
                    }
                }
                _ => panic!("Expected Float attribute"),
            }
        }
    }

    /// Test 10: Testing nested array attributes with maximum depth to test stack limits
    #[test]
    fn test_deeply_nested_array_attributes() {
        // Create nested arrays up to a reasonable depth
        let mut nested_attr = Attribute::Int(42);
        
        // Create 100 levels of nesting - sufficient to test recursion without causing stack overflow
        for _ in 0..100 {
            nested_attr = Attribute::Array(vec![nested_attr]);
        }
        
        // Verify the nesting was created properly by checking we can clone it without issue
        let cloned_nested = nested_attr.clone();
        assert_eq!(nested_attr, cloned_nested);
        
        // Test a smaller depth to verify functionality
        match &cloned_nested {
            Attribute::Array(inner) => {
                // The innermost element should be our original Int(42), nested 100 levels deep
                // We'll just verify that the top level is indeed an array
                assert!(!inner.is_empty());
            },
            _ => panic!("Expected nested array structure"),
        }
    }
}