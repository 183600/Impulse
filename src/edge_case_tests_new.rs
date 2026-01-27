//! Additional edge case tests for the Impulse compiler
//! This file contains 10 carefully selected tests covering unique edge cases

#[cfg(test)]
mod edge_case_tests_new {
    use crate::ir::{Value, Type, Operation, Attribute, Module, TypeExtensions};
    
    /// Test 1: Testing operations with empty string names and special unicode names
    #[test]
    fn test_operation_with_special_names() {
        // Test with empty name
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        
        // Test with unicode name
        let unicode_op = Operation::new("opération_⚡_测试");
        assert_eq!(unicode_op.op_type, "opération_⚡_测试");
        
        // Test with very long unicode name
        let long_unicode = "α".repeat(1000) + "_beta_gamma";
        let long_op = Operation::new(&long_unicode);
        assert_eq!(long_op.op_type, long_unicode);
    }

    /// Test 2: Testing recursive tensor type operations (creation, cloning, comparison)
    #[test]
    fn test_recursive_tensor_operations() {
        // Create a recursive tensor type
        let mut current_type = Type::F32;
        
        // Create several levels with different shapes to test complexity
        for i in 0..10 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i + 1],
            };
        }
        
        // Test cloning works correctly
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Test creation of two equivalent types
        let mut type_a = Type::F32;
        for i in 0..10 {
            type_a = Type::Tensor {
                element_type: Box::new(type_a),
                shape: vec![i + 1],
            };
        }
        
        assert_eq!(current_type, type_a);
    }

    /// Test 3: Testing Value with maximum possible number of dimensions
    #[test]
    fn test_value_with_maximum_dimensions() {
        // Create a value with many dimensions
        let many_dims = (1..=100).collect::<Vec<_>>();
        let value = Value {
            name: "many_dim_tensor".to_string(),
            ty: Type::F32,
            shape: many_dims.clone(),
        };
        
        // Verify all dimensions are preserved
        assert_eq!(value.shape, many_dims);
        
        // The total number of dimensions should be 100
        assert_eq!(value.shape.len(), 100);
        
        // Test accessing specific dimensions
        assert_eq!(value.shape[0], 1);   // First dimension
        assert_eq!(value.shape[99], 100); // Last dimension
    }

    /// Test 4: Testing attribute arrays with maximum nesting levels
    #[test]
    fn test_attribute_array_maximum_nesting() {
        // Create deeply nested arrays
        let mut nested_array = Attribute::Int(42);
        
        // Nest 20 levels deep
        for _ in 0..20 {
            nested_array = Attribute::Array(vec![nested_array]);
        }
        
        // Verify the structure by unwrapping
        let mut current_attr = &nested_array;
        for _ in 0..20 {
            match current_attr {
                Attribute::Array(arr) => {
                    assert_eq!(arr.len(), 1);
                    current_attr = &arr[0];
                },
                _ => panic!("Expected Array at nesting level"),
            }
        }
        
        // Should end up with the original Int
        match current_attr {
            Attribute::Int(42) => { /* Success */ },
            _ => panic!("Expected Int(42) at deepest level"),
        }
    }

    /// Test 5: Testing Value::num_elements() with potential overflow scenarios
    #[test]
    fn test_num_elements_overflow_scenarios() {
        // Test with a shape that would overflow if calculated naively
        // Use try_fold to handle potential overflow gracefully
        let value_with_zero = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 0, 5000],  // Contains zero, should return Some(0)
        };
        assert_eq!(value_with_zero.num_elements(), Some(0));
        
        // Test with safe large values
        let large_safe_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],  // 1 million elements
        };
        assert_eq!(large_safe_value.num_elements(), Some(1_000_000));
        
        // Test with scalar (empty shape)
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        assert_eq!(scalar_value.num_elements(), Some(1));
    }

    /// Test 6: Testing module with operations that have conflicting attribute keys
    #[test]
    fn test_operations_with_conflicting_attributes() {
        // Test case 1: operations with duplicate keys
        {
            let mut op = Operation::new("test_op");
            let keys = vec!["padding", "stride", "dilation", "padding", "stride"];
            
            for (i, key) in keys.iter().enumerate() {
                op.attributes.insert(
                    key.to_string(),
                    Attribute::Int(i as i64)
                );
            }
            
            // Should have unique keys only (HashMap behavior)
            let unique_keys: std::collections::HashSet<_> = keys.iter().filter(|&&k| !k.is_empty()).collect();
            assert_eq!(op.attributes.len(), unique_keys.len());
            
            // Should have 3 unique keys (padding, stride, dilation)
            assert_eq!(op.attributes.len(), 3);
        }
        
        // Test case 2: operations with empty keys
        {
            let mut op = Operation::new("test_op2");
            let keys = vec!["", "normal_key", "", "another_key", ""];
            
            for (i, key) in keys.iter().enumerate() {
                op.attributes.insert(
                    key.to_string(),
                    Attribute::Int(i as i64)
                );
            }
            
            // Should have unique keys only (HashMap behavior)
            let unique_keys: std::collections::HashSet<_> = keys.iter().filter(|&&k| !k.is_empty()).collect();
            assert_eq!(op.attributes.len(), unique_keys.len() + 1); // +1 for empty key
            
            // Empty keys should be allowed as keys in HashMap
            assert!(op.attributes.contains_key(""));
            assert!(op.attributes.contains_key("normal_key"));
            assert!(op.attributes.contains_key("another_key"));
        }
    }

    /// Test 7: Testing deeply nested operations with input/output dependencies
    #[test]
    fn test_deeply_nested_operation_dependencies() {
        let mut module = Module::new("dependency_chain");
        
        // Create a chain of operations where each depends on the previous one
        for i in 0..100 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Each operation takes output from previous as input (except first)
            if i > 0 {
                op.inputs.push(Value {
                    name: format!("output_{}", i - 1),
                    ty: Type::F32,
                    shape: vec![10, 10],
                });
            }
            
            // Each operation produces an output
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 100);
        
        // Verify first operation has no inputs
        assert_eq!(module.operations[0].inputs.len(), 0);
        assert_eq!(module.operations[0].outputs.len(), 1);
        
        // Verify subsequent operations have inputs
        for i in 1..100 {
            assert_eq!(module.operations[i].inputs.len(), 1, "Operation {} should have 1 input", i);
            assert_eq!(module.operations[i].outputs.len(), 1, "Operation {} should have 1 output", i);
        }
    }

    /// Test 8: Testing type validation edge cases with complex recursive types
    #[test]
    fn test_type_validation_edge_cases() {
        use crate::ir::TypeExtensions;
        
        // Test a valid deeply nested type
        let mut valid_type = Type::F32;
        for _ in 0..50 {
            valid_type = Type::Tensor {
                element_type: Box::new(valid_type),
                shape: vec![2],
            };
        }
        assert!(valid_type.is_valid_type());
        
        // Test a valid type with different element types alternating
        let mut alternating_type = Type::I32;
        for i in 0..25 {
            let element_type = if i % 2 == 0 { Type::F32 } else { Type::I64 };
            alternating_type = Type::Tensor {
                element_type: Box::new(element_type),
                shape: vec![i + 1],
            };
        }
        assert!(alternating_type.is_valid_type());
        
        // Validate that cloning preserves validity
        let cloned = alternating_type.clone();
        assert!(cloned.is_valid_type());
        assert_eq!(alternating_type, cloned);
    }

    /// Test 9: Testing values with extreme shape configurations
    #[test]
    fn test_extreme_shape_configurations() {
        // Case 1: all ones
        {
            let shape = vec![1, 1, 1, 1, 1];
            let value = Value {
                name: "extreme_shape".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let elements = value.num_elements();
            assert_eq!(elements, Some(1)); // 1*1*1*1*1 = 1
        }
        
        // Case 2: max first
        {
            let shape = vec![std::usize::MAX, 1];
            let value = Value {
                name: "extreme_shape".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let elements = value.num_elements();
            assert_eq!(elements, Some(std::usize::MAX)); // MAX * 1 = MAX
        }
        
        // Case 3: max second
        {
            let shape = vec![1, std::usize::MAX];
            let value = Value {
                name: "extreme_shape".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let elements = value.num_elements();
            assert_eq!(elements, Some(std::usize::MAX)); // 1 * MAX = MAX
        }
        
        // Case 4: sparse large
        {
            let shape = vec![100, 1, 100, 1, 100];
            let value = Value {
                name: "extreme_shape".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let elements = value.num_elements();
            assert_eq!(elements, Some(100 * 1 * 100 * 1 * 100)); // 1,000,000
        }
        
        // Case 5: ten twos
        {
            let shape = vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2]; // 2^10 = 1024
            let value = Value {
                name: "extreme_shape".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let elements = value.num_elements();
            assert_eq!(elements, Some(1024)); // 2^10
        }
    }

    /// Test 10: Testing attribute comparisons with special floating point values
    #[test]
    fn test_special_float_attribute_comparisons() {
        // Create attributes with special floating point values
        let pos_inf_attr = Attribute::Float(std::f64::INFINITY);
        let neg_inf_attr = Attribute::Float(std::f64::NEG_INFINITY);
        let nan_attr = Attribute::Float(std::f64::NAN);
        let neg_zero_attr = Attribute::Float(-0.0);
        let pos_zero_attr = Attribute::Float(0.0);
        
        // Test that equals works properly for infinities
        assert_eq!(pos_inf_attr, Attribute::Float(std::f64::INFINITY));
        assert_eq!(neg_inf_attr, Attribute::Float(std::f64::NEG_INFINITY));
        
        // Test that NaN behaves correctly (NaN != NaN by IEEE standard, but our implementation might differ)
        // Since we're using derive(PartialEq), NaN comparison will be false
        assert_ne!(nan_attr.clone(), Attribute::Float(std::f64::NAN));
        
        // Test that positive and negative zero
        // In Rust, -0.0 == 0.0 is true for f64, so they should be equal
        assert_eq!(neg_zero_attr, pos_zero_attr);
        
        // Test that positive/negative infinity are different
        assert_ne!(pos_inf_attr, neg_inf_attr);
        
        // Create an extra infinity attribute to double check
        let pos_inf_attr2 = Attribute::Float(std::f64::INFINITY);
        assert_eq!(pos_inf_attr, pos_inf_attr2);
    }
}