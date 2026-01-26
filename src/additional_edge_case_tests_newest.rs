//! Additional edge case tests for Impulse compiler
//! Covering more boundary conditions and edge cases

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;
    use crate::ir::{Module, Value, Type, Operation};
    use crate::ImpulseCompiler;
    use std::collections::HashMap;

    /// Test 1: Empty value names handling
    #[test]
    fn test_empty_value_names() {
        let value = Value {
            name: "".to_string(),  // Empty name
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(value.name, "");
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape.len(), 0);
    }

    /// Test 2: Operations with empty input/output vectors
    #[test]
    fn test_operation_empty_vectors() {
        let op = Operation::new("empty_test");
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
        
        // Test with empty vectors explicitly
        let mut op_with_empty_fields = Operation::new("empty_fields");
        op_with_empty_fields.inputs = vec![];
        op_with_empty_fields.outputs = vec![];
        op_with_empty_fields.attributes = HashMap::new();
        
        assert!(op_with_empty_fields.inputs.is_empty());
        assert!(op_with_empty_fields.outputs.is_empty());
        assert!(op_with_empty_fields.attributes.is_empty());
    }

    /// Test 3: Large number of recursive type nestings
    #[test]
    fn test_very_deep_type_nesting() {
        let mut current_type = Type::I32;
        // Create a deeply nested type with 500 levels
        for _ in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type.clone()), // Clone at each step
                shape: vec![1],
            };
        }
        
        let result_type = current_type;
        
        // Verify it's still a valid tensor type
        match &result_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![1]);
            },
            _ => panic!("Expected a tensor type after deep nesting"),
        }
        
        // Test cloning of the deeply nested type
        let cloned = result_type.clone();
        assert_eq!(result_type, cloned);
    }

    /// Test 4: Maximum attribute count in operations
    #[test]
    fn test_operation_with_max_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attrs");
        let mut attrs = HashMap::new();
        
        // Add 100,000 attributes to test memory and performance
        for i in 0..100_000 {
            attrs.insert(
                format!("attr_{}", i),
                crate::ir::Attribute::Int(i as i64)
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 100_000);
        
        // Verify we can still access attributes
        for i in (0..100_000).step_by(10000) {
            let key = format!("attr_{}", i);
            assert!(op.attributes.contains_key(&key));
        }
    }

    /// Test 5: Value with maximum possible shape dimensions
    #[test]
    fn test_value_max_shape_dimensions() {
        // Create a value with 100 dimensions to test limits
        let shape: Vec<usize> = (0..100).map(|_| 1).collect();
        let value = Value {
            name: "max_dims".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape.len(), 100);
        assert_eq!(value.shape, shape);
        
        // Calculate the number of elements (should be 1^100 = 1)
        let elements = value.num_elements().unwrap();
        assert_eq!(elements, 1);
    }

    /// Test 6: Testing extreme numerical values in attributes
    #[test]
    fn test_extreme_numerical_attributes() {
        use crate::ir::Attribute;
        
        // Test with extreme numerical values
        let attrs = [
            Attribute::Int(i64::MAX),
            Attribute::Int(i64::MIN),
            Attribute::Float(f64::MAX),
            Attribute::Float(f64::MIN),
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
        ];
        
        match attrs[0] {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }
        
        match attrs[1] {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected Int attribute"),
        }
        
        match attrs[2] {
            Attribute::Float(val) => assert_eq!(val, f64::MAX),
            _ => panic!("Expected Float attribute"),
        }
        
        match attrs[3] {
            Attribute::Float(val) => assert_eq!(val, f64::MIN),
            _ => panic!("Expected Float attribute"),
        }
        
        match attrs[4] {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected positive infinity"),
        }
        
        match attrs[5] {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
            _ => panic!("Expected negative infinity"),
        }
    }

    /// Test 7: Compiler with invalid byte patterns
    #[test]
    fn test_compiler_with_invalid_bytes() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with invalid byte sequences that might cause parsing issues
        let invalid_bytes = vec![
            0xFF, 0xFF, 0xFF, 0xFF,  // Invalid UTF-8-like sequence
            0x00, 0x00, 0x00, 0x00,  // All nulls
            0xDE, 0xAD, 0xBE, 0xEF,  // Classic hex values
        ];
        
        // This should not cause a panic, even if it results in an error
        let result = compiler.compile(&invalid_bytes, "cpu");
        
        // The result can be either success or failure, but should not panic
        assert!(result.is_ok() || result.is_err());
    }

    /// Test 8: Module with extremely deeply nested operations
    #[test]
    fn test_module_with_deeply_nested_tensors() {
        let mut module = Module::new("deeply_nested_module");
        
        // Create operation with deeply nested tensor type
        let mut current_type = Type::F32;
        for i in 0..200 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 2 + 1], // Alternate between shapes [1] and [2]
            };
        }
        
        let value = Value {
            name: "deeply_nested_value".to_string(),
            ty: current_type,
            shape: vec![1, 1, 1, 1, 1], // Small shape
        };
        
        let mut op = Operation::new("deeply_nested_op");
        op.inputs.push(value);
        
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].inputs.len(), 1);
    }

    /// Test 9: Test value with empty string tensor name
    #[test]
    fn test_value_with_special_names() {
        let special_names = vec![
            "",                    // Empty string
            " ",                   // Single space
            "\t\n\r",             // Whitespace characters
            "特殊字符",              // Unicode characters
            "🚀🔥🌟",              // Emoji characters
            "a".repeat(1_000_000), // Very long string (reduced for performance)
        ];
        
        for (i, name) in special_names.iter().enumerate() {
            let value = Value {
                name: name.clone(),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![i + 1],
            };
            
            assert_eq!(value.name, *name);
        }
    }

    /// Test 10: Large sparse tensor shapes (with zeros in non-leading positions)
    #[test]
    fn test_sparse_tensor_shapes() {
        let test_cases = vec![
            (vec![10, 0, 20], 0),         // Zero in middle causes 0 product
            (vec![0, 10, 20], 0),         // Zero at beginning
            (vec![10, 20, 0], 0),         // Zero at end
            (vec![10, 1, 0, 1], 0),       // Zero in multiple positions
            (vec![1, 1, 1, 0], 0),        // Zero at end with ones
            (vec![0], 0),                 // Single zero dimension
            (vec![10, 1, 1], 10),         // Ones don't affect product
            (vec![2, 3, 4], 24),          // Normal case
        ];
        
        for (shape, expected_product) in test_cases {
            let value = Value {
                name: "sparse_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            // Calculate actual product
            let actual_product: usize = value.shape.iter().product();
            assert_eq!(actual_product, expected_product, "Failed for shape {:?}", shape);
            
            // Also test with the num_elements method
            match value.num_elements() {
                Some(elements) => assert_eq!(elements, expected_product),
                None if expected_product == 0 => (), // This is expected for zero products
                None => panic!("num_elements returned None unexpectedly for shape {:?}", shape),
            }
        }
    }
    
    // Use rstest for parameterized testing
    #[rstest::rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    fn test_basic_types_validation(#[case] basic_type: Type) {
        assert!(basic_type.is_valid_type());
        
        // Create a tensor with the basic type and verify it's still valid
        let tensor_type = Type::Tensor {
            element_type: Box::new(basic_type.clone()),
            shape: vec![1, 2, 3],
        };
        assert!(tensor_type.is_valid_type());
    }
}