//! Additional tests focusing on deep recursion edge cases in nested tensor types

#[cfg(test)]
mod deep_recursion_edge_case_tests {
    use crate::ir::{Type, Value, TypeExtensions};

    #[test]
    fn test_very_deep_tensor_nesting() {
        // Test creating very deeply nested tensor types to check for stack overflow
        let mut current_type = Type::F32;
        
        // Create a very deep nesting (careful to avoid actual stack overflow)
        for _ in 0..500 {  // 500 levels deep
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the type is still valid
        assert!(current_type.is_valid_type());
        
        // Test cloning of deeply nested type
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }

    #[test]
    fn test_extremely_deep_tensor_nesting_clone() {
        // Test cloning a very deeply nested type
        let mut current_type = Type::F64;
        
        // Create deeper nesting than previous test
        for _ in 0..1000 {  // 1000 levels deep
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![3, 2],
            };
        }
        
        // This should not cause stack overflow during cloning
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    #[test]
    fn test_comparing_deeply_nested_types() {
        // Create two identical deeply nested types and compare them
        let mut type1 = Type::I32;
        let mut type2 = Type::I32;  // Same starting point
        
        // Build them identically to depth 100
        for _ in 0..100 {
            type1 = Type::Tensor {
                element_type: Box::new(type1),
                shape: vec![4, 4],
            };
            
            type2 = Type::Tensor {
                element_type: Box::new(type2),
                shape: vec![4, 4],
            };
        }
        
        // They should be equal
        assert_eq!(type1, type2);
        
        // Modify one slightly and they should not be equal
        let mut type3 = Type::I32;
        for _ in 0..100 {
            type3 = Type::Tensor {
                element_type: Box::new(type3),
                shape: vec![4, 5],  // Different shape
            };
        }
        
        assert_ne!(type1, type3);
    }

    #[test]
    fn test_deeply_nested_with_different_paths() {
        // Test creating deeply nested types with different paths but same depth
        let mut type1 = Type::F32;
        let mut type2 = Type::F32;
        
        // Both will be 50 levels deep but with different intermediate types
        for i in 0..50 {
            if i % 2 == 0 {
                type1 = Type::Tensor {
                    element_type: Box::new(type1),
                    shape: vec![2],
                };
                type2 = Type::Tensor {
                    element_type: Box::new(type2),
                    shape: vec![3],
                };
            } else {
                type1 = Type::Tensor {
                    element_type: Box::new(type1),
                    shape: vec![3],
                };
                type2 = Type::Tensor {
                    element_type: Box::new(type2),
                    shape: vec![2],
                };
            }
        }
        
        // They should not be equal due to different shapes at different levels
        assert_ne!(type1, type2);
        
        // Verify both are valid
        assert!(type1.is_valid_type());
        assert!(type2.is_valid_type());
    }

    #[test]
    fn test_deeply_nested_with_complex_types() {
        // Create a complex deeply nested structure
        let mut complex_type = Type::F32;
        
        // Build a complex nested structure with varying element types and shapes
        for i in 0..100 {
            match i % 3 {
                0 => complex_type = Type::Tensor {
                    element_type: Box::new(complex_type),
                    shape: vec![2, 2],
                },
                1 => complex_type = Type::Tensor {
                    element_type: Box::new(Type::I64),
                    shape: vec![i + 1],
                },
                _ => complex_type = Type::Tensor {
                    element_type: Box::new(complex_type),
                    shape: vec![i % 5 + 1],
                },
            }
        }
        
        // This complex structure should still be valid
        assert!(complex_type.is_valid_type());
    }

    #[test]
    fn test_deeply_nested_tensor_with_value_creation() {
        // Test creating a Value with a deeply nested tensor type
        let mut nested_type = Type::Bool;
        
        for _ in 0..200 {  // 200 levels deep
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![2],
            };
        }
        
        // Create a value with this deeply nested type
        let value = Value {
            name: "deeply_nested_value".to_string(),
            ty: nested_type,
            shape: vec![1, 1],
        };
        
        // Verify the value was created successfully
        assert_eq!(value.name, "deeply_nested_value");
        assert_eq!(value.shape, vec![1, 1]);
    }

    #[test]
    fn test_deep_recursion_in_type_validation() {
        // Test that deeply nested types validate correctly
        let mut deep_type = Type::I64;
        
        for _ in 0..500 {
            deep_type = Type::Tensor {
                element_type: Box::new(deep_type),
                shape: vec![1],
            };
        }
        
        // Validate that the deep nesting works correctly
        assert!(deep_type.is_valid_type());
        
        // Test the recursive validation by checking a few examples
        let simple_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![5, 5],
        };
        assert!(simple_tensor.is_valid_type());
        
        let deeper_tensor = Type::Tensor {
            element_type: Box::new(simple_tensor),
            shape: vec![3, 3],
        };
        assert!(deeper_tensor.is_valid_type());
    }

    #[test]
    fn test_pattern_based_deep_nesting() {
        // Create nested types following a pattern to test various algorithms
        let mut pattern_type = Type::F64;
        
        // Create a complex pattern across 300 levels
        for i in 0..300 {
            let shape = if i % 4 == 0 {
                vec![2, 3]
            } else if i % 4 == 1 {
                vec![5]
            } else if i % 4 == 2 {
                vec![1, 2, 3]
            } else {
                vec![7]
            };
            
            let element_type = match i % 3 {
                0 => Box::new(pattern_type),
                1 => Box::new(Type::I32),
                _ => Box::new(Type::Bool),
            };
            
            pattern_type = Type::Tensor {
                element_type,
                shape,
            };
        }
        
        // Verify this complex pattern still validates
        assert!(pattern_type.is_valid_type());
    }

    #[test]
    fn test_alternating_deep_nesting() {
        // Create alternating nested types
        let mut alt_type = Type::F32;
        
        for i in 0..100 {
            if i % 2 == 0 {
                alt_type = Type::Tensor {
                    element_type: Box::new(alt_type),
                    shape: vec![2, 2],
                };
            } else {
                alt_type = Type::Tensor {
                    element_type: Box::new(Type::F64),
                    shape: vec![3, 3],
                };
            }
        }
        
        assert!(alt_type.is_valid_type());
    }

    #[test]
    fn test_deep_nesting_with_large_shapes() {
        // Test deeply nested types with large individual shapes
        let mut large_shape_type = Type::I64;
        
        for i in 0..50 {
            // Use increasingly large shapes to test memory allocation
            let shape_size = (i % 10) + 1;
            let shape = vec![5; shape_size]; // Vector of 5s
            
            large_shape_type = Type::Tensor {
                element_type: Box::new(large_shape_type),
                shape,
            };
        }
        
        assert!(large_shape_type.is_valid_type());
    }
}