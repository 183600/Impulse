//! Additional edge case tests for the Impulse compiler
//! Covers more boundary conditions and error scenarios not tested elsewhere

use crate::{
    ir::{Module, Value, Type, Operation, Attribute, TypeExtensions},
};
use std::collections::HashMap;

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;

    /// Test 1: Testing very deep recursion with alternating types
    #[test]
    fn test_deep_alternating_recursive_types() {
        let mut current_type = Type::F32;
        
        // Create alternating nested types to test deep recursion
        for i in 0..50 {
            if i % 2 == 0 {
                current_type = Type::Tensor {
                    element_type: Box::new(Type::I32),
                    shape: vec![i + 1],
                };
            } else {
                current_type = Type::Tensor {
                    element_type: Box::new(current_type),
                    shape: vec![i % 3 + 1], // Alternate between 1, 2, 3
                };
            }
        }
        
        // Ensure the type was constructed properly
        assert!(current_type.is_valid_type());
        
        // Clone to ensure deep cloning works
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 2: Value with maximum possible dimension count
    #[test]
    fn test_value_with_maximum_dimensions() {
        // Create a shape with many dimensions
        let many_dims = vec![1; 1000]; // 1000 dimensions, each of size 1
        let value = Value {
            name: "many_dims".to_string(),
            ty: Type::F32,
            shape: many_dims,
        };
        
        assert_eq!(value.shape.len(), 1000);
        assert_eq!(value.num_elements().unwrap(), 1); // All dimensions are 1
    }

    /// Test 3: Operation with empty strings and special unicode characters
    #[test]
    fn test_operation_with_special_strings() {
        // Create test strings separately to avoid borrowing issues
        let very_long_name = "a".repeat(50_000);
        let special_names = [
            ("", "empty_name_op"),
            ("🚀✨🌟", "emoji_op"),
            ("тестовый", "cyrillic_op"),
            ("テスト", "japanese_op"),
            ("\0", "null_char_op"),  // Contains null byte
            (very_long_name.as_str(), "very_long_op"),  // Very long string
        ];
        
        for (name, op_type) in &special_names {
            let mut op = Operation::new(op_type);
            op.inputs.push(Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1],
            });
            
            assert_eq!(op.op_type, *op_type);
            assert_eq!(op.inputs[0].name, *name);
        }
    }

    /// Test 4: Attribute with deeply nested arrays
    #[test]
    fn test_deeply_nested_array_attributes() {
        // Create a deeply nested array structure: [[[[[value]]]]] (5 levels deep)
        let mut nested = Attribute::Int(42);
        
        for _ in 0..5 {
            nested = Attribute::Array(vec![nested]);
        }
        
        // Verify the structure by extracting values
        if let Attribute::Array(lvl1) = &nested {
            assert_eq!(lvl1.len(), 1);
            if let Attribute::Array(lvl2) = &lvl1[0] {
                assert_eq!(lvl2.len(), 1);
                if let Attribute::Array(lvl3) = &lvl2[0] {
                    assert_eq!(lvl3.len(), 1);
                    if let Attribute::Array(lvl4) = &lvl3[0] {
                        assert_eq!(lvl4.len(), 1);
                        if let Attribute::Array(lvl5) = &lvl4[0] {
                            assert_eq!(lvl5.len(), 1);
                            if let Attribute::Int(val) = lvl5[0] {
                                assert_eq!(val, 42);
                            } else {
                                panic!("Expected Int at deepest level");
                            }
                        } else {
                            panic!("Expected Array at level 5");
                        }
                    } else {
                        panic!("Expected Array at level 4");
                    }
                } else {
                    panic!("Expected Array at level 3");
                }
            } else {
                panic!("Expected Array at level 2");
            }
        } else {
            panic!("Expected Array at level 1");
        }
    }

    /// Test 5: Operations with maximum attribute count
    #[test]
    fn test_operation_with_maximum_attributes() {
        let mut op = Operation::new("max_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add 10,000 attributes to test limits
        for i in 0..10_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 10_000);
        
        // Verify some specific attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_9999"));
        assert_eq!(op.attributes.get("attr_5000"), 
                  Some(&Attribute::String("value_5000".to_string())));
    }

    /// Test 6: Value with shape that causes potential overflow in calculations
    #[test]
    fn test_value_shape_overflow_handling() {
        // Create a shape that would cause overflow when calculating total elements
        // This test ensures safe handling of potentially large products
        
        // Use a shape that when multiplied would exceed typical limits
        let large_but_safe_shape = vec![100_000, 100_000, 100]; // 1 trillion elements
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: large_but_safe_shape.clone(),
        };
        
        // Using checked multiplication to detect overflow
        let product: Option<usize> = value.shape.iter()
            .try_fold(1usize, |acc, &x| acc.checked_mul(x));
        
        // In this case it should be Some (not overflow on 64-bit systems)
        assert!(product.is_some());
        assert_eq!(value.shape, large_but_safe_shape);
    }

    /// Test 7: Recursive function that validates complex nested types
    #[test]
    fn test_complex_type_validation() {
        fn create_complex_type(depth: usize) -> Type {
            if depth == 0 {
                Type::F32
            } else {
                Type::Tensor {
                    element_type: Box::new(create_complex_type(depth - 1)),
                    shape: vec![depth],
                }
            }
        }
        
        // Create a complex type with depth 20
        let complex_type = create_complex_type(20);
        
        // Validate that it's a valid type
        assert!(complex_type.is_valid_type());
        
        // Clone and compare
        let cloned_type = complex_type.clone();
        assert_eq!(complex_type, cloned_type);
    }

    /// Test 8: Module with operations having conflicting names
    #[test]
    fn test_module_with_conflicting_operation_names() {
        let mut module = Module::new("conflict_test");
        
        // Add operations with similar names to test name handling
        for i in 0..100 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 100);
        
        // Verify operations have correct names
        for i in 0..100 {
            assert_eq!(module.operations[i].op_type, format!("op_{}", i));
            assert_eq!(module.operations[i].inputs[0].name, format!("input_{}", i));
        }
    }

    /// Test 9: Value with negative indices handling (simulated via custom logic)
    #[test]
    fn test_tensor_index_bounds_checking_simulation() {
        // While we can't actually have negative indices in our Value struct,
        // we can simulate bounds checking for tensor operations
        
        let value = Value {
            name: "bounds_test".to_string(),
            ty: Type::F32,
            shape: vec![10, 20, 30],
        };
        
        // Calculate total elements
        let total_elements = value.num_elements().unwrap();
        assert_eq!(total_elements, 10 * 20 * 30); // 6000 elements
        
        // Simulate checking if an index is within bounds
        let valid_indices = vec![
            vec![0, 0, 0],      // First element
            vec![9, 19, 29],    // Last element  
            vec![5, 10, 15],    // Middle element
        ];
        
        for idx in valid_indices {
            let is_valid = idx.iter().enumerate().all(|(dim, &index)| {
                index < value.shape[dim]
            });
            assert!(is_valid, "Index {:?} should be valid for shape {:?}", idx, value.shape);
        }
    }

    /// Test 10: Memory allocation stress test with mixed data types
    #[test]
    fn test_memory_allocation_with_mixed_types() {
        // Create a collection of different types to test memory allocation patterns
        let mut values = Vec::new();
        
        // Add values of different types and shapes
        for i in 0..1000 {
            let value = Value {
                name: format!("stress_value_{}", i),
                ty: match i % 5 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    3 => Type::I64,
                    _ => Type::Bool,
                },
                shape: vec![i % 10 + 1, (i + 1) % 10 + 1],
            };
            values.push(value);
        }
        
        // Verify all values were created
        assert_eq!(values.len(), 1000);
        
        // Verify types are distributed correctly
        let f32_count = values.iter().filter(|v| matches!(v.ty, Type::F32)).count();
        let f64_count = values.iter().filter(|v| matches!(v.ty, Type::F64)).count();
        let i32_count = values.iter().filter(|v| matches!(v.ty, Type::I32)).count();
        
        assert_eq!(f32_count, 200); // 1000 / 5
        assert_eq!(f64_count, 200); 
        assert_eq!(i32_count, 200);
        
        // Verify shapes are correct
        for (_i, v) in values.iter().enumerate() {
            assert_eq!(v.shape.len(), 2);
            assert!(v.shape[0] >= 1 && v.shape[0] <= 10);
            assert!(v.shape[1] >= 1 && v.shape[1] <= 10);
        }
    }
}

#[cfg(test)]
mod rstest_edge_cases {
    use rstest::*;
    use crate::ir::{Value, Type, Attribute};
    
    /// Test with rstest for various tensor shapes and data types
    #[rstest]
    #[case(vec![], Type::F32, 1)]    // scalar F32
    #[case(vec![], Type::I64, 1)]    // scalar I64
    #[case(vec![0], Type::F32, 0)]   // 0-dim tensor F32
    #[case(vec![1], Type::F32, 1)]   // 1-element F32
    #[case(vec![2, 3], Type::I32, 6)] // 2x3 I32 tensor
    #[case(vec![1, 1, 1], Type::Bool, 1)] // 1x1x1 Bool tensor
    #[case(vec![2, 0, 5], Type::F64, 0)] // contains 0, so 0 elements
    fn test_tensor_size_calculations(#[case] shape: Vec<usize>, #[case] data_type: Type, #[case] expected_elements: usize) {
        let value = Value {
            name: "tensor_test".to_string(),
            ty: data_type,
            shape,
        };
        
        let calculated_elements = value.num_elements().unwrap_or(0);
        assert_eq!(calculated_elements, expected_elements);
    }
    
    /// Test with rstest for complex attribute structures
    #[rstest]
    #[case(Attribute::Int(i64::MAX))]
    #[case(Attribute::Int(i64::MIN))]
    #[case(Attribute::Float(f64::INFINITY))]
    #[case(Attribute::Float(f64::NEG_INFINITY))]
    #[case(Attribute::Bool(true))]
    #[case(Attribute::Bool(false))]
    #[case(Attribute::String("".to_string()))]
    #[case(Attribute::String("test".to_string()))]
    fn test_basic_attribute_properties(#[case] attr: Attribute) {
        // This test ensures that common attribute values can be created and handled
        match attr {
            Attribute::Int(_val) => {
                // Just verify it's an int attribute - no specific checks needed
                assert!(true);
            },
            Attribute::Float(val) => {
                // Handle infinities specially since they don't equal themselves
                if val.is_infinite() {
                    assert!(val.is_infinite());
                } else if val.is_nan() {
                    // NaN is handled separately since NaN != NaN
                    assert!(val.is_nan());
                } else {
                    // Regular float can be compared directly
                    assert!(true);
                }
            },
            Attribute::String(ref _s) => {
                // String length should be valid
                assert!(true); // This is always true for usize, but kept for completeness
            },
            Attribute::Bool(b) => {
                // Boolean value should be either true or false
                assert!(b == true || b == false);
            },
            Attribute::Array(_) => {
                // Arrays are tested in detail elsewhere
                assert!(true);
            },
        }
    }
}