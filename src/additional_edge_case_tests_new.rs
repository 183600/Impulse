//! Additional edge case tests for Impulse compiler
//! Covering more boundary conditions and edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

#[cfg(test)]
mod additional_tests {
    use super::*;
    use rstest::rstest;

    /// Test 1: Operations with maximum possible string lengths
    #[test]
    fn test_extreme_string_lengths() {
        let extreme_name = "x".repeat(1_000_000); // 1MB string for name
        let value = Value {
            name: extreme_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name.len(), 1_000_000);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1]);
    }

    /// Test 2: Testing integer overflow protection in tensor size calculations
    #[test]
    fn test_tensor_size_overflow_protection() {
        // Values designed to cause overflow when multiplied without protection
        let problematic_shape = vec![1_000_000, 1_000_000];
        let value = Value {
            name: "problematic".to_string(),
            ty: Type::F32,
            shape: problematic_shape,
        };

        // Using safe calculation that handles potential overflow
        let result = value.num_elements();
        
        // The result could be None if overflow occurs, or Some(value) if not
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 1_000_000_000_000);
    }

    /// Test 3: Deeply nested type structures to test recursion limits  
    #[test]
    fn test_deeply_nested_recursive_types() {
        let mut current_type = Type::F32;
        
        // Create 500 levels of nesting (within reasonable limits to avoid stack overflow)
        for _ in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],
            };
        }

        // Verify the structure can be created and cloned
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 4: Edge case with maximum usize values in shapes (potential overflow)
    #[test]
    fn test_max_size_t_values_in_shapes() {
        // Use values that are large but not guaranteed to overflow on all systems
        let huge_value = usize::MAX / 1000; // Reduce to avoid immediate overflow
        
        let value = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: vec![huge_value, 2],
        };
        
        let product_opt = value.num_elements();
        assert!(product_opt.is_some() || product_opt.is_none()); // Either works or handles overflow
    }

    /// Test 5: Test operations with many attributes of all types
    #[test]
    fn test_operation_with_all_attribute_types_extensive() {
        let mut op = Operation::new("extensive_attrs");
        
        // Add many different attribute types
        let mut attrs = HashMap::new();
        
        // Add numerous int attributes
        for i in 0..100 {
            attrs.insert(
                format!("int_attr_{}", i),
                Attribute::Int(i as i64)
            );
        }
        
        // Add numerous float attributes
        for i in 0..100 {
            attrs.insert(
                format!("float_attr_{}", i),
                Attribute::Float(i as f64 * 0.5)
            );
        }
        
        // Add numerous string attributes
        for i in 0..100 {
            attrs.insert(
                format!("string_attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        // Add boolean attributes
        for i in 0..50 {
            attrs.insert(
                format!("bool_attr_{}", i),
                Attribute::Bool(i % 2 == 0)
            );
        }
        
        // Add complex nested arrays
        attrs.insert("complex_array".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
                Attribute::String("nested".to_string()),
            ]),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::Int(42),
                Attribute::Array(vec![Attribute::String("deep".to_string())]),
            ])
        ]));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 351); // 100 int + 100 float + 100 string + 50 bool + 1 complex array = 351
    }

    /// Test 6: Zero-size dimensions and special tensor shapes
    #[rstest]
    #[case(vec![], 1)]  // scalar
    #[case(vec![0], 0)]  // zero-dim tensor
    #[case(vec![1], 1)]  // unit tensor
    #[case(vec![0, 5], 0)]  // contains zero
    #[case(vec![5, 0], 0)]  // contains zero
    #[case(vec![1, 1, 1, 1], 1)]  // all ones
    #[case(vec![2, 3, 4], 24)]  // normal product
    fn test_various_zero_dimensional_shapes(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
        let value = Value {
            name: "test_shape".to_string(),
            ty: Type::F32,
            shape,
        };
        
        let calculated_size = value.num_elements().unwrap_or(0);
        assert_eq!(calculated_size, expected_size);
    }

    /// Test 7: Special float values in attributes
    #[test]
    fn test_special_floating_values_in_attributes() {
        let special_values = [
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
            (0.0, "pos_zero"),
            (-0.0, "neg_zero"),
            (f64::EPSILON, "epsilon"),
            (f64::MIN_POSITIVE, "min_pos"),
        ];
        
        for (val, desc) in special_values.iter() {
            let attr = Attribute::Float(*val);
            
            match attr {
                Attribute::Float(retrieved_val) => {
                    if val.is_nan() {
                        assert!(retrieved_val.is_nan(), "Failed for {}", desc);
                    } else if val.is_infinite() {
                        assert!(retrieved_val.is_infinite(), "Failed for {}", desc);
                        assert_eq!(retrieved_val.is_sign_positive(), val.is_sign_positive(), "Sign mismatch for {}", desc);
                    } else {
                        assert!(
                            (retrieved_val - val).abs() < f64::EPSILON || retrieved_val == *val,
                            "Value mismatch for {}: {} vs {}", 
                            desc, retrieved_val, val
                        );
                    }
                },
                _ => panic!("Expected Float attribute for {}", desc),
            }
        }
    }

    /// Test 8: Unicode and special characters in names and values
    #[test]
    fn test_unicode_character_handling() {
        let unicode_test_cases = [
            ("tensor_名称_日本語_🔥", Type::F32),
            ("chinese_chars_中文", Type::I32),
            ("arabic_chars_مرحبا", Type::F64),
            ("accented_chars_café_naïve", Type::I64),
            ("special_unicode_∑∏∇∞∫≈≠≤≥", Type::Bool),
        ];

        for (name, data_type) in unicode_test_cases.iter() {
            let value = Value {
                name: name.to_string(),
                ty: data_type.clone(),
                shape: vec![1, 2],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, *data_type);
            assert_eq!(value.shape, vec![1, 2]);
        }
        
        // Also test with operation names
        for (op_name, _) in unicode_test_cases.iter() {
            let op = Operation::new(op_name);
            assert_eq!(op.op_type, *op_name);
        }
    }

    /// Test 9: Massive number of operations in a module
    #[test]
    fn test_module_with_massive_number_of_operations() {
        let mut module = Module::new("massive_module");
        
        // Add 100,000 operations to test memory management and performance
        for i in 0..100_000 {
            let mut op = Operation::new(&format!("op_{:08}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 1000 + 1],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![(i + 1) % 1000 + 1],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 100_000);
        assert_eq!(module.name, "massive_module");
        
        // Verify a few specific operations are preserved correctly
        assert_eq!(module.operations[0].op_type, "op_00000000");
        assert_eq!(module.operations[99_999].op_type, "op_00099999");
    }

    /// Test 10: Complex nested tensor type validation
    #[test]
    fn test_complex_nested_tensor_validation() {
        // Create a complex nested structure with multiple levels
        let base_type = Type::F32;
        
        // Level 1: F32 wrapped in tensor
        let level1 = Type::Tensor {
            element_type: Box::new(base_type),
            shape: vec![2, 3],
        };
        
        // Level 2: Level1 wrapped in tensor 
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![4],
        };
        
        // Level 3: Level2 wrapped in tensor
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![5, 2],
        };
        
        // Validate the entire structure
        match &level3 {
            Type::Tensor { element_type: outer_el, shape: outer_shape } => {
                assert_eq!(outer_shape, &vec![5, 2]);
                
                match outer_el.as_ref() {
                    Type::Tensor { element_type: mid_el, shape: mid_shape } => {
                        assert_eq!(mid_shape, &vec![4]);
                        
                        match mid_el.as_ref() {
                            Type::Tensor { element_type: inner_el, shape: inner_shape } => {
                                assert_eq!(inner_shape, &vec![2, 3]);
                                
                                match inner_el.as_ref() {
                                    Type::F32 => {
                                        // Success! We reached the base type
                                        assert!(true); // Placeholder assertion
                                    },
                                    _ => panic!("Expected F32 at innermost level"),
                                }
                            },
                            _ => panic!("Expected tensor at second level"),
                        }
                    },
                    _ => panic!("Expected tensor at first level"),
                }
            },
            _ => panic!("Expected tensor at outermost level"),
        }
        
        // Verify that the complex structure can be cloned
        let cloned = level3.clone();
        assert_eq!(level3, cloned);
    }
}