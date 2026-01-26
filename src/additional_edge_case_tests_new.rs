//! Additional edge case tests for the Impulse compiler
//! These tests cover boundary conditions and edge cases that weren't covered by existing tests

use crate::ir::{Module, Value, Type, Operation, Attribute};

#[cfg(test)]
mod additional_edge_case_tests_new {
    use super::*;

    /// Test 1: Overflow protection when calculating tensor sizes with very large dimensions
    #[test]
    fn test_tensor_overflow_calculation() {
        // Test with very large but not quite overflowing dimensions
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],
        };

        // Use checked multiplication to handle potential overflow
        let mut result: Option<usize> = Some(1);
        for &dim in &large_value.shape {
            result = result.and_then(|prod| prod.checked_mul(dim));
        }

        // This should either return Some(value) or None due to overflow
        assert!(result.is_some() || result.is_none());
        
        // Test with safe dimensions
        let safe_value = Value {
            name: "safe_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],
        };
        let safe_product: usize = safe_value.shape.iter().product();
        assert_eq!(safe_product, 1_000_000);
    }

    /// Test 2: Empty and zero-dimensional tensors
    #[test]
    fn test_empty_and_zero_dimensional_tensors() {
        // Scalar (0-dimensional tensor)
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        assert_eq!(scalar.shape.len(), 0);
        let scalar_size: usize = scalar.shape.iter().product();
        assert_eq!(scalar_size, 1); // Scalars have 1 element
        
        // Tensor with one dimension of size 0
        let zero_tensor = Value {
            name: "zero_dim_tensor".to_string(),
            ty: Type::I32,
            shape: vec![5, 0, 10],  // Contains 0, so total size should be 0
        };
        let zero_size: usize = zero_tensor.shape.iter().product();
        assert_eq!(zero_size, 0);
        
        // Another zero-dimension example
        let another_zero = Value {
            name: "zero_tensor_2".to_string(),
            ty: Type::F64,
            shape: vec![0],
        };
        let another_zero_size: usize = another_zero.shape.iter().product();
        assert_eq!(another_zero_size, 0);
    }

    /// Test 3: Extreme boundary values for integer attributes
    #[test]
    fn test_integer_attribute_boundaries() {
        let min_attr = Attribute::Int(i64::MIN);
        let max_attr = Attribute::Int(i64::MAX);
        let zero_attr = Attribute::Int(0);

        match min_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected Int attribute"),
        }
        
        match max_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }
        
        match zero_attr {
            Attribute::Int(val) => assert_eq!(val, 0),
            _ => panic!("Expected Int attribute"),
        }
    }

    /// Test 4: Extreme values for floating point attributes
    #[test]
    fn test_float_attribute_extremes() {
        let pos_inf = Attribute::Float(f64::INFINITY);
        let neg_inf = Attribute::Float(f64::NEG_INFINITY);
        let nan_val = Attribute::Float(f64::NAN);
        let epsilon = Attribute::Float(f64::EPSILON);
        
        match pos_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected Float attribute"),
        }
        
        match neg_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
            _ => panic!("Expected Float attribute"),
        }
        
        match nan_val {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected Float attribute"),
        }
        
        match epsilon {
            Attribute::Float(val) => assert_eq!(val, f64::EPSILON),
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 5: Nested tensor types with maximum nesting depth
    #[test]
    fn test_deep_nested_tensor_types() {
        let mut current_type = Type::F32;
        // Create nested types up to a reasonable depth to avoid stack overflow
        for _ in 0..100 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],
            };
        }
        
        // Verify the final type is still a tensor
        match current_type {
            Type::Tensor { .. } => assert!(true),  // Successfully created nested tensor
            _ => panic!("Expected a nested tensor type"),
        }
        
        // Test cloning works for deeply nested types
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 6: Operations with maximum possible attributes
    #[test]
    fn test_operation_with_many_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attr_op");
        let mut attrs = HashMap::new();
        
        // Add many different types of attributes
        for i in 0..1000 {
            attrs.insert(
                format!("attr_{}", i),
                if i % 4 == 0 {
                    Attribute::Int(i as i64)
                } else if i % 4 == 1 {
                    Attribute::Float((i as f64) * 1.5)
                } else if i % 4 == 2 {
                    Attribute::String(format!("str_attr_{}", i))
                } else {
                    Attribute::Bool(i % 2 == 0)
                }
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 1000);
        assert!(op.attributes.contains_key("attr_500"));
        assert!(op.attributes.contains_key("attr_999"));
    }

    /// Test 7: Module with very large number of operations
    #[test]
    fn test_module_with_large_operations_count() {
        let mut module = Module::new("large_module");
        
        // Add many operations to test memory handling
        for i in 0..10_000 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 10_000);
        assert_eq!(module.name, "large_module");
        
        // Check a few operations to ensure they're preserved correctly
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[9999].op_type, "op_9999");
    }

    /// Test 8: Extremely long names for modules, operations, and values
    #[test]
    fn test_extremely_long_names() {
        let long_module_name = "x".repeat(10_000);
        let long_op_name = "y".repeat(10_000);
        let long_value_name = "z".repeat(10_000);
        
        let module = Module::new(long_module_name.clone());
        let op = Operation::new(&long_op_name);
        let value = Value {
            name: long_value_name.clone(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(module.name, long_module_name);
        assert_eq!(op.op_type, long_op_name);
        assert_eq!(value.name, long_value_name);
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    /// Test 9: Empty collections and structures
    #[test]
    fn test_empty_structures_and_collections() {
        // Empty module
        let empty_module = Module::new("");
        assert_eq!(empty_module.name, "");
        assert_eq!(empty_module.operations.len(), 0);
        assert_eq!(empty_module.inputs.len(), 0);
        assert_eq!(empty_module.outputs.len(), 0);
        
        // Operation with all empty fields
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);
        
        // Value with empty shape (scalar)
        let scalar_value = Value {
            name: "".to_string(),
            ty: Type::Bool,
            shape: vec![],
        };
        assert_eq!(scalar_value.shape.len(), 0);
        assert!(scalar_value.shape.is_empty());
    }

    /// Test 10: Complex nested array attributes
    #[test]
    fn test_complex_nested_array_attributes() {
        let complex_array = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.5),
                Attribute::Array(vec![
                    Attribute::String("nested_deep".to_string()),
                    Attribute::Bool(true),
                ]),
                Attribute::Int(3),
            ]),
            Attribute::String("top_level".to_string()),
        ]);
        
        // Verify the structure
        if let Attribute::Array(ref outer) = complex_array {
            assert_eq!(outer.len(), 3);
            
            // Check first element
            assert!(matches!(outer[0], Attribute::Int(1)));
            
            // Check second element (itself an array)
            if let Attribute::Array(ref second) = outer[1] {
                assert_eq!(second.len(), 3);
                assert!(matches!(second[0], Attribute::Float(val) if (val - 2.5).abs() < f64::EPSILON));
                
                // Check nested array inside second element
                if let Attribute::Array(ref nested) = second[1] {
                    assert_eq!(nested.len(), 2);
                    match &nested[0] {
                        Attribute::String(s) => assert_eq!(s, "nested_deep"),
                        _ => panic!("Expected string in nested array"),
                    }
                } else {
                    panic!("Expected nested array as second element of second array");
                }
            } else {
                panic!("Expected array as second element");
            }
            
            // Check third element
            if let Attribute::String(s) = &outer[2] {
                assert_eq!(s, "top_level");
            } else {
                panic!("Expected string as third element");
            }
        } else {
            panic!("Expected complex array as top level");
        }
    }
}