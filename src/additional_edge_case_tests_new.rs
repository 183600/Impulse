//! Additional edge case tests for the Impulse compiler
//! This file contains extra tests focusing on boundary conditions and edge cases

use rstest::*;
use crate::ir::{Module, Value, Type, Operation, Attribute};

// Test for extreme numerical values in attributes
#[rstest]
#[case(i64::MAX, "max_i64")]
#[case(i64::MIN, "min_i64")]
#[case(0, "zero_i64")]
fn test_extreme_integer_attributes(#[case] value: i64, #[case] description: &str) {
    let attr = Attribute::Int(value);
    match attr {
        Attribute::Int(v) => assert_eq!(v, value, "Integer attribute should preserve {}", description),
        _ => panic!("Expected Integer attribute"),
    }
}

// Test for extreme float values in attributes
#[rstest]
#[case(f64::INFINITY, "infinity_f64")]
#[case(f64::NEG_INFINITY, "neg_infinity_f64")]
#[case(f64::NAN, "nan_f64")]
#[case(f64::EPSILON, "epsilon_f64")]
#[case(-0.0, "negative_zero_f64")]
fn test_extreme_float_attributes(#[case] value: f64, #[case] description: &str) {
    let attr = Attribute::Float(value);
    match attr {
        Attribute::Float(v) => {
            if value.is_nan() {
                assert!(v.is_nan(), "Float attribute should preserve {}", description);
            } else {
                assert_eq!(v, value, "Float attribute should preserve {}", description);
            }
        },
        _ => panic!("Expected Float attribute"),
    }
}

// Test for empty and extreme string attributes
#[rstest]
#[case("", "empty_string")]
#[case("a".repeat(100_000), "very_long_string")]
fn test_string_attribute_edge_cases(#[case] value: String, #[case] description: &str) {
    let attr = Attribute::String(value.clone());
    match attr {
        Attribute::String(v) => assert_eq!(v, value, "String attribute should preserve {}", description),
        _ => panic!("Expected String attribute"),
    }
}

// Test for empty and extreme boolean attributes
#[rstest]
#[case(true, "true_bool")]
#[case(false, "false_bool")]
fn test_boolean_attribute_edge_cases(#[case] value: bool, #[case] description: &str) {
    let attr = Attribute::Bool(value);
    match attr {
        Attribute::Bool(v) => assert_eq!(v, value, "Boolean attribute should preserve {}", description),
        _ => panic!("Expected Boolean attribute"),
    }
}

// Test for value with extreme tensor dimensions
#[rstest]
#[case(vec![], 1, "scalar")]  // Scalar has one element
#[case(vec![0], 0, "zero_dim")]  // Zero-dimensional tensor
#[case(vec![1], 1, "unit_dim")]  // Unit tensor
#[case(vec![1, 1, 1, 1], 1, "multi_unit_dims")]  // Multiple unit dimensions
#[case(vec![2, 2, 2, 2], 16, "multi_dims")]  // Multiple small dimensions
fn test_tensor_shape_edge_cases(#[case] shape: Vec<usize>, #[case] expected_size: usize, #[case] description: &str) {
    let value = Value {
        name: format!("tensor_{}", description),
        ty: Type::F32,
        shape,
    };
    
    assert_eq!(value.name, format!("tensor_{}", description));
    let actual_size: usize = value.shape.iter().product();
    assert_eq!(actual_size, expected_size, "Size calculation should work for {}", description);
}

// Test for operation with extreme numbers of attributes
#[rstest]
#[case(0, "no_attributes")]
#[case(1, "one_attribute")]
#[case(10, "ten_attributes")]
#[case(100, "hundred_attributes")]
fn test_operation_attribute_counts(#[case] count: usize, #[case] description: &str) {
    use std::collections::HashMap;
    
    let mut op = Operation::new(&format!("op_{}", description));
    let mut attrs = HashMap::new();
    
    for i in 0..count {
        attrs.insert(
            format!("attr_{}", i), 
            Attribute::Int(i as i64)
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), count, "Operation should have {} attributes", description);
}

// Test for deeply nested tensor types
#[rstest]
#[case(1, "shallow_nested")]
#[case(5, "medium_nested")]
#[case(10, "deep_nested")]
fn test_deep_tensor_nesting(#[case] depth: usize, #[case] description: &str) {
    let mut current_type = Type::F32;
    
    // Build nested tensor type
    for _ in 0..depth {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the type can be compared with itself
    assert_eq!(current_type, current_type, "Deeply nested type should be equal to itself");
    
    // Verify the type can be cloned
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned, "Cloned deeply nested type should be equal");
}

// Test for operation with extreme numbers of inputs/outputs
#[rstest]
#[case(0, 0, "zero_io")]
#[case(1, 1, "single_io")]
#[case(10, 5, "many_inputs_few_outputs")]
#[case(5, 10, "few_inputs_many_outputs")]
#[case(50, 50, "balanced_many_io")]
fn test_operation_io_counts(#[case] input_count: usize, #[case] output_count: usize, #[case] description: &str) {
    let mut op = Operation::new(&format!("op_{}", description));
    
    // Add inputs
    for i in 0..input_count {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    // Add outputs
    for i in 0..output_count {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(op.inputs.len(), input_count, "Operation should have correct number of inputs for {}", description);
    assert_eq!(op.outputs.len(), output_count, "Operation should have correct number of outputs for {}", description);
}

// Test for module with extreme operation counts
#[rstest]
#[case(0, "empty_module")]
#[case(1, "single_op_module")]
#[case(10, "few_ops_module")]
#[case(100, "many_ops_module")]
fn test_module_operation_counts(#[case] op_count: usize, #[case] description: &str) {
    let mut module = Module::new(&format!("module_{}", description));
    
    // Add operations
    for i in 0..op_count {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), op_count, "Module should have correct number of operations for {}", description);
    assert_eq!(module.name, format!("module_{}", description));
}

// Test for recursive tensor type equality with various combinations
#[test]
fn test_recursive_tensor_type_equality() {
    // Different ways to construct the same nested type
    let type1 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    
    let type2 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    
    let type3 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I32),  // Different base type
            shape: vec![2],
        }),
        shape: vec![3],
    };
    
    let type4 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],  // Same as type1
        }),
        shape: vec![4],  // Different outer shape
    };
    
    // Same types should be equal
    assert_eq!(type1, type2);
    
    // Different base types should not be equal
    assert_ne!(type1, type3);
    
    // Different outer shapes should not be equal
    assert_ne!(type1, type4);
}