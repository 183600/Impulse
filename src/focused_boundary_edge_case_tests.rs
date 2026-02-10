//! Focused boundary edge case tests for the Impulse compiler
//! Tests specific edge cases and boundary conditions for IR components

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

#[test]
fn test_num_elements_with_overflow_protection() {
    // Test num_elements() method with values that would overflow
    // On 64-bit systems, use values that would exceed usize::MAX
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 3], // Would overflow
    };
    // Should return None due to overflow protection
    assert_eq!(value.num_elements(), None);
}

#[test]
fn test_num_elements_with_valid_large_values() {
    // Test num_elements() with large but valid values
    let value = Value {
        name: "large_valid".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000, 10], // 10 million elements
    };
    assert_eq!(value.num_elements(), Some(10_000_000));
}

#[test]
fn test_num_elements_with_zero_dimension() {
    // Test num_elements() when shape contains zero
    let value = Value {
        name: "zero_dim".to_string(),
        ty: Type::I32,
        shape: vec![100, 0, 50],
    };
    // Should return Some(0) when any dimension is zero
    assert_eq!(value.num_elements(), Some(0));
}

#[test]
fn test_num_elements_with_scalar() {
    // Test num_elements() for scalar (empty shape)
    let value = Value {
        name: "scalar".to_string(),
        ty: Type::F64,
        shape: vec![],
    };
    // Empty shape product is 1
    assert_eq!(value.num_elements(), Some(1));
}

#[test]
fn test_type_is_valid_with_deeply_nested_tensors() {
    // Test TypeExtensions trait with deeply nested tensors
    let deep_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2],
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    assert!(deep_nested.is_valid_type());
}

#[test]
fn test_module_with_empty_operation_list() {
    // Test module behavior with no operations
    let module = Module::new("empty_ops");
    assert_eq!(module.operations.len(), 0);
    assert!(module.operations.is_empty());
    // Should be able to clone without issues
    let cloned = module.clone();
    assert_eq!(cloned.name, "empty_ops");
    assert_eq!(cloned.operations.len(), 0);
}

#[test]
fn test_operation_with_empty_attributes() {
    // Test operation with empty but not null attributes
    let mut op = Operation::new("empty_attrs");
    op.attributes = HashMap::new();
    assert_eq!(op.attributes.len(), 0);
    assert!(op.attributes.is_empty());
    // Should be able to insert after initialization
    op.attributes.insert("key".to_string(), Attribute::Int(1));
    assert_eq!(op.attributes.len(), 1);
}

#[test]
fn test_value_with_max_usize_dimension() {
    // Test value with usize::MAX as a dimension
    let value = Value {
        name: "max_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };
    assert_eq!(value.shape[0], usize::MAX);
    // num_elements should return None due to overflow (usize::MAX * 2 would overflow)
    assert_eq!(value.num_elements(), None);
}

#[test]
fn test_attribute_with_nan_float() {
    // Test attribute with NaN float value
    let nan_attr = Attribute::Float(f64::NAN);
    match nan_attr {
        Attribute::Float(val) => {
            assert!(val.is_nan());
        }
        _ => panic!("Expected Float attribute"),
    }
}

#[test]
fn test_attribute_with_infinity_float() {
    // Test attribute with infinity float values
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    
    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected positive infinity"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected negative infinity"),
    }
}