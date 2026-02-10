//! Critical edge case tests focusing on memory safety and overflow protection
//! These tests cover important boundary conditions for the Impulse compiler IR

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: num_elements() with shape containing zero
#[test]
fn test_num_elements_with_zero_dimension() {
    let value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(value.num_elements(), Some(0));
}

/// Test 2: num_elements() with empty shape (scalar)
#[test]
fn test_num_elements_scalar() {
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::I32,
        shape: vec![],
    };
    // Empty shape product is 1 (identity)
    assert_eq!(scalar.num_elements(), Some(1));
}

/// Test 3: num_elements() with potential overflow (usize::MAX * 2)
#[test]
fn test_num_elements_overflow_protection() {
    let value = Value {
        name: "overflow_risk".to_string(),
        ty: Type::F64,
        shape: vec![usize::MAX, 2],
    };
    // Should return None for overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 4: num_elements() with large but safe dimensions
#[test]
fn test_num_elements_large_safe() {
    let value = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000, 100], // 100M elements
    };
    assert_eq!(value.num_elements(), Some(100_000_000));
}

/// Test 5: Operation with special Unicode characters in op_type
#[test]
fn test_operation_with_unicode_op_type() {
    let op = Operation::new("注意力_加法_operation");
    assert_eq!(op.op_type, "注意力_加法_operation");
}

/// Test 6: Attribute with NaN float value
#[test]
fn test_attribute_nan_value() {
    let nan_attr = Attribute::Float(f64::NAN);
    let another_nan = Attribute::Float(f64::NAN);
    // NaN != NaN in floating point
    assert_ne!(nan_attr, another_nan);
}

/// Test 7: Attribute with infinity values
#[test]
fn test_attribute_infinity_values() {
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    let finite = Attribute::Float(1.0);
    
    assert_ne!(pos_inf, neg_inf);
    assert_ne!(pos_inf, finite);
    assert_ne!(neg_inf, finite);
}

/// Test 8: Module with operations having zero attributes
#[test]
fn test_module_operations_without_attributes() {
    let mut module = Module::new("no_attrs");
    let mut op = Operation::new("simple_op");
    op.attributes = HashMap::new();
    module.add_operation(op);
    
    assert_eq!(module.operations[0].attributes.len(), 0);
    assert!(module.operations[0].attributes.is_empty());
}

/// Test 9: Nested tensor type with zero in inner shape
#[test]
fn test_nested_tensor_with_zero_inner_shape() {
    let inner = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![0], // Zero inner shape
    };
    let outer = Type::Tensor {
        element_type: Box::new(inner),
        shape: vec![2, 3],
    };
    
    if let Type::Tensor { shape, .. } = outer {
        assert_eq!(shape, vec![2, 3]);
    }
}

/// Test 10: Module with empty string names
#[test]
fn test_module_with_empty_string_names() {
    let module = Module::new("");
    assert_eq!(module.name, "");
    
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
}