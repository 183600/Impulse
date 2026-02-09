//! Focused boundary tests for the Impulse compiler
//! These tests cover specific edge cases and boundary conditions

use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
use std::collections::HashMap;

#[test]
fn test_value_num_elements_with_zero_dimension() {
    // Test that num_elements returns None when shape contains zero
    let value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    
    // Shape with zero should have 0 total elements
    let result = value.num_elements();
    assert_eq!(result, Some(0));
}

#[test]
fn test_value_num_elements_with_empty_shape() {
    // Test scalar (empty shape)
    let value = Value {
        name: "scalar".to_string(),
        ty: Type::I32,
        shape: vec![],
    };
    
    // Empty shape represents a scalar with 1 element
    let result = value.num_elements();
    assert_eq!(result, Some(1));
}

#[test]
fn test_value_num_elements_potential_overflow() {
    // Test with dimensions that could cause overflow
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000],
    };
    
    // Should return Some if no overflow occurs
    let result = value.num_elements();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), 10_000_000_000);
}

#[test]
fn test_operation_attributes_with_min_max_values() {
    let mut op = Operation::new("minmax_test");
    
    let mut attrs = HashMap::new();
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
}

#[test]
fn test_nested_tensor_type_validation() {
    // Test nested tensor types with TypeExtensions trait
    let inner_type = Type::F32;
    let tensor_type = Type::Tensor {
        element_type: Box::new(inner_type),
        shape: vec![2, 3],
    };
    
    assert!(tensor_type.is_valid_type());
}

#[test]
fn test_deeply_nested_tensor_validation() {
    // Create a deeply nested tensor type
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };
    
    assert!(level3.is_valid_type());
}

#[test]
fn test_operation_with_single_dimensional_tensors() {
    let mut op = Operation::new("vector_ops");
    
    // Test with 1D tensors (vectors)
    op.inputs.push(Value {
        name: "vector_1d".to_string(),
        ty: Type::F32,
        shape: vec![1024],
    });
    
    op.outputs.push(Value {
        name: "output_1d".to_string(),
        ty: Type::F32,
        shape: vec![1024],
    });
    
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.inputs[0].shape, vec![1024]);
    assert_eq!(op.outputs[0].shape, vec![1024]);
}

#[test]
fn test_module_with_empty_operations() {
    let module = Module::new("empty_module");
    
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
}

#[test]
fn test_value_with_high_dimensional_tensor() {
    // Test with a tensor that has many dimensions
    let value = Value {
        name: "high_dim".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1, 1, 1, 1, 1],  // 8 dimensions
    };
    
    assert_eq!(value.shape.len(), 8);
    let result = value.num_elements();
    assert_eq!(result, Some(1));
}

#[test]
fn test_operation_attributes_with_empty_values() {
    let mut op = Operation::new("empty_attrs");
    
    let mut attrs = HashMap::new();
    attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    attrs.insert("zero_int".to_string(), Attribute::Int(0));
    attrs.insert("zero_float".to_string(), Attribute::Float(0.0));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    assert_eq!(op.attributes.get("empty_string"), Some(&Attribute::String("".to_string())));
    assert_eq!(op.attributes.get("empty_array"), Some(&Attribute::Array(vec![])));
}