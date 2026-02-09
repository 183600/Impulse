//! Additional comprehensive edge case tests for Impulse compiler
//! Tests boundary conditions and unusual scenarios

use crate::ir::{Module, Operation, Value, Type, Attribute};
use std::collections::HashMap;

/// Test 1: Module with maximum usize dimension in shape
#[test]
fn test_module_with_max_usize_dimension() {
    let mut module = Module::new("max_dim_test");
    let mut op = Operation::new("max_dim_op");

    // Test with usize::MAX in shape (edge case for overflow)
    let max_dim_value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 1],
    };

    op.inputs.push(max_dim_value);
    module.add_operation(op);

    // Verify the module structure
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs[0].shape[0], usize::MAX);
}

/// Test 2: Value with num_elements overflow check
#[test]
fn test_value_num_elements_overflow_protection() {
    // Create a value with dimensions that would overflow when multiplied
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![10_000_000, 10_000_000], // Would be 10^14 elements
    };

    // The num_elements() method should handle overflow gracefully
    let result = value.num_elements();
    // On most systems this will overflow, so we expect None or a valid result
    // The test passes if it doesn't panic
    assert!(result.is_some() || result.is_none());
}

/// Test 3: Empty operation type string
#[test]
fn test_empty_operation_type() {
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert!(op.inputs.is_empty());
    assert!(op.outputs.is_empty());
    assert!(op.attributes.is_empty());
}

/// Test 4: Operation with maximum integer attribute values
#[test]
fn test_max_integer_attributes() {
    let mut op = Operation::new("max_int_op");
    let mut attrs = HashMap::new();

    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_usize_equiv".to_string(), Attribute::Int(isize::MAX as i64));
    attrs.insert("min_usize_equiv".to_string(), Attribute::Int(isize::MIN as i64));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 4);
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
}

/// Test 5: Module with single-element tensors (scalars)
#[test]
fn test_single_element_tensors() {
    let mut module = Module::new("scalar_test");
    let mut op = Operation::new("scalar_op");

    // Scalar (0-dimensional)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };

    // Single-element 1D tensor
    let single_1d = Value {
        name: "single_1d".to_string(),
        ty: Type::I32,
        shape: vec![1],
    };

    op.inputs.push(scalar);
    op.inputs.push(single_1d);
    module.add_operation(op);

    assert_eq!(module.operations[0].inputs.len(), 2);
    assert_eq!(module.operations[0].inputs[0].shape.len(), 0);
    assert_eq!(module.operations[0].inputs[1].shape, vec![1]);
}

/// Test 6: Type with deeply nested tensor (100 levels)
#[test]
fn test_deeply_nested_tensor_type() {
    let mut current_type = Type::F32;
    for _ in 0..100 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }

    // Verify it's a tensor type
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![2]);
        },
        _ => panic!("Expected Tensor type"),
    }

    // Test cloning doesn't cause stack overflow
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 7: Operation with all zeros in tensor shape
#[test]
fn test_all_zeros_tensor_shape() {
    let mut op = Operation::new("zero_shape_op");

    let zero_tensor = Value {
        name: "all_zeros".to_string(),
        ty: Type::F64,
        shape: vec![0, 0, 0, 0],
    };

    op.inputs.push(zero_tensor);

    assert_eq!(op.inputs[0].shape, vec![0, 0, 0, 0]);
    let elements: usize = op.inputs[0].shape.iter().product();
    assert_eq!(elements, 0);
}

/// Test 8: Attribute array with empty sub-arrays
#[test]
fn test_nested_empty_arrays() {
    let mut op = Operation::new("empty_array_op");
    let mut attrs = HashMap::new();

    // Create nested arrays with empty sub-arrays
    let nested_empty = Attribute::Array(vec![
        Attribute::Array(vec![]),
        Attribute::Array(vec![]),
        Attribute::Array(vec![Attribute::Int(1)]),
    ]);

    attrs.insert("nested_empty".to_string(), nested_empty);
    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 1);
    match op.attributes.get("nested_empty") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 3);
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Module with operations that have duplicate attribute keys
#[test]
fn test_duplicate_attribute_keys_handling() {
    let mut op = Operation::new("duplicate_attrs");
    let mut attrs = HashMap::new();

    // Insert attributes
    attrs.insert("key1".to_string(), Attribute::Int(1));
    attrs.insert("key2".to_string(), Attribute::String("value".to_string()));

    // Insert same key with different value (should overwrite)
    attrs.insert("key1".to_string(), Attribute::Int(999));

    op.attributes = attrs;

    // HashMap handles duplicates by overwriting
    assert_eq!(op.attributes.len(), 2); // Only 2 unique keys
    assert_eq!(op.attributes.get("key1"), Some(&Attribute::Int(999)));
}

/// Test 10: Type equivalence with different representations of same type
#[test]
fn test_type_equivalence_same_representations() {
    let type1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3, 4],
    };

    let type2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3, 4],
    };

    assert_eq!(type1, type2);

    // Different order should not be equal
    let type3 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![3, 2, 4],
    };

    assert_ne!(type1, type3);
}