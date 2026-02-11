/// Focused edge case tests v3 - Additional boundary scenarios with standard library assertions
/// 
/// This module provides concise test cases for critical edge cases using assert! and assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with num_elements() overflow detection
#[test]
fn test_value_overflow_detection() {
    // Shape with product that would overflow usize on most systems
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };
    
    // num_elements should return None due to overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 2: Type::Tensor with empty shape (0-dimensional tensor)
#[test]
fn test_tensor_zero_dimensional() {
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![],
    };
    
    match tensor_type {
        Type::Tensor { shape, .. } => {
            assert!(shape.is_empty());
            assert_eq!(shape.len(), 0);
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 3: Module with name containing only whitespace
#[test]
fn test_module_whitespace_name() {
    let whitespace_names = [" ", "\t", "\n", "   ", "\t\t"];
    
    for name in whitespace_names.iter() {
        let module = Module::new(*name);
        assert_eq!(module.name, *name);
        assert!(!module.name.is_empty());
    }
}

/// Test 4: Value with negative-like pattern in type validation
#[test]
fn test_type_validation_negative_like() {
    // Test Type::I64 with extreme negative value as attribute
    let attr = Attribute::Int(i64::MIN);
    assert_eq!(attr, Attribute::Int(i64::MIN));
    
    // Test Type::F64 with negative infinity equivalent
    let negative_large = Attribute::Float(f64::MIN);
    match negative_large {
        Attribute::Float(val) => assert!(val < 0.0),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 5: Operation with empty op_type string
#[test]
fn test_operation_empty_op_type() {
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert!(op.op_type.is_empty());
    assert!(op.inputs.is_empty());
    assert!(op.outputs.is_empty());
    assert!(op.attributes.is_empty());
}

/// Test 6: Value with shape containing usize::MAX but valid product
#[test]
fn test_value_max_usize_dimension() {
    // Single dimension with MAX, other dimensions are 1 or 0
    let valid_shape1 = vec![usize::MAX, 1];
    let value1 = Value {
        name: "max_dim_one".to_string(),
        ty: Type::F32,
        shape: valid_shape1,
    };
    assert_eq!(value1.num_elements(), Some(usize::MAX));
    
    // Zero dimension makes product 0
    let zero_shape = vec![usize::MAX, 0];
    let value2 = Value {
        name: "max_dim_zero".to_string(),
        ty: Type::F64,
        shape: zero_shape,
    };
    assert_eq!(value2.num_elements(), Some(0));
}

/// Test 7: Attribute with extreme float values
#[test]
fn test_attribute_extreme_floats() {
    // Positive infinity
    let pos_inf = Attribute::Float(f64::INFINITY);
    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float(infinity)"),
    }
    
    // Negative infinity
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float(-infinity)"),
    }
    
    // NaN
    let nan_attr = Attribute::Float(f64::NAN);
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float(NaN)"),
    }
    
    // Verify NaN != NaN (floating point property)
    assert_ne!(nan_attr, nan_attr);
}

/// Test 8: Nested Tensor with recursive structure depth test
#[test]
fn test_nested_tensor_depth_4() {
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![1],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![1],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![1],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![1],
    };
    
    match &level4 {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![1]);
        }
        _ => panic!("Expected nested tensor"),
    }
    
    // Cloning deeply nested type should work
    let cloned = level4.clone();
    assert_eq!(level4, cloned);
}

/// Test 9: Attribute::Array with single element vs empty
#[test]
fn test_array_edge_sizes() {
    let empty_array = Attribute::Array(vec![]);
    match &empty_array {
        Attribute::Array(arr) => assert!(arr.is_empty()),
        _ => panic!("Expected empty array"),
    }
    
    let single_element = Attribute::Array(vec![Attribute::Int(42)]);
    match &single_element {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected single element array"),
    }
    
    // Compare empty arrays
    let empty_array2 = Attribute::Array(vec![]);
    assert_eq!(empty_array, empty_array2);
}

/// Test 10: Module operations with identical inputs but different names
#[test]
fn test_operations_identical_inputs_different_names() {
    let mut module = Module::new("test_ops");
    
    let shared_input = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![5],
    };
    
    let mut op1 = Operation::new("op1");
    op1.inputs.push(shared_input.clone());
    op1.outputs.push(Value {
        name: "out1".to_string(),
        ty: Type::F32,
        shape: vec![5],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(shared_input);
    op2.outputs.push(Value {
        name: "out2".to_string(),
        ty: Type::F32,
        shape: vec![5],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].op_type, "op1");
    assert_eq!(module.operations[1].op_type, "op2");
    
    // Both operations share the same input name
    assert_eq!(module.operations[0].inputs[0].name, module.operations[1].inputs[0].name);
}
