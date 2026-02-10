//! Final comprehensive edge tests - covering advanced boundary scenarios
//! 
//! This module provides comprehensive edge case coverage for the Impulse compiler IR,
//! focusing on numerical precision, overflow detection, and memory safety scenarios.

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Value with shape product approaching usize::MAX overflow
#[test]
fn test_value_shape_overflow_safety() {
    // Test that checked_mul handles potential overflow correctly
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 2], // Would overflow on unchecked multiplication
    };
    
    // num_elements uses checked_mul, should return None for overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 2: Attribute with negative and positive infinity
#[test]
fn test_infinity_attributes() {
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    
    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float with positive infinity"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float with negative infinity"),
    }
}

/// Test 3: Attribute with NaN values
#[test]
fn test_nan_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float with NaN"),
    }
    
    // NaN is not equal to itself
    match nan_attr {
        Attribute::Float(val) => {
            assert!(val != val); // NaN != NaN property
        },
        _ => panic!("Expected Float with NaN"),
    }
}

/// Test 4: Value with shape containing usize::MAX
#[test]
fn test_value_with_usize_max_dimension() {
    let value = Value {
        name: "max_dim".to_string(),
        ty: Type::I32,
        shape: vec![usize::MAX, 0], // Contains usize::MAX but product is 0
    };
    
    // Since one dimension is 0, product should be Some(0), not None
    assert_eq!(value.num_elements(), Some(0));
}

/// Test 5: Module with operations having cyclic-like naming patterns
#[test]
fn test_module_cyclic_naming_pattern() {
    let mut module = Module::new("cyclic_module");
    
    // Create operations with cyclic naming pattern
    let names = ["op_a", "op_b", "op_c", "op_a_2", "op_b_2", "op_c_2"];
    
    for name in &names {
        let mut op = Operation::new(name);
        op.inputs.push(Value {
            name: format!("{}_input", name),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 6);
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[3].op_type, "op_a_2");
}

/// Test 6: Operation with attribute keys that are very long
#[test]
fn test_operation_long_attribute_keys() {
    let mut op = Operation::new("long_keys_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with extremely long keys
    for i in 0..5 {
        let long_key = "attr_".repeat(1000) + &format!("_{}", i);
        attrs.insert(long_key, Attribute::Int(i));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
}

/// Test 7: Value with negative-like shape patterns (using large values)
#[test]
fn test_value_large_shape_values() {
    // Test with shapes that use large but valid usize values
    let shapes = vec![
        vec![100_000_000],  // 100 million elements
        vec![10_000, 10_000],  // 100 million elements
        vec![1_000, 1_000, 100],  // 100 million elements
    ];
    
    for shape in shapes {
        let value = Value {
            name: "large_shape".to_string(),
            ty: Type::F64,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        assert!(value.num_elements().is_some());
    }
}

/// Test 8: Type with deeply nested structure validation
#[test]
fn test_deeply_nested_type_validation() {
    // Create a 5-level nested tensor type using cloning to avoid move issues
    let level5 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level5.clone()),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level4.clone()),
        shape: vec![4],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level3.clone()),
        shape: vec![5],
    };
    let level1 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![6],
    };
    
    // All nested types should be valid
    assert!(level1.is_valid_type());
    assert!(level2.is_valid_type());
    assert!(level3.is_valid_type());
    assert!(level4.is_valid_type());
    assert!(level5.is_valid_type());
}

/// Test 9: Module with inputs/outputs of identical names
#[test]
fn test_module_identical_input_output_names() {
    let mut module = Module::new("same_names_module");
    
    let same_value = Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![100],
    };
    
    // Add same value as both input and output
    module.inputs.push(same_value.clone());
    module.outputs.push(same_value);
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, module.outputs[0].name);
    assert_eq!(module.inputs[0].name, "data");
}

/// Test 10: Value with empty name
#[test]
fn test_value_with_empty_name() {
    let value = Value {
        name: "".to_string(),  // Empty name
        ty: Type::F32,
        shape: vec![10],
    };
    
    assert_eq!(value.name, "");
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![10]);
}