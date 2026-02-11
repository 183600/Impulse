//! Comprehensive boundary tests for edge cases with standard library assertions
//! 
//! This module contains additional test cases covering boundary conditions
//! that may not have been fully tested in existing test suites.

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value shape overflow protection using checked arithmetic
#[test]
fn test_value_shape_overflow_protection() {
    // Test that num_elements() returns None for shapes that would overflow
    let overflow_shape = vec![usize::MAX, 2];  // Would overflow on multiplication
    let value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: overflow_shape,
    };
    
    // The num_elements() method uses checked_mul and should return None
    assert_eq!(value.num_elements(), None);
    
    // Normal case should return Some
    let normal_value = Value {
        name: "normal_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    };
    assert_eq!(normal_value.num_elements(), Some(200));
}

/// Test 2: Negative integer attribute values (edge case for signed integers)
#[test]
fn test_negative_integer_attribute_values() {
    let mut op = Operation::new("negative_attrs_op");
    let mut attrs = HashMap::new();
    
    // Test various negative integer values
    attrs.insert("neg_one".to_string(), Attribute::Int(-1));
    attrs.insert("neg_large".to_string(), Attribute::Int(-1_000_000));
    attrs.insert("neg_max".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.get("neg_one"), Some(&Attribute::Int(-1)));
    assert_eq!(op.attributes.get("neg_large"), Some(&Attribute::Int(-1_000_000)));
    assert_eq!(op.attributes.get("neg_max"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("zero"), Some(&Attribute::Int(0)));
}

/// Test 3: Empty string vs non-empty string attribute distinction
#[test]
fn test_empty_vs_non_empty_string_attributes() {
    let empty_attr = Attribute::String(String::new());
    let non_empty_attr = Attribute::String("test".to_string());
    let whitespace_attr = Attribute::String("   ".to_string());
    
    match &empty_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 0);
            assert!(s.is_empty());
        },
        _ => panic!("Expected empty string"),
    }
    
    match &non_empty_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 4);
            assert!(!s.is_empty());
        },
        _ => panic!("Expected non-empty string"),
    }
    
    match &whitespace_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 3);
            assert!(!s.is_empty());
        },
        _ => panic!("Expected whitespace string"),
    }
    
    // They should not be equal
    assert_ne!(empty_attr, non_empty_attr);
}

/// Test 4: Very large float attribute values (infinity and near-infinity)
#[test]
fn test_large_float_attribute_values() {
    let max_float = Attribute::Float(f64::MAX);
    let large_positive = Attribute::Float(1e300);
    let very_small = Attribute::Float(1e-300);
    
    match max_float {
        Attribute::Float(val) => {
            assert!(val.is_finite());
            assert!(val > 0.0);
        },
        _ => panic!("Expected Float"),
    }
    
    match large_positive {
        Attribute::Float(val) => {
            assert!(val.is_finite());
            assert!(val > 1e299);
        },
        _ => panic!("Expected Float"),
    }
    
    match very_small {
        Attribute::Float(val) => {
            // May underflow to zero on some systems
            assert!(val >= 0.0 && val < 1e-299);
        },
        _ => panic!("Expected Float"),
    }
}

/// Test 5: Attribute array with duplicate values
#[test]
fn test_attribute_array_with_duplicates() {
    let dup_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Int(42),
        Attribute::Int(42),
    ]);
    
    match dup_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], arr[1]);
            assert_eq!(arr[1], arr[2]);
            assert_eq!(arr[0], Attribute::Int(42));
        },
        _ => panic!("Expected Array"),
    }
}

/// Test 6: Boolean attribute truth table behavior
#[test]
fn test_boolean_attribute_behavior() {
    let true_attr = Attribute::Bool(true);
    let false_attr = Attribute::Bool(false);
    
    match true_attr {
        Attribute::Bool(b) => assert!(b),
        _ => panic!("Expected Bool(true)"),
    }
    
    match false_attr {
        Attribute::Bool(b) => assert!(!b),
        _ => panic!("Expected Bool(false)"),
    }
    
    // Test inequality
    assert_ne!(true_attr, false_attr);
    
    // Test equality with same values
    assert_eq!(true_attr, Attribute::Bool(true));
    assert_eq!(false_attr, Attribute::Bool(false));
}

/// Test 7: Module with identical operation names (name collision scenario)
#[test]
fn test_module_with_identical_operation_names() {
    let mut module = Module::new("name_collision_test");
    
    // Add multiple operations with the same op_type
    for _ in 0..5 {
        let op = Operation::new("duplicate_op");
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    // All operations should have the same op_type
    for op in &module.operations {
        assert_eq!(op.op_type, "duplicate_op");
    }
    
    // Operations should be stored in order
    assert_eq!(module.operations[0].op_type, "duplicate_op");
    assert_eq!(module.operations[4].op_type, "duplicate_op");
}

/// Test 8: Empty array attribute vs absent attribute
#[test]
fn test_empty_array_vs_absent_attribute() {
    let mut op1 = Operation::new("empty_array_op");
    let mut op2 = Operation::new("no_attr_op");
    
    op1.attributes.insert("empty_list".to_string(), Attribute::Array(vec![]));
    
    // op1 has an empty array attribute
    assert!(op1.attributes.contains_key("empty_list"));
    match op1.attributes.get("empty_list") {
        Some(Attribute::Array(arr)) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected empty array"),
    }
    
    // op2 has no such attribute
    assert!(!op2.attributes.contains_key("empty_list"));
    assert_eq!(op2.attributes.get("empty_list"), None);
}

/// Test 9: Value with single-element shape (scalar tensor distinction)
#[test]
fn test_single_element_tensor_vs_scalar() {
    // Scalar: empty shape vector
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    
    // Single element tensor: shape = [1]
    let single_elem_tensor = Value {
        name: "single_elem".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    // Both should have 1 element
    assert_eq!(scalar.num_elements(), Some(1));
    assert_eq!(single_elem_tensor.num_elements(), Some(1));
    
    // But their shapes are different
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(single_elem_tensor.shape.len(), 1);
    assert_eq!(single_elem_tensor.shape[0], 1);
    
    // They are not equal
    assert_ne!(scalar.shape, single_elem_tensor.shape);
}

/// Test 10: Module with operations but no inputs or outputs (orphan module)
#[test]
fn test_module_orphan_operations() {
    let mut module = Module::new("orphan_module");
    
    // Add operations without connecting them to module inputs/outputs
    let mut op1 = Operation::new("internal_op1");
    op1.inputs.push(Value {
        name: "internal_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op1.outputs.push(Value {
        name: "internal_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op1);
    
    // Module has operations but no inputs/outputs
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
    
    // Operation has its own inputs/outputs
    assert_eq!(module.operations[0].inputs.len(), 1);
    assert_eq!(module.operations[0].outputs.len(), 1);
}