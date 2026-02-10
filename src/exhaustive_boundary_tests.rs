//! Exhaustive boundary tests covering edge cases not covered by existing test suites
//! 
//! This module focuses on:
//! - Numeric boundary conditions (MIN/MAX values)
//! - String boundary cases (empty, special characters, Unicode)
//! - Array nesting and size limits
//! - Type conversion edge cases
//! - Memory and overflow scenarios

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with maximum safe dimension size before overflow
#[test]
fn test_value_with_max_safe_dimensions() {
    // Test with dimensions that are close to but won't overflow when multiplied
    // 65535 * 65535 = 4294836225 (fits in u64, but check usize behavior)
    let value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![65535, 1],  // Safe from overflow
    };
    
    // Should calculate without overflow
    let num_elements = value.num_elements();
    assert_eq!(num_elements, Some(65535));
}

/// Test 2: Attribute with infinity and NaN float values
#[test]
fn test_special_float_values() {
    let inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    let nan = Attribute::Float(f64::NAN);
    
    // Verify attributes are created (even with special values)
    match inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val > 0.0),
        _ => panic!("Expected positive infinity"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val < 0.0),
        _ => panic!("Expected negative infinity"),
    }
    
    match nan {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected NaN"),
    }
}

/// Test 3: Operation with extremely long attribute names
#[test]
fn test_operation_with_long_attribute_names() {
    let mut op = Operation::new("long_attr_op");
    let long_name = "a".repeat(1000);  // 1000 character name
    
    op.attributes.insert(long_name.clone(), Attribute::Int(42));
    
    assert!(op.attributes.contains_key(&long_name));
    assert_eq!(op.attributes.len(), 1);
}

/// Test 4: Module with operations having special characters in names
#[test]
fn test_module_with_special_char_operation_names() {
    let mut module = Module::new("special_chars");
    
    let special_names = [
        "op-with-dashes",
        "op_with_underscores",
        "op.with.dots",
        "op:with:colons",
        "op/with/slashes",
    ];
    
    for name in special_names.iter() {
        let op = Operation::new(name);
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for (i, name) in special_names.iter().enumerate() {
        assert_eq!(module.operations[i].op_type, *name);
    }
}

/// Test 5: Value with shape containing only zeros
#[test]
fn test_value_with_all_zero_dimensions() {
    let value = Value {
        name: "all_zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0, 0],
    };
    
    assert_eq!(value.num_elements(), Some(0));
    assert_eq!(value.shape.iter().product::<usize>(), 0);
}

/// Test 6: Attribute array with alternating types
#[test]
fn test_attribute_array_with_alternating_types() {
    let alternating = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(1.5),
        Attribute::Int(2),
        Attribute::Float(2.5),
        Attribute::Int(3),
        Attribute::Float(3.5),
    ]);
    
    match alternating {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 6);
            // Verify pattern: Int, Float, Int, Float, Int, Float
            match &arr[0] { Attribute::Int(1) => (), _ => panic!() }
            match &arr[1] { Attribute::Float(v) if (v - 1.5).abs() < 1e-9 => (), _ => panic!() }
            match &arr[2] { Attribute::Int(2) => (), _ => panic!() }
            match &arr[3] { Attribute::Float(v) if (v - 2.5).abs() < 1e-9 => (), _ => panic!() }
            match &arr[4] { Attribute::Int(3) => (), _ => panic!() }
            match &arr[5] { Attribute::Float(v) if (v - 3.5).abs() < 1e-9 => (), _ => panic!() }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 7: Module with single operation that has many inputs and outputs
#[test]
fn test_operation_with_many_inputs_outputs() {
    let mut op = Operation::new("multi_io_op");
    
    // Add 100 inputs
    for i in 0..100 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    // Add 100 outputs
    for i in 0..100 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(op.inputs.len(), 100);
    assert_eq!(op.outputs.len(), 100);
    assert_eq!(op.inputs[0].name, "input_0");
    assert_eq!(op.inputs[99].name, "input_99");
    assert_eq!(op.outputs[0].name, "output_0");
    assert_eq!(op.outputs[99].name, "output_99");
}

/// Test 8: Attribute with empty and single-element arrays
#[test]
fn test_empty_and_single_element_arrays() {
    let empty = Attribute::Array(vec![]);
    let single = Attribute::Array(vec![Attribute::Int(1)]);
    
    match empty {
        Attribute::Array(arr) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected empty array"),
    }
    
    match single {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(1) => (),
                _ => panic!("Expected Int(1)"),
            }
        }
        _ => panic!("Expected single-element array"),
    }
}

/// Test 9: Value with very small non-zero dimensions
#[test]
fn test_value_with_small_dimensions() {
    let shapes = [
        vec![1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![1, 1, 1, 1],
        vec![1, 1, 1, 1, 1],
    ];
    
    for shape in shapes.iter() {
        let value = Value {
            name: "small_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.num_elements(), Some(1));
        assert_eq!(value.shape.iter().product::<usize>(), 1);
    }
}

/// Test 10: Module with operations using all integer boundary values
#[test]
fn test_operation_with_integer_boundaries() {
    let mut op = Operation::new("int_boundary_op");
    let mut attrs = HashMap::new();
    
    // Test various integer boundary values
    attrs.insert("i64_max".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("i64_min".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("i32_max".to_string(), Attribute::Int(i32::MAX as i64));
    attrs.insert("i32_min".to_string(), Attribute::Int(i32::MIN as i64));
    attrs.insert("u32_max".to_string(), Attribute::Int(u32::MAX as i64));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("one".to_string(), Attribute::Int(1));
    attrs.insert("neg_one".to_string(), Attribute::Int(-1));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 8);
    assert_eq!(op.attributes.get("i64_max"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("i64_min"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("zero"), Some(&Attribute::Int(0)));
}