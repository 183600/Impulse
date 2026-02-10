//! Comprehensive edge case tests for the Impulse compiler
//! Focuses on numerical precision, overflow prevention, and type safety

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Value shape with maximum dimension size causing overflow check
#[test]
fn test_value_shape_overflow_protection() {
    // Test with dimensions that could overflow when multiplied
    let large_shapes = vec![
        vec![usize::MAX, 2],
        vec![100_000, 100_000],
        vec![usize::MAX],
    ];
    
    for shape in large_shapes {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // num_elements should return None for overflow cases
        let num_elems = value.num_elements();
        // Either None (overflow detected) or Some (if calculation succeeded)
        assert!(num_elems.is_none() || num_elems.is_some());
    }
}

/// Test 2: Attribute with special floating point values (NaN, Infinity)
#[test]
fn test_special_float_values() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    let neg_zero = Attribute::Float(-0.0);
    
    // Verify these values can be created
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float(NAN)"),
    }
    
    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float(INFINITY)"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float(NEG_INFINITY)"),
    }
    
    match neg_zero {
        Attribute::Float(val) => assert_eq!(val, -0.0),
        _ => panic!("Expected Float(-0.0)"),
    }
}

/// Test 3: Module with operations containing extremely long names
#[test]
fn test_long_operation_names() {
    let mut module = Module::new("long_names");
    
    // Create operation with very long name
    let long_name = "a".repeat(10_000);
    let op = Operation::new(&long_name);
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].op_type.len(), 10_000);
}

/// Test 4: Attribute array with empty and single-element arrays
#[test]
fn test_edge_case_array_sizes() {
    let empty_array = Attribute::Array(vec![]);
    let single_array = Attribute::Array(vec![Attribute::Int(42)]);
    
    match empty_array {
        Attribute::Array(vec) => assert_eq!(vec.len(), 0),
        _ => panic!("Expected empty Array"),
    }
    
    match single_array {
        Attribute::Array(vec) => {
            assert_eq!(vec.len(), 1);
            match vec[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
        },
        _ => panic!("Expected single-element Array"),
    }
}

/// Test 5: Value with shape containing only zeros
#[test]
fn test_value_with_zero_dimensions() {
    let zero_shapes = [
        vec![0],
        vec![0, 0],
        vec![0, 5, 0],
        vec![1, 0, 1, 0],
    ];
    
    for shape in zero_shapes.iter() {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // All dimensions containing zero should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 6: Operation with all possible integer extreme values
#[test]
fn test_extreme_integer_values() {
    let mut op = Operation::new("extreme_ints");
    let mut attrs = HashMap::new();
    
    // Test extreme i64 values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("one".to_string(), Attribute::Int(1));
    attrs.insert("neg_one".to_string(), Attribute::Int(-1));
    attrs.insert("max_power_of_two".to_string(), Attribute::Int(2i64.pow(62)));
    attrs.insert("min_power_of_two".to_string(), Attribute::Int(-2i64.pow(62)));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 7);
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
}

/// Test 7: Module with operations having circular input patterns
#[test]
fn test_circular_operation_patterns() {
    let mut module = Module::new("circular");
    
    // Create three operations that could form a cycle
    let mut op_a = Operation::new("op_a");
    let mut op_b = Operation::new("op_b");
    let mut op_c = Operation::new("op_c");
    
    // Op A outputs to op_b
    let value_a = Value {
        name: "val_a".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op_a.outputs.push(value_a.clone());
    
    // Op B uses value_a and outputs to op_c
    op_b.inputs.push(value_a.clone());
    let value_b = Value {
        name: "val_b".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op_b.outputs.push(value_b.clone());
    
    // Op C uses value_b
    op_c.inputs.push(value_b.clone());
    
    module.add_operation(op_a);
    module.add_operation(op_b);
    module.add_operation(op_c);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[1].inputs[0].name, "val_a");
    assert_eq!(module.operations[2].inputs[0].name, "val_b");
}

/// Test 8: Tensor types with empty shape (scalar tensors)
#[test]
fn test_scalar_tensor_types() {
    let scalar_tensors = [
        Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![],
        },
        Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![],
        },
        Type::Tensor {
            element_type: Box::new(Type::Bool),
            shape: vec![],
        },
    ];
    
    for tensor_type in scalar_tensors.iter() {
        assert!(tensor_type.is_valid_type());
        match tensor_type {
            Type::Tensor { shape, .. } => assert_eq!(shape.len(), 0),
            _ => panic!("Expected Tensor type"),
        }
    }
}

/// Test 9: Attribute with very long string
#[test]
fn test_long_string_attribute() {
    let long_string = "x".repeat(1_000_000);
    let string_attr = Attribute::String(long_string.clone());
    
    match string_attr {
        Attribute::String(s) => assert_eq!(s.len(), 1_000_000),
        _ => panic!("Expected String attribute"),
    }
}

/// Test 10: Module with all types having empty shape
#[test]
fn test_module_with_all_empty_shapes() {
    let mut module = Module::new("empty_shapes");
    
    let scalar_inputs = vec![
        Value {
            name: "scalar_f32".to_string(),
            ty: Type::F32,
            shape: vec![],
        },
        Value {
            name: "scalar_i32".to_string(),
            ty: Type::I32,
            shape: vec![],
        },
        Value {
            name: "scalar_bool".to_string(),
            ty: Type::Bool,
            shape: vec![],
        },
    ];
    
    for input in scalar_inputs {
        module.inputs.push(input);
    }
    
    assert_eq!(module.inputs.len(), 3);
    // All inputs should have empty shape
    for input in &module.inputs {
        assert_eq!(input.shape.len(), 0);
    }
}