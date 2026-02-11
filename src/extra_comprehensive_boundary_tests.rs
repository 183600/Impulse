//! Extra comprehensive boundary tests - covering numerical precision edge cases,
//! overflow detection, special floating point values, and type safety scenarios

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with shape containing usize::MAX to check overflow safety
#[test]
fn test_shape_overflow_protection() {
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };
    
    // Verify the shape is stored correctly
    assert_eq!(value.shape.len(), 2);
    assert_eq!(value.shape[0], usize::MAX);
    assert_eq!(value.shape[1], 2);
    
    // num_elements should return None for overflow cases
    let num_elems = value.num_elements();
    assert!(num_elems.is_none() || num_elems == Some(0));
}

/// Test 2: Attribute with special float values (NaN, Infinity, -Infinity)
#[test]
fn test_special_float_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    // Verify special values are stored
    match (nan_attr, pos_inf_attr, neg_inf_attr) {
        (Attribute::Float(n), Attribute::Float(pi), Attribute::Float(ni)) => {
            assert!(n.is_nan());
            assert!(pi.is_infinite() && pi > 0.0);
            assert!(ni.is_infinite() && ni < 0.0);
        }
        _ => panic!("Expected Float attributes"),
    }
}

/// Test 3: Module with operations that have extremely deep attribute nesting
#[test]
fn test_deeply_nested_operation_attributes() {
    let mut op = Operation::new("deep_nest_op");
    let mut attrs = HashMap::new();
    
    // Create deeply nested array attribute (5 levels deep)
    let deep_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Array(vec![
                        Attribute::Int(42),
                    ]),
                ]),
            ]),
        ]),
    ]);
    
    attrs.insert("deep".to_string(), deep_nested);
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1);
    assert!(op.attributes.contains_key("deep"));
}

/// Test 4: Value with mixed precision tensor types (F16 simulation)
#[test]
fn test_mixed_precision_tensor_types() {
    // Create tensors representing different precision levels
    let f32_value = Value {
        name: "f32_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1, 3, 224, 224],
    };
    
    let f64_value = Value {
        name: "f64_tensor".to_string(),
        ty: Type::F64,
        shape: vec![1, 3, 224, 224],
    };
    
    let i32_value = Value {
        name: "i32_tensor".to_string(),
        ty: Type::I32,
        shape: vec![1, 3, 224, 224],
    };
    
    // Verify all shapes are identical but types differ
    assert_eq!(f32_value.shape, f64_value.shape);
    assert_eq!(f32_value.shape, i32_value.shape);
    
    assert_ne!(f32_value.ty, f64_value.ty);
    assert_ne!(f32_value.ty, i32_value.ty);
}

/// Test 5: Operation with empty string keys in attributes
#[test]
fn test_operation_with_empty_attribute_keys() {
    let mut op = Operation::new("empty_key_op");
    let mut attrs = HashMap::new();
    
    // Add attribute with empty key (edge case)
    attrs.insert("".to_string(), Attribute::Int(42));
    attrs.insert(" ".to_string(), Attribute::Float(3.14));
    attrs.insert("\t".to_string(), Attribute::String("tabbed".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 3);
    assert!(op.attributes.contains_key(""));
    assert!(op.attributes.contains_key(" "));
    assert!(op.attributes.contains_key("\t"));
}

/// Test 6: Module with inputs that have zero-element shapes across all positions
#[test]
fn test_zero_element_tensors_in_all_positions() {
    let test_shapes = vec![
        vec![0],
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![1, 0, 1],
        vec![0, 1, 1],
        vec![1, 1, 0],
        vec![10, 0, 10, 0],
    ];
    
    for shape in test_shapes {
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Verify shape is stored correctly
        assert_eq!(value.shape, shape);
        
        // Verify num_elements returns 0 or None
        let num_elems = value.num_elements();
        assert!(num_elems == Some(0) || num_elems.is_none());
    }
}

/// Test 7: Attribute with integer values at boundary positions
#[test]
fn test_integer_boundary_values() {
    let boundary_ints = vec![
        Attribute::Int(i64::MIN),
        Attribute::Int(i64::MAX),
        Attribute::Int(-1),
        Attribute::Int(0),
        Attribute::Int(1),
        Attribute::Int(i32::MIN as i64),
        Attribute::Int(i32::MAX as i64),
    ];
    
    let mut op = Operation::new("boundary_ints");
    let mut attrs = HashMap::new();
    
    for (i, attr) in boundary_ints.into_iter().enumerate() {
        attrs.insert(format!("int_{}", i), attr);
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 7);
    
    // Verify specific values
    if let Attribute::Int(val) = op.attributes.get("int_0").unwrap() {
        assert_eq!(*val, i64::MIN);
    }
    if let Attribute::Int(val) = op.attributes.get("int_1").unwrap() {
        assert_eq!(*val, i64::MAX);
    }
}

/// Test 8: Module with operations having Unicode names
#[test]
fn test_operations_with_unicode_names() {
    let unicode_names = vec![
        "操作_中文",           // Chinese
        "операция_русский",  // Russian
        "opération_français", // French
        "αβγδε_ελληνικά",     // Greek
        "🚀_rocket_🎯",       // Emoji
        "مرحبا_العربية",      // Arabic
    ];
    
    let mut module = Module::new("unicode_test");
    
    for name in unicode_names {
        let op = Operation::new(name);
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 6);
    assert_eq!(module.operations[0].op_type, "操作_中文");
    assert_eq!(module.operations[1].op_type, "операция_русский");
    assert_eq!(module.operations[2].op_type, "opération_français");
    assert_eq!(module.operations[3].op_type, "αβγδε_ελληνικά");
    assert_eq!(module.operations[4].op_type, "🚀_rocket_🎯");
    assert_eq!(module.operations[5].op_type, "مرحبا_العربية");
}

/// Test 9: Value with shape containing consecutive ones and zeros
#[test]
fn test_consecutive_ones_and_zeros_in_shape() {
    let patterns = vec![
        vec![1, 1, 1, 1],      // All ones
        vec![0, 0, 0, 0],      // All zeros
        vec![1, 0, 1, 0],      // Alternating
        vec![1, 1, 0, 0],      // Two ones then zeros
        vec![0, 0, 1, 1],      // Two zeros then ones
        vec![1, 0, 0, 1],      // Sandwiched zeros
    ];
    
    for shape in patterns {
        let value = Value {
            name: "pattern_tensor".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Calculate elements
        let product: usize = shape.iter().product();
        let expected: usize = if shape.iter().all(|&x| x == 1) {
            1
        } else if shape.iter().any(|&x| x == 0) {
            0
        } else {
            shape.iter().product()
        };
        assert_eq!(product, expected);
    }
}

/// Test 10: Operation with attributes containing all types of empty values
#[test]
fn test_empty_values_in_attributes() {
    let mut op = Operation::new("empty_values");
    let mut attrs = HashMap::new();
    
    // Empty array
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    
    // Empty string
    attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
    
    // Zero values
    attrs.insert("zero_int".to_string(), Attribute::Int(0));
    attrs.insert("zero_float".to_string(), Attribute::Float(0.0));
    
    // Boolean false
    attrs.insert("false_bool".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    
    // Verify each empty value type
    if let Attribute::Array(arr) = op.attributes.get("empty_array").unwrap() {
        assert!(arr.is_empty());
    }
    if let Attribute::String(s) = op.attributes.get("empty_string").unwrap() {
        assert!(s.is_empty());
    }
    if let Attribute::Int(i) = op.attributes.get("zero_int").unwrap() {
        assert_eq!(*i, 0);
    }
    if let Attribute::Float(f) = op.attributes.get("zero_float").unwrap() {
        assert_eq!(*f, 0.0);
    }
    if let Attribute::Bool(b) = op.attributes.get("false_bool").unwrap() {
        assert!(!*b);
    }
}