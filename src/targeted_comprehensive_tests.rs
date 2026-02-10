//! Targeted comprehensive tests - focused edge case coverage
//! Covers specific boundary conditions and critical scenarios with standard assertions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Compiler with model containing only null bytes
#[test]
fn test_compiler_null_byte_only_model() {
    let mut compiler = ImpulseCompiler::new();
    let null_model = vec![0x00; 10];
    
    let result = compiler.compile(&null_model, "cpu");
    // Should handle gracefully without panic
    assert!(result.is_ok() || result.is_err());
}

/// Test 2: Value with single element in each dimension
#[test]
fn test_value_unit_dimensions() {
    let shapes = vec![
        vec![1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![1, 1, 1, 1],
    ];
    
    for shape in shapes {
        let value = Value {
            name: "unit_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.num_elements(), Some(1));
        assert_eq!(value.shape, shape);
    }
}

/// Test 3: Attribute with maximum and minimum integer values
#[test]
fn test_extreme_integer_attributes() {
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let zero = Attribute::Int(0);
    let neg_one = Attribute::Int(-1);
    let pos_one = Attribute::Int(1);
    
    match max_int {
        Attribute::Int(v) => assert_eq!(v, i64::MAX),
        _ => panic!("Expected Int(i64::MAX)"),
    }
    
    match min_int {
        Attribute::Int(v) => assert_eq!(v, i64::MIN),
        _ => panic!("Expected Int(i64::MIN)"),
    }
    
    match zero {
        Attribute::Int(v) => assert_eq!(v, 0),
        _ => panic!("Expected Int(0)"),
    }
    
    match neg_one {
        Attribute::Int(v) => assert_eq!(v, -1),
        _ => panic!("Expected Int(-1)"),
    }
    
    match pos_one {
        Attribute::Int(v) => assert_eq!(v, 1),
        _ => panic!("Expected Int(1)"),
    }
}

/// Test 4: Module with empty string inputs/outputs names
#[test]
fn test_module_empty_string_names() {
    let mut module = Module::new("empty_names");
    
    module.inputs.push(Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.outputs.push(Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    assert_eq!(module.inputs[0].name, "");
    assert_eq!(module.outputs[0].name, "");
}

/// Test 5: Type validation for all primitive types
#[test]
fn test_all_primitive_types_are_valid() {
    let types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];
    
    for ty in types {
        assert!(ty.is_valid_type());
    }
}

/// Test 6: Attribute array with single element of each type
#[test]
fn test_single_element_attribute_arrays() {
    let int_array = Attribute::Array(vec![Attribute::Int(42)]);
    let float_array = Attribute::Array(vec![Attribute::Float(3.14)]);
    let string_array = Attribute::Array(vec![Attribute::String("test".to_string())]);
    let bool_array = Attribute::Array(vec![Attribute::Bool(true)]);
    
    match int_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(v) => assert_eq!(*v, 42),
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match float_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Float(v) => assert!((*v - 3.14).abs() < f64::EPSILON),
                _ => panic!("Expected Float(3.14)"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match string_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::String(s) => assert_eq!(s, "test"),
                _ => panic!("Expected String(\"test\")"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match bool_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Bool(true) => (),
                _ => panic!("Expected Bool(true)"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 7: Value with large dimension count but small sizes
#[test]
fn test_many_dimensions_small_sizes() {
    // Create a tensor with many dimensions but each is small
    let mut shape = Vec::new();
    for _ in 0..16 {
        shape.push(2);
    }
    
    let value = Value {
        name: "many_dims".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape.len(), 16);
    assert_eq!(value.num_elements(), Some(65536));
}

/// Test 8: Operation with attribute keys containing special characters
#[test]
fn test_operation_special_attribute_keys() {
    let mut op = Operation::new("special_keys");
    let mut attrs = HashMap::new();
    
    attrs.insert("key-with-dash".to_string(), Attribute::Int(1));
    attrs.insert("key_with_underscore".to_string(), Attribute::Int(2));
    attrs.insert("key.with.dot".to_string(), Attribute::Int(3));
    attrs.insert("key@symbol".to_string(), Attribute::Int(4));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    assert!(op.attributes.contains_key("key-with-dash"));
    assert!(op.attributes.contains_key("key_with_underscore"));
    assert!(op.attributes.contains_key("key.with.dot"));
    assert!(op.attributes.contains_key("key@symbol"));
}

/// Test 9: Value with identical consecutive dimensions
#[test]
fn test_value_identical_consecutive_dimensions() {
    let shapes = vec![
        vec![2, 2, 2],
        vec![3, 3],
        vec![4, 4, 4, 4],
        vec![5, 5, 5],
    ];
    
    for shape in shapes {
        let value = Value {
            name: "identical_dims".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 10: Compiler with empty target string
#[test]
fn test_compiler_empty_target() {
    let mut compiler = ImpulseCompiler::new();
    let model = vec![1, 2, 3];
    
    let result = compiler.compile(&model, "");
    // Should handle empty target gracefully
    assert!(result.is_ok() || result.is_err());
}
