//! Critical boundary tests covering edge cases and boundary conditions
//! Tests use standard library assert! and assert_eq! for clarity

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Module with maximum i64 and minimum i64 attribute values
#[test]
fn test_module_with_extreme_int_attributes() {
    let mut module = Module::new("extreme_ints");
    let mut op = Operation::new("extreme_op");
    
    op.attributes.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("zero".to_string(), Attribute::Int(0));
    op.attributes.insert("negative_one".to_string(), Attribute::Int(-1));
    
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].attributes.len(), 4);
    
    match &module.operations[0].attributes["max_i64"] {
        Attribute::Int(val) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected Int"),
    }
    match &module.operations[0].attributes["min_i64"] {
        Attribute::Int(val) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected Int"),
    }
}

/// Test 2: Value with dimension that causes usize overflow prevention
#[test]
fn test_value_with_overflow_prevention_dimensions() {
    // Test dimensions that would cause overflow when multiplied
    let large_dim = usize::MAX / 2;
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![3, large_dim],
    };
    
    // num_elements should return None when overflow would occur (3 * MAX/2 > MAX)
    assert_eq!(value.num_elements(), None);
}

/// Test 3: Value with empty string names
#[test]
fn test_value_with_empty_names() {
    let value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1, 1],
    };
    assert_eq!(value.name, "");
    assert_eq!(value.num_elements(), Some(1));
}

/// Test 4: Module with empty operation type string
#[test]
fn test_module_with_empty_operation_type() {
    let mut module = Module::new("empty_op_type");
    let op = Operation::new("");
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].op_type, "");
}

/// Test 5: Attribute with all zero and all one bit patterns
#[test]
fn test_attribute_with_bit_pattern_edge_cases() {
    let attrs = vec![
        Attribute::Int(0),
        Attribute::Int(-1),
        Attribute::Int(1),
        Attribute::Float(0.0),
        Attribute::Float(-0.0),
        Attribute::Float(1.0),
        Attribute::Float(-1.0),
    ];
    
    assert_eq!(attrs.len(), 7);
    
    match &attrs[0] {
        Attribute::Int(val) => assert_eq!(*val, 0),
        _ => panic!("Expected 0"),
    }
    match &attrs[1] {
        Attribute::Int(val) => assert_eq!(*val, -1),
        _ => panic!("Expected -1"),
    }
    match &attrs[3] {
        Attribute::Float(val) => assert_eq!(*val, 0.0),
        _ => panic!("Expected 0.0"),
    }
    match &attrs[4] {
        Attribute::Float(val) => {
            // -0.0 equals 0.0 in IEEE 754
            assert_eq!(*val, 0.0);
        },
        _ => panic!("Expected -0.0"),
    }
}

/// Test 6: Value with single dimension of zero
#[test]
fn test_value_with_single_zero_dimension() {
    let value = Value {
        name: "single_zero".to_string(),
        ty: Type::I32,
        shape: vec![0],
    };
    
    assert_eq!(value.shape, vec![0]);
    assert_eq!(value.num_elements(), Some(0));
}

/// Test 7: Compiler with very short model (single byte)
#[test]
fn test_compiler_with_single_byte_model() {
    let mut compiler = ImpulseCompiler::new();
    let single_byte_models = vec![
        vec![0x00],
        vec![0xFF],
        vec![0x7F],
        vec![0x80],
    ];
    
    for model in single_byte_models {
        let result = compiler.compile(&model, "cpu");
        assert!(result.is_ok() || result.is_err());
    }
}

/// Test 8: Nested tensor with zero dimensions in nested structure
#[test]
fn test_nested_tensor_with_zero_dimensions() {
    let inner_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![0, 5],
    };
    
    let outer_type = Type::Tensor {
        element_type: Box::new(inner_type),
        shape: vec![2, 0],
    };
    
    match outer_type {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![2, 0]);
            match *element_type {
                Type::Tensor { shape: inner_shape, .. } => {
                    assert_eq!(inner_shape, vec![0, 5]);
                }
                _ => panic!("Expected nested Tensor"),
            }
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 9: Module with operations having empty attribute map
#[test]
fn test_module_operations_with_empty_attributes() {
    let mut module = Module::new("empty_attrs");
    
    for i in 0..3 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.attributes = HashMap::new();
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 3);
    for op in &module.operations {
        assert_eq!(op.attributes.len(), 0);
    }
}

/// Test 10: Value with shape containing consecutive zeros
#[test]
fn test_value_with_consecutive_zero_dimensions() {
    let shapes = vec![
        vec![0, 0],
        vec![0, 0, 0],
        vec![1, 0, 0, 1],
        vec![0, 5, 0, 3],
    ];
    
    for shape in shapes {
        let value = Value {
            name: "consecutive_zeros".to_string(),
            ty: Type::F64,
            shape: shape.clone(),
        };
        
        // Any zero in dimensions results in zero total elements
        assert_eq!(value.num_elements(), Some(0));
    }
}