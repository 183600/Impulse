//! Focused boundary case tests covering specific edge scenarios
//! Tests use standard library assert! and assert_eq! macros

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Value with maximum dimension count (stress test)
#[test]
fn test_value_max_dimensions() {
    let mut shape = Vec::new();
    for i in 1..=10 {
        shape.push(i);
    }
    
    let value = Value {
        name: "high_dim_tensor".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape.len(), 10);
    assert!(value.num_elements().is_some());
    assert_eq!(value.num_elements(), Some(3_628_800));
}

/// Test 2: Compiler with extremely large input buffer size
#[test]
fn test_compiler_large_buffer() {
    let mut compiler = ImpulseCompiler::new();
    let large_model = vec![0xFFu8; 100_000];
    
    let result = compiler.compile(&large_model, "cpu");
    // Should handle large buffers without panicking
    assert!(result.is_ok() || result.is_err());
    
    // Compiler should remain functional
    assert_eq!(compiler.passes.passes.len(), 0);
}

/// Test 3: Operation with very long operation name (boundary length)
#[test]
fn test_operation_long_name() {
    let long_name = "x".repeat(1024);
    let op = Operation::new(&long_name);
    
    assert_eq!(op.op_type.len(), 1024);
    assert_eq!(op.op_type, long_name);
}

/// Test 4: Module with maximum number of operations (stress test)
#[test]
fn test_module_many_operations() {
    let mut module = Module::new("large_module");
    
    for i in 0..1000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[999].op_type, "op_999");
}

/// Test 5: Attribute with integer overflow boundary values
#[test]
fn test_attribute_overflow_boundaries() {
    let boundary_attrs = vec![
        Attribute::Int(i64::MAX),
        Attribute::Int(i64::MIN),
        Attribute::Int(i64::MAX - 1),
        Attribute::Int(i64::MIN + 1),
        Attribute::Int(0),
    ];
    
    assert_eq!(boundary_attrs.len(), 5);
    
    if let Attribute::Int(val) = &boundary_attrs[0] {
        assert_eq!(*val, i64::MAX);
    }
    
    if let Attribute::Int(val) = &boundary_attrs[1] {
        assert_eq!(*val, i64::MIN);
    }
}

/// Test 6: Value with single dimension (1D tensor)
#[test]
fn test_value_single_dimension() {
    let test_sizes = [1, 10, 100, 1000, 10000];
    
    for size in test_sizes.iter() {
        let value = Value {
            name: "1d_tensor".to_string(),
            ty: Type::F32,
            shape: vec![*size],
        };
        
        assert_eq!(value.shape.len(), 1);
        assert_eq!(value.num_elements(), Some(*size));
    }
}

/// Test 7: Operation with empty attributes (edge case)
#[test]
fn test_operation_empty_attributes() {
    let op = Operation::new("empty_attrs");
    assert_eq!(op.attributes.len(), 0);
    assert!(op.attributes.is_empty());
}

/// Test 8: Value with very large tensor size (boundary check)
#[test]
fn test_value_large_tensor_size() {
    let large_dim = 1000usize;
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![large_dim, large_dim],
    };
    
    // Check if num_elements can handle the size
    let elements = value.num_elements();
    assert!(elements.is_some());
    assert_eq!(elements, Some(1_000_000));
}

/// Test 9: Module with duplicate input names (boundary case)
#[test]
fn test_module_duplicate_input_names() {
    let mut module = Module::new("duplicate_inputs");
    
    // Add inputs with same name
    for _ in 0..5 {
        module.inputs.push(Value {
            name: "shared_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
    }
    
    assert_eq!(module.inputs.len(), 5);
    for input in &module.inputs {
        assert_eq!(input.name, "shared_input");
    }
}

/// Test 10: Attribute with very long string value (boundary length)
#[test]
fn test_attribute_long_string() {
    let long_string = "a".repeat(10_000);
    let attr = Attribute::String(long_string.clone());
    
    if let Attribute::String(s) = attr {
        assert_eq!(s.len(), 10_000);
        assert_eq!(s, long_string);
    }
}