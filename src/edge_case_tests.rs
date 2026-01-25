//! Additional edge case tests for the Impulse compiler
//! Focusing on boundary conditions, error handling, and extreme values

use rstest::*;
use crate::{
    ir::{Value, Type, Operation, Attribute, Module},
    ImpulseCompiler,
    utils::ir_utils
};

// Test 1: Edge cases with tensor dimensions containing zeros
#[rstest]
#[case(vec![], 1)]  // scalar has 1 element
#[case(vec![0], 0)] // zero dimension results in 0 elements
#[case(vec![0, 10], 0)] // any dimension of 0 results in 0 elements
#[case(vec![10, 0], 0)]
#[case(vec![1, 1, 0, 100], 0)]
#[case(vec![5, 5], 25)] // normal case
#[case(vec![1], 1)]     // single dimension
fn test_tensor_shape_products(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
    let actual_product: usize = shape.iter().product();
    assert_eq!(actual_product, expected_product);
}

// Test 2: Operations with many inputs but no outputs (sink operations)
#[test]
fn test_sink_operation() {
    let mut op = Operation::new("sink");
    
    // Add many inputs but no outputs
    for i in 0..10 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i + 1],
        });
    }
    
    assert_eq!(op.inputs.len(), 10);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.op_type, "sink");
}

// Test 3: Operations with no inputs but many outputs (source/generator operations)
#[test]
fn test_source_operation() {
    let mut op = Operation::new("source");
    
    // Add no inputs but many outputs
    for i in 0..10 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![i + 1],
        });
    }
    
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 10);
    assert_eq!(op.op_type, "source");
}

// Test 4: Extreme values in tensor shapes (close to usize::MAX)
#[test]
fn test_extreme_tensor_shapes() {
    // Test with the largest possible values that still allow calculation
    // (not actually using usize::MAX as it would cause overflow)
    let moderate_extreme = 1_000_000;
    let value = Value {
        name: "extreme_tensor".to_string(),
        ty: Type::F32,
        shape: vec![moderate_extreme, 1],
    };
    
    assert_eq!(value.shape[0], moderate_extreme);
    assert_eq!(value.shape[1], 1);
    
    // Calculate size without overflowing
    let size_check = value.shape[0].checked_mul(value.shape[1]);
    assert!(size_check.is_some());
}

// Test 5: Test deeply nested attribute arrays
#[test]
fn test_deeply_nested_attribute_arrays() {
    // Create a deeply nested array structure
    let mut nested = Attribute::Int(42);
    
    // Nest 5 levels deep
    for level in 1..=5 {
        nested = Attribute::Array(vec![
            Attribute::Int(level),
            nested,
        ]);
    }
    
    // Verify the structure can be matched without stack overflow
    match &nested {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 2);
            match &arr[0] {
                Attribute::Int(val) => assert_eq!(*val, 5), // Top level value
                _ => panic!("Expected Int at top level"),
            }
        },
        _ => panic!("Expected Array at top level"),
    }
}

// Test 6: Zero-sized tensors of different types
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_zero_sized_tensors(#[case] tensor_type: Type) {
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: tensor_type,
        shape: vec![10, 0, 5], // Contains 0, so total size is 0
    };
    
    assert_eq!(zero_tensor.shape, vec![10, 0, 5]);
    
    // Size calculation should be 0 regardless of type
    let size: usize = zero_tensor.shape.iter().product();
    assert_eq!(size, 0);
    
    // Test tensor size calculation function as well
    let calculated_size = ir_utils::calculate_tensor_size(&zero_tensor.ty, &zero_tensor.shape).unwrap();
    assert_eq!(calculated_size, 0);
}

// Test 7: Operations with maximum possible attributes
#[test]
fn test_operation_with_maximum_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attr_op");
    let mut attrs = HashMap::new();
    
    // Add many attributes
    for i in 0..100 {
        attrs.insert(
            format!("attribute_{:03}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 100);
    assert_eq!(op.op_type, "max_attr_op");
    
    // Verify we can access some attributes
    assert!(op.attributes.contains_key("attribute_000"));
    assert!(op.attributes.contains_key("attribute_099"));
}

// Test 8: Module with extreme numbers of operations
#[test]
fn test_module_with_many_operations() {
    let mut module = Module::new("many_ops_module");
    
    // Add many operations
    for i in 0..500 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 500);
    assert_eq!(module.name, "many_ops_module");
    
    // Verify first and last were added correctly
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[499].op_type, "op_499");
}

// Test 9: Empty name edge cases
#[rstest]
#[case("", "empty_name")]
#[case(" ", "space_name")]
#[case("\t", "tab_name")]
#[case("\n", "newline_name")]
fn test_empty_and_whitespace_names(#[case] name: &str, #[case] desc: &str) {
    // Test empty/whitespace operation name
    let op = Operation::new(name);
    assert_eq!(op.op_type, name);
    assert_eq!(op.inputs.len(), 0);
    
    // Test empty/whitespace value name
    let value = Value {
        name: name.to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    assert_eq!(value.name, name);
    
    // Test empty/whitespace module name
    let module = Module::new(name);
    assert_eq!(module.name, name);
}

// Test 10: Compiler behavior with invalid UTF-8-like byte sequences
#[test]
fn test_compiler_with_invalid_utf8_sequences() {
    let mut compiler = ImpulseCompiler::new();
    
    // Create byte sequence that looks like invalid UTF-8 continuation bytes
    let mut invalid_like_bytes = vec![0u8; 10];
    invalid_like_bytes[5] = 0xFF;  // Invalid UTF-8 start byte
    invalid_like_bytes[6] = 0xFE;  // Another invalid sequence
    
    // This should not panic, regardless of result
    let result = compiler.compile(&invalid_like_bytes, "cpu");
    
    // Validate that the result is either success or a proper error (not a panic)
    if result.is_err() {
        let err_msg = result.unwrap_err().to_string();
        assert!(!err_msg.is_empty());  // Should have some error message
    }
}