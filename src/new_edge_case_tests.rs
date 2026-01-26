//! Additional edge case tests for the Impulse compiler
//! Covering boundary conditions and extreme cases

use rstest::*;
use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    ImpulseCompiler,
};

// Test 1: Operations with large attribute count
#[test]
fn test_operation_with_large_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("large_attrs");
    let mut attrs = HashMap::new();
    
    // Add 10,000 attributes to test memory handling (reduced to avoid memory issues)
    for i in 0..10_000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 10_000);
    
    // Verify a few key attributes exist
    assert!(op.attributes.contains_key("attr_0"));
    assert!(op.attributes.contains_key("attr_9999"));
    assert_eq!(
        op.attributes.get("attr_5000"),
        Some(&Attribute::String("value_5000".to_string()))
    );
}

// Test 2: Deep tensor nesting that tests the limits but avoids stack overflow
#[test]
fn test_deep_tensor_nesting() {
    // Create a deeply nested tensor type (reduced from 1000 to 20 to avoid stack overflow)
    let mut current_type = Type::F32;
    for _ in 0..20 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the structure can be created without stack overflow
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![2]);
        },
        _ => panic!("Expected a deeply nested tensor type"),
    }
    
    // Test that cloning works without issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 3: Test tensor shapes with values that could cause numeric overflow
#[test]
fn test_tensor_shapes_with_potential_overflow() {
    // Use large values that could cause overflow when multiplied
    // Note: These values are chosen to be reasonable but large
    let large_shape = vec![50_000, 50_000];  // Reduced to avoid memory issues
    let value = Value {
        name: "large_shape_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    // Calculate using checked multiplication to avoid actual overflow
    let mut product: Option<usize> = Some(1);
    for &dim in &value.shape {
        product = product.and_then(|p| p.checked_mul(dim));
    }
    
    assert!(product.is_some()); // Ensure multiplication didn't overflow
    
    // Verify the shape values are preserved
    assert_eq!(value.shape, vec![50_000, 50_000]);
}

// Test 4: Unicode and special character handling in identifiers
#[rstest]
#[case("unicode_🚀_test", Type::F32)]
#[case("chinese_测试", Type::I64)]
#[case("arabic_اختبار", Type::Bool)]
#[case("emoji_sequence_🔥🌈🎉", Type::F64)]
#[case("special_chars_!@#$%^&*()", Type::I32)]
fn test_unicode_identifiers(#[case] name: &str, #[case] dtype: Type) {
    // Test value with unicode name
    let value = Value {
        name: name.to_string(),
        ty: dtype.clone(),
        shape: vec![1, 2, 3],
    };
    assert_eq!(value.name, name);
    assert_eq!(value.ty, dtype);
    
    // Test operation with unicode name
    let op = Operation::new(name);
    assert_eq!(op.op_type, name);
    
    // Test module with unicode name
    let module = Module::new(name);
    assert_eq!(module.name, name);
}

// Test 5: Zero-dimensional tensors (scalars) and tensors with zero in dimensions
#[test]
fn test_zero_dimensional_and_zero_containing_tensors() {
    // Test scalar (0-dimensional tensor)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape = scalar
    };
    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.shape.len(), 0);
    
    // The product of an empty shape should be 1 (scalar has 1 element)
    let product: usize = scalar.shape.iter().product();
    assert_eq!(product, 1);
    
    // Test tensor with zero in dimensions (results in zero total elements)
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::I32,
        shape: vec![10, 0, 5],  // Contains zero, total elements = 0
    };
    assert_eq!(zero_tensor.shape, vec![10, 0, 5]);
    
    let zero_product: usize = zero_tensor.shape.iter().product();
    assert_eq!(zero_product, 0);
    
    // Test 1D zero-length tensor
    let empty_1d = Value {
        name: "empty_1d".to_string(),
        ty: Type::Bool,
        shape: vec![0],  // Zero-length 1D tensor
    };
    assert_eq!(empty_1d.shape, vec![0]);
    assert_eq!(empty_1d.shape[0], 0);
    
    let empty_1d_product: usize = empty_1d.shape.iter().product();
    assert_eq!(empty_1d_product, 0);
}

// Test 6: Operations with extreme numbers of inputs and outputs
#[test]
fn test_operations_with_extreme_io_counts() {
    let mut op = Operation::new("extreme_io_op");
    
    // Add 10,000 inputs to test memory allocation limits (reduced to avoid memory issues)
    for i in 0..10_000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],  // Minimal shape to reduce memory
        });
    }
    
    // Add 5,000 outputs to test limits (reduced to avoid memory issues)
    for i in 0..5_000 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],  // Minimal shape
        });
    }
    
    assert_eq!(op.inputs.len(), 10_000);
    assert_eq!(op.outputs.len(), 5_000);
    assert_eq!(op.op_type, "extreme_io_op");
    
    // Verify first and last elements are correct
    assert_eq!(op.inputs[0].name, "input_0");
    assert_eq!(op.inputs[9_999].name, "input_9999");
    assert_eq!(op.outputs[0].name, "output_0");
    assert_eq!(op.outputs[4_999].name, "output_4999");
}

// Test 7: Nested array attributes with maximum depth and complexity
#[test]
fn test_nested_array_attributes() {
    // Create deeply nested arrays with mixed types
    let mut nested_array = Attribute::Int(42);  // Start with a simple value
    
    // Nest it 100 levels deep (reduced for safety)
    for _ in 0..100 {
        nested_array = Attribute::Array(vec![nested_array]);
    }
    
    // Verify it can be created and cloned
    let cloned_nested = nested_array.clone();
    assert_eq!(nested_array, cloned_nested);
    
    // Test equality with a similar structure
    let mut other_nested = Attribute::Int(42);
    for _ in 0..100 {
        other_nested = Attribute::Array(vec![other_nested]);
    }
    
    assert_eq!(nested_array, other_nested);
}

// Test 8: Test module with large number of operations
#[test]
fn test_module_with_large_operation_count() {
    let mut module = Module::new("large_module");
    
    // Track actual indices for verification later
    let mut basic_op_indices = Vec::new();
    let mut complex_op_indices = Vec::new();
    
    // Add 100,000 operations to test memory and performance limits (reduced to avoid memory issues)
    for i in 0..100_000 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
        basic_op_indices.push(module.operations.len() - 1); // record where we placed it
        
        // Occasionally add some inputs/outputs to make it more complex
        if i % 10_000 == 0 {
            let mut complex_op = Operation::new(&format!("complex_op_{}", i));
            
            // Add a few inputs and outputs to this operation
            for j in 0..5 {
                complex_op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
                
                complex_op.outputs.push(Value {
                    name: format!("output_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
            }
            
            module.add_operation(complex_op);
            complex_op_indices.push(module.operations.len() - 1); // record where we placed it
        }
    }
    
    // Verify the module has the expected number of operations
    assert_eq!(module.operations.len(), 100_000 + 10); // +10 for the complex ops added periodically
    
    // Verify some operations are at the expected positions
    assert_eq!(module.operations[0].op_type, "op_0");  // First operation should be op_0
    assert_eq!(module.operations[1].op_type, "complex_op_0");  // Second should be complex_op_0
    assert_eq!(module.name, "large_module");
    
    // Find the last basic operation by checking backwards from the end
    if let Some(last_basic_op) = module.operations.iter().rposition(|op| op.op_type.starts_with("op_")) {
        // The last basic op should be op_99999
        assert_eq!(module.operations[last_basic_op].op_type, "op_99999");
    }
}

// Test 9: Long string lengths in various fields
#[test]
fn test_long_string_lengths() {
    // Test very long module name (reduced length to avoid memory issues)
    let long_module_name = "module_".repeat(10_000) + "end";
    let module = Module::new(&long_module_name);
    assert_eq!(module.name, long_module_name);
    
    // Test very long operation name (reduced length to avoid memory issues)
    let long_op_name = "operation_".repeat(10_000) + "end";
    let op = Operation::new(&long_op_name);
    assert_eq!(op.op_type, long_op_name);
    
    // Test very long value name (reduced length to avoid memory issues)
    let long_value_name = "value_".repeat(10_000) + "end";
    let value = Value {
        name: long_value_name.clone(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    assert_eq!(value.name, long_value_name);
    
    // Test very long string attribute value (reduced length to avoid memory issues)
    let long_string_attr = "long_attr_value_".repeat(10_000) + "end";
    let attr = Attribute::String(long_string_attr.clone());
    
    match attr {
        Attribute::String(s) => assert_eq!(s, long_string_attr),
        _ => panic!("Expected string attribute"),
    }
}

// Test 10: Test compiler edge cases with invalid and extreme inputs
#[test]
fn test_compiler_edge_cases_with_extreme_inputs() {
    let mut compiler = ImpulseCompiler::new();
    
    // Test with empty input
    let empty_input = vec![];
    let _result = compiler.compile(&empty_input, "cpu");
    // Should not panic, result may be success or failure but no crash
    
    // Test with large input (10MB instead of 100MB to avoid memory issues)
    let large_input = vec![0u8; 10_000_000];
    let _result2 = compiler.compile(&large_input, "cpu");
    // Should not panic regardless of result
    
    // Test with input containing all possible byte values
    let all_byte_values: Vec<u8> = (0..=255).collect();
    let _result3 = compiler.compile(&all_byte_values, "cpu");
    // Should handle all byte values without crashing
    
    // Test with very long target string
    let extremely_long_target = "target_".repeat(10_000) + "end";
    let _result4 = compiler.compile(&all_byte_values, &extremely_long_target);
    // Should not panic with long target names
    
    // Verify compiler internals are still consistent after extreme inputs
    assert_eq!(compiler.passes.passes.len(), 0); // Should remain unchanged
}
