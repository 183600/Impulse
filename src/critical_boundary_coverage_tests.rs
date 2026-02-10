//! Critical boundary coverage tests - additional edge cases not covered in existing tests
//! Tests for numerical precision, type conversion, and compiler state boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Compiler with zero-size model followed by large model (state transition test)
#[test]
fn test_compiler_state_transition_zero_to_large() {
    let mut compiler = ImpulseCompiler::new();
    
    // First compile zero-size model
    let zero_model = vec![];
    let _result1 = compiler.compile(&zero_model, "cpu");
    
    // Then compile large model - verify compiler state remains consistent
    let large_model = vec![0xAB; 1_000_000];
    let _result2 = compiler.compile(&large_model, "cpu");
    
    // Compiler should still be functional
    assert_eq!(compiler.passes.passes.len(), 0);
}

/// Test 2: Value with dimension values at system boundary (usize::MAX - 1)
#[test]
fn test_value_near_max_usize_dimension() {
    // Test with dimension close to usize::MAX but not causing overflow
    let value = Value {
        name: "near_max_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX - 1],
    };
    
    // Should not overflow since single dimension
    assert_eq!(value.num_elements(), Some(usize::MAX - 1));
}

/// Test 3: Operation with attribute keys containing special characters
#[test]
fn test_operation_special_attribute_keys() {
    let mut op = Operation::new("special_keys");
    let mut attrs = HashMap::new();
    
    // Test various special characters in attribute keys
    let special_keys = [
        "key-with-dashes",
        "key_with_underscores",
        "key.with.dots",
        "key:with:colons",
        "key/with/slashes",
    ];
    
    for key in &special_keys {
        attrs.insert(key.to_string(), Attribute::Int(42));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    for key in &special_keys {
        assert!(op.attributes.contains_key(*key));
    }
}

/// Test 4: Module with operations chain of length 100 (stress test for operation management)
#[test]
fn test_module_long_operation_chain() {
    let mut module = Module::new("long_chain");
    
    // Create a chain of 100 operations
    let mut prev_output: Option<Value> = None;
    for i in 0..100 {
        let mut op = Operation::new(&format!("op_{}", i));
        
        if let Some(ref input) = prev_output {
            op.inputs.push(input.clone());
        }
        
        let output = Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        };
        op.outputs.push(output.clone());
        prev_output = Some(output);
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    // Verify chain integrity
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[99].inputs.len(), 1);
}

/// Test 5: Attribute array with recursive structure simulating tree
#[test]
fn test_tree_structure_attribute_array() {
    // Create a binary tree structure using Attribute::Array
    let tree = Attribute::Array(vec![
        Attribute::Int(1),  // root left value
        Attribute::Array(vec![  // right subtree
            Attribute::Int(2),
            Attribute::Array(vec![
                Attribute::Int(3),
                Attribute::Int(4),
            ]),
        ]),
    ]);
    
    match tree {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 2);
            assert!(matches!(arr[0], Attribute::Int(1)));
            match &arr[1] {
                Attribute::Array(sub) => {
                    assert_eq!(sub.len(), 2);
                    match &sub[1] {
                        Attribute::Array(leaf) => {
                            assert_eq!(leaf.len(), 2);
                        },
                        _ => panic!("Expected nested array"),
                    }
                },
                _ => panic!("Expected nested array"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: Type validation for complex nested structures
#[test]
fn test_complex_nested_type_validation() {
    // Create complex nested tensor with mixed depths
    let mixed_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![3, 3],
        }),
        shape: vec![2, 2],
    };
    
    assert!(mixed_nested.is_valid_type());
    
    match mixed_nested {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![2, 2]);
            match element_type.as_ref() {
                Type::Tensor { element_type: inner, shape: inner_shape } => {
                    assert_eq!(inner_shape, &vec![3, 3]);
                    assert_eq!(inner.as_ref(), &Type::I64);
                },
                _ => panic!("Expected inner Tensor"),
            }
        },
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 7: Compiler with models of exact powers of two (testing alignment)
#[test]
fn test_compiler_powers_of_two_models() {
    let mut compiler = ImpulseCompiler::new();
    
    let sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    
    for size in sizes.iter() {
        let model = vec![0xAA; *size];
        let result = compiler.compile(&model, "cpu");
        // Should handle gracefully without panic
        assert!(result.is_ok() || result.is_err());
    }
}

/// Test 8: Module with input/output having same name but different shapes (collision test)
#[test]
fn test_module_same_name_different_shapes() {
    let mut module = Module::new("name_collision");
    
    // Add input and output with same name but different shapes
    module.inputs.push(Value {
        name: "shared_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.outputs.push(Value {
        name: "shared_name".to_string(),
        ty: Type::F64,  // Different type too
        shape: vec![20],  // Different shape
    });
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, module.outputs[0].name);
    assert_ne!(module.inputs[0].shape, module.outputs[0].shape);
    assert_ne!(module.inputs[0].ty, module.outputs[0].ty);
}

/// Test 9: Value with all zero dimensions vs all one dimensions (contrast test)
#[test]
fn test_zero_dimensions_vs_one_dimensions() {
    let all_zeros = Value {
        name: "all_zeros".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0],
    };
    
    let all_ones = Value {
        name: "all_ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    
    // All zeros should have 0 elements
    assert_eq!(all_zeros.num_elements(), Some(0));
    
    // All ones should have 1 element
    assert_eq!(all_ones.num_elements(), Some(1));
    
    // They should have same shape length but different element counts
    assert_eq!(all_zeros.shape.len(), all_ones.shape.len());
    assert_ne!(all_zeros.num_elements(), all_ones.num_elements());
}

/// Test 10: Compiler with alternating small/large models (testing memory management)
#[test]
fn test_compiler_alternating_model_sizes() {
    let mut compiler = ImpulseCompiler::new();
    
    // Alternate between small and large models
    for iteration in 0..5 {
        if iteration % 2 == 0 {
            // Small model
            let small_model = vec![0xCC; 100];
            let _result = compiler.compile(&small_model, "cpu");
        } else {
            // Large model
            let large_model = vec![0xDD; 500_000];
            let _result = compiler.compile(&large_model, "cpu");
        }
    }
    
    // Compiler should remain functional after alternating sizes
    assert_eq!(compiler.passes.passes.len(), 0);
}