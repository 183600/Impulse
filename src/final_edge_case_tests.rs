//! Final set of edge case tests for the Impulse compiler
//! Focus on missing coverage areas identified from existing tests

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};
use std::collections::HashMap;

/// Test 1: Operations with extremely large integer values in attributes that could cause overflow
#[test]
fn test_extremely_large_integer_attributes() {
    let mut op = Operation::new("large_int_test");
    let mut attrs = HashMap::new();
    
    // Test with extremities of i64 range
    attrs.insert("i64_max".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("i64_min".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("i64_middle".to_string(), Attribute::Int(i64::MAX / 2));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.get("i64_max"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("i64_min"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("i64_middle"), Some(&Attribute::Int(i64::MAX / 2)));
}

/// Test 2: Values with empty string names (edge case for debugging/serialization)
#[test]
fn test_empty_string_names() {
    // Test value with empty name
    let value = Value {
        name: "".to_string(),  // Empty name
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    assert_eq!(value.name, "");
    assert_eq!(value.shape, vec![1, 2, 3]);
    
    // Test operation with empty name
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs.len(), 0);
    
    // Test module with empty name
    let module = Module::new("");
    assert_eq!(module.name, "");
    assert_eq!(module.operations.len(), 0);
}

/// Test 3: Tensor types with maximum depth but minimum width to test recursion limits
#[test]
fn test_maximum_depth_minimum_width_tensors() {
    let mut current_type = Type::F32;
    // Create deeply nested tensor with smallest possible inner dimensions
    for _ in 0..200 {  // Use 200 to avoid potential stack overflow issues
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1],  // Minimum width
        };
    }

    // Should still be a valid type
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![1]);
        },
        _ => panic!("Expected tensor type after deep nesting"),
    }
    
    // Should be able to clone without issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 4: Operations with all possible attribute type combinations
#[test]
fn test_all_attribute_type_combinations() {
    let mut op = Operation::new("all_attr_types");
    let mut attrs = HashMap::new();
    
    // Add all attribute types in one operation
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14159));
    attrs.insert("string_attr".to_string(), Attribute::String("hello".to_string()));
    attrs.insert("bool_true".to_string(), Attribute::Bool(true));
    attrs.insert("bool_false".to_string(), Attribute::Bool(false));
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.0),
        Attribute::String("nested".to_string()),
        Attribute::Bool(true),
    ]));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 6);
    assert!(matches!(op.attributes.get("int_attr"), Some(Attribute::Int(42))));
    assert!(matches!(op.attributes.get("float_attr"), Some(Attribute::Float(v)) if (v - 3.14159).abs() < f64::EPSILON));
    assert!(matches!(op.attributes.get("string_attr"), Some(Attribute::String(s)) if s == "hello"));
    assert!(matches!(op.attributes.get("bool_true"), Some(Attribute::Bool(true))));
    assert!(matches!(op.attributes.get("bool_false"), Some(Attribute::Bool(false))));
    if let Some(Attribute::Array(arr)) = op.attributes.get("array_attr") {
        assert_eq!(arr.len(), 4);
        assert!(matches!(&arr[0], Attribute::Int(1)));
    } else {
        panic!("Expected array attribute at 'array_attr'");
    }
}

/// Test 5: Complex tensor operations with mixed primitive and tensor types
#[test]
fn test_mixed_primitive_tensor_type_operations() {
    // Create an operation with inputs of mixed types
    let mut op = Operation::new("mixed_types_op");
    
    // Add a primitive type input
    op.inputs.push(Value {
        name: "primitive_input".to_string(),
        ty: Type::F32,
        shape: vec![],
    });
    
    // Add a tensor input
    op.inputs.push(Value {
        name: "tensor_input".to_string(),
        ty: Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![10, 20],
        },
        shape: vec![5, 5],
    });
    
    // Add another nested tensor
    op.inputs.push(Value {
        name: "nested_tensor_input".to_string(),
        ty: Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Bool),
                shape: vec![3],
            }),
            shape: vec![2, 2],
        },
        shape: vec![1, 2, 3],
    });
    
    assert_eq!(op.inputs.len(), 3);
    assert_eq!(op.op_type, "mixed_types_op");
    
    // Verify types were preserved
    assert_eq!(op.inputs[0].ty, Type::F32);
    if let Type::Tensor { shape, .. } = &op.inputs[1].ty {
        assert_eq!(shape, &vec![10, 20]);
    } else {
        panic!("Expected tensor type for second input");
    }
}

/// Test 6: Shape calculations with potential integer overflows using checked arithmetic
#[rstest]
#[case(vec![], 1)]  // Scalar has 1 element
#[case(vec![0], 0)]  // Contains 0: 0 elements
#[case(vec![1], 1)]  // Single dimension
#[case(vec![2, 3, 4], 24)]  // Multiple dimensions
#[case(vec![100, 200, 50], 1_000_000)]  // Larger numbers
fn test_safe_shape_calculations(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
    let value = Value {
        name: "safe_calc".to_string(),
        ty: Type::F32,
        shape,
    };
    
    // Calculate using regular product
    let regular_product: usize = value.shape.iter().product();
    assert_eq!(regular_product, expected_product);
    
    // Calculate using checked arithmetic to ensure no overflow
    let checked_result: Option<usize> = value.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    if expected_product == 0 {
        // If expected is 0 (due to a 0 in shape), result should be Some(0)
        assert_eq!(checked_result, Some(0));
    } else {
        // Otherwise, should match expected result if no overflow
        assert_eq!(checked_result, Some(expected_product));
    }
}

/// Test 7: Nested attribute arrays with maximum depth
#[test]
fn test_maximum_depth_attribute_arrays() {
    // Create a deeply nested array structure
    let mut nested = Attribute::Int(42);
    
    // Nest 15 levels deep (avoiding potential stack overflow with too much nesting)
    for _ in 0..15 {
        nested = Attribute::Array(vec![nested]);
    }
    
    // The structure should still be valid
    match &nested {
        Attribute::Array(inner) => {
            assert_eq!(inner.len(), 1);
            // Can access but don't traverse deeply to avoid stack issues
        },
        _ => panic!("Expected nested array structure"),
    }
    
    // Should be able to clone
    let cloned = nested.clone();
    assert_eq!(nested, cloned);
}

/// Test 8: Values with high dynamic range in tensor shapes
#[test]
fn test_dynamic_range_tensor_shapes() {
    // Test tensor shapes with high variance in dimension sizes
    let extreme_ratio_tensor = Value {
        name: "dynamic_range_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1, 1_000_000, 1], // One very large dimension, others small
    };
    
    assert_eq!(extreme_ratio_tensor.shape, vec![1, 1_000_000, 1]);
    let product: usize = extreme_ratio_tensor.shape.iter().product();
    assert_eq!(product, 1_000_000);
    
    // Another extreme: wide and narrow mix
    let mixed_extreme_tensor = Value {
        name: "mixed_extreme_tensor".to_string(),
        ty: Type::I64,
        shape: vec![1_000, 1, 1_000, 1], // Alternating large and small
    };
    
    assert_eq!(mixed_extreme_tensor.shape, vec![1_000, 1, 1_000, 1]);
    let mixed_product: usize = mixed_extreme_tensor.shape.iter().product();
    assert_eq!(mixed_product, 1_000_000);
}

/// Test 9: Operations with maximum attribute key-value size ratio
#[test]
fn test_extreme_key_value_size_ratios() {
    let mut op = Operation::new("key_value_ratio_test");
    let mut attrs = HashMap::new();
    
    // Short key, very long value
    attrs.insert("k".to_string(), Attribute::String("v".repeat(1_000_000))); // 1MB value
    
    // Long key, short value  
    let long_key = "a".repeat(100_000); // 100KB key
    attrs.insert(long_key.clone(), Attribute::Int(12345));
    
    // Medium key, medium value
    attrs.insert("medium_key".to_string(), Attribute::String("medium_val".repeat(50_000)));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 3);
    assert_eq!(op.attributes.get("k").unwrap(),
               &Attribute::String("v".repeat(1_000_000)));
    assert_eq!(op.attributes.get(&long_key).unwrap(),
               &Attribute::Int(12345));
}

/// Test 10: Comprehensive cleanup verification after extreme allocations
#[test]
fn test_comprehensive_cleanup_after_extreme_allocations() {
    // Create and store many objects to test memory cleanup
    let mut all_modules = Vec::new();
    
    for i in 0..500 {  // Reduced from extreme values to prevent timeouts
        let mut module = Module::new(&format!("cleanup_test_module_{}", i));
        
        // Add operations to each module
        for j in 0..50 {
            let mut op = Operation::new(&format!("cleanup_op_{}_{}", i, j));
            
            // Add a few inputs with different types
            for k in 0..5 {
                op.inputs.push(Value {
                    name: format!("cleanup_input_{}_{}_{}", i, j, k),
                    ty: match k % 4 {
                        0 => Type::F32,
                        1 => Type::I32,
                        2 => Type::Bool,
                        _ => Type::F64,
                    },
                    shape: vec![k + 1, j + 1],
                });
            }
            
            module.add_operation(op);
        }
        
        all_modules.push(module);
    }
    
    // Verify we created the expected number
    assert_eq!(all_modules.len(), 500);
    assert_eq!(all_modules[0].operations.len(), 50);
    
    // This test passes if no memory issues occur during creation/destruction
    // The real test is whether this completes without memory issues
    drop(all_modules);  // Explicitly drop to trigger cleanup
    
    // If we get here without crashing, the test passes
    assert!(true);
}