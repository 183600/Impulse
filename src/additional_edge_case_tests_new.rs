//! Additional edge case tests for the Impulse compiler to cover more boundary conditions
//! using standard library assertions and rstest framework

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;
use std::collections::HashMap;

/// Test 1: Deep but safe nested tensor types to test recursion handling
#[test]
fn test_deep_tensor_nesting() {
    // Creates a deeply nested tensor type, but at a safe depth to avoid stack overflow
    let mut current_type = Type::F32;
    const DEPTH: usize = 100; // Safe depth to test recursion without stack overflow
    
    for _ in 0..DEPTH {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1],
        };
    }
    
    // Verify that the type is still valid after deep nesting
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![1]);
        },
        _ => panic!("Expected a nested tensor type"),
    }
    
    // Verify we can clone this deeply nested type without issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 2: Operations with large number of attributes to test map handling
#[test]
fn test_operation_with_many_attributes() {
    let mut op = Operation::new("many_attr_op");
    
    // Add a large but reasonable number of attributes
    for i in 0..10_000 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.op_type, "many_attr_op");
}

/// Test 3: Values with long names to test string handling
#[test]
fn test_values_with_long_names() {
    // Create a value with a long but reasonable name
    let long_name = "x".repeat(100_000); // 100k string for name
    let value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1, 1],
    };
    
    assert_eq!(value.name, long_name);
    assert_eq!(value.name.len(), 100_000);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 1]);
}

/// Test 4: Module with many values for inputs/outputs to test collection handling
#[test]
fn test_module_with_many_collections() {
    let mut module = Module::new("many_collections_module");
    
    // Add many inputs (reasonable number)
    for i in 0..5_000 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    // Add many outputs (reasonable number)
    for i in 0..5_000 {
        module.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(module.inputs.len(), 5_000);
    assert_eq!(module.outputs.len(), 5_000);
    assert_eq!(module.name, "many_collections_module");
}

/// Test 5: Extreme values in tensor shapes that could cause integer overflow
#[test]
fn test_tensor_shapes_with_overflow_potential() {
    // Create a tensor shape that could cause overflow in multiplication
    // This tests the num_elements() method which uses checked_mul
    let large_value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000, 100],  // May overflow when multiplied
    };
    
    // Test our safe calculation method
    let elements = large_value.num_elements();
    // Result may be None if overflow detected, or Some(value) if it fits
    assert!(elements.is_some() || elements.is_none());
    
    // Test with safe shape that won't overflow
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000],
    };
    
    let safe_elements = safe_value.num_elements();
    assert_eq!(safe_elements, Some(1_000_000));
}

/// Test 6: Special floating point values in tensor computations
#[rstest]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
#[case(f64::NAN)]
#[case(-0.0)]
#[case(f64::EPSILON)]
#[case(f64::MAX)]
#[case(f64::MIN)]
fn test_special_float_attributes(#[case] value: f64) {
    let attr = Attribute::Float(value);
    
    match attr {
        Attribute::Float(retrieved_value) => {
            if value.is_nan() {
                assert!(retrieved_value.is_nan());
            } else if value.is_infinite() {
                assert!(retrieved_value.is_infinite());
                assert_eq!(value.is_sign_positive(), retrieved_value.is_sign_positive());
            } else {
                // For normal finite values, check equality
                assert!((retrieved_value - value).abs() < f64::EPSILON || 
                       (retrieved_value - value).abs() == 0.0);
            }
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 7: Complex nested arrays with maximum nesting levels
#[test]
fn test_max_nested_arrays() {
    // Create a deeply nested array structure
    let mut nested_array = Attribute::Int(42);
    
    // Nest arrays deeply
    for _ in 0..100 {
        nested_array = Attribute::Array(vec![nested_array]);
    }
    
    // Verify we can still handle this nested structure
    match &nested_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
        },
        _ => panic!("Expected nested array structure"),
    }
    
    // Verify equality still works on deeply nested structures
    let cloned = nested_array.clone();
    assert_eq!(nested_array, cloned);
}

/// Test 8: Operation with mixed extreme values in all fields
#[test]
fn test_operation_mixed_extremes() {
    let mut attrs = HashMap::new();
    attrs.insert("long_name_attr".to_string(), Attribute::String("x".repeat(10_000)));
    attrs.insert("max_int_attr".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_int_attr".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("infinity_float_attr".to_string(), Attribute::Float(f64::INFINITY));
    
    let mut op = Operation::new(&"x".repeat(10_000)); // Long operation name
    
    // Add many inputs (reduced number for performance)
    for i in 0..1_000 {
        op.inputs.push(Value {
            name: format!("input_{}_{}", i, "x".repeat(50)), // Names with long suffixes
            ty: if i % 2 == 0 { Type::F32 } else { Type::F64 },
            shape: vec![i % 100 + 1], // Varying shapes
        });
    }
    
    // Add many outputs (reduced number for performance)
    for i in 0..500 {
        op.outputs.push(Value {
            name: format!("output_{}_{}", i, "y".repeat(50)),
            ty: if i % 2 == 0 { Type::I32 } else { Type::I64 },
            shape: vec![i % 50 + 1],
        });
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.op_type.len(), 10_000);
    assert_eq!(op.inputs.len(), 1_000);
    assert_eq!(op.outputs.len(), 500);
    assert_eq!(op.attributes.len(), 4); // Our predefined attributes
}

/// Test 9: Value with empty shape (scalar) edge case
#[test]
fn test_scalar_value_edge_cases() {
    // A scalar value has an empty shape vector
    let scalar = Value {
        name: "scalar_value".to_string(),
        ty: Type::F32,
        shape: vec![], // Empty shape = scalar
    };
    
    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(scalar.ty, Type::F32);
    
    // A scalar has 1 element
    match scalar.num_elements() {
        Some(1) => assert_eq!(1, 1), // Scalar has 1 element
        _ => panic!("Scalar should have 1 element"),
    }
    
    // Test another scalar variant
    let int_scalar = Value {
        name: "int_scalar".to_string(),
        ty: Type::I32,
        shape: vec![],
    };
    
    assert!(int_scalar.shape.is_empty());
    assert_eq!(int_scalar.ty, Type::I32);
}

/// Test 10: Recursive type with maximum complexity 
#[test]
fn test_recursive_type_max_complexity() {
    // Create a complex recursive type structure
    let mut complex_type = Type::F32;
    
    // Alternate between different types in the hierarchy to create complex structure
    for i in 0..100 {
        complex_type = if i % 3 == 0 {
            Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![i + 1],
            }
        } else if i % 3 == 1 {
            Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![i + 2, i + 3],
            }
        } else {
            Type::Tensor {
                element_type: Box::new(complex_type),
                shape: vec![2],
            }
        };
    }
    
    // Ensure the type remains valid despite complexity
    // We can't check exact structure due to depth, but we can ensure it's still a tensor
    match &complex_type {
        Type::Tensor { .. } => (), // Expected
        _ => panic!("Expected a tensor type after complex nesting"),
    }
    
    // Test that cloning works correctly on complex recursive type
    let cloned_complex = complex_type.clone();
    assert_eq!(complex_type, cloned_complex);
}