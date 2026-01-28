//! Additional boundary condition edge case tests for the Impulse compiler
//! Focuses on edge cases that weren't covered in the existing tests, particularly
//! around overflow protection, extreme boundary conditions, and error handling

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;

/// Test 1: Testing the num_elements function with shapes that could overflow
#[test]
fn test_num_elements_overflow_potential() {
    // Test large dimensions to ensure checked_mul works properly
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 1_000_000], // Would overflow usize if multiplied naively
    };
    
    // Using try_fold with checked_mul, this should not panic
    let result = value.num_elements();
    // The result should be None since it overflows
    assert!(result.is_none());
    
    // Test a safe size that doesn't overflow
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000], // Safe to multiply
    };
    
    let result = safe_value.num_elements();
    assert_eq!(result, Some(1_000_000));
}

/// Test 2: Testing tensor shapes with zero dimensions using num_elements
#[test]
fn test_num_elements_with_zero_dimensions() {
    let zero_values = vec![
        Value {
            name: "zero_tensor_1".to_string(),
            ty: Type::F32,
            shape: vec![0],
        },
        Value {
            name: "zero_tensor_2".to_string(),
            ty: Type::F32,
            shape: vec![5, 0, 3],
        },
        Value {
            name: "zero_tensor_3".to_string(),
            ty: Type::F32,
            shape: vec![0, 100, 200],
        },
    ];
    
    for value in zero_values {
        let elements = value.num_elements().unwrap_or(0);
        assert_eq!(elements, 0, "Value with shape {:?} should have 0 elements", value.shape);
    }
    
    // Test scalar (empty shape) 
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![], // Scalar
    };
    
    let elements = scalar.num_elements().unwrap_or(0);
    assert_eq!(elements, 1, "Scalar should have 1 element");
}

/// Test 3: Testing with maximum possible dimension sizes
#[test]
fn test_max_dimension_sizes() {
    let max_dim = usize::MAX;
    let value = Value {
        name: "max_tensor".to_string(),
        ty: Type::F32,
        shape: vec![max_dim, 1], // Max size in first dim with 1 in second to avoid immediate overflow
    };
    
    let result = value.num_elements();
    assert_eq!(result, Some(max_dim), "Should handle max dimension properly");
    
    // Test with max dimensions that would overflow when multiplied
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![max_dim, 2], // This should overflow
    };
    
    let result = overflow_value.num_elements();
    assert!(result.is_none(), "Multiplication with max_dim * 2 should overflow");
}

/// Test 4: Test operations with extremely large attribute counts using rstest with different types
#[rstest]
#[case(Attribute::Int(i64::MAX))]
#[case(Attribute::Int(i64::MIN))]
#[case(Attribute::Float(f64::MAX))]
#[case(Attribute::Float(f64::MIN))]
fn test_extreme_attribute_values(#[case] attr: Attribute) {
    let mut op = Operation::new("test_op");
    op.attributes.insert("extreme_attr".to_string(), attr.clone());
    
    assert!(op.attributes.contains_key("extreme_attr"));
    assert_eq!(op.attributes.get("extreme_attr"), Some(&attr));
}

/// Test 5: Test deeply nested tensor type validation
#[test]
fn test_deep_tensor_validation() {
    // Create a deeply nested tensor type
    let mut nested_type = Type::F32;
    for i in 0..50 { // Create 50 levels of nesting
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![i % 10 + 1], // Varying small shapes to keep it reasonable
        };
    }
    
    // Validate that it's still a valid type
    assert!(nested_type.is_valid_type(), "Deeply nested type should still be valid");
    
    // Clone it to test deep cloning
    let cloned_nested = nested_type.clone();
    assert_eq!(nested_type, cloned_nested, "Cloned deep nested type should be equal");
}

/// Test 6: Boundary case for very long tensor names
#[test]
fn test_extremely_long_tensor_names() {
    let long_name = "a".repeat(100_000); // Very long name
    let value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    
    assert_eq!(value.name.len(), 100_000);
    assert_eq!(value.name, long_name);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 1, 1]);
}

/// Test 7: Test edge case with empty strings and special Unicode characters in attributes
#[test]
fn test_special_string_attributes() {
    let special_strings = vec![
        ("empty_string", "".to_string()),
        ("unicode_emoji", "Hello 🌍 World 🦀".to_string()),
        ("unicode_chinese", "你好世界".to_string()),
        ("unicode_arabic", "مرحبا بالعالم".to_string()),
        ("control_chars", "\n\t\r\0".to_string()),
        ("max_unicode", "\u{10FFFF}".to_string()), // Maximum valid Unicode
    ];
    
    for (label, test_string) in special_strings {
        let attr = Attribute::String(test_string.clone());
        
        match attr {
            Attribute::String(s) => assert_eq!(s, test_string, "Failed for {}", label),
            _ => panic!("Expected String attribute for {}", label),
        }
    }
}

/// Test 8: Test attribute array edge cases with nested structures
#[test]
fn test_nested_attribute_array_edge_cases() {
    // Deeply nested arrays
    let deep_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(42),
            ]),
        ]),
    ]);
    
    // Verify structure
    match &deep_nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 1);
            match &outer[0] {
                Attribute::Array(middle) => {
                    assert_eq!(middle.len(), 1);
                    match &middle[0] {
                        Attribute::Array(inner) => {
                            assert_eq!(inner.len(), 1);
                            match inner[0] {
                                Attribute::Int(42) => {}, // Success
                                _ => panic!("Expected inner value to be Int(42)"),
                            }
                        },
                        _ => panic!("Expected middle level to be Array"),
                    }
                },
                _ => panic!("Expected outer level to be Array"),
            }
        },
        _ => panic!("Expected outermost to be Array"),
    }
    
    // Empty nested arrays
    let empty_nested = Attribute::Array(vec![
        Attribute::Array(vec![]),
        Attribute::Array(vec![]),
    ]);
    
    match &empty_nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            for item in outer {
                match item {
                    Attribute::Array(inner) => assert_eq!(inner.len(), 0),
                    _ => panic!("Expected all items to be empty arrays"),
                }
            }
        },
        _ => panic!("Expected outermost to be Array"),
    }
}

/// Test 9: Test operations with zero inputs and outputs but with attributes
#[test]
fn test_operation_with_no_ios_has_attributes() {
    let mut op = Operation::new("param_op");
    op.attributes.insert("param1".to_string(), Attribute::Int(100));
    op.attributes.insert("param2".to_string(), Attribute::Float(3.14));
    op.attributes.insert("param3".to_string(), Attribute::String("test".to_string()));
    
    assert_eq!(op.op_type, "param_op");
    assert_eq!(op.inputs.len(), 0, "Should have no inputs");
    assert_eq!(op.outputs.len(), 0, "Should have no outputs");
    assert_eq!(op.attributes.len(), 3, "Should have 3 attributes");
    
    // Verify individual attributes
    assert_eq!(op.attributes.get("param1"), Some(&Attribute::Int(100)));
    assert_eq!(op.attributes.get("param2"), Some(&Attribute::Float(3.14)));
    assert_eq!(op.attributes.get("param3"), Some(&Attribute::String("test".to_string())));
}

/// Test 10: Test boundary conditions for tensor type creation and comparison
#[rstest]
#[case(vec![], 1)]  // scalar -> 1 element
#[case(vec![0], 0)] // zero dim -> 0 elements  
#[case(vec![1], 1)] // single unit -> 1 element
#[case(vec![1, 1, 1, 1, 1], 1)] // multiple units -> 1 element
#[case(vec![2, 3], 6)] // 2x3 -> 6 elements
#[case(vec![2, 3, 4], 24)] // 2x3x4 -> 24 elements
fn test_specific_shape_products(#[case] shape: Vec<usize>, #[case] expected: usize) {
    let value = Value {
        name: "shape_test".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    let calculated = value.shape.iter().product::<usize>();
    assert_eq!(calculated, expected, "Shape {:?} should have {} elements", shape, expected);
    
    // Also test with the safe num_elements function
    if expected == 0 || calculated == expected {
        let safe_result = value.num_elements();
        if expected == 0 {
            assert_eq!(safe_result.unwrap_or(0), 0);
        } else {
            assert_eq!(safe_result, Some(expected));
        }
    }
}