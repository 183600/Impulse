//! Additional edge case tests for the Impulse compiler
//! Covering boundary conditions and extreme values not covered in other test files

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Operations and values with UTF-8 edge cases in names
#[test]
fn test_utf8_edge_case_names() {
    // Test with various Unicode characters that might cause issues
    let utf8_test_cases = [
        "simple_ascii",
        "汉字",           // Chinese characters
        "café naïve",     // Accented Latin characters  
        "🐍🚀💰",        // Emojis
        "\u{0000}",       // Null character (valid in Rust strings)
        "D800_surrogate", // Surrogate half represented as string (may cause issues in other systems)
        &"a".repeat(1000), // Very long ASCII string
        &"αβγδε".repeat(200), // Very long Unicode string
    ];
    
    for test_case in utf8_test_cases.iter() {
        // Test value with UTF-8 name
        let value = Value {
            name: test_case.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, *test_case);
        
        // Test operation with UTF-8 name
        let op = Operation::new(test_case);
        assert_eq!(op.op_type, *test_case);
        
        // Test module with UTF-8 name
        let module = Module::new(test_case.to_string());
        assert_eq!(module.name, *test_case);
    }
}

/// Test 2: Floating point values with precision edge cases
#[test]
fn test_floating_point_precision_edge_cases() {
    use std::f64;
    
    // Test edge cases for floating point values in attributes
    let fp_test_cases = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::EPSILON,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        0.0,
        -0.0,  // Negative zero
        1.0,
        -1.0,
    ];
    
    for (i, &val) in fp_test_cases.iter().enumerate() {
        let attr = Attribute::Float(val);
        
        match attr {
            Attribute::Float(retrieved_val) => {
                if val.is_nan() {
                    assert!(retrieved_val.is_nan(), "Test case {}: Expected NaN", i);
                } else if val.is_infinite() {
                    assert!(retrieved_val.is_infinite(), "Test case {}: Expected infinite", i);
                    assert_eq!(val.is_sign_positive(), retrieved_val.is_sign_positive(), 
                              "Test case {}: Sign mismatch for infinite value", i);
                } else {
                    assert!((retrieved_val - val).abs() <= f64::EPSILON.max((val + retrieved_val).abs() * f64::EPSILON),
                           "Test case {}: Precision issue - expected {}, got {}", i, val, retrieved_val);
                }
            },
            _ => panic!("Test case {}: Expected Float attribute", i),
        }
    }
}

/// Test 3: Integer values with boundary cases
#[test]
fn test_integer_boundary_edge_cases() {
    use std::i64;
    
    // Test edge cases for integer values in attributes
    let int_test_cases = [
        i64::MIN,
        i64::MIN + 1,
        -1,
        0,
        1,
        i64::MAX - 1,
        i64::MAX,
    ];
    
    for (i, &val) in int_test_cases.iter().enumerate() {
        let attr = Attribute::Int(val);
        
        match attr {
            Attribute::Int(retrieved_val) => {
                assert_eq!(retrieved_val, val, "Test case {}: Integer boundary mismatch", i);
            },
            _ => panic!("Test case {}: Expected Int attribute", i),
        }
    }
    
    // Test operations with integer attributes at boundaries
    let mut op = Operation::new("boundary_test");
    let mut attrs = HashMap::new();
    
    for (i, &val) in int_test_cases.iter().enumerate() {
        attrs.insert(format!("int_attr_{}", i), Attribute::Int(val));
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), int_test_cases.len());
}

/// Test 4: Extremely large collections causing potential memory issues
#[test]
fn test_memory_allocation_edge_cases() {
    const LARGE_SIZE: usize = 50_000;
    
    // Test creating a module with a very large number of small operations
    let mut large_module = Module::new("large_allocation_test");
    
    for i in 0..LARGE_SIZE {
        let op = Operation::new(&format!("op_{:08}", i));
        large_module.add_operation(op);
        
        // Periodic check to make sure the test is progressing
        if i % 10_000 == 0 {
            assert!(large_module.operations.len() <= i + 1);
        }
    }
    
    assert_eq!(large_module.operations.len(), LARGE_SIZE);
    
    // Test creating a single operation with a large number of attributes
    let mut large_op = Operation::new("large_attr_op");
    let mut attrs = HashMap::new();
    
    for i in 0..LARGE_SIZE {
        attrs.insert(
            format!("large_attr_{:08}", i),
            Attribute::String(format!("value_{:08}", i))
        );
    }
    
    large_op.attributes = attrs;
    assert_eq!(large_op.attributes.len(), LARGE_SIZE);
}

/// Test 5: Boolean attribute edge cases
#[test]
fn test_boolean_attribute_edge_cases() {
    // Test boolean attributes with both true and false
    let true_attr = Attribute::Bool(true);
    let false_attr = Attribute::Bool(false);
    
    match true_attr {
        Attribute::Bool(val) => assert_eq!(val, true),
        _ => panic!("Expected Bool(true)"),
    }
    
    match false_attr {
        Attribute::Bool(val) => assert_eq!(val, false),
        _ => panic!("Expected Bool(false)"),
    }
    
    // Test operations with many boolean attributes
    let mut op = Operation::new("bool_test_op");
    let mut attrs = HashMap::new();
    
    for i in 0..10_000 {
        attrs.insert(
            format!("bool_attr_{}", i),
            Attribute::Bool(i % 2 == 0)  // Alternate true/false
        );
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 10_000);
    
    // Verify the pattern is maintained
    for i in 0..10_000 {
        let key = format!("bool_attr_{}", i);
        if let Some(Attribute::Bool(val)) = op.attributes.get(&key) {
            assert_eq!(val, &(i % 2 == 0));
        } else {
            panic!("Missing or incorrect attribute for key: {}", key);
        }
    }
}

/// Test 6: Array attributes with complex nesting and edge cases
#[test]
fn test_array_attribute_complex_nesting() {
    // Create deeply nested arrays
    let nested_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![
            Attribute::Float(2.5),
            Attribute::Array(vec![
                Attribute::String("deeply_nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Int(42),
                ])
            ])
        ])
    ]);
    
    // Verify the nested structure
    match &nested_array {
        Attribute::Array(level1) => {
            assert_eq!(level1.len(), 2);
            
            // Check first element
            match &level1[0] {
                Attribute::Int(1) => (),
                _ => panic!("Expected Int(1) at top level"),
            }
            
            // Check second element (nested array)
            match &level1[1] {
                Attribute::Array(level2) => {
                    assert_eq!(level2.len(), 2);
                    
                    match &level2[0] {
                        Attribute::Float(val) if (val - 2.5).abs() < f64::EPSILON => (),
                        _ => panic!("Expected Float(2.5) at level 2"),
                    }
                    
                    match &level2[1] {
                        Attribute::Array(level3) => {
                            assert_eq!(level3.len(), 2);
                            
                            match &level3[0] {
                                Attribute::String(s) if s == "deeply_nested" => (),
                                _ => panic!("Expected String at level 3"),
                            }
                            
                            match &level3[1] {
                                Attribute::Array(level4) => {
                                    assert_eq!(level4.len(), 2);
                                    
                                    match &level4[0] {
                                        Attribute::Bool(true) => (),
                                        _ => panic!("Expected Bool(true) at level 4"),
                                    }
                                    
                                    match &level4[1] {
                                        Attribute::Int(42) => (),
                                        _ => panic!("Expected Int(42) at level 4"),
                                    }
                                }
                                _ => panic!("Expected Array at level 4"),
                            }
                        }
                        _ => panic!("Expected Array at level 3"),
                    }
                }
                _ => panic!("Expected Array at level 2"),
            }
        }
        _ => panic!("Expected Array at top level"),
    }
    
    // Test very large array with simple elements
    let large_array = Attribute::Array(
        (0..20_000)
            .map(|i| Attribute::Int(i as i64))
            .collect()
    );
    
    match large_array {
        Attribute::Array(arr) => assert_eq!(arr.len(), 20_000),
        _ => panic!("Expected large array"),
    }
}

/// Test 7: Edge cases for tensor shape calculations and overflow protection
#[rstest]
#[case(vec![], 1)]  // Scalar has 1 element
#[case(vec![0], 0)] // Any dimension with 0 gives 0 total
#[case(vec![0, 1], 0)]
#[case(vec![1, 0], 0)]
#[case(vec![0, 0, 0], 0)]
#[case(vec![1], 1)]
#[case(vec![1, 1, 1], 1)]
#[case(vec![2, 3], 6)]
#[case(vec![2, 3, 4], 24)]
#[case(vec![10, 10, 10], 1000)]
#[case(vec![100, 100], 10_000)]
fn test_tensor_shape_calculation_edge_cases(#[case] shape: Vec<usize>, #[case] expected_elements: usize) {
    let value = Value {
        name: "shape_test".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let calculated_elements: usize = value.shape.iter().product();
    assert_eq!(calculated_elements, expected_elements, 
               "Shape {:?} should have {} elements", value.shape, expected_elements);
    
    // Also test with checked multiplication
    let checked_result = value.shape.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim)
    });
    
    if expected_elements == 0 {
        // If any dimension is 0, result should be Some(0) due to 0 multiplication
        assert_eq!(checked_result, Some(0));
    } else if calculated_elements <= usize::MAX {
        assert_eq!(checked_result, Some(expected_elements));
    } else {
        // This case shouldn't happen with our test cases, but would result in None
        assert_eq!(checked_result, None);
    }
}

/// Test 8: Deeply nested tensor types with validation
#[test]
fn test_deeply_nested_tensor_validation() {
    // Create a deeply nested tensor type and test validation
    let mut current_type = Type::F32;
    
    // Create 100 levels of nesting
    for _ in 0..100 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Validate the deeply nested type
    assert!(current_type.is_valid_type(), "Deeply nested type should be valid");
    
    // Test that it can be cloned without issues
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
    
    // Test with a more complex nested structure
    let complex_type = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![3, 4],
        }),
        shape: vec![5, 6],
    };
    
    assert!(complex_type.is_valid_type(), "Complex nested type should be valid");
    
    // Test equality comparison
    let complex_type_copy = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![3, 4],
        }),
        shape: vec![5, 6],
    };
    
    assert_eq!(complex_type, complex_type_copy, "Identical complex types should be equal");
}

/// Test 9: Large string attributes that could cause memory pressure
#[test]
fn test_large_string_attribute_memory_pressure() {
    // Test with strings of different sizes that could cause memory issues
    let sizes = [1_000, 10_000, 100_000, 1_000_000]; // 1KB, 10KB, 100KB, 1MB
    
    for &size in sizes.iter() {
        let large_string = "A".repeat(size);
        let string_attr = Attribute::String(large_string.clone());
        
        match string_attr {
            Attribute::String(retrieved) => {
                assert_eq!(retrieved.len(), size);
                assert_eq!(retrieved, large_string);
            },
            _ => panic!("Expected String attribute for size {}", size),
        }
        
        // Test operation with large string attribute
        let mut op = Operation::new("large_string_op");
        let mut attrs = HashMap::new();
        attrs.insert("large_content".to_string(), Attribute::String(large_string.clone()));
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 1);
        if let Some(Attribute::String(attr_str)) = op.attributes.get("large_content") {
            assert_eq!(attr_str.len(), size);
        } else {
            panic!("Failed to retrieve large string attribute");
        }
    }
}

/// Test 10: Mixed edge cases combining multiple extreme conditions
#[test]
fn test_mixed_extreme_conditions() {
    // Combine multiple edge cases in one test
    let mut module = Module::new("extreme_conditions_module_🚀🔥");
    
    // Add an operation with extreme conditions
    let mut extreme_op = Operation::new("extreme_op_名称_日本語");
    
    // Add many inputs with different types and extreme shapes
    for i in 0..100 {
        let value = Value {
            name: format!("extreme_value_{}.🐍", i),  // Unicode in name
            ty: match i % 5 {
                0 => Type::F32,
                1 => Type::F64,
                2 => Type::I32,
                3 => Type::I64,
                _ => Type::Bool,
            },
            shape: if i % 10 == 0 {
                // Sometimes have zero-dimension (scalar)
                vec![]
            } else if i % 7 == 0 {
                // Sometimes have zero in dimensions
                vec![10, 0, 5]
            } else {
                // Normal dimension
                vec![2, 3]
            },
        };
        extreme_op.inputs.push(value);
    }
    
    // Add many outputs with different conditions
    for i in 0..50 {
        let value = Value {
            name: format!("output_{}.tensor_🔥", i),  // Unicode in name
            ty: match i % 3 {
                0 => Type::F32,
                1 => Type::I64,
                _ => Type::Bool,
            },
            shape: if i % 5 == 0 {
                // Sometimes have zero-dimension (scalar)
                vec![]
            } else {
                // Normal dimension
                vec![4, 5]
            },
        };
        extreme_op.outputs.push(value);
    }
    
    // Add many attributes with different types and extremes
    let mut attrs = HashMap::new();
    for i in 0..1000 {
        attrs.insert(
            format!("extreme_attr_{}.🚀", i),  // Unicode in attribute key
            match i % 7 {
                0 => Attribute::Int(i64::MAX),
                1 => Attribute::Int(i64::MIN),
                2 => Attribute::Float(std::f64::INFINITY),
                3 => Attribute::Float(std::f64::NEG_INFINITY),
                4 => Attribute::Float(std::f64::NAN),
                5 => Attribute::Bool(i % 2 == 0),
                _ => Attribute::String(format!("extreme_string_value_{}.🌟", i)),
            }
        );
    }
    extreme_op.attributes = attrs;
    
    // Add the operation to the module
    module.add_operation(extreme_op);
    
    // Verify the setup
    assert_eq!(module.name, "extreme_conditions_module_🚀🔥");
    assert_eq!(module.operations.len(), 1);
    
    let op = &module.operations[0];
    assert_eq!(op.inputs.len(), 100);
    assert_eq!(op.outputs.len(), 50);
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify some of the inputs
    for (i, input) in op.inputs.iter().enumerate() {
        assert!(input.name.starts_with(&format!("extreme_value_{}.🐍", i)));
        
        // Check type assignment is as expected
        let expected_type = match i % 5 {
            0 => Type::F32,
            1 => Type::F64,
            2 => Type::I32,
            3 => Type::I64,
            _ => Type::Bool,
        };
        assert_eq!(input.ty, expected_type);
        
        // Check shape assignment is as expected
        let expected_shape = if i % 10 == 0 {
            vec![]
        } else if i % 7 == 0 {
            vec![10, 0, 5]
        } else {
            vec![2, 3]
        };
        assert_eq!(input.shape, expected_shape);
    }
    
    // Verify some attributes
    for i in 0..10 {
        let attr_name = format!("extreme_attr_{}.🚀", i);
        assert!(op.attributes.contains_key(&attr_name));
    }
}