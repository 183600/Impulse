//! Additional focused edge case tests for the Impulse compiler
//! Covers specific boundary scenarios not addressed by other tests

use crate::ir::{Module, Value, Type, Operation, Attribute};

// Test 1: Empty collections and minimal structures
#[test]
fn test_empty_collections_and_minimal_structures() {
    // Test empty module
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
    assert!(empty_module.operations.is_empty());
    assert!(empty_module.inputs.is_empty());
    assert!(empty_module.outputs.is_empty());

    // Test operation with no fields populated
    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    assert!(empty_op.inputs.is_empty());
    assert!(empty_op.outputs.is_empty());
    assert!(empty_op.attributes.is_empty());

    // Test value with minimal content
    let minimal_value = Value {
        name: "".to_string(),
        ty: Type::F32,  // Basic type
        shape: vec![],  // Scalar/empty shape
    };
    assert_eq!(minimal_value.name, "");
    assert_eq!(minimal_value.ty, Type::F32);
    assert!(minimal_value.shape.is_empty());
    assert_eq!(minimal_value.num_elements(), Some(1));  // Scalar has 1 element
}

// Test 2: Extremely large but valid integer attributes that could cause overflow in operations
#[test]
fn test_extreme_integer_attribute_values() {
    let extreme_attrs = [
        ("max_i64", Attribute::Int(i64::MAX)),
        ("min_i64", Attribute::Int(i64::MIN)),
        ("max_usize_as_i64", Attribute::Int(usize::MAX as i64)),
        ("negative_max_usize_as_i64", Attribute::Int(-(usize::MAX as i64))),
        ("power_of_two_large", Attribute::Int(2i64.pow(60))),
    ];

    for (name, attr) in &extreme_attrs {
        match attr {
            Attribute::Int(value) => {
                if *name == "max_i64" {
                    assert_eq!(*value, i64::MAX);
                } else if *name == "min_i64" {
                    assert_eq!(*value, i64::MIN);
                } else if *name == "max_usize_as_i64" {
                    assert_eq!(*value, usize::MAX as i64);
                } else if *name == "negative_max_usize_as_i64" {
                    assert_eq!(*value, -(usize::MAX as i64));
                } else if *name == "power_of_two_large" {
                    assert_eq!(*value, 2i64.pow(60));
                } else {
                    panic!("Unknown attribute: {}", name);
                }
            },
            _ => panic!("Expected Int attribute for {}", name),
        }
    }
}

// Test 3: Boolean attributes edge cases and array structures
#[test]
fn test_boolean_attribute_edge_cases() {
    let bool_attrs = [
        ("true", Attribute::Bool(true)),
        ("false", Attribute::Bool(false)),
        ("array_single_true", Attribute::Array(vec![Attribute::Bool(true)])),
        ("array_single_false", Attribute::Array(vec![Attribute::Bool(false)])),
        ("mixed_bool_array", Attribute::Array(vec![
            Attribute::Bool(true),
            Attribute::Bool(false),
            Attribute::Bool(true),
        ])),
    ];

    for (name, attr) in &bool_attrs {
        match attr {
            Attribute::Bool(b) => {
                if *name == "true" {
                    assert_eq!(*b, true);
                } else if *name == "false" {
                    assert_eq!(*b, false);
                } else {
                    panic!("Unexpected single boolean for {}", name);
                }
            },
            Attribute::Array(arr) => {
                if *name == "array_single_true" {
                    assert_eq!(arr.len(), 1);
                    if let Attribute::Bool(val) = arr[0] {
                        assert_eq!(val, true);
                    } else {
                        panic!("Expected boolean in array");
                    }
                } else if *name == "array_single_false" {
                    assert_eq!(arr.len(), 1);
                    if let Attribute::Bool(val) = arr[0] {
                        assert_eq!(val, false);
                    } else {
                        panic!("Expected boolean in array");
                    }
                } else if *name == "mixed_bool_array" {
                    assert_eq!(arr.len(), 3);
                    if let Attribute::Bool(val) = arr[0] { assert_eq!(val, true); } else { panic!("Expected true"); }
                    if let Attribute::Bool(val) = arr[1] { assert_eq!(val, false); } else { panic!("Expected false"); }
                    if let Attribute::Bool(val) = arr[2] { assert_eq!(val, true); } else { panic!("Expected true"); }
                } else {
                    panic!("Unexpected array for {}", name);
                }
            },
            _ => panic!("Expected Bool or Array attribute for {}", name),
        }
    }
}

// Test 4: Zero-sized tensor operations edge cases
#[test]
fn test_zero_sized_tensor_edge_cases() {
    let zero_tensors = [
        ("zero_1d", vec![0]),
        ("zero_2d_first", vec![0, 10]),
        ("zero_2d_second", vec![10, 0]),
        ("zero_3d_middle", vec![5, 0, 3]),
        ("all_zeros", vec![0, 0, 0]),
        ("multiple_zeros", vec![1, 0, 1, 0, 1]),
    ];

    for (name, shape) in &zero_tensors {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, *shape);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.num_elements(), Some(0), "Tensor with zero in shape should have 0 elements: {}", name);
    }

    // Test special case: empty shape (scalar) should have 1 element, not 0
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.num_elements(), Some(1));  // Scalar has 1 element
}

// Test 5: Deep recursive type equality without stack overflow
#[test]
fn test_deep_recursive_type_equality() {
    fn create_deep_type(depth: usize) -> Type {
        if depth == 0 {
            Type::F32
        } else {
            Type::Tensor {
                element_type: Box::new(create_deep_type(depth - 1)),
                shape: vec![1],
            }
        }
    }

    // Create two identical deep types
    let deep_type_1 = create_deep_type(500);
    let deep_type_2 = create_deep_type(500);

    // They should be equal
    assert_eq!(deep_type_1, deep_type_2);

    // Cloning should preserve equality
    let deep_type_3 = deep_type_1.clone();
    assert_eq!(deep_type_1, deep_type_3);

    // A type with different depth should not be equal
    let shallow_type = create_deep_type(100);
    assert_ne!(deep_type_1, shallow_type);
}

// Test 6: Mixed attribute array edge cases
#[test]
fn test_mixed_attribute_array_edge_cases() {
    let complex_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14159),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
            Attribute::Array(vec![Attribute::Int(3)])
        ]),
    ]);

    match &complex_array {
        Attribute::Array(outer_arr) => {
            assert_eq!(outer_arr.len(), 5);
            
            // Validate each element
            match &outer_arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("First element should be Int(42)"),
            }
            
            match &outer_arr[1] {
                Attribute::Float(f) if (f - 3.14159).abs() < f64::EPSILON => {},
                _ => panic!("Second element should be Float(3.14159)"),
            }
            
            match &outer_arr[2] {
                Attribute::String(s) if s == "hello" => {},
                _ => panic!("Third element should be String(\"hello\")"),
            }
            
            match &outer_arr[3] {
                Attribute::Bool(true) => {},
                _ => panic!("Fourth element should be Bool(true)"),
            }
            
            match &outer_arr[4] {
                Attribute::Array(nested_arr) => {
                    assert_eq!(nested_arr.len(), 3);
                    match &nested_arr[2] {
                        Attribute::Array(deeply_nested) => {
                            assert_eq!(deeply_nested.len(), 1);
                            match &deeply_nested[0] {
                                Attribute::Int(3) => {},
                                _ => panic!("Deep nested element should be Int(3)"),
                            }
                        },
                        _ => panic!("Last element of nested array should be another array"),
                    }
                },
                _ => panic!("Fifth element should be Array"),
            }
        },
        _ => panic!("complex_array should be an Array"),
    }
}

// Test 7: Unicode and special character handling in names
#[test]
fn test_unicode_and_special_character_names() {
    let unicode_names = [
        "🚀_tensor",
        "café_résumé",
        "数据_tensor",  // Chinese characters
        "тест_tensor",  // Cyrillic characters  
        "áéíóú_tensor",  // Accented Latin
        "multi\nline\tname",
        "\0null_char_tensor",  // Null character
        "control\x01\x02\x03chars",  // Control characters
    ];

    for name in &unicode_names {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };

        assert_eq!(value.name, *name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    // Test with operations too
    for name in &unicode_names {
        let mut op = Operation::new(name);
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::I32,
            shape: vec![10],
        });

        assert_eq!(op.op_type, *name);
        assert_eq!(op.inputs.len(), 1);
    }
}

// Test 8: Very high-rank tensors (many dimensions)
#[test]
fn test_very_high_rank_tensors() {
    // Test tensors with many dimensions
    let high_rank_shapes = [
        vec![1; 10],      // 10 dimensions, all size 1
        vec![1; 100],     // 100 dimensions, all size 1
        vec![2; 10],      // 10 dimensions, all size 2 
        vec![2, 1, 1, 1, 1, 1, 1, 1, 1, 1], // 10 dimensions, mostly 1s
    ];

    for (i, shape) in high_rank_shapes.iter().enumerate() {
        let value = Value {
            name: format!("high_rank_tensor_{}", i),
            ty: Type::F64,
            shape: shape.clone(),
        };

        assert_eq!(value.shape.len(), shape.len());
        assert_eq!(value.shape, *shape);
        assert_eq!(value.ty, Type::F64);

        // Calculate expected number of elements
        let expected_elements: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected_elements));
    }
}

// Test 9: Special floating-point values in context of tensor calculations
#[test]
fn test_special_float_values_in_tensor_context() {
    let special_values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,  // Negative zero
        f64::EPSILON,
        f64::MIN_POSITIVE,
    ];

    for (i, &special_val) in special_values.iter().enumerate() {
        let attr = Attribute::Float(special_val);
        
        match attr {
            Attribute::Float(retrieved_val) => {
                if retrieved_val.is_nan() {
                    assert!(special_val.is_nan(), "Value {} should be NaN", i);
                } else {
                    assert_eq!(retrieved_val, special_val);
                    
                    if special_val.is_infinite() {
                        assert!(retrieved_val.is_infinite());
                        assert_eq!(retrieved_val.is_sign_positive(), special_val.is_sign_positive());
                    }
                }
            },
            _ => panic!("Expected Float attribute"),
        }
    }
    
    // Test that special floats don't affect tensor size calculations
    let nan_tensor = Value {
        name: "nan_tensor".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    };
    
    assert_eq!(nan_tensor.num_elements(), Some(25)); // Should still be 25 despite potential NaN values in data
}

// Test 10: Maximum length string attributes for memory stress
#[test]
fn test_maximum_length_string_attributes() {
    // Test progressively larger strings to test memory handling
    let sizes = [100, 1_000, 10_000, 100_000]; // 100 bytes to 100KB
    
    for size in &sizes {
        let large_string = "x".repeat(*size);
        let attr = Attribute::String(large_string.clone());
        
        match attr {
            Attribute::String(retrieved_str) => {
                assert_eq!(retrieved_str.len(), *size);
                assert_eq!(retrieved_str, large_string);
                
                // Verify content integrity
                for c in retrieved_str.chars() {
                    assert_eq!(c, 'x');
                }
            },
            _ => panic!("Expected String attribute"),
        }
    }
    
    // Test also with the longest possible string (within reason)
    let very_large_string = "A".repeat(1_000_000); // 1MB string
    let large_attr = Attribute::String(very_large_string.clone());
    
    match large_attr {
        Attribute::String(retrieved_str) => {
            assert_eq!(retrieved_str.len(), 1_000_000);
            assert_eq!(retrieved_str, very_large_string);
        },
        _ => panic!("Expected String attribute for large string"),
    }
}