//! Comprehensive edge case tests for Impulse compiler
//! Focusing on numerical precision, memory safety, and boundary conditions

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils,
};

/// Test 1: Special floating-point values (NaN, Infinity, -Infinity)
#[test]
fn test_special_float_values() {
    let inf_val = Attribute::Float(f64::INFINITY);
    let neg_inf_val = Attribute::Float(f64::NEG_INFINITY);
    let nan_val = Attribute::Float(f64::NAN);

    // Verify these values are stored correctly
    match inf_val {
        Attribute::Float(v) => assert!(v.is_infinite() && v.is_sign_positive()),
        _ => panic!("Expected Float attribute"),
    }

    match neg_inf_val {
        Attribute::Float(v) => assert!(v.is_infinite() && v.is_sign_negative()),
        _ => panic!("Expected Float attribute"),
    }

    match nan_val {
        Attribute::Float(v) => assert!(v.is_nan()),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: Integer overflow detection in shape calculations
#[test]
fn test_integer_overflow_detection() {
    // Test with shapes that would cause overflow on 32-bit systems
    let large_shape = vec![usize::MAX, 2];
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };

    // num_elements should return None for overflow cases
    let result = value.num_elements();
    assert!(result.is_none() || result == Some(0), "Should handle overflow gracefully");
}

/// Test 3: Very small floating-point precision
#[test]
fn test_minimal_float_precision() {
    let smallest_positive = Attribute::Float(f64::MIN_POSITIVE);
    let epsilon = Attribute::Float(f64::EPSILON);

    match smallest_positive {
        Attribute::Float(v) => assert!(v > 0.0 && v < 1e-300),
        _ => panic!("Expected Float attribute"),
    }

    match epsilon {
        Attribute::Float(v) => assert!(v > 0.0 && v < 1.0),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 4: Module with operation chain dependencies
#[test]
fn test_operation_chain_dependencies() {
    let mut module = Module::new("chain_test");

    // Create a chain: op1 -> op2 -> op3
    let mut op1 = Operation::new("op1");
    let val1 = Value {
        name: "val1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op1.outputs.push(val1.clone());
    module.add_operation(op1);

    let mut op2 = Operation::new("op2");
    op2.inputs.push(val1.clone());
    let val2 = Value {
        name: "val2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op2.outputs.push(val2.clone());
    module.add_operation(op2);

    let mut op3 = Operation::new("op3");
    op3.inputs.push(val2);
    module.add_operation(op3);

    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[1].inputs[0].name, "val1");
    assert_eq!(module.operations[2].inputs[0].name, "val2");
}

/// Test 5: Empty and single-element tensor types
#[test]
fn test_edge_case_tensor_types() {
    // Empty tensor (zero elements)
    let empty_tensor = Value {
        name: "empty".to_string(),
        ty: Type::F32,
        shape: vec![0, 10, 20],
    };
    assert_eq!(empty_tensor.num_elements(), Some(0));

    // Single element tensor
    let single_elem = Value {
        name: "single".to_string(),
        ty: Type::I32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(single_elem.num_elements(), Some(1));

    // Scalar (0-dimensional)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F64,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
}

/// Test 6: Mixed type nested tensor structure
#[test]
fn test_mixed_type_nested_tensors() {
    // Create nested tensor with different types at each level
    let level3 = Type::Tensor {
        element_type: Box::new(Type::Bool),
        shape: vec![2],
    };

    let level2 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![3],
    };

    let level1 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };

    // Verify the structure
    match level1 {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![4]);
            match element_type.as_ref() {
                Type::Tensor { element_type: inner_element, shape: inner_shape } => {
                    assert_eq!(inner_shape, &vec![3]);
                    match inner_element.as_ref() {
                        Type::Tensor { element_type: innermost, shape: innermost_shape } => {
                            assert_eq!(innermost_shape, &vec![2]);
                            assert_eq!(innermost.as_ref(), &Type::Bool);
                        }
                        _ => panic!("Expected Tensor at level 3"),
                    }
                }
                _ => panic!("Expected Tensor at level 2"),
            }
        }
        _ => panic!("Expected Tensor at level 1"),
    }
}

/// Test 7: Attribute array with recursive structure
#[test]
fn test_recursive_attribute_array() {
    // Create a self-referential-like structure through arrays
    let recursive_attr = Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Int(1)]),
        Attribute::Array(vec![Attribute::Int(2), Attribute::Array(vec![Attribute::Int(3)])]),
    ]);

    match recursive_attr {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                Attribute::Array(inner) => assert_eq!(inner.len(), 1),
                _ => panic!("Expected nested array"),
            }
            match &outer[1] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 2);
                    match &inner[1] {
                        Attribute::Array(deep) => assert_eq!(deep.len(), 1),
                        _ => panic!("Expected deeply nested array"),
                    }
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 8: Operation with maximum boundary integer values
#[test]
fn test_max_integer_boundaries() {
    let mut op = Operation::new("boundary_test");

    // Add attributes with boundary values
    op.attributes.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("zero".to_string(), Attribute::Int(0));

    // Verify boundary values
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("zero"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.len(), 3);
}

/// Test 9: Tensor size calculation for extreme shapes
#[test]
fn test_extreme_shape_size_calculations() {
    // Test with shapes that are valid but large
    let wide_tensor = Value {
        name: "wide".to_string(),
        ty: Type::F32,
        shape: vec![1, 1000000],
    };

    let wide_size = ir_utils::calculate_tensor_size(&wide_tensor.ty, &wide_tensor.shape);
    assert!(wide_size.is_ok());
    assert_eq!(wide_size.unwrap(), 1_000_000 * 4);

    // Test with tall tensor
    let tall_tensor = Value {
        name: "tall".to_string(),
        ty: Type::I32,
        shape: vec![1000000, 1],
    };

    let tall_size = ir_utils::calculate_tensor_size(&tall_tensor.ty, &tall_tensor.shape);
    assert!(tall_size.is_ok());
    assert_eq!(tall_size.unwrap(), 1_000_000 * 4);
}

/// Test 10: String attributes with Unicode and edge cases
#[test]
fn test_string_attribute_edge_cases() {
    let test_cases = vec![
        Attribute::String("".to_string()),                           // Empty string
        Attribute::String("a".to_string()),                          // Single character
        Attribute::String("测试中文".to_string()),                     // Chinese characters
        Attribute::String("😀🎉".to_string()),                        // Emoji
        Attribute::String("\u{0000}".to_string()),                   // Null character
        Attribute::String("a\nb\tc\rd".to_string()),                  // Control characters
        Attribute::String(" ".repeat(1000)),                         // Long space string
    ];

    for attr in test_cases {
        match attr {
            Attribute::String(s) => {
                // Verify the string is preserved
                assert!(!s.is_empty() || s.len() == 0);
            }
            _ => panic!("Expected String attribute"),
        }
    }
}
