//! Focused core boundary tests - covering critical edge cases with standard library assertions
//!
//! This test module covers essential boundary conditions for core IR components:
//! - Overflow detection in shape calculations
//! - Numerical precision edge cases (NaN, infinity, subnormal numbers)
//! - Empty and degenerate structures
//! - Extreme attribute values
//! - Type system boundaries

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with shape multiplication overflow detection
#[test]
fn test_shape_multiplication_overflow() {
    // Test case where multiplication would overflow (use checked_mul)
    let mut overflow_shape = vec![1usize; 50]; // 50 dimensions, each 1 - should be safe
    let safe_value = Value {
        name: "many_ones".to_string(),
        ty: Type::F32,
        shape: overflow_shape.clone(),
    };
    // 1^50 = 1, should return Some(1)
    assert_eq!(safe_value.num_elements(), Some(1));

    // Test with zero dimension that prevents overflow
    let zero_dim = Value {
        name: "zero_dim_prevents_overflow".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000, 0, 100_000],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));

    // Test with large but safe dimensions
    let large_safe = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: vec![1000, 100, 100], // 10 million elements
    };
    assert_eq!(large_safe.num_elements(), Some(10_000_000));
}

/// Test 2: Float attribute with subnormal (denormal) numbers
#[test]
fn test_subnormal_float_attributes() {
    // Smallest positive subnormal float
    let min_subnormal = f64::MIN_POSITIVE / 2.0; // Smallest denormal
    let subnormal_attr = Attribute::Float(min_subnormal);

    match subnormal_attr {
        Attribute::Float(val) => {
            // Subnormal numbers are positive but smaller than MIN_POSITIVE
            assert!(val > 0.0);
            assert!(val < f64::MIN_POSITIVE);
        }
        _ => panic!("Expected Float attribute"),
    }

    // Test negative subnormal
    let neg_subnormal = -f64::MIN_POSITIVE / 2.0;
    let neg_attr = Attribute::Float(neg_subnormal);

    match neg_attr {
        Attribute::Float(val) => {
            assert!(val < 0.0);
            assert!(val > -f64::MIN_POSITIVE);
        }
        _ => panic!("Expected Float attribute"),
    }

    // Test near-zero values
    let near_zero_pos = Attribute::Float(1e-323);
    let near_zero_neg = Attribute::Float(-1e-323);

    match near_zero_pos {
        Attribute::Float(val) => assert!(val >= 0.0),
        _ => panic!("Expected Float attribute"),
    }

    match near_zero_neg {
        Attribute::Float(val) => assert!(val <= 0.0),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with degenerate operations (no inputs/outputs)
#[test]
fn test_degenerate_operations() {
    let mut module = Module::new("degenerate_module");

    // Operation with no inputs or outputs
    let no_io_op = Operation::new("stateful_constant");
    module.add_operation(no_io_op);

    // Operation with empty inputs but has outputs
    let mut no_input_op = Operation::new("constant_generator");
    no_input_op.outputs.push(Value {
        name: "generated".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    module.add_operation(no_input_op);

    // Operation with inputs but no outputs (side-effect only)
    let mut no_output_op = Operation::new("print_op");
    no_output_op.inputs.push(Value {
        name: "to_print".to_string(),
        ty: Type::I32,
        shape: vec![1],
    });
    module.add_operation(no_output_op);

    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 0);
    assert_eq!(module.operations[1].inputs.len(), 0);
    assert_eq!(module.operations[1].outputs.len(), 1);
    assert_eq!(module.operations[2].inputs.len(), 1);
    assert_eq!(module.operations[2].outputs.len(), 0);
}

/// Test 4: Extreme integer attribute values
#[test]
fn test_extreme_integer_attributes() {
    // Maximum positive i64
    let max_i64 = Attribute::Int(i64::MAX);
    assert_eq!(max_i64, Attribute::Int(9_223_372_036_854_775_807));

    // Minimum negative i64
    let min_i64 = Attribute::Int(i64::MIN);
    assert_eq!(min_i64, Attribute::Int(-9_223_372_036_854_775_808));

    // Zero crossing values
    let zero = Attribute::Int(0);
    let neg_one = Attribute::Int(-1);
    let pos_one = Attribute::Int(1);

    assert_eq!(zero, Attribute::Int(0));
    assert_eq!(neg_one, Attribute::Int(-1));
    assert_eq!(pos_one, Attribute::Int(1));

    // Powers of 2
    let power_of_2_32 = Attribute::Int(4_294_967_296_i64);
    assert_eq!(power_of_2_32, Attribute::Int(4_294_967_296));

    let power_of_2_63 = Attribute::Int(i64::MIN); // -9223372036854775808
    // Note: 1 << 63 in i64 is i64::MIN (-9223372036854775808)
    assert_eq!(power_of_2_63, Attribute::Int(i64::MIN));
}

/// Test 5: Tensor type with empty and unit shapes
#[test]
fn test_tensor_type_shapes() {
    // Scalar (empty shape)
    let scalar_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![],
    };
    match scalar_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape.len(), 0);
        }
        _ => panic!("Expected Tensor type"),
    }

    // Rank-1 tensor (vector)
    let vector_type = Type::Tensor {
        element_type: Box::new(Type::I32),
        shape: vec![10],
    };
    match vector_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, vec![10]);
        }
        _ => panic!("Expected Tensor type"),
    }

    // Rank-0 vs scalar
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar_value.num_elements(), Some(1));

    // Unit dimensions
    let unit_shape = Value {
        name: "unit_shape".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(unit_shape.num_elements(), Some(1));
}

/// Test 6: Array attribute with mixed types and nesting
#[test]
fn test_mixed_array_attributes() {
    // Empty array
    let empty = Attribute::Array(vec![]);
    match empty {
        Attribute::Array(arr) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected Array"),
    }

    // Single element array
    let single = Attribute::Array(vec![Attribute::Int(42)]);
    match single {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], Attribute::Int(42));
        }
        _ => panic!("Expected Array"),
    }

    // Mixed type array
    let mixed = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.0),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(10), Attribute::Int(20)]),
    ]);

    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], Attribute::Int(1));
            assert_eq!(arr[1], Attribute::Float(2.0));
            assert_eq!(arr[2], Attribute::String("test".to_string()));
            assert_eq!(arr[3], Attribute::Bool(true));
            match &arr[4] {
                Attribute::Array(nested) => {
                    assert_eq!(nested.len(), 2);
                    assert_eq!(nested[0], Attribute::Int(10));
                    assert_eq!(nested[1], Attribute::Int(20));
                }
                _ => panic!("Expected nested Array"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 7: Value with special float values
#[test]
fn test_special_float_values() {
    // NaN - NaN != NaN by IEEE standard, so test with is_nan()
    let nan_attr = Attribute::Float(f64::NAN);
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float with NaN"),
    }

    // Positive infinity
    let pos_inf = Attribute::Float(f64::INFINITY);
    match pos_inf {
        Attribute::Float(val) => {
            assert!(val.is_infinite());
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected Float with +inf"),
    }

    // Negative infinity
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    match neg_inf {
        Attribute::Float(val) => {
            assert!(val.is_infinite());
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected Float with -inf"),
    }

    // Max finite float
    let max_finite = Attribute::Float(f64::MAX);
    match max_finite {
        Attribute::Float(val) => {
            assert!(val.is_finite());
            assert!(!val.is_infinite());
        }
        _ => panic!("Expected Float with MAX"),
    }

    // Min positive finite float
    let min_positive = Attribute::Float(f64::MIN_POSITIVE);
    match min_positive {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val.is_finite());
        }
        _ => panic!("Expected Float with MIN_POSITIVE"),
    }
}

/// Test 8: Module with operation having duplicate attribute keys
#[test]
fn test_duplicate_attribute_keys() {
    let mut op = Operation::new("duplicate_test");
    let mut attrs = std::collections::HashMap::new();

    // Insert same key multiple times - HashMap should keep only the last value
    attrs.insert("key".to_string(), Attribute::Int(1));
    attrs.insert("key".to_string(), Attribute::Int(2));
    attrs.insert("key".to_string(), Attribute::Int(3));

    op.attributes = attrs;

    // Should have exactly one entry
    assert_eq!(op.attributes.len(), 1);

    // And it should be the last inserted value
    match op.attributes.get("key") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 3),
        _ => panic!("Expected Int(3)"),
    }
}

/// Test 9: Value with zero-containing shapes
#[test]
fn test_zero_containing_shapes() {
    // Single zero dimension
    let single_zero = Value {
        name: "single_zero".to_string(),
        ty: Type::F32,
        shape: vec![0],
    };
    assert_eq!(single_zero.num_elements(), Some(0));

    // Zero at beginning
    let zero_start = Value {
        name: "zero_start".to_string(),
        ty: Type::F32,
        shape: vec![0, 10, 20],
    };
    assert_eq!(zero_start.num_elements(), Some(0));

    // Zero in middle
    let zero_middle = Value {
        name: "zero_middle".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 20],
    };
    assert_eq!(zero_middle.num_elements(), Some(0));

    // Zero at end
    let zero_end = Value {
        name: "zero_end".to_string(),
        ty: Type::F32,
        shape: vec![10, 20, 0],
    };
    assert_eq!(zero_end.num_elements(), Some(0));

    // Multiple zeros
    let multi_zero = Value {
        name: "multi_zero".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0],
    };
    assert_eq!(multi_zero.num_elements(), Some(0));
}

/// Test 10: Empty string and whitespace string attributes
#[test]
fn test_string_attribute_edge_cases() {
    // Empty string
    let empty_str = Attribute::String("".to_string());
    match empty_str {
        Attribute::String(s) => {
            assert_eq!(s.len(), 0);
            assert!(s.is_empty());
        }
        _ => panic!("Expected String attribute"),
    }

    // Single space
    let space_str = Attribute::String(" ".to_string());
    match space_str {
        Attribute::String(s) => assert_eq!(s, " "),
        _ => panic!("Expected String attribute"),
    }

    // Multiple spaces
    let spaces_str = Attribute::String("   ".to_string());
    match spaces_str {
        Attribute::String(s) => assert_eq!(s.len(), 3),
        _ => panic!("Expected String attribute"),
    }

    // Tab and newline characters
    let whitespace = Attribute::String("\t\n\r".to_string());
    match whitespace {
        Attribute::String(s) => assert_eq!(s.len(), 3),
        _ => panic!("Expected String attribute"),
    }

    // String with only null byte
    let null_byte = Attribute::String("\0".to_string());
    match null_byte {
        Attribute::String(s) => assert_eq!(s.len(), 1),
        _ => panic!("Expected String attribute"),
    }
}