//! Focused boundary tests - essential edge cases with standard library assertions
//! Tests cover: overflow detection, NaN/infinity, extreme values, and memory safety

use crate::ir::{Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Overflow detection in num_elements using checked_mul
#[test]
fn test_overflow_detection_in_num_elements() {
    // Create a shape that will definitely overflow on 64-bit systems
    // usize::MAX ≈ 1.84e19, so we need a product larger than that
    // 10000000^3 = 1e21 > usize::MAX
    let large_val = 10_000_000usize;
    let multi_dim_overflow = Value {
        name: "multi_overflow".to_string(),
        ty: Type::I32,
        shape: vec![large_val, large_val, large_val], // 1e21 > usize::MAX
    };
    
    // Should return None due to overflow
    assert_eq!(multi_dim_overflow.num_elements(), None);
    
    // Test another overflow case
    // 100000 * 200000 * 100000 = 2e15 which fits in 64-bit
    // But 1000000 * 2000000 * 1000000 = 2e18 still fits
    // Use even larger values
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 2], // Will overflow
    };
    
    assert_eq!(overflow_value.num_elements(), None);
}

/// Test 2: Float NaN and infinity comparisons in attributes
#[test]
fn test_float_nan_infinity_comparisons() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    
    // Clone before match to avoid borrow issues
    let nan_attr_clone = nan_attr.clone();
    
    // NaN should not equal itself in standard float comparison
    match (nan_attr, nan_attr_clone) {
        (Attribute::Float(a), Attribute::Float(b)) => {
            // Standard float equality: NaN != NaN
            assert!(a != b);
            // But both should be NaN
            assert!(a.is_nan() && b.is_nan());
        }
        _ => panic!("Expected Float attributes"),
    }
    
    // Positive infinity should be greater than any finite value
    match pos_inf {
        Attribute::Float(inf) => {
            assert!(inf.is_infinite());
            assert!(inf.is_sign_positive());
            assert!(inf > f64::MAX);
        }
        _ => panic!("Expected Float attribute"),
    }
    
    // Negative infinity should be less than any finite value
    match neg_inf {
        Attribute::Float(inf) => {
            assert!(inf.is_infinite());
            assert!(inf.is_sign_negative());
            assert!(inf < f64::MIN);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Maximum safe shape without overflow
#[test]
fn test_max_safe_shape_without_overflow() {
    // Largest square that doesn't overflow in usize (sqrt(MAX) on 64-bit)
    let safe_square = Value {
        name: "safe_square".to_string(),
        ty: Type::F64,
        shape: vec![46340, 46340], // 46340^2 ≈ 2.1B < 2^32
    };
    
    assert_eq!(safe_square.num_elements(), Some(46340 * 46340));
    
    // Shape that fits exactly within 32-bit range
    let u32_max_shape = Value {
        name: "u32_max_shape".to_string(),
        ty: Type::I32,
        shape: vec![65536, 65536], // 2^16 * 2^16 = 2^32
    };
    
    let result = u32_max_shape.num_elements();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), 65536 * 65536);
}

/// Test 4: Empty arrays and nested structures
#[test]
fn test_empty_arrays_and_nested_structures() {
    let empty_array = Attribute::Array(vec![]);
    
    match empty_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 0);
            assert!(arr.is_empty());
        }
        _ => panic!("Expected empty Array"),
    }
    
    // Nested array with all empty sub-arrays
    let nested_empty = Attribute::Array(vec![
        Attribute::Array(vec![]),
        Attribute::Array(vec![]),
        Attribute::Array(vec![]),
    ]);
    
    match nested_empty {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
            for inner in outer {
                match inner {
                    Attribute::Array(arr) => assert!(arr.is_empty()),
                    _ => panic!("Expected inner Array"),
                }
            }
        }
        _ => panic!("Expected nested Array"),
    }
}

/// Test 5: Value with all zero dimensions
#[test]
fn test_value_with_all_zero_dimensions() {
    let zero_dim_value = Value {
        name: "zero_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0, 0],
    };
    
    // Product of all zeros is zero
    assert_eq!(zero_dim_value.num_elements(), Some(0));
    
    // Test with single zero dimension
    let single_zero = Value {
        name: "single_zero".to_string(),
        ty: Type::I64,
        shape: vec![0],
    };
    
    assert_eq!(single_zero.num_elements(), Some(0));
    assert_eq!(single_zero.shape[0], 0);
}

/// Test 6: Attribute with extreme integer values
#[test]
fn test_extreme_integer_values_in_attributes() {
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let zero = Attribute::Int(0);
    let neg_one = Attribute::Int(-1);
    let one = Attribute::Int(1);
    
    match max_int {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected MAX Int"),
    }
    
    match min_int {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected MIN Int"),
    }
    
    match (zero, neg_one, one) {
        (Attribute::Int(z), Attribute::Int(n), Attribute::Int(o)) => {
            assert_eq!(z, 0);
            assert_eq!(n, -1);
            assert_eq!(o, 1);
            // Verify ordering
            assert!(n < z && z < o);
        }
        _ => panic!("Expected Int attributes"),
    }
}

/// Test 7: Module with operation containing all attribute types
#[test]
fn test_operation_with_all_attribute_types() {
    let mut op = Operation::new("all_attrs");
    op.attributes.insert("int_attr".to_string(), Attribute::Int(42));
    op.attributes.insert("float_attr".to_string(), Attribute::Float(3.14159));
    op.attributes.insert("string_attr".to_string(), Attribute::String("test".to_string()));
    op.attributes.insert("bool_attr".to_string(), Attribute::Bool(true));
    op.attributes.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Int(2),
        Attribute::Int(3),
    ]));
    
    assert_eq!(op.attributes.len(), 5);
    assert!(op.attributes.contains_key("int_attr"));
    assert!(op.attributes.contains_key("float_attr"));
    assert!(op.attributes.contains_key("string_attr"));
    assert!(op.attributes.contains_key("bool_attr"));
    assert!(op.attributes.contains_key("array_attr"));
}

/// Test 8: Type validation for all basic types
#[test]
fn test_type_validation_basic_types() {
    let types = vec![
        (Type::F32, "F32"),
        (Type::F64, "F64"),
        (Type::I32, "I32"),
        (Type::I64, "I64"),
        (Type::Bool, "Bool"),
    ];
    
    for (ty, expected_name) in types {
        assert!(ty.is_valid_type());
        // Type implements Debug, so we can check string representation
        let debug_str = format!("{:?}", ty);
        assert!(debug_str.contains(expected_name));
    }
}

/// Test 9: Single-element and scalar tensors
#[test]
fn test_single_element_and_scalar_tensors() {
    // Scalar (0-dimensional)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    
    assert_eq!(scalar.num_elements(), Some(1));
    assert!(scalar.shape.is_empty());
    
    // Single element 1D tensor
    let single_1d = Value {
        name: "single_1d".to_string(),
        ty: Type::I32,
        shape: vec![1],
    };
    
    assert_eq!(single_1d.num_elements(), Some(1));
    assert_eq!(single_1d.shape.len(), 1);
    
    // Single element multi-dimensional tensor
    let single_3d = Value {
        name: "single_3d".to_string(),
        ty: Type::Bool,
        shape: vec![1, 1, 1],
    };
    
    assert_eq!(single_3d.num_elements(), Some(1));
    assert_eq!(single_3d.shape, vec![1, 1, 1]);
}

/// Test 10: Deep nesting without stack overflow
#[test]
fn test_deep_nesting_without_stack_overflow() {
    // Create a deeply nested tensor type to test stack limits
    let mut current_type = Type::F32;
    
    // Create 50 levels of nesting (reasonable depth)
    for _ in 0..50 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the type is valid
    assert!(current_type.is_valid_type());
    
    // Clone should work without issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}