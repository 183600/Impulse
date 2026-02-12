//! Comprehensive boundary edge tests - critical edge cases with standard library assertions

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Tensor with shape that would cause multiplication overflow (checked_mul)
#[test]
fn test_tensor_shape_overflow_protection() {
    // Large shape that would overflow if multiplied naively
    let overflow_shape = vec![usize::MAX, 2];
    let value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: overflow_shape,
    };
    // num_elements uses checked_mul, should return None for overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 2: Float attribute with denormal/subnormal values
#[test]
fn test_denormal_float_attributes() {
    let denormal_min = f64::MIN_POSITIVE; // Smallest positive normal
    let denormal_tiny = 1e-310_f64; // Subnormal range

    let attr1 = Attribute::Float(denormal_min);
    let attr2 = Attribute::Float(denormal_tiny);

    match attr1 {
        Attribute::Float(v) => assert!(v > 0.0),
        _ => panic!("Expected Float attribute"),
    }

    match attr2 {
        Attribute::Float(v) => {
            // May underflow to 0
            assert!(v >= 0.0);
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Type::Tensor with empty element shape (scalar tensor)
#[test]
fn test_tensor_with_scalar_element() {
    let scalar_tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![], // Empty shape for scalar element
    };

    match scalar_tensor_type {
        Type::Tensor { element_type, shape } => {
            let expected_shape: Vec<usize> = vec![];
            assert_eq!(shape, expected_shape);
            match *element_type {
                Type::F32 => {},
                _ => panic!("Expected F32 element type"),
            }
        },
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 4: Module with operations using maximum i64 attribute values
#[test]
fn test_module_with_extreme_i64_attributes() {
    let mut module = Module::new("extreme_attrs");
    let mut op = Operation::new("extreme_op");

    op.attributes.insert("max".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("zero".to_string(), Attribute::Int(0));
    op.attributes.insert("neg_one".to_string(), Attribute::Int(-1));

    module.add_operation(op);

    assert_eq!(module.operations[0].attributes.get("max"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(module.operations[0].attributes.get("min"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(module.operations[0].attributes.get("zero"), Some(&Attribute::Int(0)));
    assert_eq!(module.operations[0].attributes.get("neg_one"), Some(&Attribute::Int(-1)));
}

/// Test 5: Operation with all Bool attribute variations
#[test]
fn test_operation_all_bool_attributes() {
    let mut op = Operation::new("bool_test");

    op.attributes.insert("true_val".to_string(), Attribute::Bool(true));
    op.attributes.insert("false_val".to_string(), Attribute::Bool(false));

    assert_eq!(op.attributes.get("true_val"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("false_val"), Some(&Attribute::Bool(false)));
}

/// Test 6: Array attribute containing all different types mixed
#[test]
fn test_mixed_type_array_attribute() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]);

    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            // Verify each element type
            match &arr[0] { Attribute::Int(v) => assert_eq!(*v, 42), _ => panic!() };
            match &arr[1] { Attribute::Float(v) => assert!((v - 3.14).abs() < 1e-10), _ => panic!() };
            match &arr[2] { Attribute::String(s) => assert_eq!(s, "test"), _ => panic!() };
            match &arr[3] { Attribute::Bool(b) => assert_eq!(*b, true), _ => panic!() };
            match &arr[4] { Attribute::Array(_) => {}, _ => panic!() };
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: Value with shape containing usize::MAX in a safe context
#[test]
fn test_value_with_max_usize_dimension() {
    // Single dimension with usize::MAX
    let max_dim_value = Value {
        name: "max_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };

    assert_eq!(max_dim_value.shape, vec![usize::MAX]);
    assert_eq!(max_dim_value.num_elements(), Some(usize::MAX));
}

/// Test 8: Empty string attribute handling
#[test]
fn test_empty_string_attribute() {
    let empty_str = Attribute::String("".to_string());

    match empty_str {
        Attribute::String(s) => {
            assert_eq!(s.len(), 0);
            assert_eq!(s, "");
        },
        _ => panic!("Expected String attribute"),
    }
}

/// Test 9: Module with many operations to test vector capacity
#[test]
fn test_module_with_many_operations() {
    let mut module = Module::new("many_ops");

    // Add 1000 operations
    for i in 0..1000 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[999].op_type, "op_999");
}

/// Test 10: Nested tensor types with empty shapes at different levels
#[test]
fn test_nested_tensors_with_empty_shapes() {
    // Level 1: tensor<f32, []> (scalar)
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![],
    };

    // Level 2: tensor<tensor<f32, []>, []> (scalar of scalars)
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![],
    };

    match level2 {
        Type::Tensor { element_type, shape } => {
            let expected_shape: Vec<usize> = vec![];
            assert_eq!(shape, expected_shape);
            match element_type.as_ref() {
                Type::Tensor { element_type: inner, shape: inner_shape } => {
                    assert_eq!(inner_shape, &expected_shape);
                    match inner.as_ref() {
                        Type::F32 => {},
                        _ => panic!("Expected F32 at innermost level"),
                    }
                },
                _ => panic!("Expected nested Tensor"),
            }
        },
        _ => panic!("Expected Tensor type"),
    }
}