//! New boundary coverage tests - testing edge cases with standard assertions
//! This module covers boundary scenarios including overflow, precision, and extreme values

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Module with name containing special characters and Unicode
#[test]
fn test_module_special_characters_name() {
    let special_names = [
        "test-module_2024",
        "测试模块",
        "module@special#chars",
        "module with spaces",
        "module/with/slashes",
    ];

    for name in special_names {
        let module = Module::new(name);
        assert_eq!(module.name, name);
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }
}

/// Test 2: Value with shape that would cause multiplication overflow
#[test]
fn test_value_overflow_detection() {
    // Shape that would overflow when multiplied
    let large_shape = vec![usize::MAX, 2];
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: large_shape.clone(),
    };

    // num_elements should return None for overflow
    assert_eq!(value.num_elements(), None);

    // Non-overflowing case
    let safe_shape = vec![1000, 1000];
    let safe_value = Value {
        name: "safe_test".to_string(),
        ty: Type::F32,
        shape: safe_shape.clone(),
    };
    assert_eq!(safe_value.num_elements(), Some(1_000_000));
}

/// Test 3: Attribute with extreme numeric values
#[test]
fn test_attribute_extreme_numeric_values() {
    let extreme_attrs = [
        Attribute::Int(i64::MAX),
        Attribute::Int(i64::MIN),
        Attribute::Int(0),
        Attribute::Float(f64::MAX),
        Attribute::Float(f64::MIN),
        Attribute::Float(-0.0),
        Attribute::Float(0.0),
        Attribute::Float(std::f64::consts::PI),
        Attribute::Float(std::f64::consts::E),
    ];

    for attr in extreme_attrs {
        match attr {
            Attribute::Int(val) => {
                assert!(val == i64::MAX || val == i64::MIN || val == 0);
            }
            Attribute::Float(val) => {
                assert!(!val.is_nan() || val.is_finite());
            }
            _ => panic!("Expected Int or Float attribute"),
        }
    }
}

/// Test 4: Operation with empty name string
#[test]
fn test_operation_empty_name() {
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert!(op.inputs.is_empty());
    assert!(op.outputs.is_empty());
    assert!(op.attributes.is_empty());
}

/// Test 5: Module with very long operation chain
#[test]
fn test_module_long_operation_chain() {
    let mut module = Module::new("chain_test");
    const CHAIN_LENGTH: usize = 100;

    for i in 0..CHAIN_LENGTH {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), CHAIN_LENGTH);
    for i in 0..CHAIN_LENGTH {
        assert_eq!(module.operations[i].op_type, format!("op_{}", i));
    }
}

/// Test 6: Type validation for deeply nested tensor types
#[test]
fn test_deeply_nested_tensor_types() {
    let depth1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![1],
    };

    let depth2 = Type::Tensor {
        element_type: Box::new(depth1.clone()),
        shape: vec![1],
    };

    let depth3 = Type::Tensor {
        element_type: Box::new(depth2.clone()),
        shape: vec![1],
    };

    // All should be valid
    assert!(depth1.is_valid_type());
    assert!(depth2.is_valid_type());
    assert!(depth3.is_valid_type());

    // They should not be equal
    assert_ne!(depth1, depth2);
    assert_ne!(depth2, depth3);
}

/// Test 7: Value with single dimension of size 1 (vector)
#[test]
fn test_value_single_dimension_vectors() {
    let test_cases = [
        (vec![1], 1),
        (vec![10], 10),
        (vec![1000000], 1000000),
    ];

    for (shape, expected_elements) in test_cases {
        let value = Value {
            name: "vector".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };
        assert_eq!(value.num_elements(), Some(expected_elements));
        assert_eq!(value.shape, shape);
    }
}

/// Test 8: Attribute array with mixed types including nested structures
#[test]
fn test_attribute_mixed_nested_array() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.5),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Float(3.14),
        ]),
    ]);

    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            // Verify the nested array is the 5th element
            match &arr[4] {
                Attribute::Array(nested) => {
                    assert_eq!(nested.len(), 2);
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Operation with large number of attributes
#[test]
fn test_operation_many_attributes() {
    let mut op = Operation::new("many_attrs");
    let mut attrs = HashMap::new();

    const NUM_ATTRS: usize = 50;
    for i in 0..NUM_ATTRS {
        let key = format!("attr_{}", i);
        let value = match i % 4 {
            0 => Attribute::Int(i as i64),
            1 => Attribute::Float(i as f64),
            2 => Attribute::String(format!("value_{}", i)),
            _ => Attribute::Bool(i % 2 == 0),
        };
        attrs.insert(key, value);
    }

    op.attributes = attrs;
    assert_eq!(op.attributes.len(), NUM_ATTRS);
}

/// Test 10: Module with inputs and outputs of same type but different shapes
#[test]
fn test_module_same_type_different_shapes() {
    let mut module = Module::new("shape_conversion");

    // Input: 1D tensor
    module.inputs.push(Value {
        name: "input_1d".to_string(),
        ty: Type::F32,
        shape: vec![100],
    });

    // Output: 2D tensor (reshaped)
    module.outputs.push(Value {
        name: "output_2d".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });

    // Another output: 3D tensor (reshaped differently)
    module.outputs.push(Value {
        name: "output_3d".to_string(),
        ty: Type::F32,
        shape: vec![2, 5, 10],
    });

    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].ty, module.outputs[0].ty);
    assert_eq!(module.inputs[0].ty, module.outputs[1].ty);

    // All should have same number of elements
    assert_eq!(module.inputs[0].num_elements(), Some(100));
    assert_eq!(module.outputs[0].num_elements(), Some(100));
    assert_eq!(module.outputs[1].num_elements(), Some(100));
}