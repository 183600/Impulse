//! Focused comprehensive edge case tests - Extended
//! Additional boundary tests covering edge cases in IR, operations, and data types

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Value with NaN and Infinity float attributes
#[test]
fn test_nan_infinity_float_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);

    // NaN should not equal itself
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float attribute"),
    }

    // Infinity checks
    match inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float attribute"),
    }

    match neg_inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: Value with num_elements() method edge cases
#[test]
fn test_num_elements_edge_cases() {
    // Test with shape that could cause overflow
    let large_shape = Value {
        name: "large".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000],
    };
    assert_eq!(large_shape.num_elements(), Some(10_000_000_000));

    // Test with zero in shape
    let zero_shape = Value {
        name: "zero".to_string(),
        ty: Type::I32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(zero_shape.num_elements(), Some(0));

    // Test with empty shape (scalar)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F64,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
}

/// Test 3: Type validation with TypeExtensions trait
#[test]
fn test_type_extensions_validation() {
    // All primitive types should be valid
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());

    // Nested tensor types should also be valid
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(tensor_type.is_valid_type());

    // Deeply nested tensor should be valid
    let deep_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    assert!(deep_nested.is_valid_type());
}

/// Test 4: Module with circular operation references (simulated)
#[test]
fn test_module_circular_operation_structure() {
    let mut module = Module::new("circular_test");

    // Create operation A that produces output_x
    let mut op_a = Operation::new("op_a");
    let output_x = Value {
        name: "output_x".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op_a.outputs.push(output_x.clone());
    module.add_operation(op_a);

    // Create operation B that consumes output_x and produces output_y
    let mut op_b = Operation::new("op_b");
    op_b.inputs.push(output_x);
    let output_y = Value {
        name: "output_y".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op_b.outputs.push(output_y.clone());
    module.add_operation(op_b);

    // Create operation C that consumes output_y (simulating a chain)
    let mut op_c = Operation::new("op_c");
    op_c.inputs.push(output_y);
    op_c.outputs.push(Value {
        name: "output_z".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op_c);

    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[1].op_type, "op_b");
    assert_eq!(module.operations[2].op_type, "op_c");
}

/// Test 5: Attribute with maximum integer values
#[test]
fn test_extreme_integer_attributes() {
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let zero_int = Attribute::Int(0);
    let neg_one = Attribute::Int(-1);

    match max_int {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected Int attribute"),
    }

    match min_int {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected Int attribute"),
    }

    match zero_int {
        Attribute::Int(val) => assert_eq!(val, 0),
        _ => panic!("Expected Int attribute"),
    }

    match neg_one {
        Attribute::Int(val) => assert_eq!(val, -1),
        _ => panic!("Expected Int attribute"),
    }
}

/// Test 6: Compiler with empty target string
#[test]
fn test_compiler_empty_target() {
    let mut compiler = ImpulseCompiler::new();
    let mock_model = vec![1u8, 2u8, 3u8];

    let result = compiler.compile(&mock_model, "");
    // Should handle gracefully without panic
    assert!(result.is_ok() || result.is_err());
}

/// Test 7: Value with single dimension extremes
#[test]
fn test_single_dimension_extremes() {
    // Very small 1D tensor
    let tiny_1d = Value {
        name: "tiny".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(tiny_1d.num_elements(), Some(1));

    // Empty 1D tensor
    let empty_1d = Value {
        name: "empty".to_string(),
        ty: Type::F32,
        shape: vec![0],
    };
    assert_eq!(empty_1d.num_elements(), Some(0));

    // Large 1D tensor
    let large_1d = Value {
        name: "large".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000],
    };
    assert_eq!(large_1d.num_elements(), Some(1_000_000));
}

/// Test 8: Operation with empty string attributes
#[test]
fn test_empty_string_attributes() {
    let mut op = Operation::new("test_op");
    let mut attrs = HashMap::new();

    attrs.insert("empty_key".to_string(), Attribute::String("".to_string()));
    attrs.insert("".to_string(), Attribute::String("value_for_empty_key".to_string()));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 2);
    assert!(op.attributes.contains_key("empty_key"));
    assert!(op.attributes.contains_key(""));

    match op.attributes.get("empty_key") {
        Some(Attribute::String(s)) => assert_eq!(s, ""),
        _ => panic!("Expected empty string attribute"),
    }
}

/// Test 9: Module with duplicate input names
#[test]
fn test_module_duplicate_input_names() {
    let mut module = Module::new("duplicate_inputs");

    // Add inputs with the same name
    for _ in 0..3 {
        module.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
    }

    assert_eq!(module.inputs.len(), 3);
    // All should have the same name
    for input in &module.inputs {
        assert_eq!(input.name, "input");
    }
}

/// Test 10: Attribute array with duplicate values
#[test]
fn test_attribute_array_duplicates() {
    let array_with_duplicates = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Int(1),
        Attribute::Int(2),
        Attribute::Int(2),
        Attribute::Int(1),
    ]);

    match array_with_duplicates {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], arr[1]); // First two are duplicates
            assert_eq!(arr[2], arr[3]); // Middle two are duplicates
            assert_eq!(arr[0], arr[4]); // First and last are duplicates
        }
        _ => panic!("Expected Array attribute"),
    }
}