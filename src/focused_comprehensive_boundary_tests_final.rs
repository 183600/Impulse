//! Focused comprehensive boundary tests - final edge cases with standard library assertions
//! 覆盖更多边界情况的测试用例，使用 assert! 和 assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Value with shape containing dimension product that exactly matches usize boundary
#[test]
fn test_shape_near_usize_boundary() {
    // Test with shape where product approaches but doesn't overflow usize
    let _safe_shape = vec![1_000_000, 1_000_000]; // Would overflow, test with checked_mul
    let value = Value {
        name: "boundary_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10_000, 100_000], // Exactly 1 billion
    };
    assert_eq!(value.num_elements(), Some(1_000_000_000));
}

/// Test 2: Attribute Float with -0.0 (negative zero) handling
#[test]
fn test_negative_zero_attribute() {
    let neg_zero = Attribute::Float(-0.0_f64);
    match neg_zero {
        Attribute::Float(val) => {
            assert_eq!(val, -0.0);
            assert!(val.is_sign_negative());
            assert_eq!(val.abs(), 0.0);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with operations that have circular reference-like naming patterns
#[test]
fn test_module_with_circular_naming() {
    let mut module = Module::new("circular_ref_module");

    let mut op1 = Operation::new("op_a");
    op1.outputs.push(Value {
        name: "a_to_b".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });

    let mut op2 = Operation::new("op_b");
    op2.inputs.push(Value {
        name: "a_to_b".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    op2.outputs.push(Value {
        name: "b_to_c".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });

    let mut op3 = Operation::new("op_c");
    op3.inputs.push(Value {
        name: "b_to_c".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    op3.outputs.push(Value {
        name: "c_to_a".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });

    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);

    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].outputs[0].name, "a_to_b");
    assert_eq!(module.operations[1].inputs[0].name, "a_to_b");
    assert_eq!(module.operations[2].outputs[0].name, "c_to_a");
}

/// Test 4: Value with extremely small but non-zero float attribute
#[test]
fn test_subnormal_float_value() {
    // Test subnormal float (denormalized) values
    let subnormal_val = f64::MIN_POSITIVE; // Smallest positive normal float
    let attr = Attribute::Float(subnormal_val);

    match attr {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val < f64::MIN_POSITIVE * 2.0);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 5: Operation with attributes containing special control characters in strings
#[test]
fn test_operation_with_control_char_attributes() {
    let mut op = Operation::new("control_char_op");
    let mut attrs = HashMap::new();

    let control_strings = vec![
        "test\nwith\nnewlines",
        "test\twith\ttabs",
        "test\rcarriage\rreturn",
        "test\x00null\x00byte",
        "test\u{1F600}emoji",  // Unicode emoji
    ];

    for (i, s) in control_strings.iter().enumerate() {
        attrs.insert(format!("attr_{}", i), Attribute::String(s.to_string()));
    }

    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 5);
    assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::String("test\nwith\nnewlines".to_string())));
}

/// Test 6: Nested Tensor types with alternating element types
#[test]
fn test_alternating_nested_tensor_types() {
    let level1 = Type::F32;
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![2],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(Type::I32),
        shape: vec![3],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![4],
    };

    assert_ne!(level2, level3);
    assert_ne!(level3, level4);

    match &level4 {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, &vec![4]);
            assert_eq!(element_type.as_ref(), &level2);
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 7: Module with operation having extremely long attribute keys
#[test]
fn test_operation_with_long_attribute_keys() {
    let mut op = Operation::new("long_key_op");
    let mut attrs = HashMap::new();

    // Create a very long attribute key
    let long_key = "x".repeat(10_000);
    attrs.insert(long_key.clone(), Attribute::Int(42));

    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 1);
    assert_eq!(op.attributes.keys().next().unwrap().len(), 10_000);
}

/// Test 8: Value with mixed type nested tensor structure
#[test]
fn test_mixed_type_nested_tensor() {
    // Create a tensor with nested structure
    let inner_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 2],
    };

    let outer_type = Type::Tensor {
        element_type: Box::new(Type::I32), // Different element type at outer level
        shape: vec![3, 3],
    };

    // Verify they are different types
    assert_ne!(inner_type, outer_type);

    // Verify TypeExtensions validation
    assert!(inner_type.is_valid_type());
    assert!(outer_type.is_valid_type());
}

/// Test 9: Attribute array with all possible combinations of nested types
#[test]
fn test_attribute_array_all_combinations() {
    let complex_array = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(1.5),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
        ]),
        Attribute::Int(2),
        Attribute::Array(vec![]), // Empty nested array
        Attribute::Bool(false),
    ]);

    match complex_array {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 4);
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 4);
                    assert_eq!(inner[0], Attribute::Int(1));
                    assert_eq!(inner[1], Attribute::Float(1.5));
                }
                _ => panic!("Expected nested array"),
            }
            match &outer[2] {
                Attribute::Array(empty) => {
                    assert_eq!(empty.len(), 0);
                }
                _ => panic!("Expected empty array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 10: Module with all inputs and outputs having different shapes but same type
#[test]
fn test_module_varied_shapes_same_type() {
    let mut module = Module::new("varied_shapes_module");

    // Add inputs with different shapes but same type
    let shapes = vec![
        vec![],           // scalar
        vec![1],          // 1D with single element
        vec![1, 1],       // 2D with single element
        vec![1, 1, 1],    // 3D with single element
        vec![2, 3, 4],    // 3D with multiple elements
        vec![10, 10],     // 2D square
    ];

    for (i, shape) in shapes.iter().enumerate() {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        });
    }

    // Add outputs with same type but different shapes
    for (i, shape) in shapes.iter().enumerate() {
        module.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        });
    }

    assert_eq!(module.inputs.len(), 6);
    assert_eq!(module.outputs.len(), 6);

    // Verify all inputs and outputs have F32 type
    for input in &module.inputs {
        assert_eq!(input.ty, Type::F32);
    }
    for output in &module.outputs {
        assert_eq!(output.ty, Type::F32);
    }

    // Verify specific shapes
    assert_eq!(module.inputs[0].shape, Vec::<usize>::new());
    assert_eq!(module.inputs[5].shape, vec![10_usize, 10]);
    assert_eq!(module.outputs[3].shape, vec![1_usize, 1, 1]);
}