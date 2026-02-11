//! Focused edge boundary comprehensive tests - Additional boundary scenarios with assert! and assert_eq!
//! This module contains 10 test cases covering various edge cases and boundary conditions

use crate::{
    ir::{Module, Value, Type, Operation, Attribute, TypeExtensions},
    ImpulseCompiler,
    utils::ir_utils,
};

/// Test 1: NaN and Infinity float attribute handling
#[test]
fn test_nan_infinity_float_attributes() {
    // Test with positive infinity
    let pos_inf = Attribute::Float(f64::INFINITY);
    if let Attribute::Float(val) = pos_inf {
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
    }

    // Test with negative infinity
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    if let Attribute::Float(val) = neg_inf {
        assert!(val.is_infinite());
        assert!(val.is_sign_negative());
    }

    // Test with NaN
    let nan_val = Attribute::Float(f64::NAN);
    if let Attribute::Float(val) = nan_val {
        assert!(val.is_nan());
    }

    // Test with negative zero
    let neg_zero = Attribute::Float(-0.0);
    if let Attribute::Float(val) = neg_zero {
        assert_eq!(val, 0.0);
        assert!(val.is_sign_negative());
    }
}

/// Test 2: Operation with extremely large attribute values
#[test]
fn test_operation_extreme_attribute_values() {
    use std::collections::HashMap;

    let mut op = Operation::new("extreme_attr_op");
    let mut attrs = HashMap::new();

    // Test with i64 boundary values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("negative_one".to_string(), Attribute::Int(-1));

    // Test with f64 boundary values
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_positive_f64".to_string(), Attribute::Float(f64::MIN_POSITIVE));

    // Test with extremely long string
    let long_string = "a".repeat(10000);
    attrs.insert("long_string".to_string(), Attribute::String(long_string));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 6);
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("negative_one"), Some(&Attribute::Int(-1)));

    if let Attribute::String(s) = op.attributes.get("long_string").unwrap() {
        assert_eq!(s.len(), 10000);
    }
}

/// Test 3: Value with overflow-safe shape calculation
#[test]
fn test_value_overflow_safe_shape_calculation() {
    // Test with shapes that could potentially overflow
    let shapes = [
        vec![1, 1, 1, 1, 1],      // Small shape
        vec![100, 100, 100],      // Medium shape (1,000,000 elements)
        vec![2, 2, 2, 2, 2, 2],   // 64 elements
        vec![2, usize::MAX],      // Would overflow
    ];

    for shape in shapes.iter() {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // Use the safe num_elements method
        let num_elements = value.num_elements();

        // Verify the calculation is safe
        if shape.len() > 1 && shape.iter().any(|&dim| dim == usize::MAX) {
            // Should return None for overflow case
            assert_eq!(num_elements, None);
        } else {
            // Should return Some for valid cases
            assert!(num_elements.is_some());
        }
    }

    // Test specifically with usize::MAX single element (doesn't overflow)
    let max_single = Value {
        name: "max_single".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };
    assert_eq!(max_single.num_elements(), Some(usize::MAX));
}

/// Test 4: Module with special Unicode characters in names
#[test]
fn test_module_unicode_names() {
    // Test with various Unicode characters
    let unicode_names = [
        "模块_测试",           // Chinese
        "модуль_тест",        // Russian
        "モジュール_テスト",   // Japanese
        "모듈_테스트",         // Korean
        "🚀rocket_module",    // Emoji
        "αβγ_δ_ε",            // Greek
        "test_ñ_ç_é",         // Accented characters
    ];

    for name in unicode_names.iter() {
        let module = Module::new(*name);
        assert_eq!(module.name, *name);

        // Add an operation with Unicode name
        let op_name = format!("{}_op", name);
        let op = Operation::new(&op_name);
        assert_eq!(op.op_type, op_name);
    }
}

/// Test 5: Compiler with empty and whitespace targets
#[test]
fn test_compiler_empty_whitespace_targets() {
    let mut compiler = ImpulseCompiler::new();
    let mock_model = vec![1u8, 2u8, 3u8];

    let targets = [
        "",           // Empty string
        "   ",        // Only spaces
        "\t\t",       // Only tabs
        "\n\n",       // Only newlines
        "  cpu  ",    // Spaces around
        "\tcpu\t",    // Tabs around
    ];

    for target in targets.iter() {
        let result = compiler.compile(&mock_model, target);
        // Should handle gracefully without panic
        match result {
            Ok(_) => (),
            Err(e) => {
                assert!(e.to_string().len() > 0);
            }
        }
    }
}

/// Test 6: Attribute with deeply nested arrays (nesting limit)
#[test]
fn test_deeply_nested_attribute_arrays() {
    // Create a deeply nested array structure
    let level1 = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(42),
                ]),
            ]),
        ]),
    ]);

    // Verify the structure
    match level1 {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 1);
            match &outer[0] {
                Attribute::Array(l2) => {
                    assert_eq!(l2.len(), 1);
                    match &l2[0] {
                        Attribute::Array(l3) => {
                            assert_eq!(l3.len(), 1);
                            match &l3[0] {
                                Attribute::Array(l4) => {
                                    assert_eq!(l4.len(), 1);
                                    assert_eq!(l4[0], Attribute::Int(42));
                                }
                                _ => panic!("Expected level 4 array"),
                            }
                        }
                        _ => panic!("Expected level 3 array"),
                    }
                }
                _ => panic!("Expected level 2 array"),
            }
        }
        _ => panic!("Expected outer array"),
    }
}

/// Test 7: Value with all single-digit shapes
#[test]
fn test_value_all_single_digit_shapes() {
    // Test all combinations of shapes with single digits 0-9
    let single_digit_shapes = [
        vec![0], vec![1], vec![2], vec![3], vec![4],
        vec![5], vec![6], vec![7], vec![8], vec![9],
        vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3],
        vec![0, 1, 0], vec![1, 0, 1],
    ];

    for shape in single_digit_shapes.iter() {
        let value = Value {
            name: "single_digit".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, *shape);

        // Calculate expected elements
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 8: Module with consecutive operations of same type
#[test]
fn test_module_consecutive_same_type_operations() {
    let mut module = Module::new("consecutive_ops");

    // Add 10 consecutive operations of the same type
    for i in 0..10 {
        let mut op = Operation::new("repeat_op");
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 10);

    // Verify all operations have the same type
    for op in &module.operations {
        assert_eq!(op.op_type, "repeat_op");
    }

    // Count operations by type
    let counts = ir_utils::count_operations_by_type(&module);
    assert_eq!(counts.get("repeat_op"), Some(&10));
}

/// Test 9: Type validity for all types
#[test]
fn test_type_validity_check() {

    // Test all basic types
    let types = [
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];

    for typ in types.iter() {
        assert!(typ.is_valid_type());
    }

    // Test nested tensor types
    let tensor_f32 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(tensor_f32.is_valid_type());

    // Test deeply nested tensor
    let nested_tensor = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        }),
        shape: vec![3],
    };
    assert!(nested_tensor.is_valid_type());
}

/// Test 10: Value with alternating dimension pattern
#[test]
fn test_value_alternating_dimension_pattern() {
    // Test shapes with alternating patterns
    let alternating_shapes = [
        vec![1, 2, 1, 2, 1],           // Alternating 1 and 2
        vec![2, 3, 2, 3, 2, 3],        // Alternating 2 and 3
        vec![1, 1, 1, 1],              // All 1s
        vec![2, 2, 2],                 // All 2s
        vec![1, 2, 3, 1, 2, 3],        // Repeating 1,2,3
        vec![0, 1, 0, 1, 0],           // Alternating 0 and 1
    ];

    for shape in alternating_shapes.iter() {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::F64,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, *shape);

        // Verify num_elements calculation
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));

        // Verify shape pattern is preserved
        assert_eq!(value.shape.len(), shape.len());
        for (i, &dim) in shape.iter().enumerate() {
            assert_eq!(value.shape[i], dim);
        }
    }
}