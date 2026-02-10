//! Unique boundary case tests for Impulse compiler
//! This module contains edge case tests that complement existing test suites

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with dimension product approaching usize::MAX boundary
#[test]
fn test_value_dimension_product_near_max() {
    // Test shapes that multiply to values near power-of-2 boundaries
    let test_cases = vec![
        vec![65536, 65536],     // 2^16 * 2^16 = 2^32 ≈ 4.3 billion
        vec![256, 256, 256],     // 2^8 * 2^8 * 2^8 = 2^24 ≈ 16.8 million
        vec![1024, 1024, 1024],  // 2^10 * 2^10 * 2^10 = 2^30 ≈ 1.07 billion
    ];

    for shape in test_cases {
        let value = Value {
            name: "boundary_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // Verify shape is preserved
        assert_eq!(value.shape, shape);

        // Calculate and verify product
        let product: usize = value.shape.iter().product();
        assert!(product > 0);
    }
}

/// Test 2: Operation with extremely large negative integer attribute
#[test]
fn test_operation_with_extreme_negative_int_attribute() {
    let mut op = Operation::new("extreme_int_op");
    let mut attrs = HashMap::new();

    // Test extreme negative values near i64::MIN
    attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("very_negative".to_string(), Attribute::Int(-9_223_372_036_854_775_807));
    attrs.insert("neg_1e18".to_string(), Attribute::Int(-1_000_000_000_000_000_000));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 3);
    assert_eq!(op.attributes.get("min_int"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("very_negative"), Some(&Attribute::Int(-9_223_372_036_854_775_807)));
}

/// Test 3: Module with single operation chain of maximum depth
#[test]
fn test_module_with_maximum_depth_operation_chain() {
    let mut module = Module::new("deep_chain_module");

    // Create a chain where each operation's output is the next's input
    let mut previous_output = Value {
        name: "input_0".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };

    for i in 0..20 {
        let mut op = Operation::new(&format!("chain_op_{}", i));
        op.inputs.push(previous_output.clone());

        previous_output = Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        };
        op.outputs.push(previous_output.clone());

        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 20);

    // Verify chain connectivity
    for i in 0..19 {
        assert_eq!(module.operations[i].outputs[0].name, module.operations[i + 1].inputs[0].name);
    }
}

/// Test 4: Value with alternating dimension pattern [1, 100, 1, 100, 1]
#[test]
fn test_value_alternating_extreme_dimensions() {
    let patterns = vec![
        vec![1, 100_000, 1, 100_000, 1],
        vec![100_000, 1, 100_000, 1, 100_000],
        vec![1, 1, 1, 1_000_000, 1],
        vec![1_000_000, 1, 1, 1, 1],
    ];

    for shape in patterns {
        let value = Value {
            name: "alternating_dim".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, shape);

        // Calculate product - should equal product of non-1 dimensions
        let product: usize = value.shape.iter().product();
        assert!(product > 0);
    }
}

/// Test 5: Attribute array with all boolean combinations
#[test]
fn test_attribute_array_with_all_boolean_combinations() {
    // Create array with all 4 possible boolean combinations
    let bool_array = Attribute::Array(vec![
        Attribute::Bool(true),
        Attribute::Bool(false),
        Attribute::Bool(true),
        Attribute::Bool(false),
    ]);

    match bool_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Attribute::Bool(true));
            assert_eq!(arr[1], Attribute::Bool(false));
            assert_eq!(arr[2], Attribute::Bool(true));
            assert_eq!(arr[3], Attribute::Bool(false));
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: Module with operations having identical attributes but different names
#[test]
fn test_module_operations_with_identical_attributes() {
    let mut module = Module::new("identical_attrs_module");

    // Create identical attribute sets
    let mut attrs = HashMap::new();
    attrs.insert("learning_rate".to_string(), Attribute::Float(0.001));
    attrs.insert("momentum".to_string(), Attribute::Float(0.9));
    attrs.insert("epsilon".to_string(), Attribute::Float(1e-8));

    // Add multiple operations with same attributes but different names
    for i in 0..5 {
        let mut op = Operation::new(&format!("layer_{}", i));
        op.attributes = attrs.clone();
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);

    // Verify all operations have identical attributes
    for op in &module.operations {
        assert_eq!(op.attributes.len(), 3);
        assert_eq!(op.attributes.get("learning_rate"), Some(&Attribute::Float(0.001)));
        assert_eq!(op.attributes.get("momentum"), Some(&Attribute::Float(0.9)));
    }
}

/// Test 7: Value with shape containing power-of-2 dimensions
#[test]
fn test_value_with_power_of_two_dimensions() {
    let power_of_two_shapes = vec![
        vec![2, 4, 8, 16],              // 2^1, 2^2, 2^3, 2^4
        vec![32, 64, 128],              // 2^5, 2^6, 2^7
        vec![256, 512, 1024, 2048],     // 2^8, 2^9, 2^10, 2^11
        vec![4096, 8192],               // 2^12, 2^13
    ];

    for shape in power_of_two_shapes {
        let value = Value {
            name: "power_of_two".to_string(),
            ty: Type::F64,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, shape);

        // Verify all dimensions are powers of 2
        for dim in &value.shape {
            assert!(*dim > 0 && dim.is_power_of_two());
        }
    }
}

/// Test 8: Operation with nested array attribute containing all types
#[test]
fn test_operation_with_mixed_nested_array_attribute() {
    let mut op = Operation::new("mixed_nested_op");
    let mut attrs = HashMap::new();

    // Create a complex nested array with all attribute types
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Float(3.14),
        ]),
        Attribute::Array(vec![
            Attribute::String("nested".to_string()),
            Attribute::Bool(false),
        ]),
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
            ]),
        ]),
    ]);

    attrs.insert("complex_nested".to_string(), nested);
    op.attributes = attrs;

    match op.attributes.get("complex_nested") {
        Some(Attribute::Array(outer)) => {
            assert_eq!(outer.len(), 3);
        }
        _ => panic!("Expected nested array attribute"),
    }
}

/// Test 9: Module with input/output name collision
#[test]
fn test_module_with_input_output_name_collision() {
    let mut module = Module::new("name_collision_module");

    // Add input and output with the same name
    let shared_value = Value {
        name: "shared_value".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    };

    module.inputs.push(shared_value.clone());
    module.outputs.push(shared_value);

    // Both should exist independently
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, "shared_value");
    assert_eq!(module.outputs[0].name, "shared_value");
}

/// Test 10: Value with single very large dimension (1D tensor)
#[test]
fn test_value_with_single_large_dimension() {
    let large_1d_shapes = vec![
        vec![100_000],          // 100k elements
        vec![1_000_000],        // 1M elements
        vec![10_000_000],       // 10M elements
        vec![100_000_000],      // 100M elements
    ];

    for shape in large_1d_shapes {
        let value = Value {
            name: "large_1d_tensor".to_string(),
            ty: Type::I64,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, shape);
        assert_eq!(value.shape.len(), 1);

        // Verify num_elements matches the single dimension
        assert_eq!(value.num_elements(), Some(shape[0]));
    }
}