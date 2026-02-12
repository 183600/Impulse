//! Core edge case tests for Impulse compiler
//! This file contains additional tests covering edge cases not previously tested

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::math_utils;

/// Test 1: Module with extremely large tensor dimensions (close to overflow boundary)
#[test]
fn test_large_tensor_dimensions() {
    // Test dimensions that are large but don't overflow when multiplied
    let large_shapes = [
        vec![100000, 10],       // 1 million elements
        vec![1000, 1000],       // 1 million elements
        vec![5000, 500, 400],   // 1 billion elements
    ];

    for shape in large_shapes.iter() {
        let value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // Should return Some if no overflow
        let num_elements = value.num_elements();
        assert!(num_elements.is_some());

        // Verify the calculation is correct
        let expected: usize = shape.iter().product();
        assert_eq!(num_elements, Some(expected));
    }
}

/// Test 2: Round up to multiple with extreme values
#[test]
fn test_round_up_to_multiple_extreme_values() {
    // Test with large values
    assert_eq!(math_utils::round_up_to_multiple(1000000, 1024), 1000448);
    assert_eq!(math_utils::round_up_to_multiple(999999, 1000), 1000000);

    // Test when value is already a multiple
    assert_eq!(math_utils::round_up_to_multiple(1024, 1024), 1024);
    assert_eq!(math_utils::round_up_to_multiple(1000000, 1000000), 1000000);

    // Test with multiple of 1
    assert_eq!(math_utils::round_up_to_multiple(42, 1), 42);

    // Test with multiple larger than value
    assert_eq!(math_utils::round_up_to_multiple(5, 100), 100);
    assert_eq!(math_utils::round_up_to_multiple(999, 1024), 1024);
}

/// Test 3: GCD with power of 2 values
#[test]
fn test_gcd_power_of_2_values() {
    // Test GCD with powers of 2
    assert_eq!(math_utils::gcd(1024, 512), 512);
    assert_eq!(math_utils::gcd(4096, 256), 256);
    assert_eq!(math_utils::gcd(2048, 2048), 2048);
    assert_eq!(math_utils::gcd(16384, 8192), 8192);

    // Test GCD with mixed powers of 2 and other values
    assert_eq!(math_utils::gcd(1024, 768), 256);
    assert_eq!(math_utils::gcd(2048, 1536), 512);
}

/// Test 4: LCM with large values
#[test]
fn test_lcm_large_values() {
    // Test LCM with moderately large values
    assert_eq!(math_utils::lcm(1000, 250), 1000);
    assert_eq!(math_utils::lcm(1024, 512), 1024);
    assert_eq!(math_utils::lcm(999, 111), 999);

    // Test LCM with co-prime values
    assert_eq!(math_utils::lcm(17, 19), 323);
    assert_eq!(math_utils::lcm(23, 29), 667);
}

/// Test 5: Value with single-element shape variations
#[test]
fn test_single_element_shape_variations() {
    // All these shapes represent a single element
    let single_element_shapes = [
        vec![1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![1, 1, 1, 1],
        vec![1, 1, 1, 1, 1],
    ];

    for shape in single_element_shapes.iter() {
        let value = Value {
            name: "single".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(value.num_elements(), Some(1));
    }
}

/// Test 6: Operation with deeply nested attribute arrays
#[test]
fn test_deeply_nested_attribute_arrays() {
    let mut op = Operation::new("nested_attr_op");

    // Create a deeply nested array structure
    let nested_attr = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::String("deep".to_string()),
            ]),
            Attribute::Float(3.14),
        ]),
        Attribute::Bool(true),
    ]);

    op.attributes.insert("deep_structure".to_string(), nested_attr);

    assert_eq!(op.attributes.len(), 1);

    // Verify the structure can be retrieved
    if let Some(Attribute::Array(outer)) = op.attributes.get("deep_structure") {
        assert_eq!(outer.len(), 2);
        if let Attribute::Array(inner) = &outer[0] {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("Expected nested array");
        }
    } else {
        panic!("Expected Array attribute");
    }
}

/// Test 7: Module with mixed data types in operations
#[test]
fn test_module_mixed_data_types() {
    let mut module = Module::new("mixed_types_module");

    let types = [Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool];

    for (i, ty) in types.iter().enumerate() {
        let mut op = Operation::new(&format!("mixed_op_{}", i));

        // Add input with this type
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: ty.clone(),
            shape: vec![2, 2],
        });

        // Add output with the same type
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: ty.clone(),
            shape: vec![2, 2],
        });

        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);

    // Verify each operation has the correct type
    for (i, op) in module.operations.iter().enumerate() {
        assert_eq!(op.inputs[0].ty, types[i]);
        assert_eq!(op.outputs[0].ty, types[i]);
    }
}

/// Test 8: Value with alternating dimension pattern
#[test]
fn test_value_alternating_dimension_pattern() {
    // Test shapes with alternating patterns
    let alternating_shapes = [
        vec![1, 2, 1, 2],  // Alternating 1 and 2
        vec![2, 4, 2, 4],  // Alternating 2 and 4
        vec![1, 10, 1, 10], // Alternating 1 and 10
    ];

    for shape in alternating_shapes.iter() {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };

        // Verify the alternating pattern
        for i in 2..shape.len() {
            assert_eq!(shape[i], shape[i - 2]);
        }

        // Verify element count
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 9: Operation with empty string attribute
#[test]
fn test_operation_empty_string_attribute() {
    let mut op = Operation::new("empty_string_op");

    // Add an empty string attribute
    op.attributes.insert("empty".to_string(), Attribute::String("".to_string()));

    assert_eq!(op.attributes.len(), 1);

    if let Some(Attribute::String(s)) = op.attributes.get("empty") {
        assert_eq!(s.len(), 0);
    } else {
        panic!("Expected String attribute");
    }
}

/// Test 10: Module with operations having only attributes
#[test]
fn test_module_operations_only_attributes() {
    let mut module = Module::new("attrs_only_module");

    // Add operations that only have attributes (no inputs or outputs)
    for i in 0..3 {
        let mut op = Operation::new(&format!("attr_op_{}", i));

        let mut attrs = std::collections::HashMap::new();
        attrs.insert("index".to_string(), Attribute::Int(i as i64));
        attrs.insert("enabled".to_string(), Attribute::Bool(i % 2 == 0));

        op.attributes = attrs;
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);

    // Verify all operations have no inputs or outputs
    for (i, op) in module.operations.iter().enumerate() {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 2);

        if let Some(Attribute::Int(idx)) = op.attributes.get("index") {
            assert_eq!(*idx, i as i64);
        }
    }
}