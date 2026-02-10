//! Focused edge boundary tests v2 - additional edge cases with standard assertions
//! Using assert! and assert_eq! for clear, explicit testing

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with overflow-prone large dimensions
#[test]
fn test_value_overflow_large_dimensions() {
    // Test dimensions that will cause overflow when multiplied cumulatively
    // try_fold starts with 1, so we need dimensions where the cumulative product overflows
    let overflow_shapes = vec![
        vec![usize::MAX, 2],              // 1 * usize::MAX * 2 will overflow
        vec![usize::MAX / 2 + 1, 3],      // 1 * (usize::MAX/2+1) * 3 will overflow
    ];

    for shape in overflow_shapes {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // num_elements should return None for overflow cases
        let num_elem = value.num_elements();
        assert!(num_elem.is_none(), "Should return None for overflow: {:?}", shape);
    }
}

/// Test 2: Module with empty operation names
#[test]
fn test_module_empty_operation_names() {
    let mut module = Module::new("empty_name_module");

    // Add operations with empty names
    for _ in 0..3 {
        let op = Operation::new("");
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);
    // All operations should have empty op_type
    for op in &module.operations {
        assert_eq!(op.op_type, "");
    }
}

/// Test 3: Value with very long tensor names
#[test]
fn test_value_very_long_name() {
    // Create a very long name (stress test for string handling)
    let long_name = "a".repeat(1000);

    let value = Value {
        name: long_name.clone(),
        ty: Type::I32,
        shape: vec![10],
    };

    assert_eq!(value.name.len(), 1000);
    assert_eq!(value.name, long_name);
}

/// Test 4: Attribute with special characters in string values
#[test]
fn test_attribute_special_characters() {
    let special_strings = vec![
        "",                              // Empty string
        "\0",                            // Null character
        "\n\t\r",                        // Newline, tab, carriage return
        "日本語",                        // Unicode characters
        "😀🎉",                          // Emojis
        "path/to/file.txt",             // Path-like string
        "key=value&other=data",          // URL-like string
    ];

    for s in special_strings {
        let attr = Attribute::String(s.to_string());
        match attr {
            Attribute::String(val) => {
                assert_eq!(val, s);
            }
            _ => panic!("Expected String attribute"),
        }
    }
}

/// Test 5: Type validity check with nested tensor types
#[test]
fn test_type_validity_deeply_nested_tensors() {
    use crate::ir::TypeExtensions;

    // Create deeply nested tensor types
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![1],
    };

    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![1],
    };

    let level3 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![1],
    };

    // All should be valid types
    assert!(level1.is_valid_type());
    assert!(level2.is_valid_type());
    assert!(level3.is_valid_type());
}

/// Test 6: Operation with very large number of attributes
#[test]
fn test_operation_many_attributes() {
    let mut op = Operation::new("many_attrs");
    let mut attrs = std::collections::HashMap::new();

    // Add 100 attributes
    for i in 0..100 {
        attrs.insert(format!("attr_{}", i), Attribute::Int(i as i64));
    }

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 100);

    // Verify all attributes exist
    for i in 0..100 {
        let key = format!("attr_{}", i);
        assert!(op.attributes.contains_key(&key));
        if let Attribute::Int(val) = op.attributes.get(&key).unwrap() {
            assert_eq!(*val, i as i64);
        }
    }
}

/// Test 7: Value with alternating dimension pattern
#[test]
fn test_value_alternating_dimensions() {
    let patterns = vec![
        vec![1, 2, 1, 2, 1],    // Alternating small values
        vec![0, 1, 0, 1, 0],    // Alternating with zeros
        vec![10, 0, 10, 0, 10], // Large with zeros
    ];

    for (idx, shape) in patterns.iter().enumerate() {
        let value = Value {
            name: format!("pattern_{}", idx),
            ty: Type::F64,
            shape: shape.clone(),
        };

        // Calculate expected elements
        let product: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(product));
    }
}

/// Test 8: Attribute with empty and single-element arrays
#[test]
fn test_attribute_empty_and_single_arrays() {
    // Empty array
    let empty = Attribute::Array(vec![]);
    match empty {
        Attribute::Array(arr) => {
            assert!(arr.is_empty());
            assert_eq!(arr.len(), 0);
        }
        _ => panic!("Expected empty Array"),
    }

    // Single-element arrays
    let single_int = Attribute::Array(vec![Attribute::Int(42)]);
    let single_float = Attribute::Array(vec![Attribute::Float(3.14)]);
    let single_string = Attribute::Array(vec![Attribute::String("test".to_string())]);
    let single_bool = Attribute::Array(vec![Attribute::Bool(true)]);

    match single_int {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array with Int"),
    }

    match single_float {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
        }
        _ => panic!("Expected Array with Float"),
    }

    match single_string {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
        }
        _ => panic!("Expected Array with String"),
    }

    match single_bool {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
        }
        _ => panic!("Expected Array with Bool"),
    }
}

/// Test 9: Module with inputs only (no operations, no outputs)
#[test]
fn test_module_inputs_only() {
    let mut module = Module::new("inputs_only");

    // Add only inputs, no operations or outputs
    for i in 0..5 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
    }

    assert_eq!(module.inputs.len(), 5);
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.outputs.len(), 0);

    // Verify all input names
    for i in 0..5 {
        assert_eq!(module.inputs[i].name, format!("input_{}", i));
    }
}

/// Test 10: Module with outputs only (no inputs, no operations)
#[test]
fn test_module_outputs_only() {
    let mut module = Module::new("outputs_only");

    // Add only outputs, no inputs or operations
    for i in 0..3 {
        module.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::I64,
            shape: vec![1],
        });
    }

    assert_eq!(module.outputs.len(), 3);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.operations.len(), 0);

    // Verify all output names and types
    for i in 0..3 {
        assert_eq!(module.outputs[i].name, format!("output_{}", i));
        assert_eq!(module.outputs[i].ty, Type::I64);
    }
}