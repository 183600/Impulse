//! Critical edge boundary coverage tests - unique edge cases for compiler robustness
//! Tests covering memory safety, numerical precision, overflow prevention, and edge conditions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Value with dimension product at usize boundary (overflow edge)
#[test]
fn test_dimension_product_at_usize_boundary() {
    // Test values that cause actual overflow when multiplied
    let overflow_shape = Value {
        name: "overflow_shape".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 2], // Would overflow when multiplied
    };
    // Should return None due to overflow
    assert_eq!(overflow_shape.num_elements(), None);

    // Test valid large shape just below boundary
    let safe_shape = Value {
        name: "safe_large".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000, 1000],
    };
    assert_eq!(safe_shape.num_elements(), Some(1_000_000_000));
}

/// Test 2: Attribute with negative zero float
#[test]
fn test_negative_zero_float() {
    let neg_zero = Attribute::Float(-0.0f64);
    let pos_zero = Attribute::Float(0.0f64);

    match neg_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val.is_sign_negative()); // -0.0 has negative sign
        }
        _ => panic!("Expected Float attribute"),
    }

    match pos_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with operation count approaching practical limits
#[test]
fn test_module_operation_capacity() {
    let mut module = Module::new("large_ops_module");

    // Add many operations (stress test)
    for i in 0..1000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.attributes.insert("id".to_string(), Attribute::Int(i as i64));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[999].op_type, "op_999");
}

/// Test 4: Value with empty name and zero dimensions
#[test]
fn test_value_empty_name_zero_dims() {
    let empty_name_value = Value {
        name: "".to_string(),
        ty: Type::I32,
        shape: vec![],
    };
    assert_eq!(empty_name_value.name, "");
    assert_eq!(empty_name_value.shape.len(), 0);
    assert_eq!(empty_name_value.num_elements(), Some(1)); // Scalar = 1 element
}

/// Test 5: Deeply nested tensor type (type system stress test)
#[test]
fn test_deeply_nested_tensor_type() {
    // Create a tensor type with depth 4
    let level1 = Type::F32;
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![2],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![3],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![4],
    };

    assert!(level4.is_valid_type());
}

/// Test 6: Attribute array with circular structure detection
#[test]
fn test_attribute_array_consistency() {
    // Create an array with duplicate references
    let inner = Attribute::Int(42);
    let array = Attribute::Array(vec![
        inner.clone(),
        inner.clone(),
        inner.clone(),
    ]);

    match array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 3);
            match &arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
            match &arr[1] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
            match &arr[2] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: Module with single operation having maximum attribute count
#[test]
fn test_operation_max_attributes() {
    let mut op = Operation::new("attr_stress");

    // Add many attributes
    for i in 0..100 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::Int(i as i64)
        );
    }

    assert_eq!(op.attributes.len(), 100);
    assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("attr_99"), Some(&Attribute::Int(99)));
}

/// Test 8: Value with alternating zero and non-zero dimensions
#[test]
fn test_value_alternating_zero_dimensions() {
    let patterns = vec![
        vec![0, 1, 0, 1],
        vec![1, 0, 1, 0],
        vec![0, 0, 1, 1],
        vec![1, 1, 0, 0],
    ];

    for pattern in patterns {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::F64,
            shape: pattern.clone(),
        };
        // Any zero dimension should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 9: String attribute with null bytes and special characters
#[test]
fn test_string_special_characters() {
    let special_strings = vec![
        String::from("test\x00null"),      // Contains null byte
        String::from("emoji🔥✨test"),      // Unicode emojis
        String::from("tab\there"),         // Tab character
        String::from("new\nline"),         // Newline
        String::from("quote\"test"),       // Quote
        String::from("backslash\\test"),   // Backslash
    ];

    for s in special_strings {
        let attr = Attribute::String(s.clone());
        match attr {
            Attribute::String(val) => assert_eq!(val, s),
            _ => panic!("Expected String attribute"),
        }
    }
}

/// Test 10: Module with all operations having identical structure
#[test]
fn test_module_identical_operations() {
    let mut module = Module::new("identical_ops");

    // Add operations with identical structure
    for _ in 0..10 {
        let mut op = Operation::new("identical");
        op.inputs.push(Value {
            name: "in".to_string(),
            ty: Type::F32,
            shape: vec![2],
        });
        op.outputs.push(Value {
            name: "out".to_string(),
            ty: Type::F32,
            shape: vec![2],
        });
        op.attributes.insert("param".to_string(), Attribute::Float(1.0));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 10);
    // Verify all have same structure
    for op in &module.operations {
        assert_eq!(op.op_type, "identical");
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.attributes.len(), 1);
    }
}