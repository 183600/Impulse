//! New edge case boundary tests - 覆盖尚未充分测试的边界情况
//! 使用标准库 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Value with exactly i32::MAX elements in single dimension
#[test]
fn test_value_with_i32_max_elements() {
    let value = Value {
        name: "i32_max_dim".to_string(),
        ty: Type::F32,
        shape: vec![i32::MAX as usize],
    };
    assert_eq!(value.num_elements(), Some(i32::MAX as usize));
}

/// Test 2: Attribute with exact float precision boundaries (0.5, -0.5)
#[test]
fn test_float_half_boundaries() {
    let positive_half = Attribute::Float(0.5);
    let negative_half = Attribute::Float(-0.5);

    match positive_half {
        Attribute::Float(val) => assert_eq!(val, 0.5),
        _ => panic!("Expected Float(0.5)"),
    }

    match negative_half {
        Attribute::Float(val) => assert_eq!(val, -0.5),
        _ => panic!("Expected Float(-0.5)"),
    }
}

/// Test 3: Module with operations that have only attributes (no inputs/outputs)
#[test]
fn test_module_operations_with_only_attributes() {
    let mut module = Module::new("attr_only_ops");

    for i in 0..3 {
        let mut op = Operation::new(&format!("attr_op_{}", i));
        op.attributes.insert("value".to_string(), Attribute::Int(i * 10));
        op.attributes.insert("enabled".to_string(), Attribute::Bool(true));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);
    for (i, op) in module.operations.iter().enumerate() {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 2);
    }
}

/// Test 4: Value with alternating 0 and 1 dimension pattern
#[test]
fn test_value_alternating_zero_one_pattern() {
    let patterns = [
        vec![0, 1, 0, 1],
        vec![1, 0, 1, 0],
        vec![0, 1, 0, 1, 0],
    ];

    for shape in patterns.iter() {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::F32,
            shape: shape.to_vec(),
        };
        // Any shape containing 0 should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 5: Tensor type with recursive nesting depth of 4
#[test]
fn test_deep_nested_tensor_type() {
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

    // Verify it's a valid type
    assert!(level4.is_valid_type());

    match level4 {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![4]);
            match *element_type {
                Type::Tensor { element_type: inner_elem, shape: inner_shape } => {
                    assert_eq!(inner_shape, vec![3]);
                    // Continue nesting verification
                    assert!(inner_elem.is_valid_type());
                }
                _ => panic!("Expected nested Tensor"),
            }
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 6: Array attribute with mixed primitive types
#[test]
fn test_mixed_type_array_attribute() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Int(-100),
        Attribute::Float(-2.5),
    ]);

    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 6);
            assert_eq!(arr[0], Attribute::Int(42));
            assert_eq!(arr[2], Attribute::String("test".to_string()));
            assert_eq!(arr[3], Attribute::Bool(true));
            assert_eq!(arr[5], Attribute::Float(-2.5));
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: Module with duplicate input names
#[test]
fn test_module_duplicate_input_names() {
    let mut module = Module::new("dup_inputs");

    // Add inputs with the same name
    for i in 0..3 {
        module.inputs.push(Value {
            name: "shared_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
    }

    assert_eq!(module.inputs.len(), 3);
    for input in &module.inputs {
        assert_eq!(input.name, "shared_input");
    }
}

/// Test 8: Value with dimensions that are powers of 2
#[test]
fn test_value_power_of_2_dimensions() {
    let power_of_2_shapes = [
        vec![2, 4, 8],
        vec![16, 32],
        vec![64, 128, 256],
        vec![512, 1024],
    ];

    for shape in power_of_2_shapes.iter() {
        let value = Value {
            name: "power_of_2".to_string(),
            ty: Type::F64,
            shape: shape.to_vec(),
        };

        // Verify each dimension is a power of 2
        for &dim in shape.iter() {
            assert!(dim.is_power_of_two());
        }
    }
}

/// Test 9: Operation with attribute key being empty string
#[test]
fn test_operation_empty_attribute_key() {
    let mut op = Operation::new("empty_key_op");

    // Insert attribute with empty key (edge case)
    op.attributes.insert("".to_string(), Attribute::Int(999));

    assert_eq!(op.attributes.len(), 1);
    assert!(op.attributes.contains_key(""));

    match op.attributes.get("") {
        Some(Attribute::Int(999)) => (),
        _ => panic!("Expected Int(999) for empty key"),
    }
}

/// Test 10: Module with operations count exactly matching usize boundary
#[test]
fn test_module_operation_count_boundary() {
    let mut module = Module::new("boundary_ops");

    // Add exactly 256 operations (power of 2 boundary)
    for i in 0..=255u8 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::I32,
            shape: vec![1],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 256);
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[255].op_type, "op_255");
}