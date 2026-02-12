//! Comprehensive Unique Boundary Tests
//! 覆盖独特的边界情况，使用标准库 assert! 和 assert_eq!
//! 这些测试专注于 IR 核心组件的边界场景

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with shape containing usize::MAX dimension
#[test]
fn test_value_with_max_usize_dimension() {
    let value = Value {
        name: "max_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };
    assert_eq!(value.shape, vec![usize::MAX]);
    // num_elements should return None due to overflow
    assert_eq!(value.num_elements(), Some(usize::MAX));
}

/// Test 2: Operation with all primitive attribute types
#[test]
fn test_operation_all_primitive_attributes() {
    let mut op = Operation::new("primitive_attrs");

    op.attributes.insert("int_attr".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("float_attr".to_string(), Attribute::Float(f64::MAX));
    op.attributes.insert("string_attr".to_string(), Attribute::String("test".to_string()));
    op.attributes.insert("bool_attr".to_string(), Attribute::Bool(true));

    assert_eq!(op.attributes.len(), 4);
    assert!(op.attributes.contains_key("int_attr"));
    assert!(op.attributes.contains_key("float_attr"));
    assert!(op.attributes.contains_key("string_attr"));
    assert!(op.attributes.contains_key("bool_attr"));
}

/// Test 3: Module with operations sharing the same input value
#[test]
fn test_module_shared_inputs() {
    let mut module = Module::new("shared_inputs");
    let shared_input = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };

    // Multiple operations using the same input
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(shared_input.clone());
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.inputs[0].name, "shared");
    }
}

/// Test 4: Value with shape containing multiple zeros
#[test]
fn test_value_multiple_zero_dimensions() {
    let value = Value {
        name: "multi_zero".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0, 0],
    };
    assert_eq!(value.shape, vec![0, 0, 0, 0]);
    assert_eq!(value.num_elements(), Some(0));
}

/// Test 5: Operation with deeply nested array attribute
#[test]
fn test_deeply_nested_array_attribute() {
    let mut op = Operation::new("nested_array");
    let deep_array = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
            ]),
        ]),
    ]);
    op.attributes.insert("deep".to_string(), deep_array);

    match &op.attributes["deep"] {
        Attribute::Array(outer) => {
            match &outer[0] {
                Attribute::Array(middle) => {
                    match &middle[0] {
                        Attribute::Array(inner) => {
                            assert_eq!(inner.len(), 2);
                        }
                        _ => panic!("Expected inner array"),
                    }
                }
                _ => panic!("Expected middle array"),
            }
        }
        _ => panic!("Expected outer array"),
    }
}

/// Test 6: Value with alternating dimension pattern
#[test]
fn test_value_alternating_dimensions() {
    let value = Value {
        name: "alternating".to_string(),
        ty: Type::F32,
        shape: vec![1, 100, 1, 100, 1, 100],
    };
    assert_eq!(value.shape, vec![1, 100, 1, 100, 1, 100]);
    assert_eq!(value.num_elements(), Some(100 * 100 * 100));
}

/// Test 7: Module with empty operation names
#[test]
fn test_module_empty_operation_names() {
    let mut module = Module::new("empty_names");
    let op = Operation::new("");
    module.add_operation(op);

    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].op_type, "");
}

/// Test 8: Value with very small non-zero dimensions
#[test]
fn test_value_small_nonzero_dimensions() {
    let value = Value {
        name: "small_dims".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1, 1, 1, 1, 1],
    };
    assert_eq!(value.shape.len(), 8);
    assert_eq!(value.num_elements(), Some(1));
}

/// Test 9: Operation with attribute value changes
#[test]
fn test_operation_attribute_modification() {
    let mut op = Operation::new("mod_attr");
    op.attributes.insert("key".to_string(), Attribute::Int(1));

    assert_eq!(op.attributes.len(), 1);

    // Modify the attribute
    op.attributes.insert("key".to_string(), Attribute::Int(2));

    assert_eq!(op.attributes.len(), 1);
    match &op.attributes["key"] {
        Attribute::Int(val) => assert_eq!(*val, 2),
        _ => panic!("Expected Int attribute"),
    }
}

/// Test 10: Module with inputs and outputs having same names
#[test]
fn test_module_input_output_same_names() {
    let mut module = Module::new("same_names");

    // Add input
    module.inputs.push(Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    // Add output with same name
    module.outputs.push(Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, "data");
    assert_eq!(module.outputs[0].name, "data");
}