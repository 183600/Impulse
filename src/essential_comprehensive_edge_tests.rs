//! Essential comprehensive edge tests - 10边界情况测试用例
//! 使用标准库 assert! 和 assert_eq! 覆盖更多边界情况

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Value with zero dimensions and single element (scalar)
#[test]
fn test_scalar_value() {
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(scalar.num_elements(), Some(1));
}

/// Test 2: Operation with boolean attributes
#[test]
fn test_boolean_attributes() {
    let mut op = Operation::new("bool_op");
    op.attributes.insert("flag_true".to_string(), Attribute::Bool(true));
    op.attributes.insert("flag_false".to_string(), Attribute::Bool(false));

    assert_eq!(op.attributes.len(), 2);
    match op.attributes.get("flag_true") {
        Some(Attribute::Bool(true)) => {},
        _ => panic!("Expected Bool(true)"),
    }
    match op.attributes.get("flag_false") {
        Some(Attribute::Bool(false)) => {},
        _ => panic!("Expected Bool(false)"),
    }
}

/// Test 3: Module with maximum single dimension
#[test]
fn test_max_single_dimension() {
    let max_dim = usize::MAX;
    let value = Value {
        name: "max_dim".to_string(),
        ty: Type::F32,
        shape: vec![max_dim],
    };
    assert_eq!(value.shape[0], max_dim);
    assert_eq!(value.num_elements(), Some(max_dim));
}

/// Test 4: Value with shape containing zero
#[test]
fn test_zero_shape_dimension() {
    let value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(value.shape, vec![10, 0, 5]);
    assert_eq!(value.num_elements(), Some(0));
}

/// Test 5: Attribute array with mixed types
#[test]
fn test_mixed_type_array() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.5),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
    ]);
    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 4);
            assert!(matches!(arr[0], Attribute::Int(1)));
            assert!(matches!(arr[1], Attribute::Float(_)));
            assert!(matches!(arr[2], Attribute::String(_)));
            assert!(matches!(arr[3], Attribute::Bool(true)));
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: Type validation for all valid types
#[test]
fn test_type_validation() {
    let types = vec![Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool];
    for ty in types {
        assert!(ty.is_valid_type());
    }
}

/// Test 7: Operation with nested array attribute
#[test]
fn test_nested_array_attribute() {
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
        Attribute::Array(vec![Attribute::Int(3), Attribute::Int(4)]),
    ]);
    match nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            if let Attribute::Array(inner) = &outer[0] {
                assert_eq!(inner.len(), 2);
            } else {
                panic!("Expected nested array");
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 8: Module with single operation
#[test]
fn test_module_single_operation() {
    let mut module = Module::new("single_op");
    let op = Operation::new("add");
    module.add_operation(op);
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].op_type, "add");
}

/// Test 9: Value with unit dimensions (all ones)
#[test]
fn test_unit_dimensions() {
    let value = Value {
        name: "unit".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(value.num_elements(), Some(1));
}

/// Test 10: Module with multiple same-type inputs
#[test]
fn test_multiple_same_type_inputs() {
    let mut module = Module::new("multi_input");
    module.inputs.push(Value {
        name: "in1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.inputs.push(Value {
        name: "in2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.inputs.push(Value {
        name: "in3".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::F32);
    assert_eq!(module.inputs[2].ty, Type::F32);
}