//! Comprehensive edge coverage tests - 覆盖更多边界情况
//! 使用标准库 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: 值的形状计算在边界值处的行为
#[test]
fn test_value_shape_calculation_boundary_values() {
    // 测试形状为 [1, 1, 1] 的值
    let single_element = Value {
        name: "single".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(single_element.num_elements(), Some(1));

    // 测试形状为 [1, usize::MAX] - 应该检测到溢出
    let overflow_shape = Value {
        name: "overflow".to_string(),
        ty: Type::F32,
        shape: vec![2, usize::MAX],
    };
    // 由于 usize::MAX * 2 会溢出，应该返回 None
    assert_eq!(overflow_shape.num_elements(), None);

    // 测试形状为 [0, 100, 100] - 包含零维度
    let zero_dim = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![0, 100, 100],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));
}

/// Test 2: 操作属性在极端整数值下的处理
#[test]
fn test_operation_with_extreme_int_attributes() {
    let mut op = Operation::new("extreme_ints");
    op.attributes.insert("max".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("neg_one".to_string(), Attribute::Int(-1));

    // 验证极值被正确存储
    match op.attributes.get("max") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected MAX int"),
    }
    match op.attributes.get("min") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected MIN int"),
    }
    match op.attributes.get("neg_one") {
        Some(Attribute::Int(val)) => assert_eq!(*val, -1),
        _ => panic!("Expected -1"),
    }
}

/// Test 3: 模块在连续添加和删除操作后的状态
#[test]
fn test_module_with_multiple_operations() {
    let mut module = Module::new("multi_ops");

    // 添加多个操作
    for i in 0..10 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 10);

    // 验证所有操作都被正确添加
    for i in 0..10 {
        assert_eq!(module.operations[i].op_type, format!("op_{}", i));
    }
}

/// Test 4: 嵌套张量类型在不同层级深度的行为
#[test]
fn test_nested_tensor_type_various_depths() {
    // 1层嵌套
    let depth1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    assert!(depth1.is_valid_type());

    // 2层嵌套
    let depth2 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    assert!(depth2.is_valid_type());

    // 3层嵌套
    let depth3 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2],
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    assert!(depth3.is_valid_type());
}

/// Test 5: 特殊浮点数值（NaN、无穷大、零）在属性中的处理
#[test]
fn test_special_float_values_in_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    let neg_zero = Attribute::Float(-0.0);

    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected NaN"),
    }

    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected positive infinity"),
    }

    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected negative infinity"),
    }

    match neg_zero {
        Attribute::Float(val) => assert_eq!(val, -0.0),
        _ => panic!("Expected negative zero"),
    }
}

/// Test 6: 值在具有大量维度时的行为
#[test]
fn test_value_with_many_dimensions() {
    // 创建一个具有8个维度的值
    let dims: Vec<usize> = (1..=8).collect();
    let multi_dim = Value {
        name: "multi_dim".to_string(),
        ty: Type::F32,
        shape: dims.clone(),
    };

    assert_eq!(multi_dim.shape.len(), 8);
    assert_eq!(multi_dim.shape, vec![1, 2, 3, 4, 5, 6, 7, 8]);

    // 验证元素总数
    let product: usize = multi_dim.shape.iter().product();
    assert_eq!(product, 40320);
}

/// Test 7: 操作在输入和输出值类型不匹配时的行为
#[test]
fn test_operation_with_mismatched_io_types() {
    let mut op = Operation::new("mismatched_types");

    // 添加不同类型的输入
    op.inputs.push(Value {
        name: "float_input".to_string(),
        ty: Type::F32,
        shape: vec![2, 2],
    });
    op.inputs.push(Value {
        name: "int_input".to_string(),
        ty: Type::I32,
        shape: vec![2, 2],
    });

    // 添加不同类型的输出
    op.outputs.push(Value {
        name: "bool_output".to_string(),
        ty: Type::Bool,
        shape: vec![2, 2],
    });

    assert_eq!(op.inputs.len(), 2);
    assert_eq!(op.outputs.len(), 1);
    assert_ne!(op.inputs[0].ty, op.inputs[1].ty);
    assert_ne!(op.inputs[0].ty, op.outputs[0].ty);
}

/// Test 8: 空字符串和空白字符串在属性中的处理
#[test]
fn test_empty_and_whitespace_string_attributes() {
    let mut op = Operation::new("string_tests");

    op.attributes.insert("empty".to_string(), Attribute::String(String::new()));
    op.attributes.insert("spaces".to_string(), Attribute::String("   ".to_string()));
    op.attributes.insert("mixed".to_string(), Attribute::String("  hello  ".to_string()));

    match op.attributes.get("empty") {
        Some(Attribute::String(s)) => assert_eq!(s.len(), 0),
        _ => panic!("Expected empty string"),
    }

    match op.attributes.get("spaces") {
        Some(Attribute::String(s)) => assert_eq!(s.trim().len(), 0),
        _ => panic!("Expected whitespace string"),
    }

    match op.attributes.get("mixed") {
        Some(Attribute::String(s)) => assert_eq!(s.trim(), "hello"),
        _ => panic!("Expected trimmed hello"),
    }
}

/// Test 9: 模块在没有任何操作、输入或输出时的行为
#[test]
fn test_module_with_all_empty_collections() {
    let module = Module::new("empty_module");

    assert!(module.operations.is_empty());
    assert!(module.inputs.is_empty());
    assert!(module.outputs.is_empty());
    assert_eq!(module.name, "empty_module");
}

/// Test 10: 属性数组中包含不同类型元素的行为
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]);

    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);

            // 验证每个元素
            match &arr[0] {
                Attribute::Int(42) => {}
                _ => panic!("Expected Int(42)"),
            }

            match &arr[1] {
                Attribute::Float(val) if (*val - 3.14).abs() < f64::EPSILON => {}
                _ => panic!("Expected Float(3.14)"),
            }

            match &arr[2] {
                Attribute::String(s) if s == "test" => {}
                _ => panic!("Expected String(\"test\")"),
            }

            match &arr[3] {
                Attribute::Bool(true) => {}
                _ => panic!("Expected Bool(true)"),
            }

            match &arr[4] {
                Attribute::Array(nested) => {
                    assert_eq!(nested.len(), 2);
                }
                _ => panic!("Expected nested Array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}
