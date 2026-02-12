//! Unique edge case coverage tests - 覆盖独特的边界情况
//! 使用标准库的 assert! 和 assert_eq! 宏进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: 值的形状计算 - 测试接近 usize::MAX 的边界值
#[test]
fn test_shape_calculation_near_max_usize() {
    // 测试安全的边界值，不会导致溢出
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![65536, 65536], // 约 42 亿元素
    };
    assert_eq!(safe_value.num_elements(), Some(4294967296));

    // 测试单个大维度
    let large_single_dim = Value {
        name: "large_single".to_string(),
        ty: Type::F32,
        shape: vec![100000000],
    };
    assert_eq!(large_single_dim.num_elements(), Some(100000000));

    // 测试多个维度相乘接近边界但安全
    let multi_dim = Value {
        name: "multi_dim".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000, 1000], // 10 亿元素
    };
    assert_eq!(multi_dim.num_elements(), Some(1000000000));
}

/// Test 2: 操作属性 - 测试特殊浮点值的比较
#[test]
fn test_special_float_value_attributes() {
    let mut op = Operation::new("special_floats");
    let mut attrs = HashMap::new();

    // 添加各种特殊浮点值
    attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("pos_inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("subnormal".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("neg_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("zero".to_string(), Attribute::Float(0.0));

    op.attributes = attrs;

    // 验证特殊值的属性
    match op.attributes.get("nan") {
        Some(Attribute::Float(val)) => assert!(val.is_nan()),
        _ => panic!("Expected NaN attribute"),
    }

    match op.attributes.get("pos_inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected positive infinity attribute"),
    }

    match op.attributes.get("neg_inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected negative infinity attribute"),
    }

    match op.attributes.get("neg_zero") {
        Some(Attribute::Float(val)) => {
            assert_eq!(*val, 0.0);
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected negative zero attribute"),
    }

    match op.attributes.get("zero") {
        Some(Attribute::Float(val)) => {
            assert_eq!(*val, 0.0);
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected zero attribute"),
    }
}

/// Test 3: 类型嵌套 - 测试深层嵌套的 Tensor 类型
#[test]
fn test_deeply_nested_tensor_types() {
    // 创建深度嵌套的 Tensor 类型
    let depth1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };

    let depth2 = Type::Tensor {
        element_type: Box::new(depth1.clone()),
        shape: vec![3],
    };

    let depth3 = Type::Tensor {
        element_type: Box::new(depth2.clone()),
        shape: vec![4],
    };

    let depth4 = Type::Tensor {
        element_type: Box::new(depth3.clone()),
        shape: vec![5],
    };

    // 验证所有类型都是有效的
    assert!(depth1.is_valid_type());
    assert!(depth2.is_valid_type());
    assert!(depth3.is_valid_type());
    assert!(depth4.is_valid_type());

    // 验证类型不同
    assert_ne!(depth1, depth2);
    assert_ne!(depth2, depth3);
    assert_ne!(depth3, depth4);
}

/// Test 4: 模块操作 - 测试空输入和输出的模块
#[test]
fn test_module_with_empty_io() {
    let mut module = Module::new("empty_io_module");

    // 添加一个没有输入输出的操作
    let mut op = Operation::new("no_io_op");
    op.attributes.insert("internal_state".to_string(), Attribute::Int(42));
    module.add_operation(op);

    // 验证模块状态
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 0);
    assert_eq!(module.operations[0].attributes.len(), 1);
}

/// Test 5: 数组属性 - 测试混合类型的嵌套数组
#[test]
fn test_mixed_type_nested_array_attributes() {
    let nested_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.5),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![
            Attribute::Int(10),
            Attribute::Float(20.5),
        ]),
    ]);

    match nested_array {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 5);
            assert_eq!(outer[0], Attribute::Int(1));
            assert_eq!(outer[1], Attribute::Float(2.5));
            assert_eq!(outer[2], Attribute::String("test".to_string()));
            assert_eq!(outer[3], Attribute::Bool(true));
            
            match &outer[4] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 2);
                    assert_eq!(inner[0], Attribute::Int(10));
                    assert_eq!(inner[1], Attribute::Float(20.5));
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: 值命名 - 测试包含各种特殊字符的名称
#[test]
fn test_special_character_names() {
    let special_names = vec![
        "tensor_with_underscore",
        "tensor-with-dash",
        "tensor.with.dot",
        "tensor:with:colon",
        "tensor/with/slash",
        "tensor\\with\\backslash",
        "with space",
        "with\ttab",
        "emoji🔥name",
        "cyrillicИмя",
        "chinese名称",
        "arabicاسم",
    ];

    for name in special_names {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, name);
    }
}

/// Test 7: 操作重复 - 测试添加多个相同类型的操作
#[test]
fn test_multiple_operations_same_type() {
    let mut module = Module::new("multiple_ops");

    // 添加多个相同类型的操作
    for i in 0..10 {
        let mut op = Operation::new("add");
        op.attributes.insert("id".to_string(), Attribute::Int(i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![2, 2],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 10);
    
    // 验证所有操作都是 add 类型
    for op in &module.operations {
        assert_eq!(op.op_type, "add");
    }

    // 验证每个操作有唯一的 id 属性
    for (i, op) in module.operations.iter().enumerate() {
        match op.attributes.get("id") {
            Some(Attribute::Int(val)) => assert_eq!(*val, i as i64),
            _ => panic!("Expected id attribute"),
        }
    }
}

/// Test 8: 形状边界 - 测试包含零维度的形状
#[test]
fn test_shapes_with_zero_dimensions() {
    let zero_shapes = vec![
        vec![0],
        vec![0, 10],
        vec![10, 0],
        vec![2, 0, 3],
        vec![0, 0, 0],
    ];

    for shape in zero_shapes {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // 任何包含零的形状都应该返回 0 元素
        assert_eq!(value.num_elements(), Some(0));
        assert_eq!(value.shape, shape);
    }
}

/// Test 9: 属性覆盖 - 测试 HashMap 中属性的覆盖行为
#[test]
fn test_attribute_override_behavior() {
    let mut op = Operation::new("override_test");
    let mut attrs = HashMap::new();

    // 添加初始属性
    attrs.insert("key".to_string(), Attribute::Int(1));
    attrs.insert("key".to_string(), Attribute::Int(2));
    attrs.insert("key".to_string(), Attribute::Int(3));

    op.attributes = attrs;

    // HashMap 应该只保留最后一个值
    assert_eq!(op.attributes.len(), 1);
    match op.attributes.get("key") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 3),
        _ => panic!("Expected Int(3)"),
    }
}

/// Test 10: 模块类型 - 测试模块中所有数据类型的组合
#[test]
fn test_module_with_all_primitive_types() {
    let mut module = Module::new("all_types");

    // 为每种基本类型创建值
    let types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];

    for (i, ty) in types.iter().enumerate() {
        let mut op = Operation::new(&format!("op_{}", i));
        
        // 添加输入
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: ty.clone(),
            shape: vec![2, 2],
        });

        // 添加输出
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: ty.clone(),
            shape: vec![2, 2],
        });

        // 添加类型特定的属性
        match ty {
            Type::F32 => {
                op.attributes.insert("precision".to_string(), Attribute::String("float32".to_string()));
            }
            Type::F64 => {
                op.attributes.insert("precision".to_string(), Attribute::String("float64".to_string()));
            }
            Type::I32 => {
                op.attributes.insert("precision".to_string(), Attribute::String("int32".to_string()));
            }
            Type::I64 => {
                op.attributes.insert("precision".to_string(), Attribute::String("int64".to_string()));
            }
            Type::Bool => {
                op.attributes.insert("precision".to_string(), Attribute::String("bool".to_string()));
            }
            _ => {}
        }

        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);
    
    // 验证每种类型都被正确处理
    for i in 0..5 {
        assert_eq!(module.operations[i].inputs[0].ty, types[i]);
        assert_eq!(module.operations[i].outputs[0].ty, types[i]);
    }
}
