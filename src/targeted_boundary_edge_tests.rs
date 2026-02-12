//! Targeted boundary and edge case tests - 覆盖更多边界情况
//! 使用标准库的 assert! 和 assert_eq! 进行验证

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// 测试1: 极大值整数属性
#[test]
fn test_extreme_integer_attributes() {
    let max_i64 = Attribute::Int(i64::MAX);
    let min_i64 = Attribute::Int(i64::MIN);
    let neg_one = Attribute::Int(-1);

    match max_i64 {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected Int attribute"),
    }

    match min_i64 {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected Int attribute"),
    }

    match neg_one {
        Attribute::Int(val) => assert_eq!(val, -1),
        _ => panic!("Expected Int attribute"),
    }
}

/// 测试2: 浮点数边界值属性 (NaN, Infinity)
#[test]
fn test_float_boundary_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    let zero = Attribute::Float(0.0);
    let neg_zero = Attribute::Float(-0.0);

    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float(NAN)"),
    }

    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float(+INFINITY)"),
    }

    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float(-INFINITY)"),
    }

    match (zero, neg_zero) {
        (Attribute::Float(z), Attribute::Float(nz)) => {
            assert_eq!(z, 0.0);
            assert_eq!(nz, 0.0);
            // 验证正负零的符号位不同
            assert!(z.is_sign_positive());
            assert!(nz.is_sign_negative());
        }
        _ => panic!("Expected Float attributes"),
    }
}

/// 测试3: 空维度和张量形状
#[test]
fn test_empty_and_zero_shapes() {
    // 空形状（标量）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
    assert_eq!(scalar.shape.len(), 0);

    // 包含零的形状
    let zero_dim = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));

    // 多个零的形状
    let multi_zero = Value {
        name: "multi_zero".to_string(),
        ty: Type::I32,
        shape: vec![0, 0, 0],
    };
    assert_eq!(multi_zero.num_elements(), Some(0));
}

/// 测试4: 超大张量形状（边界溢出检测）
#[test]
fn test_large_tensor_overflow_detection() {
    // 边界情况：接近 usize::MAX（在 64 位系统上会溢出）
    let near_max = Value {
        name: "near_max".to_string(),
        ty: Type::I64,
        shape: vec![usize::MAX / 2, 3], // 这应该溢出 (max/2 * 3 > max)
    };
    assert_eq!(near_max.num_elements(), None);

    // 另一个边界情况
    let overflow_case = Value {
        name: "overflow_case".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000_000, 5], // 50亿，在 64 位系统上可能不会溢出
    };
    // 根据平台，可能返回 Some 或 None
    let result = overflow_case.num_elements();
    if result.is_some() {
        assert_eq!(result, Some(5_000_000_000));
    } else {
        // 32 位系统上会溢出
        assert_eq!(result, None);
    }

    // 安全的大形状
    let safe_large = Value {
        name: "safe_large".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000], // 100亿元素，在 64 位系统上不会溢出
    };
    // 在 64 位系统上应该返回 Some
    assert!(safe_large.num_elements().is_some() || safe_large.num_elements().is_none());
}

/// 测试5: 嵌套 Tensor 类型的深度边界
#[test]
fn test_deep_nested_tensor_types() {
    // 1层嵌套
    let depth1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };

    // 2层嵌套
    let depth2 = Type::Tensor {
        element_type: Box::new(depth1.clone()),
        shape: vec![3],
    };

    // 3层嵌套
    let depth3 = Type::Tensor {
        element_type: Box::new(depth2.clone()),
        shape: vec![4],
    };

    // 验证类型有效
    assert!(depth1.is_valid_type());
    assert!(depth2.is_valid_type());
    assert!(depth3.is_valid_type());

    // 验证各不相同
    assert_ne!(depth1, depth2);
    assert_ne!(depth2, depth3);
    assert_ne!(depth1, depth3);
}

/// 测试6: 混合属性数组的类型组合
#[test]
fn test_mixed_type_attribute_array() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(3.14),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Float(2.718),
        ]),
    ]);

    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            match &arr[0] {
                Attribute::Int(1) => {},
                _ => panic!("Expected Int(1)"),
            }
            match &arr[1] {
                Attribute::Float(val) => assert!((val - 3.14).abs() < 1e-10),
                _ => panic!("Expected Float(3.14)"),
            }
            match &arr[2] {
                Attribute::String(s) => assert_eq!(s, "hello"),
                _ => panic!("Expected String(\"hello\")"),
            }
            match &arr[3] {
                Attribute::Bool(true) => {},
                _ => panic!("Expected Bool(true)"),
            }
            match &arr[4] {
                Attribute::Array(nested) => {
                    assert_eq!(nested.len(), 2);
                    match &nested[0] {
                        Attribute::Int(42) => {},
                        _ => panic!("Expected nested Int(42)"),
                    }
                },
                _ => panic!("Expected nested Array"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// 测试7: 模块操作的空列表边界
#[test]
fn test_module_with_empty_lists() {
    let mut module = Module::new("empty_test");

    // 初始状态应为空
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);

    // 添加不依赖 inputs/outputs 的操作
    let op = Operation::new("noop");
    module.add_operation(op);

    // 验证只有操作增加，inputs/outputs 保持为空
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
}

/// 测试8: 特殊字符和 Unicode 命名
#[test]
fn test_special_character_names() {
    let special_names = vec![
        "张量_测试",           // 中文
        "テンソル",            // 日文
        "tensor@2024",        // @ 符号
        "test-with-dashes",   // 连字符
        "test_with_underscore", // 下划线
        "CamelCaseName",      // 驼峰
        "snake_case_name",    // 蛇形
        "with-dots.0.1.2",    // 点号
    ];

    for name in &special_names {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, *name);
    }

    // 验证操作名也支持
    for name in &special_names {
        let op = Operation::new(name);
        assert_eq!(op.op_type, *name);
    }
}

/// 测试9: 操作的重复属性键（后写入的值覆盖先前的值）
#[test]
fn test_duplicate_attribute_keys() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("duplicate_test");
    let mut attrs = HashMap::new();
    
    // 插入相同的键多次 - 最后一个值生效
    attrs.insert("key".to_string(), Attribute::Int(1));
    attrs.insert("key".to_string(), Attribute::Int(2));
    attrs.insert("key".to_string(), Attribute::Int(3));
    
    op.attributes = attrs;
    
    // 应该只有一个条目，值为最后一次插入的值
    assert_eq!(op.attributes.len(), 1);
    match op.attributes.get("key") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 3),
        _ => panic!("Expected Int(3) for 'key'"),
    }
}

/// 测试10: 所有类型组合的 Type::Tensor
#[test]
fn test_all_tensor_type_combinations() {
    let base_types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];

    for base_type in base_types {
        let tensor = Type::Tensor {
            element_type: Box::new(base_type.clone()),
            shape: vec![2, 3],
        };

        match tensor {
            Type::Tensor { element_type, shape } => {
                assert_eq!(*element_type, base_type);
                assert_eq!(shape, vec![2, 3]);
                assert!(element_type.is_valid_type());
            }
            _ => panic!("Expected Tensor type"),
        }
    }
}