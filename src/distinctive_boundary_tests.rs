//! Distinctive boundary tests - 覆盖独特的边界情况，使用标准库 assert! 和 assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// 测试1: Value 的 num_elements 方法在接近 usize::MAX 边界时的行为
#[test]
fn test_value_num_elements_near_usize_max() {
    // 测试大数值但不会溢出的情况
    let large_but_safe = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000, 1000], // 10亿元素
    };
    assert_eq!(large_but_safe.num_elements(), Some(1_000_000_000));

    // 测试空形状（标量）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));

    // 测试单个大维度
    let single_large_dim = Value {
        name: "single_large".to_string(),
        ty: Type::I32,
        shape: vec![usize::MAX / 2],
    };
    // 应该返回 Some，因为不会溢出
    assert!(single_large_dim.num_elements().is_some());
}

/// 测试2: Operation 的属性使用各种极端浮点值
#[test]
fn test_operation_extreme_float_attributes() {
    let mut op = Operation::new("float_extremes");
    let mut attrs = HashMap::new();

    // 添加各种极端浮点值
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("neg_epsilon".to_string(), Attribute::Float(-f64::EPSILON));

    op.attributes = attrs;

    // 验证所有属性都已正确存储
    assert_eq!(op.attributes.len(), 4);
    assert!(op.attributes.contains_key("max_f64"));
    assert!(op.attributes.contains_key("min_f64"));
    assert!(op.attributes.contains_key("epsilon"));
    assert!(op.attributes.contains_key("neg_epsilon"));

    // 验证特定值
    match op.attributes.get("max_f64") {
        Some(Attribute::Float(val)) => assert_eq!(*val, f64::MAX),
        _ => panic!("Expected Float attribute"),
    }
}

/// 测试3: Module 包含多个同名但不同类型的操作
#[test]
fn test_module_same_name_different_types() {
    let mut module = Module::new("mixed_ops");

    // 添加多个相同类型名称的操作
    for _ in 0..3 {
        let op = Operation::new("matmul");
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);
    // 所有操作应该有相同的 op_type
    for op in &module.operations {
        assert_eq!(op.op_type, "matmul");
    }
}

/// 测试4: Attribute 数组的嵌套深度测试
#[test]
fn test_deeply_nested_attribute_arrays() {
    // 创建 5 层嵌套的数组
    let level5 = Attribute::Array(vec![Attribute::Int(1)]);
    let level4 = Attribute::Array(vec![level5]);
    let level3 = Attribute::Array(vec![level4]);
    let level2 = Attribute::Array(vec![level3]);
    let level1 = Attribute::Array(vec![level2]);

    match level1 {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 1);
            match &outer[0] {
                Attribute::Array(l2) => {
                    assert_eq!(l2.len(), 1);
                    match &l2[0] {
                        Attribute::Array(l3) => {
                            assert_eq!(l3.len(), 1);
                            match &l3[0] {
                                Attribute::Array(l4) => {
                                    assert_eq!(l4.len(), 1);
                                    match &l4[0] {
                                        Attribute::Array(l5) => {
                                            assert_eq!(l5.len(), 1);
                                            match &l5[0] {
                                                Attribute::Int(1) => {},
                                                _ => panic!("Expected Int at deepest level"),
                                            }
                                        },
                                        _ => panic!("Expected Array at level 4"),
                                    }
                                },
                                _ => panic!("Expected Array at level 3"),
                            }
                        },
                        _ => panic!("Expected Array at level 2"),
                    }
                },
                _ => panic!("Expected Array at level 1"),
            }
        },
        _ => panic!("Expected outer Array"),
    }
}

/// 测试5: Tensor 类型使用所有基础类型组合
#[test]
fn test_tensor_all_element_types() {
    let element_types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];

    for element_type in element_types {
        let tensor_type = Type::Tensor {
            element_type: Box::new(element_type.clone()),
            shape: vec![2, 3, 4],
        };

        match tensor_type {
            Type::Tensor { element_type: et, shape } => {
                assert_eq!(shape, vec![2, 3, 4]);
                assert_eq!(*et, element_type);
            },
            _ => panic!("Expected Tensor type"),
        }
    }
}

/// 测试6: Value 形状包含零维度的多种组合
#[test]
fn test_value_zero_dimension_variations() {
    let zero_shapes = vec![
        vec![0],           // 单个零维度
        vec![0, 0],        // 多个零维度
        vec![10, 0, 10],   // 零在中间
        vec![0, 10, 10],   // 零在开头
        vec![10, 10, 0],   // 零在末尾
    ];

    for shape in zero_shapes {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // 任何包含零的形状都应该产生 0 个元素
        assert_eq!(value.num_elements(), Some(0));
        assert_eq!(value.shape, shape);
    }
}

/// 测试7: Module 的输入和输出使用不同形状的张量
#[test]
fn test_module_variable_shape_inputs_outputs() {
    let mut module = Module::new("variable_shapes");

    // 添加不同形状的输入
    module.inputs.push(Value {
        name: "input_1d".to_string(),
        ty: Type::F32,
        shape: vec![100],
    });
    module.inputs.push(Value {
        name: "input_2d".to_string(),
        ty: Type::F32,
        shape: vec![50, 50],
    });
    module.inputs.push(Value {
        name: "input_3d".to_string(),
        ty: Type::F32,
        shape: vec![10, 10, 10],
    });

    // 添加不同形状的输出
    module.outputs.push(Value {
        name: "output_scalar".to_string(),
        ty: Type::F64,
        shape: vec![],
    });
    module.outputs.push(Value {
        name: "output_4d".to_string(),
        ty: Type::I32,
        shape: vec![2, 3, 4, 5],
    });

    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].shape, vec![100usize]);
    assert_eq!(module.inputs[1].shape, vec![50, 50]);
    assert_eq!(module.inputs[2].shape, vec![10, 10, 10]);
    assert_eq!(module.outputs[0].shape, Vec::<usize>::new());
    assert_eq!(module.outputs[1].shape, vec![2, 3, 4, 5]);
}

/// 测试8: Operation 属性使用重复的键（最后写入生效）
#[test]
fn test_operation_duplicate_keys() {
    let mut op = Operation::new("duplicate_keys");
    let mut attrs = HashMap::new();

    // 多次插入相同的键，最后一个值生效
    attrs.insert("value".to_string(), Attribute::Int(1));
    attrs.insert("value".to_string(), Attribute::Int(2));
    attrs.insert("value".to_string(), Attribute::Int(3));

    op.attributes = attrs;

    // 应该只有一个键，值为最后一个插入的值
    assert_eq!(op.attributes.len(), 1);
    match op.attributes.get("value") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 3),
        _ => panic!("Expected Int(3)"),
    }
}

/// 测试9: Value 名称使用各种特殊字符和长字符串
#[test]
fn test_value_special_names() {
    // 测试空字符串
    let empty_name = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(empty_name.name, "");

    // 测试空格
    let space_name = Value {
        name: "   ".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(space_name.name, "   ");

    // 测试非常长的名称
    let long_name_str = "a".repeat(10000);
    let long_name = Value {
        name: long_name_str.clone(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(long_name.name.len(), 10000);

    // 测试包含特殊字符的名称
    let special_chars = vec![
        "name with spaces",
        "name\twith\ttabs",
        "name\nwith\nnewlines",
        "name/with/slashes",
        "name\\with\\backslashes",
        "name\"with\"quotes",
    ];

    for name in special_chars {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, name);
    }
}

/// 测试10: Module 包含大量操作测试内存使用
#[test]
fn test_module_many_operations() {
    let mut module = Module::new("many_ops");

    // 添加大量操作以测试内存处理
    let num_ops = 1000;
    for i in 0..num_ops {
        let mut op = Operation::new(&format!("op_{}", i));
        // 为每个操作添加一些输入
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), num_ops);

    // 验证一些随机操作
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[500].op_type, "op_500");
    assert_eq!(module.operations[999].op_type, "op_999");

    // 验证所有操作都有输入
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 1);
    }
}
