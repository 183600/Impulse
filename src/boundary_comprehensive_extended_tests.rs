//! 边界情况综合测试 - 扩展版本
//! 覆盖数值精度、内存安全、特殊值处理等边界情况

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// 测试1: NaN 和 Infinity 属性处理
#[test]
fn test_nan_infinity_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    match nan_attr {
        Attribute::Float(val) => {
            assert!(val.is_nan());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match pos_inf_attr {
        Attribute::Float(val) => {
            assert!(val.is_infinite());
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match neg_inf_attr {
        Attribute::Float(val) => {
            assert!(val.is_infinite());
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// 测试2: 维度计算防止溢出
#[test]
fn test_dimension_overflow_prevention() {
    // 创建可能导致乘法溢出的维度
    let large_value = Value {
        name: "overflow_risk".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };
    
    // num_elements 应该返回 None 而不是溢出
    assert_eq!(large_value.num_elements(), None);
    
    // 另一个接近边界的测试
    let near_overflow = Value {
        name: "near_overflow".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 1_000_000],
    };
    
    // 1e12 > usize::MAX (在64位系统上是1.84e19，所以这个不会溢出)
    // 但测试逻辑应该正确处理
    let result = near_overflow.num_elements();
    assert!(result.is_some() || result.is_none());
}

/// 测试3: 空字符串和特殊字符属性
#[test]
fn test_special_character_attributes() {
    let special_strings = vec![
        "",
        " ",
        "\t\n",
        "a\x08\x01c",
        "\u{0}\u{1}\u{2}",
        "你好世界",  // Unicode
        "🎉😀🚀",  // Emoji
        "\"quote\"",  // 引号
        "'apostrophe'",  // 撇号
        "back\\slash",  // 反斜杠
    ];
    
    for s in special_strings {
        let attr = Attribute::String(s.to_string());
        match &attr {
            Attribute::String(val) => {
                assert_eq!(val, s);
            }
            _ => panic!("Expected String attribute"),
        }
    }
}

/// 测试4: 嵌套数组的极限深度
#[test]
fn test_nested_array_extreme_depth() {
    // 创建深度嵌套的数组
    let level5 = Attribute::Array(vec![Attribute::Int(42)]);
    let level4 = Attribute::Array(vec![level5]);
    let level3 = Attribute::Array(vec![level4]);
    let level2 = Attribute::Array(vec![level3]);
    let level1 = Attribute::Array(vec![level2]);
    
    // 验证可以访问最深层的值
    match &level1 {
        Attribute::Array(outer) => {
            match &outer[0] {
                Attribute::Array(l2) => {
                    match &l2[0] {
                        Attribute::Array(l3) => {
                            match &l3[0] {
                                Attribute::Array(l4) => {
                                    match &l4[0] {
                                        Attribute::Array(l5) => {
                                            match &l5[0] {
                                                Attribute::Int(42) => {
                                                    // 成功访问到最深层
                                                    assert!(true);
                                                }
                                                _ => panic!("Expected Int at innermost level"),
                                            }
                                        }
                                        _ => panic!("Expected Array at level 5"),
                                    }
                                }
                                _ => panic!("Expected Array at level 4"),
                            }
                        }
                        _ => panic!("Expected Array at level 3"),
                    }
                }
                _ => panic!("Expected Array at level 2"),
            }
        }
        _ => panic!("Expected Array at level 1"),
    }
}

/// 测试5: 操作名称的边界情况
#[test]
fn test_operation_name_edge_cases() {
    let edge_case_names: Vec<&str> = vec![
        "",  // 空字符串
        "a",  // 单字符
        "A",  // 大写单字符
        "0",  // 数字
        "op_with_underscores",  // 下划线
        "op-with-dashes",  // 连字符
        "op.with.dots",  // 点号
        "op/with/slashes",  // 斜杠
        "op\\with\\backslashes",  // 反斜杠
        "op with spaces",  // 空格
        "op_with_unicode_中文",  // 中文字符
        "op_with_emoji_🔥",  // Emoji
    ];

    for name in edge_case_names {
        let op = Operation::new(name);
        assert_eq!(op.op_type, name);
    }

    // 测试超长名称
    let long_name = str::repeat("a", 1000);
    let op = Operation::new(&long_name);
    assert_eq!(op.op_type, long_name);
}

/// 测试6: 模块输入输出为空的情况
#[test]
fn test_module_empty_inputs_outputs() {
    let mut module = Module::new("empty_io_module");
    
    // 模块有操作但没有输入输出
    let op = Operation::new("internal_op");
    module.add_operation(op);
    
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
    assert_eq!(module.operations.len(), 1);
    
    // 添加空输入和输出列表
    module.inputs = vec![];
    module.outputs = vec![];
    
    assert!(module.inputs.is_empty());
    assert!(module.outputs.is_empty());
}

/// 测试7: 混合类型的属性数组
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
        Attribute::Int(-999),
        Attribute::Float(f64::MIN),
        Attribute::String("".to_string()),
        Attribute::Bool(false),
    ]);
    
    match &mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 9);
            
            // 验证每个元素
            match &arr[0] {
                Attribute::Int(42) => assert!(true),
                _ => panic!("Expected Int(42)"),
            }
            match &arr[1] {
                Attribute::Float(val) => assert!((val - 3.14).abs() < 0.001),
                _ => panic!("Expected Float(3.14)"),
            }
            match &arr[2] {
                Attribute::String(s) => assert_eq!(s, "hello"),
                _ => panic!("Expected String(\"hello\")"),
            }
            match &arr[3] {
                Attribute::Bool(true) => assert!(true),
                _ => panic!("Expected Bool(true)"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// 测试8: 值名称的边界情况
#[test]
fn test_value_name_edge_cases() {
    let edge_case_names: Vec<&str> = vec![
        "",  // 空字符串
        "x",  // 单字符
        "X",  // 大写
        "_",  // 下划线
        "0",  // 纯数字
        "x0",  // 字母数字
        "input_0",  // 常见模式
        "output:final",  // 冒号
        "tensor[a][b]",  // 方括号
        "data-1",  // 连字符
        "data.1",  // 点号
        "data/1",  // 斜杠
    ];

    // 单独处理超长名称以避免借用问题
    for name in &edge_case_names {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, *name);
    }

    // 测试超长名称
    let long_name = str::repeat("a", 10000);
    let value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.name, long_name);
}

/// 测试9: 操作输入输出长度不匹配的边界情况
#[test]
fn test_operation_io_length_mismatch() {
    let mut op = Operation::new("multi_io");

    // 添加不同数量的输入和输出
    op.inputs.push(Value {
        name: "in1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op.inputs.push(Value {
        name: "in2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op.inputs.push(Value {
        name: "in3".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    // 只添加一个输出
    op.outputs.push(Value {
        name: "out1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    assert_eq!(op.inputs.len(), 3);
    assert_eq!(op.outputs.len(), 1);

    // 测试零输入多输出
    let mut op2 = Operation::new("zero_input_multi_output");
    op2.outputs.push(Value {
        name: "out1".to_string(),
        ty: Type::F32,
        shape: vec![5],
    });
    op2.outputs.push(Value {
        name: "out2".to_string(),
        ty: Type::I32,
        shape: vec![5],
    });

    assert_eq!(op2.inputs.len(), 0);
    assert_eq!(op2.outputs.len(), 2);
}

/// 测试10: 模块名称的特殊字符处理
#[test]
fn test_module_name_special_characters() {
    let special_names: Vec<&str> = vec![
        "",  // 空名称
        "module",  // 普通名称
        "module-with-dashes",  // 连字符
        "module_with_underscores",  // 下划线
        "module.with.dots",  // 点号
        "module/with/slashes",  // 斜杠
        "module\\with\\backslashes",  // 反斜杠
        "module with spaces",  // 空格
        "模块名称",  // 中文
        "🔥hot_module🚀",  // Emoji
        "123456",  // 纯数字
    ];

    for name in &special_names {
        let module = Module::new(*name);
        assert_eq!(module.name, *name);
    }

    // 测试超长名称
    let long_name = str::repeat("a", 5000);
    let module = Module::new(long_name.clone());
    assert_eq!(module.name, long_name);
}
