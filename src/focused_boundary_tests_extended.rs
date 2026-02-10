//! 扩展边界测试 - 覆盖尚未被充分测试的边界场景
//! Extended boundary tests covering edge cases not yet fully tested

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// 测试1: 属性数组的深度嵌套边界
#[test]
fn test_deeply_nested_attribute_array_boundaries() {
    // 创建5层深度的嵌套数组
    let deep_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Array(vec![
                        Attribute::Int(42)
                    ])
                ])
            ])
        ])
    ]);
    
    // 验证嵌套结构可以正确创建和访问
    match deep_nested {
        Attribute::Array(outer) => {
            match &outer[0] {
                Attribute::Array(l1) => {
                    match &l1[0] {
                        Attribute::Array(l2) => {
                            match &l2[0] {
                                Attribute::Array(l3) => {
                                    match &l3[0] {
                                        Attribute::Array(l4) => {
                                            match &l4[0] {
                                                Attribute::Int(val) => {
                                                    assert_eq!(*val, 42);
                                                }
                                                _ => panic!("Expected Int at deepest level"),
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
        _ => panic!("Expected Array at top level"),
    }
}

/// 测试2: 包含空字符串和特殊字符的字符串属性
#[test]
fn test_string_attributes_with_special_characters() {
    let mut op = Operation::new("string_special_test");
    let mut attrs = HashMap::new();
    
    // 空字符串
    attrs.insert("empty".to_string(), Attribute::String("".to_string()));
    
    // 仅包含空格的字符串
    attrs.insert("spaces".to_string(), Attribute::String("   ".to_string()));
    
    // 包含各种Unicode字符
    attrs.insert("unicode".to_string(), Attribute::String("你好🌍こんにちは".to_string()));
    
    // 包含转义字符的字符串
    attrs.insert("escaped".to_string(), Attribute::String("line1\nline2\ttab".to_string()));
    
    // 非常长的字符串
    let long_string = "x".repeat(10000);
    attrs.insert("long".to_string(), Attribute::String(long_string));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    
    // 验证空字符串
    match op.attributes.get("empty") {
        Some(Attribute::String(s)) => assert_eq!(s.len(), 0),
        _ => panic!("Expected empty string"),
    }
    
    // 验证Unicode字符串
    match op.attributes.get("unicode") {
        Some(Attribute::String(s)) => {
            assert!(s.contains('好'));
            assert!(s.contains('🌍'));
            assert!(s.contains('こ'));
        }
        _ => panic!("Expected unicode string"),
    }
    
    // 验证长字符串
    match op.attributes.get("long") {
        Some(Attribute::String(s)) => assert_eq!(s.len(), 10000),
        _ => panic!("Expected long string"),
    }
}

/// 测试3: 大规模操作链的Module
#[test]
fn test_module_with_large_operation_chain() {
    let mut module = Module::new("large_chain");
    
    // 添加初始输入
    module.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![100],
    });
    
    // 创建100个连续操作的链
    let mut current_name = "input".to_string();
    for i in 0..100 {
        let mut op = Operation::new(&format!("layer_{}", i));
        op.inputs.push(Value {
            name: current_name.clone(),
            ty: Type::F32,
            shape: vec![100],
        });
        current_name = format!("layer_{}_output", i);
        op.outputs.push(Value {
            name: current_name.clone(),
            ty: Type::F32,
            shape: vec![100],
        });
        module.add_operation(op);
    }
    
    // 添加最终输出
    module.outputs.push(Value {
        name: current_name,
        ty: Type::F32,
        shape: vec![100],
    });
    
    assert_eq!(module.operations.len(), 100);
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
}

/// 测试4: 混合类型属性的数组
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Int(-100),
        Attribute::Float(-2.71),
        Attribute::Bool(false),
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 7);
            
            // 验证混合类型
            match &arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
            match &arr[1] {
                Attribute::Float(val) => {
                    assert!((val - 3.14).abs() < f64::EPSILON);
                },
                _ => panic!("Expected Float(3.14)"),
            }
            match &arr[2] {
                Attribute::String(s) => {
                    assert_eq!(s, "test");
                },
                _ => panic!("Expected String(\"test\")"),
            }
            match &arr[3] {
                Attribute::Bool(true) => {},
                _ => panic!("Expected Bool(true)"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// 测试5: 包含所有零维的Tensor形状
#[test]
fn test_tensor_with_all_zero_dimensions() {
    let value = Value {
        name: "all_zeros".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0, 0],
    };
    
    // 所有维度为0应该产生0个元素
    assert_eq!(value.num_elements(), Some(0));
}

/// 测试6: 包含单个1维度的长形状
#[test]
fn test_tensor_with_single_unit_dimension() {
    let value = Value {
        name: "single_unit".to_string(),
        ty: Type::I32,
        shape: vec![1, 1, 1, 1, 1],
    };
    
    // 所有维度为1应该产生1个元素
    assert_eq!(value.num_elements(), Some(1));
}

/// 测试7: 操作属性键的边界情况
#[test]
fn test_operation_attribute_key_boundaries() {
    let mut op = Operation::new("attr_key_test");
    let mut attrs = HashMap::new();
    
    // 空键（虽然可能不推荐，但应能处理）
    attrs.insert("".to_string(), Attribute::Int(0));
    
    // 非常长的键
    let long_key = "x".repeat(1000);
    attrs.insert(long_key.clone(), Attribute::Int(1));
    
    // 包含特殊字符的键
    attrs.insert("key-with-dashes".to_string(), Attribute::Int(2));
    attrs.insert("key_with_underscores".to_string(), Attribute::Int(3));
    attrs.insert("key.with.dots".to_string(), Attribute::Int(4));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    assert!(op.attributes.contains_key(""));
    assert!(op.attributes.contains_key(&long_key));
    assert!(op.attributes.contains_key("key-with-dashes"));
}

/// 测试8: Value名称的边界情况
#[test]
fn test_value_name_boundaries() {
    let test_cases: Vec<(String, Vec<usize>)> = vec![
        ("".to_string(), vec![1]),  // 空名称
        ("a".to_string(), vec![1]),  // 单字符名称
        ("x".repeat(1000), vec![1]),  // 非常长的名称
        ("name with spaces".to_string(), vec![1]),  // 包含空格的名称
        ("name/with/slashes".to_string(), vec![1]),  // 包含斜杠的名称
    ];
    
    for (name, shape) in test_cases {
        let value = Value {
            name: name.clone(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.name, name);
        assert_eq!(value.shape, shape);
    }
}

/// 测试9: 包含多个相同类型但不同形状的输入
#[test]
fn test_operation_with_same_type_different_shapes() {
    let mut op = Operation::new("shape_variety");
    
    // 添加相同类型但不同形状的输入
    op.inputs.push(Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    });
    
    op.inputs.push(Value {
        name: "vector".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    op.inputs.push(Value {
        name: "matrix".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    });
    
    op.inputs.push(Value {
        name: "tensor3d".to_string(),
        ty: Type::F32,
        shape: vec![2, 3, 4],
    });
    
    assert_eq!(op.inputs.len(), 4);
    assert_eq!(op.inputs[0].shape.len(), 0);
    assert_eq!(op.inputs[1].shape.len(), 1);
    assert_eq!(op.inputs[2].shape.len(), 2);
    assert_eq!(op.inputs[3].shape.len(), 3);
}

/// 测试10: Module名称的边界情况
#[test]
fn test_module_name_boundaries() {
    let test_names = vec![
        "a".to_string(),  // 单字符
        "test_module".to_string(),  // 常规名称
        "Module_With_Underscores".to_string(),  // 大小写混合
        "module-with-dashes".to_string(),  // 包含连字符
        "123numbers".to_string(),  // 以数字开头
        "a".repeat(100),  // 较长的名称
    ];
    
    for name in test_names {
        let module = Module::new(&name);
        assert_eq!(module.name, name);
    }
}