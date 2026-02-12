//! 额外的关键边界测试 - 覆盖数值精度、溢出检测和类型转换的边界情况
//! Extra critical edge case tests - covering numerical precision, overflow detection, and type conversion edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// 测试1: 检查Value的num_elements()方法对溢出情况的正确处理
#[test]
fn test_num_elements_overflow_detection() {
    // 测试可能溢出的情况 - 使用足够大的值来触发溢出
    // 在64位系统上，usize是64位，最大值约为1.8e19
    let large_value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // 会溢出
    };
    
    // 应该返回None因为会溢出
    assert_eq!(large_value.num_elements(), None);
    
    // 测试刚好不溢出的情况
    let safe_value = Value {
        name: "safe_test".to_string(),
        ty: Type::F32,
        shape: vec![1_000, 1_000, 1_000], // 10亿，不会溢出
    };
    
    // 应该返回Some因为有明确的元素数
    assert_eq!(safe_value.num_elements(), Some(1_000_000_000));
}

/// 测试2: 测试带有极端小浮点数的属性
#[test]
fn test_denormalized_float_attributes() {
    // 次正规数（denormalized numbers）测试
    let denormal_min = f64::MIN_POSITIVE; // 最小正正规数
    let tiny_val = denormal_min / 2.0;    // 次正规数
    
    let attr = Attribute::Float(tiny_val);
    match attr {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val < f64::MIN_POSITIVE);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// 测试3: 测试空属性字符串的处理
#[test]
fn test_empty_string_attribute() {
    let empty_attr = Attribute::String("".to_string());
    let whitespace_attr = Attribute::String("   ".to_string());
    
    match empty_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 0);
            assert_eq!(s, "");
        }
        _ => panic!("Expected empty String attribute"),
    }
    
    match whitespace_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 3);
            assert_eq!(s, "   ");
        }
        _ => panic!("Expected whitespace String attribute"),
    }
}

/// 测试4: 测试带有混合类型的属性数组
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
            
            // 验证数组中每个元素的类型
            match &arr[0] {
                Attribute::Int(42) => {}
                _ => panic!("Expected Int(42)"),
            }
            
            match &arr[1] {
                Attribute::Float(val) => assert!((val - 3.14).abs() < f64::EPSILON),
                _ => panic!("Expected Float(3.14)"),
            }
            
            match &arr[2] {
                Attribute::String(s) => assert_eq!(s, "test"),
                _ => panic!("Expected String(\"test\")"),
            }
            
            match &arr[3] {
                Attribute::Bool(true) => {}
                _ => panic!("Expected Bool(true)"),
            }
            
            match &arr[4] {
                Attribute::Array(nested) => assert_eq!(nested.len(), 2),
                _ => panic!("Expected nested Array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// 测试5: 测试模块中操作链的正确性
#[test]
fn test_operation_chain_correctness() {
    let mut module = Module::new("chain_test");
    
    // 创建操作链: op1 -> op2 -> op3
    let mut op1 = Operation::new("op1");
    op1.outputs.push(Value {
        name: "intermediate1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(Value {
        name: "intermediate1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op2.outputs.push(Value {
        name: "intermediate2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(Value {
        name: "intermediate2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op3.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].op_type, "op1");
    assert_eq!(module.operations[1].op_type, "op2");
    assert_eq!(module.operations[2].op_type, "op3");
}

/// 测试6: 测试带有单个元素的数组和标量形状的区别
#[test]
fn test_scalar_vs_single_element_array() {
    // 标量（空形状）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
    assert!(scalar.shape.is_empty());
    
    // 单元素数组
    let single_element = Value {
        name: "single_element".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(single_element.num_elements(), Some(1));
    assert_eq!(single_element.shape.len(), 1);
    
    // 它们的num_elements应该相同，但shape不同
    assert_eq!(scalar.num_elements(), single_element.num_elements());
    assert_ne!(scalar.shape, single_element.shape);
}

/// 测试7: 测试包含布尔值true和false的属性
#[test]
fn test_boolean_attribute_values() {
    let true_attr = Attribute::Bool(true);
    let false_attr = Attribute::Bool(false);
    
    match true_attr {
        Attribute::Bool(b) => assert!(b),
        _ => panic!("Expected Bool(true)"),
    }
    
    match false_attr {
        Attribute::Bool(b) => assert!(!b),
        _ => panic!("Expected Bool(false)"),
    }
}

/// 测试8: 测试嵌套tensor类型的深度和验证
#[test]
fn test_deeply_nested_tensor_validation() {
    // 创建深层嵌套: tensor<tensor<tensor<f32, [2]>, [3]>, [4]>
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    
    // 验证level1的有效性
    assert!(level1.is_valid_type());
    
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![3],
    };
    
    // 验证level2的有效性
    assert!(level2.is_valid_type());
    
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };
    
    // 验证level3的有效性
    assert!(level3.is_valid_type());
}

/// 测试9: 测试操作属性的键值对操作
#[test]
fn test_operation_attribute_manipulation() {
    let mut op = Operation::new("attr_test");
    
    // 插入多个属性
    op.attributes.insert("key1".to_string(), Attribute::Int(1));
    op.attributes.insert("key2".to_string(), Attribute::Float(2.0));
    op.attributes.insert("key3".to_string(), Attribute::String("value".to_string()));
    
    assert_eq!(op.attributes.len(), 3);
    
    // 更新现有属性
    op.attributes.insert("key1".to_string(), Attribute::Int(10));
    assert_eq!(op.attributes.len(), 3); // 长度应该不变
    
    // 验证更新后的值
    match op.attributes.get("key1") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 10),
        _ => panic!("Expected Int(10)"),
    }
    
    // 移除属性
    op.attributes.remove("key2");
    assert_eq!(op.attributes.len(), 2);
}

/// 测试10: 测试带有特殊Unicode字符和转义序列的字符串属性
#[test]
fn test_special_character_string_attributes() {
    let special_strings = vec![
        "test\nwith\nnewlines",       // 包含换行符
        "test\twith\ttabs",           // 包含制表符
        "test\\with\\backslashes",    // 包含反斜杠
        "test\"with\"quotes",         // 包含引号
        "🚀emoji🎉test",              // 包含emoji
        "test\r\nwith\rcarriage",     // 包含回车符
    ];
    
    for test_str in special_strings {
        let attr = Attribute::String(test_str.to_string());
        match attr {
            Attribute::String(s) => {
                assert_eq!(s, test_str);
                assert_eq!(s.len(), test_str.len());
            }
            _ => panic!("Expected String attribute"),
        }
    }
}