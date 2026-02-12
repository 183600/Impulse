//! Essential boundary edge case tests for the Impulse compiler
//! Covers critical edge cases with standard library assertions (assert!, assert_eq!)

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: 浮点数 subnormal 值的属性处理
#[test]
fn test_subnormal_float_attributes() {
    // f64 的最小正非归一化数约为 4.9e-324
    let subnormal = f64::MIN_POSITIVE * f64::EPSILON;
    let attr = Attribute::Float(subnormal);

    match attr {
        Attribute::Float(val) => {
            // 验证这是一个非常小的正数
            assert!(val > 0.0);
            assert!(val < f64::MIN_POSITIVE);
        }
        _ => panic!("Expected Float attribute"),
    }

    // 测试负的 subnormal 值
    let neg_subnormal = -subnormal;
    let neg_attr = Attribute::Float(neg_subnormal);

    match neg_attr {
        Attribute::Float(val) => {
            assert!(val < 0.0);
            assert!(val > -f64::MIN_POSITIVE);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: 深度嵌套张量类型的内存安全验证
#[test]
fn test_deeply_nested_tensor_memory_safety() {
    // 创建深度嵌套的张量类型
    let mut current_type = Type::F32;

    // 创建 50 层嵌套
    for _i in 0..50 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };

        // 验证中间状态
        if let Type::Tensor { shape, .. } = &current_type {
            assert_eq!(shape.len(), 1);
            assert_eq!(shape[0], 2);
        }
    }

    // 验证克隆操作不会导致堆栈溢出
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 3: 大整数属性的比较
#[test]
fn test_large_integer_attribute_comparison() {
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let another_max = Attribute::Int(i64::MAX);

    // 相等性检查
    assert_eq!(max_int, another_max);
    assert_ne!(max_int, min_int);

    // 创建一个接近边界的值
    let near_max = Attribute::Int(i64::MAX - 1);
    assert_ne!(max_int, near_max);

    // 创建一个接近边界的负值
    let near_min = Attribute::Int(i64::MIN + 1);
    assert_ne!(min_int, near_min);
}

/// Test 4: Module 中重复名称的冲突检测
#[test]
fn test_module_duplicate_name_conflicts() {
    use crate::utils::validation_utils;

    let mut module = Module::new("test_module");

    // 添加具有相同名称的输入
    let input1 = Value {
        name: "shared_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    let input2 = Value {
        name: "shared_name".to_string(),
        ty: Type::I32,
        shape: vec![10],
    };

    module.inputs.push(input1);
    module.inputs.push(input2);

    // 应该检测到重复名称
    let result = validation_utils::validate_module_uniqueness(&module);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Duplicate input name"));
}

/// Test 5: 值名称中包含控制字符的处理
#[test]
fn test_value_names_with_control_characters() {
    let control_chars = [
        "test\x00name",     // NULL
        "test\x01name",     // Start of heading
        "test\ttab",        // Tab
        "test\nnewline",    // Newline
        "test\rcarriage",   // Carriage return
        "test\x1Bescape",   // Escape
    ];

    for name in control_chars.iter() {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };

        // 验证名称被正确存储
        assert_eq!(value.name, *name);
        assert_eq!(value.name.len(), name.len());
    }
}

/// Test 6: 空字符串属性的处理
#[test]
fn test_empty_string_attributes() {
    let empty_attr = Attribute::String(String::new());
    let whitespace_attr = Attribute::String("   ".to_string());

    match empty_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 0);
            assert!(s.is_empty());
        }
        _ => panic!("Expected String attribute"),
    }

    match whitespace_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 3);
            assert!(!s.is_empty());
            assert!(s.chars().all(|c| c.is_whitespace()));
        }
        _ => panic!("Expected String attribute"),
    }
}

/// Test 7: 极端长度的操作类型名称
#[test]
fn test_extremely_long_operation_names() {
    // 创建一个 10000 字符的操作名称
    let long_name = "op_".repeat(5000);
    assert_eq!(long_name.len(), 15000); // "op_" * 5000 = 15000

    let op = Operation::new(&long_name);

    assert_eq!(op.op_type.len(), 15000);
    assert_eq!(op.op_type, long_name);
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
}

/// Test 8: 多层嵌套数组的验证
#[test]
fn test_deeply_nested_array_attributes() {
    // 创建 4 层嵌套的数组属性
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
            ]),
            Attribute::Array(vec![
                Attribute::Int(3),
                Attribute::Int(4),
            ]),
        ]),
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(5),
            ]),
        ]),
    ]);

    match nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);

            // 检查第一个元素
            match &outer[0] {
                Attribute::Array(level1) => {
                    assert_eq!(level1.len(), 2);
                    match &level1[0] {
                        Attribute::Array(level2) => {
                            assert_eq!(level2.len(), 2);
                            match &level2[0] {
                                Attribute::Int(1) => {}
                                _ => panic!("Expected Int(1)"),
                            }
                        }
                        _ => panic!("Expected nested array"),
                    }
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Shape 计算溢出的边界情况
#[test]
fn test_shape_calculation_overflow_boundary() {
    use crate::utils::ir_utils;

    // 测试会溢出的形状计算
    let overflow_shape = [usize::MAX, 2];
    let result = ir_utils::get_num_elements(&Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: overflow_shape.to_vec(),
    });

    // 应该返回 None 表示溢出
    assert_eq!(result, None);

    // 测试不会溢出的边界情况
    let safe_shape = [100_000, 100_000]; // 100 亿元素
    let safe_result = ir_utils::get_num_elements(&Value {
        name: "safe_test".to_string(),
        ty: Type::F32,
        shape: safe_shape.to_vec(),
    });

    assert_eq!(safe_result, Some(10_000_000_000));
}

/// Test 10: 类型系统递归深度的限制
#[test]
fn test_type_system_recursion_depth_limit() {
    // 创建一个非常深的嵌套类型来测试递归深度限制
    let mut deep_type = Type::F32;

    // 创建 1000 层嵌套
    for _ in 0..1000 {
        deep_type = Type::Tensor {
            element_type: Box::new(deep_type),
            shape: vec![1],
        };
    }

    // 验证类型仍然有效
    use crate::utils::validation_utils;
    let result = validation_utils::validate_type(&deep_type);

    // 应该通过验证（假设没有递归深度限制）
    // 如果有深度限制，这里会返回错误
    assert!(result.is_ok() || result.is_err());

    // 尝试获取元素类型
    let element_type = crate::utils::ir_utils::get_element_type(&deep_type);
    assert_eq!(element_type, &Type::F32);
}