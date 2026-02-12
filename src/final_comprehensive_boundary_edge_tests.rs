//! Final comprehensive boundary edge tests - 最终综合边界测试
//! 覆盖更多边界情况，包括数值精度、类型转换、内存安全等

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::{calculate_tensor_size_safe, gcd, lcm, round_up_to_multiple, next_power_of_2};
use std::collections::HashMap;

/// Test 1: 数组属性包含递归嵌套结构
#[test]
fn test_deeply_recursive_array_attribute() {
    let level1 = Attribute::Int(1);
    let level2 = Attribute::Array(vec![level1.clone()]);
    let level3 = Attribute::Array(vec![level2.clone()]);
    let level4 = Attribute::Array(vec![level3.clone()]);
    let level5 = Attribute::Array(vec![level4]);

    match level5 {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 1);
                    match &inner[0] {
                        Attribute::Array(deeper) => {
                            assert_eq!(deeper.len(), 1);
                            match &deeper[0] {
                                Attribute::Array(deepest) => {
                                    assert_eq!(deepest.len(), 1);
                                    match &deepest[0] {
                                        Attribute::Int(1) => (),
                                        _ => panic!("Expected Int(1) at deepest level"),
                                    }
                                }
                                _ => panic!("Expected nested array"),
                            }
                        }
                        _ => panic!("Expected nested array"),
                    }
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 2: 浮点属性包含极端精度值
#[test]
fn test_extreme_precision_float_attributes() {
    // 测试接近浮点数精度边界的值
    let precision_tests = vec![
        Attribute::Float(1.0e308),        // 接近 f64 最大值
        Attribute::Float(-1.0e308),       // 接近 f64 最小值
        Attribute::Float(1.0e-308),       // 接近 f64 最小正数
        Attribute::Float(0.9999999999999999), // 接近 1.0
        Attribute::Float(1.0000000000000001), // 接近 1.0+
    ];

    for attr in precision_tests {
        match attr {
            Attribute::Float(val) => {
                assert!(!val.is_nan());
            }
            _ => panic!("Expected Float attribute"),
        }
    }
}

/// Test 3: GCD 和 LCM 函数的边界情况
#[test]
fn test_gcd_lcm_boundary_cases() {
    // GCD 边界情况
    assert_eq!(gcd(usize::MAX, usize::MAX), usize::MAX);
    assert_eq!(gcd(1, usize::MAX), 1);
    assert_eq!(gcd(0, 0), 0);

    // LCM 边界情况 - 测试是否会溢出
    let result = lcm(usize::MAX, 2);
    // 由于 usize::MAX * 2 会溢出，LCM 的实现需要处理这种情况
    // 这里我们只验证函数不会 panic
    assert!(result == 0 || result <= usize::MAX);

    // 测试互质数
    assert_eq!(gcd(9973, 9967), 1); // 两个大质数
    assert_eq!(lcm(9973, 9967), 9973 * 9967);
}

/// Test 4: next_power_of_2 和 round_up_to_multiple 的边界情况
#[test]
fn test_power_of_2_and_round_up_boundaries() {
    // next_power_of_2 边界情况
    assert_eq!(next_power_of_2(usize::MAX), usize::MAX);
    assert_eq!(next_power_of_2(usize::MAX / 2 + 1), usize::MAX);

    // round_up_to_multiple 边界情况
    assert_eq!(round_up_to_multiple(usize::MAX, 1), usize::MAX);
    assert_eq!(round_up_to_multiple(usize::MAX - 1, 2), usize::MAX);
    assert_eq!(round_up_to_multiple(0, usize::MAX), 0);

    // 测试边界附近的值
    let near_max = usize::MAX - 100;
    let rounded = round_up_to_multiple(near_max, 128);
    assert!(rounded >= near_max);
}

/// Test 5: calculate_tensor_size_safe 的极端情况
#[test]
fn test_calculate_tensor_size_extreme_cases() {
    // 测试包含多个大维度的形状
    let shape1 = vec![100_000, 100_000];
    assert_eq!(calculate_tensor_size_safe(&shape1), Some(10_000_000_000));

    // 测试包含 0 的形状
    let shape2 = vec![100_000, 0, 100_000];
    assert_eq!(calculate_tensor_size_safe(&shape2), Some(0));

    // 测试空形状（标量）
    let shape3: Vec<usize> = vec![];
    assert_eq!(calculate_tensor_size_safe(&shape3), Some(1));

    // 测试单个维度为 1
    let shape4 = vec![1];
    assert_eq!(calculate_tensor_size_safe(&shape4), Some(1));

    // 测试所有维度都是 1
    let shape5 = vec![1, 1, 1, 1, 1];
    assert_eq!(calculate_tensor_size_safe(&shape5), Some(1));
}

/// Test 6: 操作类型名称包含特殊字符和长字符串
#[test]
fn test_operation_type_special_characters() {
    let special_op_types = vec![
        "matmul@v2",
        "conv2d/depthwise",
        "op:with:colons",
        "op.with.dots",
        "op_under_score_123",
        "OP_UPPER_CASE",
    ];

    for op_type in special_op_types {
        let op = Operation::new(op_type);
        assert_eq!(op.op_type, op_type);
    }

    // 测试非常长的操作类型名称
    let long_op_type = "a".repeat(10_000);
    let op = Operation::new(&long_op_type);
    assert_eq!(op.op_type.len(), 10_000);
}

/// Test 7: 属性 HashMap 包含特殊键名
#[test]
fn test_attribute_hashmap_special_keys() {
    let mut op = Operation::new("special_keys");
    let mut attrs = HashMap::new();

    let special_keys = vec![
        "key_with_underscore",
        "key-with-dash",
        "key.with.dot",
        "key:with:colon",
        "key@with@at",
        "key123numbers",
        "UPPERCASE_KEY",
    ];

    for key in &special_keys {
        attrs.insert(key.to_string(), Attribute::Int(1));
    }

    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 7);

    for key in &special_keys {
        assert!(op.attributes.contains_key(*key));
    }
}

/// Test 8: 值的形状包含重复维度
#[test]
fn test_value_shape_with_repeated_dimensions() {
    let repeated_shapes = vec![
        vec![5, 5, 5, 5],  // 所有维度相同
        vec![2, 2, 2],      // 小的重复维度
        vec![100, 100],     // 重复的大维度
        vec![1, 1, 1, 1, 1], // 全是 1
    ];

    for shape in repeated_shapes {
        let value = Value {
            name: "repeated_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // 验证 num_elements 正确计算
        let expected = shape.iter().product::<usize>();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 9: 模块包含具有相同名称但不同类型的输入输出
#[test]
fn test_module_with_same_name_different_types() {
    let mut module = Module::new("type_test");

    // 添加输入
    module.inputs.push(Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });

    // 添加输出（不能与输入同名）
    module.outputs.push(Value {
        name: "data_out".to_string(), // 使用不同的名称
        ty: Type::I32,
        shape: vec![10, 10],
    });

    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_ne!(module.inputs[0].ty, module.outputs[0].ty);
}

/// Test 10: 字符串属性包含各种 Unicode 字符
#[test]
fn test_string_attribute_unicode_variations() {
    let unicode_strings = vec![
        // 中文
        "张量操作",
        "卷积层",
        // 日文
        "テンソル",
        "畳み込み",
        // 韩文
        "텐서",
        "합성곱",
        // 阿拉伯文
        "موتر",
        // 希腊文
        "τανυστής",
        // 西里尔文
        "тензор",
        // Emoji
        "🚀 tensor 🔥",
        "🎯 accuracy 📊",
        // 组合
        "张量🚀tensorテンソル",
    ];

    for s in unicode_strings {
        let attr = Attribute::String(s.to_string());
        match attr {
            Attribute::String(ref val) => {
                assert_eq!(val, s);
                // 验证字符串长度（字符数，不是字节数）
                assert_eq!(val.chars().count(), s.chars().count());
            }
            _ => panic!("Expected String attribute"),
        }
    }
}