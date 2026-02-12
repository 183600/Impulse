//! Unique boundary edge tests - 覆盖尚未充分测试的边界情况
//! 使用标准库的 assert! 和 assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// 测试 1: 负零浮点数属性处理
#[test]
fn test_negative_zero_float() {
    let neg_zero = Attribute::Float(-0.0);
    let pos_zero = Attribute::Float(0.0);

    // 负零和正零在位表示上不同，但相等
    if let Attribute::Float(val) = neg_zero {
        assert_eq!(val, 0.0);
        assert!(val.is_sign_negative());
    }

    if let Attribute::Float(val) = pos_zero {
        assert_eq!(val, 0.0);
        assert!(val.is_sign_positive());
    }
}

/// 测试 2: 带有单个维度为1的形状退化
#[test]
fn test_degenerate_shape_with_ones() {
    // 1x1x1...x1 的形状
    let ones_shape = vec![1; 100]; // 100个1
    let value = Value {
        name: "degenerate".to_string(),
        ty: Type::F32,
        shape: ones_shape.clone(),
    };

    // 应该正确计算元素数量为1
    assert_eq!(value.num_elements(), Some(1));
    assert_eq!(value.shape.len(), 100);
}

/// 测试 3: 操作属性中包含空字符串键
#[test]
fn test_empty_string_attribute_key() {
    let mut op = Operation::new("empty_key_op");
    let mut attrs = HashMap::new();

    // 插入空字符串键
    attrs.insert("".to_string(), Attribute::Int(42));
    attrs.insert("normal_key".to_string(), Attribute::Float(3.14));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 2);
    assert!(op.attributes.contains_key(""));
    assert_eq!(op.attributes.get(""), Some(&Attribute::Int(42)));
}

/// 测试 4: 最大安全整数的浮点数转换边界
#[test]
fn test_float_integer_boundary() {
    // 2^53 是 f64 可以精确表示的最大整数
    let exact_int = 9007199254740992.0_f64; // 2^53
    let next_int = 9007199254740993.0_f64;   // 2^53 + 1

    let attr1 = Attribute::Float(exact_int);
    let attr2 = Attribute::Float(next_int);

    if let Attribute::Float(val) = attr1 {
        // 精确表示
        assert_eq!(val, exact_int);
    }

    if let Attribute::Float(val) = attr2 {
        // 可能丢失精度（取决于实现）
        assert!(val >= 9007199254740992.0);
    }
}

/// 测试 5: 递归嵌套张量类型的极限深度
#[test]
fn test_extremely_deep_tensor_nesting() {
    // 创建深度为10的嵌套张量
    let mut nested_type: Type = Type::F32;
    for i in 1..=10 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![i],
        };
    }

    // 验证类型是有效的
    assert!(nested_type.is_valid_type());
}

/// 测试 6: 包含所有 Unicode 控制字符的字符串属性
#[test]
fn test_unicode_control_characters() {
    // 包含各种控制字符的字符串
    let control_chars = String::from_utf8(vec![
        0x00, 0x01, 0x02, 0x07, 0x08, 0x09, 0x0A, 0x0C, 0x0D,
        0x1B, // ESC
    ]).unwrap();

    let attr = Attribute::String(control_chars.clone());

    if let Attribute::String(s) = attr {
        assert_eq!(s, control_chars);
        // 每个控制字符对应一个 UTF-8 字符
        assert_eq!(s.len(), 10);
    }
}

/// 测试 7: 超大数组的元素数量计算溢出保护
#[test]
fn test_large_dimension_overflow_protection() {
    // 创建一个会导致溢出的形状
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };

    // 应该返回 None 而不是 panic
    assert_eq!(value.num_elements(), None);
}

/// 测试 8: 模块输入和输出为相同的 Value 引用模式
#[test]
fn test_module_with_shared_value_pattern() {
    let mut module = Module::new("shared_pattern");

    // 创建相同的输入和输出
    let shared_value = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    };

    module.inputs.push(shared_value.clone());
    module.outputs.push(shared_value);

    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, "shared");
    assert_eq!(module.outputs[0].name, "shared");
}

/// 测试 9: 嵌套属性数组的空元素
#[test]
fn test_nested_array_with_empty_elements() {
    let nested_array = Attribute::Array(vec![
        Attribute::Array(vec![]),           // 空内层数组
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![]),       // 嵌套空数组
        ]),
        Attribute::Array(vec![]),           // 另一个空数组
    ]);

    if let Attribute::Array(outer) = nested_array {
        assert_eq!(outer.len(), 3);

        if let Attribute::Array(inner) = &outer[0] {
            assert_eq!(inner.len(), 0);
        }

        if let Attribute::Array(inner) = &outer[1] {
            assert_eq!(inner.len(), 2);
        }
    }
}

/// 测试 10: 最小和最大浮点数的边界
#[test]
fn test_float_min_max_boundaries() {
    let min_normal = f64::MIN_POSITIVE;    // 最小正正规数
    let max_float = f64::MAX;              // 最大有限浮点数
    let min_float = f64::MIN;              // 最小有限浮点数（负数）

    let attr1 = Attribute::Float(min_normal);
    let attr2 = Attribute::Float(max_float);
    let attr3 = Attribute::Float(min_float);

    if let Attribute::Float(val) = attr1 {
        assert_eq!(val, f64::MIN_POSITIVE);
        assert!(val > 0.0);
    }

    if let Attribute::Float(val) = attr2 {
        assert_eq!(val, f64::MAX);
        assert!(val.is_finite());
    }

    if let Attribute::Float(val) = attr3 {
        assert_eq!(val, f64::MIN);
        assert!(val.is_finite());
        assert!(val < 0.0);
    }
}