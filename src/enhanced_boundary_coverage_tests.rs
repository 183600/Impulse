//! Enhanced boundary coverage tests - additional edge cases for compiler robustness
//! 覆盖数值精度、内存安全和边界条件的额外测试

use crate::ir::{Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Value with subnormal (denormal) float values
/// 测试次正规浮点数值处理
#[test]
fn test_subnormal_float_values() {
    // f64::MIN_POSITIVE 是最小的正次正规数
    let subnormal_pos = f64::MIN_POSITIVE;
    // 更小的值可能下溢到零
    let subnormal_tiny = 5e-324;
    
    let attr_pos = Attribute::Float(subnormal_pos);
    let attr_tiny = Attribute::Float(subnormal_tiny);
    
    if let Attribute::Float(val) = attr_pos {
        // MIN_POSITIVE 约为 2.2e-308，它应该大于 0
        assert!(val > 0.0);
    }
    
    if let Attribute::Float(val) = attr_tiny {
        // 非常小的值，检查它不会导致 panic
        assert!(val >= 0.0);
    }
}

/// Test 2: Attribute with negative zero float
/// 测试负零浮点数的处理
#[test]
fn test_negative_zero_float() {
    let neg_zero = -0.0_f64;
    let pos_zero = 0.0_f64;
    
    let attr_neg = Attribute::Float(neg_zero);
    let attr_pos = Attribute::Float(pos_zero);
    
    if let Attribute::Float(val) = attr_neg {
        assert!(val == 0.0);
        assert!(val.is_sign_negative());
    }
    
    if let Attribute::Float(val) = attr_pos {
        assert!(val == 0.0);
        assert!(val.is_sign_positive());
    }
}

/// Test 3: Value with shape that approaches usize limit without overflow
/// 测试接近 usize 限制的形状计算
#[test]
fn test_shape_near_usize_limit() {
    // 测试不会溢出的大形状
    let safe_shape = vec![1000, 1000, 1000]; // 10亿个元素
    
    let value = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: safe_shape.clone(),
    };
    
    // 应该成功计算元素数量
    assert_eq!(value.num_elements(), Some(1_000_000_000));
    
    // 测试包含零维的形状（结果为零）
    let zero_shape = vec![0, 1000000, 1000000];
    let zero_value = Value {
        name: "zero_elements".to_string(),
        ty: Type::F32,
        shape: zero_shape,
    };
    assert_eq!(zero_value.num_elements(), Some(0));
}

/// Test 4: Module with operation containing extremely long attribute names
/// 测试包含极长属性名称的操作
#[test]
fn test_extremely_long_attribute_names() {
    let mut op = Operation::new("long_attr_op");
    let mut attrs = HashMap::new();
    
    // 创建极长的属性名称
    let long_name = "a".repeat(10000);
    attrs.insert(long_name, Attribute::Int(42));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1);
}

/// Test 5: Value with all ones and all zeros shapes
/// 测试全1和全0形状的张量
#[test]
fn test_all_ones_and_zeros_shapes() {
    // 全1形状
    let ones_shape = vec![1, 1, 1, 1, 1];
    let ones_value = Value {
        name: "ones".to_string(),
        ty: Type::F32,
        shape: ones_shape.clone(),
    };
    assert_eq!(ones_value.num_elements(), Some(1));
    
    // 全零形状
    let zeros_shape = vec![0, 0, 0, 0, 0];
    let zeros_value = Value {
        name: "zeros".to_string(),
        ty: Type::F32,
        shape: zeros_shape,
    };
    assert_eq!(zeros_value.num_elements(), Some(0));
}

/// Test 6: Attribute array with many elements
/// 测试包含大量元素的属性数组
#[test]
fn test_large_attribute_array() {
    let mut attrs = Vec::new();
    
    // 创建包含1000个元素的数组
    for i in 0..1000 {
        attrs.push(Attribute::Int(i as i64));
    }
    
    let array_attr = Attribute::Array(attrs);
    
    if let Attribute::Array(vec) = array_attr {
        assert_eq!(vec.len(), 1000);
        assert_eq!(vec[0], Attribute::Int(0));
        assert_eq!(vec[999], Attribute::Int(999));
    }
}

/// Test 7: Module with deeply nested tensor types
/// 测试深度嵌套的张量类型
#[test]
fn test_deeply_nested_tensor_types() {
    // 创建5层嵌套的张量类型
    let level1 = Type::F32;
    
    // 先验证类型有效性
    assert!(level1.is_valid_type());
    
    let level2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    assert!(level2.is_valid_type());
    
    let level3 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    assert!(level3.is_valid_type());
    
    let level4 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2],
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    assert!(level4.is_valid_type());
    
    let level5 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        }),
        shape: vec![5],
    };
    assert!(level5.is_valid_type());
}

/// Test 8: Value with alternating dimension pattern
/// 测试交替维度的形状模式
#[test]
fn test_alternating_dimension_pattern() {
    // 交替的 0 和 1 维度
    let alternating = vec![0, 1, 0, 1, 0, 1];
    let value = Value {
        name: "alternating".to_string(),
        ty: Type::F32,
        shape: alternating,
    };
    
    // 任何包含零的维度都会导致总元素数为零
    assert_eq!(value.num_elements(), Some(0));
    
    // 全为1的维度应该保持元素数量为1
    let all_ones = vec![1, 1, 1, 1, 1, 1];
    let ones_value = Value {
        name: "all_ones".to_string(),
        ty: Type::F32,
        shape: all_ones,
    };
    assert_eq!(ones_value.num_elements(), Some(1));
}

/// Test 9: Attribute with string containing special escape sequences
/// 测试包含特殊转义序列的字符串属性
#[test]
fn test_string_with_escape_sequences() {
    let special_strings = vec![
        "test\nnewline\ttab\rcarriage",
        "quote\"single\'quote",
        "back\\slash",
        "null\x00byte",
        "unicode\u{1F600}emoji",
    ];
    
    for s in special_strings {
        let attr = Attribute::String(s.to_string());
        if let Attribute::String(str_val) = attr {
            assert_eq!(str_val, s);
        }
    }
}

/// Test 10: Module with mixed attribute types in single operation
/// 测试单个操作中混合使用多种属性类型
#[test]
fn test_mixed_attribute_types_in_single_operation() {
    let mut op = Operation::new("mixed_attrs");
    let mut attrs = HashMap::new();
    
    // 插入所有类型的属性
    attrs.insert("int_attr".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("float_attr".to_string(), Attribute::Float(std::f64::consts::E));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(false));
    attrs.insert("string_attr".to_string(), Attribute::String("mixed".to_string()));
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.0),
        Attribute::Bool(true),
    ]));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    
    // 验证每个属性类型
    match op.attributes.get("int_attr") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected Int attribute"),
    }
    
    match op.attributes.get("float_attr") {
        Some(Attribute::Float(val)) => assert!((val - std::f64::consts::E).abs() < f64::EPSILON),
        _ => panic!("Expected Float attribute"),
    }
    
    match op.attributes.get("bool_attr") {
        Some(Attribute::Bool(false)) => {},
        _ => panic!("Expected Bool(false)"),
    }
    
    match op.attributes.get("string_attr") {
        Some(Attribute::String(s)) if s == "mixed" => {},
        _ => panic!("Expected String(\"mixed\")"),
    }
    
    match op.attributes.get("array_attr") {
        Some(Attribute::Array(arr)) => assert_eq!(arr.len(), 3),
        _ => panic!("Expected Array attribute"),
    }
}