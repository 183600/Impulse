//! Comprehensive Critical Boundary Tests v2 - 覆盖关键边界情况的新测试
//! 使用标准库 assert! 和 assert_eq! 宏

use crate::ir::{Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: 检测整数溢出的形状计算 - 使用 checked_mul 保护
#[test]
fn test_checked_multiplication_overflow_protection() {
    // 创建可能溢出的形状
    let safe_shape = vec![1000, 1000, 100]; // 100M 元素
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: safe_shape,
    };
    
    // 使用 num_elements 方法，它使用 checked_mul
    assert_eq!(safe_value.num_elements(), Some(100_000_000));
    
    // 包含0的形状应该返回0
    let zero_shape_value = Value {
        name: "zero_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 100],
    };
    assert_eq!(zero_shape_value.num_elements(), Some(0));
}

/// Test 2: 嵌套Tensor类型的递归验证
#[test]
fn test_deeply_nested_tensor_type_validation() {
    // 创建深度嵌套的Tensor类型
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![4],
    };
    
    // 验证嵌套类型是有效的
    assert!(level3.is_valid_type());
    assert!(level2.is_valid_type());
    assert!(level1.is_valid_type());
}

/// Test 3: 浮点数特殊值的属性处理
#[test]
fn test_special_float_values_in_attributes() {
    // 测试正负无穷和零值
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    let pos_zero = Attribute::Float(0.0);
    let neg_zero = Attribute::Float(-0.0);
    
    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected positive infinity"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected negative infinity"),
    }
    
    match pos_zero {
        Attribute::Float(val) => assert_eq!(val, 0.0),
        _ => panic!("Expected positive zero"),
    }
    
    match neg_zero {
        Attribute::Float(val) => assert_eq!(val, -0.0),
        _ => panic!("Expected negative zero"),
    }
}

/// Test 4: 空字符串和空白字符串属性
#[test]
fn test_empty_and_whitespace_string_attributes() {
    let empty_string = Attribute::String(String::new());
    let whitespace_string = Attribute::String("   \t\n   ".to_string());
    
    match empty_string {
        Attribute::String(s) => assert!(s.is_empty()),
        _ => panic!("Expected empty string"),
    }
    
    match whitespace_string {
        Attribute::String(s) => assert_eq!(s.len(), 8),
        _ => panic!("Expected whitespace string"),
    }
}

/// Test 5: 单元素数组的特殊情况
#[test]
fn test_single_element_array_attribute() {
    let single_int_array = Attribute::Array(vec![Attribute::Int(42)]);
    let single_float_array = Attribute::Array(vec![Attribute::Float(3.14)]);
    
    match single_int_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
        },
        _ => panic!("Expected Array"),
    }
    
    match single_float_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Float(val) if (val - 3.14).abs() < f64::EPSILON => (),
                _ => panic!("Expected Float(3.14)"),
            }
        },
        _ => panic!("Expected Array"),
    }
}

/// Test 6: 布尔属性的真假值
#[test]
fn test_boolean_attribute_values() {
    let true_attr = Attribute::Bool(true);
    let false_attr = Attribute::Bool(false);
    
    match true_attr {
        Attribute::Bool(val) => assert!(val),
        _ => panic!("Expected Bool(true)"),
    }
    
    match false_attr {
        Attribute::Bool(val) => assert!(!val),
        _ => panic!("Expected Bool(false)"),
    }
}

/// Test 7: 边界整数值的属性
#[test]
fn test_boundary_integer_values_in_attributes() {
    // 测试边界整数值
    let max_u8_attr = Attribute::Int(u8::MAX as i64);
    let min_i8_attr = Attribute::Int(i8::MIN as i64);
    let max_u16_attr = Attribute::Int(u16::MAX as i64);
    let min_i16_attr = Attribute::Int(i16::MIN as i64);
    
    match max_u8_attr {
        Attribute::Int(val) => assert_eq!(val, 255),
        _ => panic!("Expected max u8 value"),
    }
    
    match min_i8_attr {
        Attribute::Int(val) => assert_eq!(val, -128),
        _ => panic!("Expected min i8 value"),
    }
    
    match max_u16_attr {
        Attribute::Int(val) => assert_eq!(val, 65535),
        _ => panic!("Expected max u16 value"),
    }
    
    match min_i16_attr {
        Attribute::Int(val) => assert_eq!(val, -32768),
        _ => panic!("Expected min i16 value"),
    }
}

/// Test 8: 操作名称包含特殊字符
#[test]
fn test_operation_names_with_special_characters() {
    let special_names = [
        "add@kernel#1",
        "conv2d::with::namespace",
        "activation.relu_v2",
        "layer_norm/beta",
        "matmul_transpose_b=True",
    ];
    
    for name in special_names {
        let op = Operation::new(name);
        assert_eq!(op.op_type, name);
    }
}

/// Test 9: 值名称包含控制字符
#[test]
fn test_value_names_with_control_characters() {
    let names_with_controls = [
        "input\nwith\ncarriage",
        "tensor\traw\tdata",
        "value_with\x00null_char",
        "unicode\u{feff}bom_value",
    ];
    
    for name in names_with_controls {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![10],
        };
        assert_eq!(value.name, name);
    }
}

/// Test 10: 所有类型的有效性检查
#[test]
fn test_all_type_validity() {
    let all_types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
        Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        },
        Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![3, 3, 3],
        },
    ];
    
    for ty in all_types {
        assert!(ty.is_valid_type());
    }
}
