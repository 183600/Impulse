//! 边界情况和极端场景测试
//! 使用标准库的 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// 测试1: 检查空操作列表的模块
#[test]
fn test_module_with_empty_operations() {
    let module = Module::new("empty_ops_module");
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
}

/// 测试2: 值形状中包含零维度的情况
#[test]
fn test_value_with_zero_dimension() {
    let value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(value.num_elements(), Some(0));
}

/// 测试3: 检查极大整数值的属性
#[test]
fn test_attribute_with_max_min_int() {
    let max_attr = Attribute::Int(i64::MAX);
    let min_attr = Attribute::Int(i64::MIN);
    
    assert_eq!(max_attr, Attribute::Int(i64::MAX));
    assert_eq!(min_attr, Attribute::Int(i64::MIN));
}

/// 测试4: 标量值（空形状）
#[test]
fn test_scalar_value_empty_shape() {
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::I32,
        shape: vec![],
    };
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(scalar.num_elements(), Some(1));
}

/// 测试5: 浮点数边界值属性
#[test]
fn test_float_boundary_attributes() {
    let max_f64 = Attribute::Float(f64::MAX);
    let min_f64 = Attribute::Float(f64::MIN);
    let nan_attr = Attribute::Float(f64::NAN);
    let inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    // 验证边界值属性创建成功
    match max_f64 {
        Attribute::Float(v) => assert!(v.is_finite()),
        _ => panic!("Expected Float attribute"),
    }
    
    match min_f64 {
        Attribute::Float(v) => assert!(v.is_finite()),
        _ => panic!("Expected Float attribute"),
    }
    
    match nan_attr {
        Attribute::Float(v) => assert!(v.is_nan()),
        _ => panic!("Expected Float attribute"),
    }
    
    match inf_attr {
        Attribute::Float(v) => assert!(v.is_infinite() && v.is_sign_positive()),
        _ => panic!("Expected Float attribute"),
    }
    
    match neg_inf_attr {
        Attribute::Float(v) => assert!(v.is_infinite() && v.is_sign_negative()),
        _ => panic!("Expected Float attribute"),
    }
}

/// 测试6: 空字符串属性
#[test]
fn test_empty_string_attribute() {
    let empty = Attribute::String(String::new());
    match empty {
        Attribute::String(s) => assert_eq!(s.len(), 0),
        _ => panic!("Expected String attribute"),
    }
}

/// 测试7: 空数组属性
#[test]
fn test_empty_array_attribute() {
    let empty = Attribute::Array(vec![]);
    match empty {
        Attribute::Array(arr) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected Array attribute"),
    }
}

/// 测试8: 深度嵌套的 Tensor 类型
#[test]
fn test_deeply_nested_tensor_type() {
    let tensor = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3, 4],
    };
    
    assert!(tensor.is_valid_type());
}

/// 测试9: 编译器处理空字节数组
#[test]
fn test_compiler_with_empty_byte_array() {
    let mut compiler = ImpulseCompiler::new();
    let empty_model: Vec<u8> = vec![];
    
    let result = compiler.compile(&empty_model, "cpu");
    // 应该处理而不panic，结果是 Ok 或 Err 都可以
    assert!(result.is_ok() || result.is_err());
}

/// 测试10: 操作包含所有布尔值属性
#[test]
fn test_operation_with_boolean_attributes() {
    let mut op = Operation::new("bool_test");
    let mut attrs = HashMap::new();
    
    attrs.insert("flag_true".to_string(), Attribute::Bool(true));
    attrs.insert("flag_false".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 2);
    assert_eq!(op.attributes.get("flag_true"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("flag_false"), Some(&Attribute::Bool(false)));
}