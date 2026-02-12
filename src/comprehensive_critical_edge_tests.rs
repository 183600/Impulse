//! 综合关键边界测试 - 覆盖尚未充分测试的边界情况
//! 使用标准库 assert! 和 assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::utils::{math_utils, validation_utils};

/// 测试1: 使用最大安全维度计算内存大小
#[test]
fn test_maximum_safe_dimensions_memory_calculation() {
    // 测试接近 usize::MAX 的维度（但不会溢出）
    let safe_value = Value {
        name: "max_safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000_000], // 10亿元素
    };

    // num_elements 应该返回正确的值
    assert_eq!(safe_value.num_elements(), Some(1_000_000_000));

    // 计算字节大小（假设 f32 是 4 字节）
    if let Some(elements) = safe_value.num_elements() {
        // 使用 checked_mul 防止溢出
        let bytes = elements.checked_mul(std::mem::size_of::<f32>());
        assert!(bytes.is_some());
        assert_eq!(bytes, Some(4_000_000_000));
    }
}

/// 测试2: 测试 round_up_to_multiple 在边界条件下的行为
#[test]
fn test_round_up_to_multiple_boundaries() {
    // 零倍数情况
    assert_eq!(math_utils::round_up_to_multiple(10, 0), 10);

    // 值已经是倍数
    assert_eq!(math_utils::round_up_to_multiple(16, 16), 16);
    assert_eq!(math_utils::round_up_to_multiple(1024, 1024), 1024);

    // 值小于倍数
    assert_eq!(math_utils::round_up_to_multiple(1, 1024), 1024);
    assert_eq!(math_utils::round_up_to_multiple(16, 32), 32);

    // 值刚刚超过倍数
    assert_eq!(math_utils::round_up_to_multiple(17, 16), 32);
    assert_eq!(math_utils::round_up_to_multiple(1025, 1024), 2048);

    // 大值情况
    assert_eq!(math_utils::round_up_to_multiple(1_000_000, 1_048_576), 1_048_576);
}

/// 测试3: 测试验证函数处理边界条件
#[test]
fn test_validation_with_edge_cases() {
    // 测试标量（空形状）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert!(validation_utils::validate_value_shape(&scalar).is_ok());

    // 测试零维张量
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![0, 10, 10],
    };
    assert!(validation_utils::validate_value_shape(&zero_tensor).is_ok());

    // 测试所有维度都是 1
    let ones_tensor = Value {
        name: "ones_tensor".to_string(),
        ty: Type::I32,
        shape: vec![1, 1, 1, 1],
    };
    assert!(validation_utils::validate_value_shape(&ones_tensor).is_ok());

    // 测试混合零和正维度
    let mixed_tensor = Value {
        name: "mixed_tensor".to_string(),
        ty: Type::F64,
        shape: vec![10, 0, 20, 0],
    };
    assert!(validation_utils::validate_value_shape(&mixed_tensor).is_ok());
}

/// 测试4: 测试嵌套 Tensor 类型的深度边界
#[test]
fn test_nested_tensor_depth_boundaries() {
    // 单层嵌套
    let nested_1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 2],
    };
    assert!(nested_1.is_valid_type());

    // 嵌套包含基础类型
    let nested_with_int = Type::Tensor {
        element_type: Box::new(Type::I64),
        shape: vec![100],
    };
    assert!(nested_with_int.is_valid_type());

    // 验证不同元素类型
    assert!(Type::Tensor {
        element_type: Box::new(Type::Bool),
        shape: vec![5],
    }.is_valid_type());

    assert!(Type::Tensor {
        element_type: Box::new(Type::F64),
        shape: vec![3, 3],
    }.is_valid_type());
}

/// 测试5: 测试操作属性的特殊值
#[test]
fn test_operation_attributes_special_values() {
    let mut op = Operation::new("special_attrs");

    // 添加特殊数值属性
    op.attributes.insert("zero_int".to_string(), Attribute::Int(0));
    op.attributes.insert("min_int".to_string(), Attribute::Int(-1));
    op.attributes.insert("max_int".to_string(), Attribute::Int(999999999));
    op.attributes.insert("zero_float".to_string(), Attribute::Float(0.0));
    op.attributes.insert("negative_float".to_string(), Attribute::Float(-3.14159));

    // 验证属性存在且正确
    assert_eq!(op.attributes.len(), 5);

    if let Some(Attribute::Int(val)) = op.attributes.get("zero_int") {
        assert_eq!(*val, 0);
    } else {
        panic!("Expected zero_int to be Int(0)");
    }

    if let Some(Attribute::Float(val)) = op.attributes.get("negative_float") {
        assert!(*val < 0.0);
    } else {
        panic!("Expected negative_float to be negative");
    }
}

/// 测试6: 测试模块在极端输入/输出数量下的行为
#[test]
fn test_module_extreme_io_counts() {
    let mut module = Module::new("extreme_io_module");

    // 添加大量输入
    for i in 0..100 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }

    // 添加大量输出
    for i in 0..50 {
        module.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::I32,
            shape: vec![1],
        });
    }

    assert_eq!(module.inputs.len(), 100);
    assert_eq!(module.outputs.len(), 50);
    assert_eq!(module.operations.len(), 0);
}

/// 测试7: 测试 gcd 函数的边界条件
#[test]
fn test_gcd_boundary_conditions() {
    // 互质数
    assert_eq!(math_utils::gcd(17, 19), 1);

    // 一个是另一个的倍数
    assert_eq!(math_utils::gcd(100, 25), 25);
    assert_eq!(math_utils::gcd(25, 100), 25);

    // 相同的数
    assert_eq!(math_utils::gcd(42, 42), 42);

    // 1 和任意数
    assert_eq!(math_utils::gcd(1, 1000000), 1);
    assert_eq!(math_utils::gcd(1000000, 1), 1);

    // 大数的 gcd
    assert_eq!(math_utils::gcd(1000000, 500000), 500000);
}

/// 测试8: 测试 lcm 函数的边界条件
#[test]
fn test_lcm_boundary_conditions() {
    // 互质数
    assert_eq!(math_utils::lcm(17, 19), 323);

    // 一个是另一个的倍数
    assert_eq!(math_utils::lcm(100, 25), 100);
    assert_eq!(math_utils::lcm(25, 100), 100);

    // 相同的数
    assert_eq!(math_utils::lcm(42, 42), 42);

    // 1 和任意数
    assert_eq!(math_utils::lcm(1, 1000000), 1000000);
    assert_eq!(math_utils::lcm(1000000, 1), 1000000);

    // 0 和任意数
    assert_eq!(math_utils::lcm(0, 100), 0);
    assert_eq!(math_utils::lcm(100, 0), 0);
}

/// 测试9: 测试模块名称的边界条件
#[test]
fn test_module_name_boundaries() {
    // 空名称（应该被验证拒绝）
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");

    // 非常长的名称
    let long_name = "x".repeat(1000);
    let long_module = Module::new(&long_name);
    assert_eq!(long_module.name.len(), 1000);

    // 包含特殊字符的名称
    let special_module = Module::new("module_with-special_chars@2024");
    assert_eq!(special_module.name, "module_with-special_chars@2024");

    // 包含 Unicode 的名称
    let unicode_module = Module::new("模块_测试_2024");
    assert_eq!(unicode_module.name, "模块_测试_2024");
}

/// 测试10: 测试值在形状变化时的 num_elements 行为
#[test]
fn test_value_num_elements_with_shape_variations() {
    // 空形状（标量）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));

    // 单维度
    let one_d = Value {
        name: "one_d".to_string(),
        ty: Type::F32,
        shape: vec![42],
    };
    assert_eq!(one_d.num_elements(), Some(42));

    // 多维度
    let multi_d = Value {
        name: "multi_d".to_string(),
        ty: Type::F32,
        shape: vec![2, 3, 4],
    };
    assert_eq!(multi_d.num_elements(), Some(24));

    // 包含零的形状
    let with_zero = Value {
        name: "with_zero".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(with_zero.num_elements(), Some(0));

    // 包含 1 的形状
    let with_ones = Value {
        name: "with_ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(with_ones.num_elements(), Some(1));
}