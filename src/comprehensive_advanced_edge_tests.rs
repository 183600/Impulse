//! Comprehensive Advanced Edge Tests - 覆盖更多边界情况的高级测试
//! 使用标准库 assert! 和 assert_eq!，以及 rstest 库
//! 包含数值精度、内存安全、类型转换、极端边界值等测试

use rstest::*;
use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// 测试1: 检查所有类型的有效性验证
#[test]
fn test_all_types_validity() {
    // 测试所有基本类型的有效性
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());

    // 测试嵌套 Tensor 类型的有效性
    let nested_f32 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(nested_f32.is_valid_type());

    let nested_bool = Type::Tensor {
        element_type: Box::new(Type::Bool),
        shape: vec![10],
    };
    assert!(nested_bool.is_valid_type());

    // 测试嵌套 Tensor 的嵌套类型有效性
    let deep_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![3, 3],
        }),
        shape: vec![2, 2],
    };
    assert!(deep_nested.is_valid_type());
}

/// 测试2: 使用 rstest 测试各种边界形状的元素数量计算
#[rstest]
fn test_element_count_calculation(
    #[values(
        vec![],                          // 标量
        vec![1],                         // 1D
        vec![1, 1],                      // 2D 全1
        vec![0],                         // 1D 零
        vec![0, 10],                     // 2D 前零
        vec![10, 0],                     // 2D 后零
        vec![1, 0, 1],                   // 3D 中间零
        vec![2, 3, 4],                   // 正常3D
        vec![100, 100, 10],              // 大型3D
        vec![1, 1, 1, 1, 1, 1, 1, 1]    // 8D 全1
    )] shape: Vec<usize>
) {
    let value = Value {
        name: "test_tensor".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };

    let expected_elements = if shape.is_empty() {
        Some(1)  // 标量
    } else if shape.iter().any(|&dim| dim == 0) {
        Some(0)  // 任何零维度都导致零元素
    } else {
        shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    };

    assert_eq!(value.num_elements(), expected_elements);
}

/// 测试3: 测试浮点数的特殊值（NaN、无穷大、负零）
#[test]
fn test_special_float_values() {
    // 测试正无穷大
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    if let Attribute::Float(val) = pos_inf_attr {
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
    }

    // 测试负无穷大
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    if let Attribute::Float(val) = neg_inf_attr {
        assert!(val.is_infinite());
        assert!(val.is_sign_negative());
    }

    // 测试 NaN
    let nan_attr = Attribute::Float(f64::NAN);
    if let Attribute::Float(val) = nan_attr {
        assert!(val.is_nan());
    }

    // 测试负零
    let neg_zero_attr = Attribute::Float(-0.0);
    if let Attribute::Float(val) = neg_zero_attr {
        assert_eq!(val, 0.0);
        assert!(val.is_sign_negative());
    }

    // 测试最小正值
    let min_positive_attr = Attribute::Float(f64::MIN_POSITIVE);
    if let Attribute::Float(val) = min_positive_attr {
        assert!(val > 0.0);
        assert!(val < 1e-300);
    }
}

/// 测试4: 使用 rstest 测试极端整数值
#[rstest]
fn test_extreme_integer_values(
    #[values(
        i64::MAX,
        i64::MIN,
        0,
        1,
        -1,
        i32::MAX as i64,
        i32::MIN as i64,
        u32::MAX as i64
    )] value: i64
) {
    let attr = Attribute::Int(value);
    if let Attribute::Int(val) = attr {
        assert_eq!(val, value);
    }
}

/// 测试5: 测试 Module 的空操作和边界条件
#[test]
fn test_module_boundary_conditions() {
    // 测试空 Module
    let empty_module = Module::new("empty");
    assert_eq!(empty_module.name, "empty");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.inputs.len(), 0);
    assert_eq!(empty_module.outputs.len(), 0);

    // 测试 Module 名称为空字符串
    let empty_name_module = Module::new("");
    assert_eq!(empty_name_module.name, "");

    // 测试 Module 名称包含特殊字符
    let special_name_module = Module::new("module_with_special_!@#$%^&*()_chars");
    assert_eq!(special_name_module.name, "module_with_special_!@#$%^&*()_chars");

    // 测试 Module 名称包含 Unicode
    let unicode_name_module = Module::new("模块_名称_测试🚀");
    assert_eq!(unicode_name_module.name, "模块_名称_测试🚀");

    // 测试 Module 名称非常长
    let long_name = "x".repeat(10000);
    let long_name_module = Module::new(long_name.clone());
    assert_eq!(long_name_module.name.len(), 10000);
}

/// 测试6: 使用 rstest 测试 Operation 的不同配置
#[rstest]
fn test_operation_configurations(
    #[values("", " ", "add", "matmul", "conv2d", "transpose", "resize", "noop")] op_type: &str,
    #[values(0, 1, 2, 5)] num_inputs: usize,
    #[values(0, 1, 2, 3)] num_outputs: usize
) {
    let mut op = Operation::new(op_type);

    // 添加指定数量的输入
    for i in 0..num_inputs {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
    }

    // 添加指定数量的输出
    for i in 0..num_outputs {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
    }

    assert_eq!(op.op_type, op_type);
    assert_eq!(op.inputs.len(), num_inputs);
    assert_eq!(op.outputs.len(), num_outputs);
}

/// 测试7: 测试 Attribute 的复杂嵌套和边界情况
#[test]
fn test_nested_attributes() {
    // 测试空数组
    let empty_array = Attribute::Array(vec![]);
    if let Attribute::Array(arr) = empty_array {
        assert_eq!(arr.len(), 0);
    }

    // 测试深层嵌套数组
    let deeply_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1)]),
        ]),
    ]);
    if let Attribute::Array(outer) = deeply_nested {
        if let Attribute::Array(inner) = &outer[0] {
            if let Attribute::Array(deepest) = &inner[0] {
                assert_eq!(deepest.len(), 1);
            }
        }
    }

    // 测试混合类型数组
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]);
    if let Attribute::Array(arr) = mixed_array {
        assert_eq!(arr.len(), 5);
    }

    // 测试非常长的字符串属性
    let long_string = "x".repeat(100000);
    let long_string_attr = Attribute::String(long_string);
    if let Attribute::String(s) = long_string_attr {
        assert_eq!(s.len(), 100000);
    }
}

/// 测试8: 测试 Value 名称的特殊字符和边界情况
#[rstest]
fn test_value_name_edge_cases(
    #[values(
        "",
        " ",
        "\t\n",
        "valid_name",
        "name with spaces",
        "name/with/slashes",
        "name\\with\\backslashes",
        "name.with.dots",
        "name-with-dashes",
        "name_with_underscores",
        "123numbers",
        "🚀emoji🎯",
        "中文名称",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  // 100个字符的字符串
        "name\twith\ttabs",
        "name\nwith\nnewlines",
        "name\rwith\rcarriage",
        "name\u{0000}null",  // 包含 null 字符
        "name\u{FFFD}replacement",  // 包含替换字符
        "!@#$%^&*()",
        "<script>alert</script>",
        "CON",  // Windows 保留名称
        "AUX",  // Windows 保留名称
        "PRN",  // Windows 保留名称
        "NUL"   // Windows 保留名称
    )] name: &str
) {
    let value = Value {
        name: name.to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.name, name);
}

/// 测试9: 测试 Module 包含大量操作的性能边界
#[test]
fn test_module_with_many_operations() {
    let mut module = Module::new("many_ops");

    // 添加大量操作
    for i in 0..10000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 10000);

    // 验证第一个和最后一个操作
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[9999].op_type, "op_9999");
}

/// 测试10: 使用 rstest 测试不同类型组合的 Module
#[rstest]
fn test_module_with_mixed_types(
    #[values(Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool)] input_type: Type,
    #[values(Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool)] output_type: Type
) {
    let mut module = Module::new("mixed_types");

    // 添加不同类型的输入
    module.inputs.push(Value {
        name: format!("input_{:?}", input_type),
        ty: input_type.clone(),
        shape: vec![10],
    });

    // 添加不同类型的输出
    module.outputs.push(Value {
        name: format!("output_{:?}", output_type),
        ty: output_type.clone(),
        shape: vec![10],
    });

    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].ty, input_type);
    assert_eq!(module.outputs[0].ty, output_type);
}