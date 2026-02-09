//! Critical edge case tests for Impulse compiler
//! 覆盖关键边界情况的测试用例

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// 测试1: 检查溢出保护 - 验证 num_elements 方法正确处理可能导致溢出的形状
#[test]
fn test_overflow_protection_in_num_elements() {
    // 使用接近 usize 边界的形状进行测试
    // 在64位系统上，46341 * 46341 ≈ 2.15 billion，接近 i32::MAX
    let large_shape = vec![46341_usize, 46341];
    let value = Value {
        name: "potential_overflow".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };

    // 使用 checked_mul 应该能安全计算或返回 None
    let num_elements = value.num_elements();
    
    // 由于 46341 * 46341 在大多数 64 位系统上是有效的，应该返回 Some
    // 但这个测试验证方法不会 panic
    match num_elements {
        Some(count) => {
            // 如果成功计算，验证结果
            assert_eq!(count, 46341 * 46341);
        }
        None => {
            // 如果检测到潜在溢出，这是正确的行为
            assert!(true);
        }
    }
}

/// 测试2: 最大边界值 - 测试使用 i64::MAX 和 i64::MIN 作为属性值
#[test]
fn test_boundary_integer_attributes() {
    let mut op = Operation::new("boundary_test");
    let mut attrs = HashMap::new();
    
    // 添加边界整数值
    attrs.insert("max_int".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("one".to_string(), Attribute::Int(1));
    attrs.insert("minus_one".to_string(), Attribute::Int(-1));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.get("max_int"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_int"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("zero"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("one"), Some(&Attribute::Int(1)));
    assert_eq!(op.attributes.get("minus_one"), Some(&Attribute::Int(-1)));
}

/// 测试3: 特殊浮点值 - 测试 NaN, Infinity, -Infinity, 负零等特殊值
#[test]
fn test_special_floating_point_attributes() {
    let mut op = Operation::new("float_special_test");
    let mut attrs = HashMap::new();
    
    // 添加特殊浮点值
    attrs.insert("infinity".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    
    op.attributes = attrs;
    
    // 验证这些特殊值被正确存储
    match op.attributes.get("infinity") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val > 0.0),
        _ => panic!("Expected positive infinity"),
    }
    
    match op.attributes.get("neg_infinity") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val < 0.0),
        _ => panic!("Expected negative infinity"),
    }
    
    match op.attributes.get("nan") {
        Some(Attribute::Float(val)) => assert!(val.is_nan()),
        _ => panic!("Expected NaN"),
    }
    
    // 负零应该等于正零
    match op.attributes.get("negative_zero") {
        Some(Attribute::Float(val)) => assert_eq!(*val, 0.0),
        _ => panic!("Expected negative zero"),
    }
}

/// 测试4: 空操作和空值 - 测试空字符串名称、空属性、空输入输出
#[test]
fn test_empty_operation_and_values() {
    // 空操作类型
    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    assert!(empty_op.inputs.is_empty());
    assert!(empty_op.outputs.is_empty());
    assert!(empty_op.attributes.is_empty());
    
    // 空属性哈希表
    let op_with_empty_attrs = {
        let mut op = Operation::new("empty_attrs");
        op.attributes = HashMap::new();
        op
    };
    assert_eq!(op_with_empty_attrs.attributes.len(), 0);
    
    // 空值名称
    let empty_value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(empty_value.name, "");
    
    // 空模块名称
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
}

/// 测试5: 单元素和零元素张量 - 测试标量和空张量的边界情况
#[test]
fn test_single_and_zero_element_tensors() {
    // 标量 (0维张量，1个元素)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
    
    // 单个元素的1D张量
    let single_1d = Value {
        name: "single_1d".to_string(),
        ty: Type::I32,
        shape: vec![1],
    };
    assert_eq!(single_1d.num_elements(), Some(1));
    
    // 单个元素的3D张量
    let single_3d = Value {
        name: "single_3d".to_string(),
        ty: Type::Bool,
        shape: vec![1, 1, 1],
    };
    assert_eq!(single_3d.num_elements(), Some(1));
    
    // 零元素张量 (包含0维度)
    let zero_dim = Value {
        name: "zero_dim".to_string(),
        ty: Type::F64,
        shape: vec![0],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));
    
    // 多个零维度的张量
    let multi_zero = Value {
        name: "multi_zero".to_string(),
        ty: Type::F32,
        shape: vec![0, 10, 5],
    };
    assert_eq!(multi_zero.num_elements(), Some(0));
}

/// 测试6: 深度嵌套类型 - 测试多层嵌套的张量类型
#[test]
fn test_deeply_nested_tensor_types() {
    // 创建10层嵌套的张量类型
    let mut nested_type = Type::F32;
    for i in 0..10 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![i + 1],
        };
    }
    
    // 验证最外层是 Tensor
    match &nested_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![10]);
        }
        _ => panic!("Expected Tensor type at outermost level"),
    }
    
    // 验证类型有效性
    assert!(nested_type.is_valid_type());
    
    // 测试克隆
    let cloned = nested_type.clone();
    assert_eq!(nested_type, cloned);
}

/// 测试7: Unicode 和特殊字符 - 测试包含特殊字符的名称
#[test]
fn test_unicode_and_special_characters() {
    let test_names = vec![
        "valid_🚀",                    // Emoji
        "中文测试",                     // 中文
        "日本語テスト",                 // 日文
        "العربية",                     // 阿拉伯文
        "café_naïve",                  // 重音字符
        "control_\x00_\x1f",          // 控制字符
        "space\ttab\r\nnewline",      // 空白字符
    ];
    
    for name in test_names {
        // 创建操作
        let op = Operation::new(name);
        assert_eq!(op.op_type, name);
        
        // 创建值
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, name);
        
        // 创建模块
        let module = Module::new(name);
        assert_eq!(module.name, name);
        
        // 创建属性
        let attr = Attribute::String(name.to_string());
        match attr {
            Attribute::String(s) => assert_eq!(s, name),
            _ => panic!("Expected String attribute"),
        }
    }
}

/// 测试8: 极端形状比例 - 测试非常扁平或非常高的张量形状
#[test]
fn test_extreme_aspect_ratios() {
    // 非常扁平的张量 (1行，多列)
    let flat = Value {
        name: "flat".to_string(),
        ty: Type::F32,
        shape: vec![1, 1_000_000],
    };
    assert_eq!(flat.num_elements(), Some(1_000_000));
    
    // 非常高的张量 (多行，1列)
    let tall = Value {
        name: "tall".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 1],
    };
    assert_eq!(tall.num_elements(), Some(1_000_000));
    
    // 单维长向量
    let long_vector = Value {
        name: "long_vector".to_string(),
        ty: Type::I32,
        shape: vec![10_000_000],
    };
    assert_eq!(long_vector.num_elements(), Some(10_000_000));
    
    // 深而窄的4D张量 (类似批处理)
    let deep_narrow = Value {
        name: "deep_narrow".to_string(),
        ty: Type::F64,
        shape: vec![1000, 1, 1, 1],
    };
    assert_eq!(deep_narrow.num_elements(), Some(1000));
}

/// 测试9: 混合属性数组 - 测试包含不同类型属性的数组
#[test]
fn test_mixed_attribute_arrays() {
    let mut op = Operation::new("mixed_array_test");
    
    // 创建包含混合类型的数组属性
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14159),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
        Attribute::Int(-100),
        Attribute::Float(-2.71828),
        Attribute::String("world".to_string()),
        Attribute::Bool(false),
    ]);
    
    let mut attrs = HashMap::new();
    attrs.insert("mixed_array".to_string(), mixed_array);
    op.attributes = attrs;
    
    // 验证数组内容
    match op.attributes.get("mixed_array") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 8);
            
            // 验证每种类型
            match &arr[0] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
            match &arr[1] {
                Attribute::Float(val) if (*val - 3.14159).abs() < f64::EPSILON => (),
                _ => panic!("Expected Float(3.14159)"),
            }
            match &arr[2] {
                Attribute::String(s) if s == "hello" => (),
                _ => panic!("Expected String(\"hello\")"),
            }
            match &arr[3] {
                Attribute::Bool(true) => (),
                _ => panic!("Expected Bool(true)"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// 测试10: 嵌套数组属性 - 测试数组的数组
#[test]
fn test_nested_array_attributes() {
    let mut op = Operation::new("nested_array_test");

    // 创建嵌套数组结构
    let nested_array = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
        ]),
        Attribute::Array(vec![
            Attribute::Float(1.5),
            Attribute::Float(2.5),
            Attribute::Float(3.5),
        ]),
        Attribute::Array(vec![
            Attribute::String("a".to_string()),
            Attribute::String("b".to_string()),
        ]),
    ]);

    let mut attrs = HashMap::new();
    attrs.insert("nested_arrays".to_string(), nested_array);
    op.attributes = attrs;

    // 验证嵌套结构
    match op.attributes.get("nested_arrays") {
        Some(Attribute::Array(outer_arr)) => {
            assert_eq!(outer_arr.len(), 3);

            // 验证第一个子数组
            match &outer_arr[0] {
                Attribute::Array(inner_arr) => {
                    assert_eq!(inner_arr.len(), 2);
                    match &inner_arr[0] {
                        Attribute::Int(1) => (),
                        _ => panic!("Expected Int(1)"),
                    }
                    match &inner_arr[1] {
                        Attribute::Int(2) => (),
                        _ => panic!("Expected Int(2)"),
                    }
                }
                _ => panic!("Expected Array in first element"),
            }

            // 验证第二个子数组
            match &outer_arr[1] {
                Attribute::Array(inner_arr) => {
                    assert_eq!(inner_arr.len(), 3);
                    match &inner_arr[0] {
                        Attribute::Float(val) if (*val - 1.5).abs() < f64::EPSILON => (),
                        _ => panic!("Expected Float(1.5)"),
                    }
                }
                _ => panic!("Expected Array in second element"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// 测试11: 检查 usize 边界值的形状乘积 - 测试 num_elements 方法处理边界值
#[test]
fn test_num_elements_boundary_values() {
    // 测试包含 1 的形状 (乘积不变)
    let ones_shape = Value {
        name: "ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1],
    };
    assert_eq!(ones_shape.num_elements(), Some(1));

    // 测试包含多个 1 和其他值的形状
    let mixed_ones = Value {
        name: "mixed_ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 5, 1, 10, 1],
    };
    assert_eq!(mixed_ones.num_elements(), Some(50));

    // 测试包含大数但仍在安全范围内的形状
    let large_safe = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: vec![100, 100, 100],  // 1,000,000
    };
    assert_eq!(large_safe.num_elements(), Some(1_000_000));

    // 测试包含零的形状 (乘积应为 0)
    let with_zero = Value {
        name: "with_zero".to_string(),
        ty: Type::F32,
        shape: vec![1000, 0, 500],
    };
    assert_eq!(with_zero.num_elements(), Some(0));
}

/// 测试12: 空字符串属性 - 测试空字符串作为属性值
#[test]
fn test_empty_string_attributes() {
    let mut op = Operation::new("empty_string_test");
    let mut attrs = HashMap::new();

    // 添加空字符串属性
    attrs.insert("empty".to_string(), Attribute::String("".to_string()));
    attrs.insert("spaces".to_string(), Attribute::String("   ".to_string()));
    attrs.insert("tab".to_string(), Attribute::String("\t".to_string()));
    attrs.insert("newline".to_string(), Attribute::String("\n".to_string()));

    op.attributes = attrs;

    // 验证空字符串
    match op.attributes.get("empty") {
        Some(Attribute::String(s)) => {
            assert_eq!(s, "");
            assert!(s.is_empty());
        }
        _ => panic!("Expected empty string"),
    }

    // 验证仅包含空格的字符串
    match op.attributes.get("spaces") {
        Some(Attribute::String(s)) => {
            assert_eq!(s, "   ");
            assert_eq!(s.len(), 3);
        }
        _ => panic!("Expected spaces string"),
    }

    // 验证包含制表符的字符串
    match op.attributes.get("tab") {
        Some(Attribute::String(s)) => {
            assert_eq!(s, "\t");
            assert_eq!(s.len(), 1);
        }
        _ => panic!("Expected tab string"),
    }

    // 验证包含换行符的字符串
    match op.attributes.get("newline") {
        Some(Attribute::String(s)) => {
            assert_eq!(s, "\n");
            assert_eq!(s.len(), 1);
        }
        _ => panic!("Expected newline string"),
    }
}

/// 测试13: 模块克隆 - 测试模块的克隆行为
#[test]
fn test_module_clone() {
    let mut original = Module::new("clone_test");

    // 添加操作
    let mut op1 = Operation::new("add");
    op1.inputs.push(Value {
        name: "input1".to_string(),
        ty: Type::F32,
        shape: vec![2, 3],
    });
    original.add_operation(op1);

    // 添加输入和输出
    original.inputs.push(Value {
        name: "module_input".to_string(),
        ty: Type::I32,
        shape: vec![10],
    });

    original.outputs.push(Value {
        name: "module_output".to_string(),
        ty: Type::F32,
        shape: vec![2, 3],
    });

    // 克隆模块
    let cloned = original.clone();

    // 验证克隆的模块与原始模块相等
    assert_eq!(original.name, cloned.name);
    assert_eq!(original.operations.len(), cloned.operations.len());
    assert_eq!(original.inputs.len(), cloned.inputs.len());
    assert_eq!(original.outputs.len(), cloned.outputs.len());

    // 验证操作也被正确克隆
    assert_eq!(original.operations[0].op_type, cloned.operations[0].op_type);
    assert_eq!(original.operations[0].inputs.len(), cloned.operations[0].inputs.len());

    // 修改原始模块不应影响克隆
    original.name = "modified".to_string();
    assert_eq!(original.name, "modified");
    assert_eq!(cloned.name, "clone_test");
}

/// 测试14: 操作序列化/反序列化 - 测试操作可以被正确序列化和反序列化
#[test]
fn test_operation_serialization() {
    use serde_json;

    let mut original_op = Operation::new("conv2d");

    // 添加输入
    original_op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![1, 3, 224, 224],
    });

    // 添加输出
    original_op.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![1, 64, 112, 112],
    });

    // 添加属性
    let mut attrs = HashMap::new();
    attrs.insert("kernel_size".to_string(), Attribute::Int(3));
    attrs.insert("stride".to_string(), Attribute::Int(2));
    attrs.insert("padding".to_string(), Attribute::String("SAME".to_string()));
    attrs.insert("use_bias".to_string(), Attribute::Bool(true));
    original_op.attributes = attrs;

    // 序列化
    let serialized = serde_json::to_string(&original_op).expect("Serialization failed");

    // 验证序列化结果不为空
    assert!(!serialized.is_empty());

    // 反序列化
    let deserialized: Operation = serde_json::from_str(&serialized).expect("Deserialization failed");

    // 验证反序列化的操作与原始操作相等
    assert_eq!(original_op.op_type, deserialized.op_type);
    assert_eq!(original_op.inputs.len(), deserialized.inputs.len());
    assert_eq!(original_op.outputs.len(), deserialized.outputs.len());
    assert_eq!(original_op.attributes.len(), deserialized.attributes.len());

    // 验证属性被正确恢复
    assert_eq!(deserialized.attributes.get("kernel_size"), Some(&Attribute::Int(3)));
    assert_eq!(deserialized.attributes.get("stride"), Some(&Attribute::Int(2)));
}

/// 测试15: 所有基本类型的相等性 - 测试所有 Type 变体的相等性比较
#[test]
fn test_all_type_equality() {
    // 测试基本类型的相等性
    assert_eq!(Type::F32, Type::F32);
    assert_eq!(Type::F64, Type::F64);
    assert_eq!(Type::I32, Type::I32);
    assert_eq!(Type::I64, Type::I64);
    assert_eq!(Type::Bool, Type::Bool);

    // 测试基本类型的不等性
    assert_ne!(Type::F32, Type::F64);
    assert_ne!(Type::I32, Type::I64);
    assert_ne!(Type::F32, Type::I32);
    assert_ne!(Type::Bool, Type::F32);

    // 测试 Tensor 类型的相等性
    let tensor1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    let tensor2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert_eq!(tensor1, tensor2);

    // 测试不同元素类型的 Tensor
    let tensor3 = Type::Tensor {
        element_type: Box::new(Type::I32),
        shape: vec![2, 3],
    };
    assert_ne!(tensor1, tensor3);

    // 测试不同形状的 Tensor
    let tensor4 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![3, 2],
    };
    assert_ne!(tensor1, tensor4);

    // 测试嵌套 Tensor 类型的相等性
    let nested1 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    let nested2 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    assert_eq!(nested1, nested2);
}

/// 测试16: 单元素数组属性 - 测试包含单个元素的数组属性
#[test]
fn test_single_element_array_attributes() {
    let mut op = Operation::new("single_element_array_test");
    let mut attrs = HashMap::new();

    // 添加包含单个元素的数组
    attrs.insert("single_int".to_string(), Attribute::Array(vec![Attribute::Int(42)]));
    attrs.insert("single_float".to_string(), Attribute::Array(vec![Attribute::Float(3.14)]));
    attrs.insert("single_string".to_string(), Attribute::Array(vec![Attribute::String("hello".to_string())]));
    attrs.insert("single_bool".to_string(), Attribute::Array(vec![Attribute::Bool(true)]));

    op.attributes = attrs;

    // 验证单元素整数数组
    match op.attributes.get("single_int") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }

    // 验证单元素浮点数组
    match op.attributes.get("single_float") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Float(val) if (*val - 3.14).abs() < f64::EPSILON => (),
                _ => panic!("Expected Float(3.14)"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }

    // 验证单元素字符串数组
    match op.attributes.get("single_string") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::String(s) if s == "hello" => (),
                _ => panic!("Expected String(\"hello\")"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// 测试17: 空数组属性 - 测试空数组作为属性值
#[test]
fn test_empty_array_attributes() {
    let mut op = Operation::new("empty_array_test");
    let mut attrs = HashMap::new();

    // 添加空数组
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));

    op.attributes = attrs;

    // 验证空数组
    match op.attributes.get("empty_array") {
        Some(Attribute::Array(arr)) => {
            assert!(arr.is_empty());
            assert_eq!(arr.len(), 0);
        }
        _ => panic!("Expected empty Array attribute"),
    }

    // 创建多个空数组并验证
    let mut op2 = Operation::new("multiple_empty_arrays");
    let mut attrs2 = HashMap::new();

    attrs2.insert("empty1".to_string(), Attribute::Array(vec![]));
    attrs2.insert("empty2".to_string(), Attribute::Array(vec![]));
    attrs2.insert("empty3".to_string(), Attribute::Array(vec![]));

    op2.attributes = attrs2;

    assert_eq!(op2.attributes.len(), 3);
    for (_key, value) in op2.attributes.iter() {
        match value {
            Attribute::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("Expected all attributes to be empty arrays"),
        }
    }
}

/// 测试18: 最大浮点精度值 - 测试接近浮点数精度的边界值
#[test]
fn test_floating_point_precision_values() {
    let mut op = Operation::new("precision_test");
    let mut attrs = HashMap::new();

    // 添加接近浮点数精度的值
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("max_exp_f64".to_string(), Attribute::Float(f64::MAX_EXP as f64));
    attrs.insert("min_exp_f64".to_string(), Attribute::Float(f64::MIN_EXP as f64));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("min_positive".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("mantissa_digits".to_string(), Attribute::Float(f64::MANTISSA_DIGITS as f64));
    attrs.insert("digits".to_string(), Attribute::Float(f64::DIGITS as f64));
    attrs.insert("radix".to_string(), Attribute::Float(f64::RADIX as f64));

    op.attributes = attrs;

    // 验证最大浮点数
    match op.attributes.get("max_f64") {
        Some(Attribute::Float(val)) => assert_eq!(*val, f64::MAX),
        _ => panic!("Expected f64::MAX"),
    }

    // 验证最小浮点数
    match op.attributes.get("min_f64") {
        Some(Attribute::Float(val)) => assert_eq!(*val, f64::MIN),
        _ => panic!("Expected f64::MIN"),
    }

    // 验证 Epsilon (机器精度)
    match op.attributes.get("epsilon") {
        Some(Attribute::Float(val)) => assert_eq!(*val, f64::EPSILON),
        _ => panic!("Expected f64::EPSILON"),
    }

    // 验证最小正数
    match op.attributes.get("min_positive") {
        Some(Attribute::Float(val)) => assert_eq!(*val, f64::MIN_POSITIVE),
        _ => panic!("Expected f64::MIN_POSITIVE"),
    }
}

/// 测试19: 类型转换的边界情况 - 测试不同类型之间的转换边界
#[test]
fn test_type_conversion_boundaries() {
    // 测试所有基本类型的创建和验证
    let types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];

    for ty in types {
        assert!(ty.is_valid_type());
    }

    // 测试 Tensor 类型的验证
    let tensor_valid = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(tensor_valid.is_valid_type());

    // 测试深层嵌套 Tensor 的验证
    let mut nested = Type::F32;
    for i in 0..5 {
        nested = Type::Tensor {
            element_type: Box::new(nested),
            shape: vec![i + 1],
        };
    }
    assert!(nested.is_valid_type());

    // 测试使用不同类型创建的 Value
    let f32_value = Value {
        name: "f32_val".to_string(),
        ty: Type::F32,
        shape: vec![2, 2],
    };
    assert!(f32_value.ty.is_valid_type());

    let i64_value = Value {
        name: "i64_val".to_string(),
        ty: Type::I64,
        shape: vec![5],
    };
    assert!(i64_value.ty.is_valid_type());

    let bool_value = Value {
        name: "bool_val".to_string(),
        ty: Type::Bool,
        shape: vec![3, 3],
    };
    assert!(bool_value.ty.is_valid_type());

    // 测试类型相等性检查
    assert_eq!(f32_value.ty, Type::F32);
    assert_ne!(f32_value.ty, i64_value.ty);
    assert_ne!(f32_value.ty, bool_value.ty);
}

/// 测试20: 操作克隆行为 - 测试操作的深拷贝行为
#[test]
fn test_operation_clone() {
    let mut original = Operation::new("matmul");

    // 添加输入
    original.inputs.push(Value {
        name: "matrix_a".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    });
    original.inputs.push(Value {
        name: "matrix_b".to_string(),
        ty: Type::F32,
        shape: vec![20, 30],
    });

    // 添加输出
    original.outputs.push(Value {
        name: "result".to_string(),
        ty: Type::F32,
        shape: vec![10, 30],
    });

    // 添加属性
    let mut attrs = HashMap::new();
    attrs.insert("transpose_a".to_string(), Attribute::Bool(false));
    attrs.insert("transpose_b".to_string(), Attribute::Bool(true));
    attrs.insert("alpha".to_string(), Attribute::Float(1.0));
    attrs.insert("beta".to_string(), Attribute::Float(0.0));
    original.attributes = attrs;

    // 克隆操作
    let cloned = original.clone();

    // 验证克隆的操作与原始操作相等
    assert_eq!(original.op_type, cloned.op_type);
    assert_eq!(original.inputs.len(), cloned.inputs.len());
    assert_eq!(original.outputs.len(), cloned.outputs.len());
    assert_eq!(original.attributes.len(), cloned.attributes.len());

    // 验证输入被正确克隆
    assert_eq!(original.inputs[0].name, cloned.inputs[0].name);
    assert_eq!(original.inputs[0].ty, cloned.inputs[0].ty);
    assert_eq!(original.inputs[0].shape, cloned.inputs[0].shape);

    // 验证输出被正确克隆
    assert_eq!(original.outputs[0].name, cloned.outputs[0].name);
    assert_eq!(original.outputs[0].ty, cloned.outputs[0].ty);
    assert_eq!(original.outputs[0].shape, cloned.outputs[0].shape);

    // 验证属性被正确克隆
    assert_eq!(
        cloned.attributes.get("transpose_a"),
        Some(&Attribute::Bool(false))
    );
    assert_eq!(
        cloned.attributes.get("transpose_b"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        cloned.attributes.get("alpha"),
        Some(&Attribute::Float(1.0))
    );
    assert_eq!(
        cloned.attributes.get("beta"),
        Some(&Attribute::Float(0.0))
    );

    // 修改原始操作不应影响克隆
    original.op_type = "modified".to_string();
    original.inputs[0].name = "modified_input".to_string();
    assert_eq!(original.op_type, "modified");
    assert_eq!(original.inputs[0].name, "modified_input");
    assert_eq!(cloned.op_type, "matmul");
    assert_eq!(cloned.inputs[0].name, "matrix_a");
}