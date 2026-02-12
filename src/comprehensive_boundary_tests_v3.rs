//! Comprehensive boundary tests v3 - 覆盖更多边界情况，使用标准库 assert! 和 assert_eq!
//! 
//! 10个测试用例，覆盖数值精度、内存安全、类型转换、极端值等边界情况

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: 测试零尺寸张量（包含零维度的形状）
#[test]
fn test_zero_dimension_tensors() {
    // 测试形状中包含零的情况
    let zero_dim_values = vec![
        Value {
            name: "zero_first_dim".to_string(),
            ty: Type::F32,
            shape: vec![0, 10],
        },
        Value {
            name: "zero_middle_dim".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 10],
        },
        Value {
            name: "zero_last_dim".to_string(),
            ty: Type::F32,
            shape: vec![10, 0],
        },
        Value {
            name: "all_zero".to_string(),
            ty: Type::F32,
            shape: vec![0, 0, 0],
        },
    ];

    // 所有包含零维度的张量应该有0个元素
    for value in zero_dim_values {
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 2: 测试超大维度的形状（但不会溢出）
#[test]
fn test_large_dimensions_without_overflow() {
    // 测试不会溢出的大尺寸形状
    let large_values = vec![
        Value {
            name: "large_1d".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX / 2], // 不会溢出
        },
        Value {
            name: "large_2d".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 10_000], // 10亿元素
        },
        Value {
            name: "large_3d".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000, 1000], // 10亿元素
        },
    ];

    // 验证这些形状不会导致溢出
    assert_eq!(large_values[0].num_elements(), Some(usize::MAX / 2));
    assert_eq!(large_values[1].num_elements(), Some(1_000_000_000));
    assert_eq!(large_values[2].num_elements(), Some(1_000_000_000));
}

/// Test 3: 测试可能导致溢出的形状
#[test]
fn test_overflow_shapes() {
    // 测试会溢出的形状 - 使用usize::MAX确保溢出
    let overflow_shapes = vec![
        vec![usize::MAX, 2],              // 肯定溢出
        vec![usize::MAX / 2 + 1, 2],      // 会溢出 (MAX/2 + 1) * 2 = MAX + 2
        vec![usize::MAX / 4 + 1, 5],      // 会溢出 (MAX/4 + 1) * 5 > MAX
        vec![usize::MAX / 10 + 1, 11],    // 会溢出 (MAX/10 + 1) * 11 > MAX
    ];

    for shape in overflow_shapes {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        // 应该返回None表示溢出
        assert_eq!(value.num_elements(), None, "Shape {:?} should overflow", shape);
    }
}

/// Test 4: 测试特殊浮点数值（NaN, Infinity, -Infinity）
#[test]
fn test_special_float_values() {
    let special_values = vec![
        Attribute::Float(f64::NAN),
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::Float(-0.0),
    ];

    // 验证NaN
    if let Attribute::Float(val) = &special_values[0] {
        assert!(val.is_nan());
    } else {
        panic!("Expected Float attribute");
    }

    // 验证正无穷
    if let Attribute::Float(val) = &special_values[1] {
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
    } else {
        panic!("Expected Float attribute");
    }

    // 验证负无穷
    if let Attribute::Float(val) = &special_values[2] {
        assert!(val.is_infinite());
        assert!(val.is_sign_negative());
    } else {
        panic!("Expected Float attribute");
    }

    // 验证-0.0
    if let Attribute::Float(val) = &special_values[3] {
        assert_eq!(*val, 0.0);
        assert!(val.is_sign_negative());
    } else {
        panic!("Expected Float attribute");
    }
}

/// Test 5: 测试嵌套Tensor类型的边界情况
#[test]
fn test_nested_tensor_types() {
    // 测试多层嵌套的Tensor类型
    let nested_type = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        }),
        shape: vec![3, 3],
    };

    // 验证嵌套类型的有效性
    assert!(nested_type.is_valid_type());

    // 测试深层嵌套（但保持合理深度）
    let deep_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![1],
            }),
            shape: vec![1],
        }),
        shape: vec![1],
    };

    assert!(deep_nested.is_valid_type());
}

/// Test 6: 测试Module包含大量操作的边界情况
#[test]
fn test_large_module_operations() {
    let mut module = Module::new("large_module");

    // 添加大量操作
    for i in 0..1000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[999].op_type, "op_999");
}

/// Test 7: 测试极端整数值
#[test]
fn test_extreme_integer_values() {
    let extreme_ints = vec![
        Attribute::Int(i64::MAX),
        Attribute::Int(i64::MIN),
        Attribute::Int(-1),
        Attribute::Int(0),
        Attribute::Int(1),
    ];

    // 验证最大值
    if let Attribute::Int(val) = &extreme_ints[0] {
        assert_eq!(*val, i64::MAX);
    }

    // 验证最小值
    if let Attribute::Int(val) = &extreme_ints[1] {
        assert_eq!(*val, i64::MIN);
    }

    // 验证边界值
    if let Attribute::Int(val) = &extreme_ints[2] {
        assert_eq!(*val, -1);
    }

    if let Attribute::Int(val) = &extreme_ints[3] {
        assert_eq!(*val, 0);
    }

    if let Attribute::Int(val) = &extreme_ints[4] {
        assert_eq!(*val, 1);
    }
}

/// Test 8: 测试包含大量属性的Operation
#[test]
fn test_operation_with_many_attributes() {
    let mut op = Operation::new("many_attrs");

    // 添加大量属性
    for i in 0..100 {
        op.attributes.insert(format!("attr_{}", i), Attribute::Int(i as i64));
    }

    assert_eq!(op.attributes.len(), 100);

    // 验证所有属性都存在且正确
    for i in 0..100 {
        let key = format!("attr_{}", i);
        assert!(op.attributes.contains_key(&key));
        if let Some(Attribute::Int(val)) = op.attributes.get(&key) {
            assert_eq!(*val, i as i64);
        }
    }
}

/// Test 9: 测试混合类型的Module输入输出
#[test]
fn test_mixed_type_module_io() {
    let mut module = Module::new("mixed_types");

    // 添加不同类型的输入
    module.inputs.push(Value {
        name: "float_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    module.inputs.push(Value {
        name: "int_input".to_string(),
        ty: Type::I64,
        shape: vec![5],
    });
    module.inputs.push(Value {
        name: "bool_input".to_string(),
        ty: Type::Bool,
        shape: vec![1],
    });

    // 添加不同类型的输出
    module.outputs.push(Value {
        name: "double_output".to_string(),
        ty: Type::F64,
        shape: vec![10, 10],
    });
    module.outputs.push(Value {
        name: "int32_output".to_string(),
        ty: Type::I32,
        shape: vec![5],
    });

    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::I64);
    assert_eq!(module.inputs[2].ty, Type::Bool);
    assert_eq!(module.outputs[0].ty, Type::F64);
    assert_eq!(module.outputs[1].ty, Type::I32);
}

/// Test 10: 测试嵌套数组的边界情况
#[test]
fn test_deeply_nested_attribute_arrays() {
    // 测试深层嵌套的数组属性
    let deeply_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
            ]),
            Attribute::String("inner".to_string()),
        ]),
        Attribute::Bool(true),
        Attribute::Int(42),
    ]);

    // 验证嵌套结构
    if let Attribute::Array(outer) = deeply_nested {
        assert_eq!(outer.len(), 3);

        // 检查第一个元素（深层嵌套）
        if let Attribute::Array(mid) = &outer[0] {
            assert_eq!(mid.len(), 2);
            if let Attribute::Array(inner) = &mid[0] {
                assert_eq!(inner.len(), 2);
                if let Attribute::Int(val) = inner[0] {
                    assert_eq!(val, 1);
                }
                if let Attribute::Float(val) = inner[1] {
                    assert_eq!(val, 2.5);
                }
            }
        }

        // 检查第二个元素（Bool）
        if let Attribute::Bool(b) = outer[1] {
            assert_eq!(b, true);
        }

        // 检查第三个元素（Int）
        if let Attribute::Int(val) = outer[2] {
            assert_eq!(val, 42);
        }
    }
}