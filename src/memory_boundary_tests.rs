//! Memory and resource boundary tests - 10 additional test cases
//! 覆盖内存边界、资源限制和极端场景

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: 检测整数溢出 - 超大形状乘积溢出检查
#[test]
fn test_shape_overflow_detection() {
    // 创建一个会导致溢出的形状
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // 这会溢出
    };
    
    // num_elements 应该返回 None，因为会发生溢出
    assert_eq!(overflow_value.num_elements(), None);
}

/// Test 2: 零维张量（标量）处理
#[test]
fn test_scalar_tensor_handling() {
    let scalar = Value {
        name: "scalar_value".to_string(),
        ty: Type::F32,
        shape: vec![], // 空形状表示标量
    };
    
    // 标量的元素数量应该是 1
    assert_eq!(scalar.num_elements(), Some(1));
    assert_eq!(scalar.shape.len(), 0);
}

/// Test 3: 深度嵌套的张量类型
#[test]
fn test_deeply_nested_tensor_type() {
    // 创建 3 层嵌套的张量类型
    let level1 = Type::F32;
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![2, 3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4, 5],
    };
    
    // 验证嵌套类型有效性
    assert!(level3.is_valid_type());
    
    match level3 {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![4, 5]);
            match element_type.as_ref() {
                Type::Tensor { element_type: inner, shape: inner_shape } => {
                    assert_eq!(inner_shape, &vec![2, 3]);
                    assert_eq!(inner.as_ref(), &Type::F32);
                }
                _ => panic!("Expected nested Tensor"),
            }
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 4: 特殊浮点值（NaN 和无穷大）属性
#[test]
fn test_special_float_values() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    let zero_attr = Attribute::Float(0.0);
    let neg_zero_attr = Attribute::Float(-0.0);
    
    // 验证 NaN
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float with NaN"),
    }
    
    // 验证正无穷
    match pos_inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float with positive infinity"),
    }
    
    // 验证负无穷
    match neg_inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float with negative infinity"),
    }
    
    // 验证零和负零
    match zero_attr {
        Attribute::Float(val) => assert_eq!(val, 0.0),
        _ => panic!("Expected Float with zero"),
    }
    
    match neg_zero_attr {
        Attribute::Float(val) => {
            assert_eq!(val, -0.0);
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected Float with negative zero"),
    }
}

/// Test 5: 模块中包含大量操作
#[test]
fn test_module_with_many_operations() {
    let mut module = Module::new("large_module");
    
    // 添加 1000 个操作
    for i in 0..1000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[999].op_type, "op_999");
}

/// Test 6: 属性数组的混合类型
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
        ]),
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            
            // 验证每个元素类型
            assert!(matches!(arr[0], Attribute::Int(42)));
            assert!(matches!(arr[1], Attribute::Float(_)));
            assert!(matches!(arr[2], Attribute::String(_)));
            assert!(matches!(arr[3], Attribute::Bool(true)));
            assert!(matches!(arr[4], Attribute::Array(_)));
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: 极端整数值属性
#[test]
fn test_extreme_integer_attributes() {
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let zero = Attribute::Int(0);
    let neg_one = Attribute::Int(-1);
    
    match max_int {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected Int with MAX value"),
    }
    
    match min_int {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected Int with MIN value"),
    }
    
    match zero {
        Attribute::Int(val) => assert_eq!(val, 0),
        _ => panic!("Expected Int with zero"),
    }
    
    match neg_one {
        Attribute::Int(val) => assert_eq!(val, -1),
        _ => panic!("Expected Int with -1"),
    }
}

/// Test 8: 空字符串和特殊字符字符串属性
#[test]
fn test_string_attributes_with_special_chars() {
    let empty_string = Attribute::String("".to_string());
    let whitespace_string = Attribute::String("   ".to_string());
    let unicode_string = Attribute::String("你好世界🚀".to_string());
    let control_chars = Attribute::String("test\t\n\r\0".to_string());
    let very_long_string = Attribute::String("a".repeat(10000));
    
    // 验证空字符串
    match empty_string {
        Attribute::String(s) => assert_eq!(s.len(), 0),
        _ => panic!("Expected empty String attribute"),
    }
    
    // 验证空白字符串
    match whitespace_string {
        Attribute::String(s) => assert_eq!(s, "   "),
        _ => panic!("Expected whitespace String attribute"),
    }
    
    // 验证 Unicode 字符串
    match unicode_string {
        Attribute::String(s) => assert!(s.contains("你好")),
        _ => panic!("Expected Unicode String attribute"),
    }
    
    // 验证包含控制字符的字符串
    match control_chars {
        Attribute::String(s) => assert!(s.contains('\t')),
        _ => panic!("Expected String with control characters"),
    }
    
    // 验证超长字符串
    match very_long_string {
        Attribute::String(s) => assert_eq!(s.len(), 10000),
        _ => panic!("Expected very long String attribute"),
    }
}

/// Test 9: 包含零维度的张量形状
#[test]
fn test_tensor_with_zero_dimension() {
    let test_cases = vec![
        (vec![0], 0),
        (vec![0, 10], 0),
        (vec![10, 0], 0),
        (vec![5, 0, 3], 0),
        (vec![1, 0, 1, 0, 1], 0),
    ];
    
    for (shape, expected_elements) in test_cases {
        let value = Value {
            name: "zero_dim_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.num_elements(), Some(expected_elements));
    }
}

/// Test 10: 模块输入输出类型一致性验证
#[test]
fn test_module_type_consistency() {
    let mut module = Module::new("type_consistency_module");
    
    // 添加输入
    module.inputs.push(Value {
        name: "input1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.inputs.push(Value {
        name: "input2".to_string(),
        ty: Type::I32,
        shape: vec![5],
    });
    
    // 添加输出
    module.outputs.push(Value {
        name: "output1".to_string(),
        ty: Type::F64,
        shape: vec![10],
    });
    module.outputs.push(Value {
        name: "output2".to_string(),
        ty: Type::Bool,
        shape: vec![5],
    });
    
    // 验证输入和输出的数量
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    
    // 验证输入类型
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::I32);
    
    // 验证输出类型
    assert_eq!(module.outputs[0].ty, Type::F64);
    assert_eq!(module.outputs[1].ty, Type::Bool);
}