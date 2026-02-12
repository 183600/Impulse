//! Practical boundary edge case tests - 覆盖实用的边界情况
//! 使用标准库 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: num_elements() 边界测试 - 测试接近 usize::MAX 的乘积
#[test]
fn test_num_elements_near_usize_max() {
    // 测试形状乘积刚好不会溢出的情况
    let value = Value {
        name: "safe_max".to_string(),
        ty: Type::F32,
        shape: vec![46340, 46340], // 46340^2 ≈ 2.1 billion, 接近 i32::MAX
    };
    assert_eq!(value.num_elements(), Some(46340 * 46340));

    // 测试形状乘积刚好会溢出的情况
    let overflow_value = Value {
        name: "overflow".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };
    assert_eq!(overflow_value.num_elements(), None);
}

/// Test 2: Value 包含单一 1 维度的张量
#[test]
fn test_single_one_dimension() {
    // 形状 [1] 表示一维张量，有1个元素
    let value = Value {
        name: "single_dim".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.shape, vec![1]);
    assert_eq!(value.num_elements(), Some(1));
    assert!(!value.shape.is_empty());

    // 形状 [1, 1, 1] 表示三维张量，有1个元素
    let triple_one = Value {
        name: "triple_one".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(triple_one.shape, vec![1, 1, 1]);
    assert_eq!(triple_one.num_elements(), Some(1));
}

/// Test 3: Attribute 浮点数精度边界
#[test]
fn test_float_precision_boundaries() {
    // 测试接近 f64::EPSILON 的值
    let tiny = Attribute::Float(f64::EPSILON);
    match tiny {
        Attribute::Float(val) => assert!(val > 0.0 && val < 1e-15),
        _ => panic!("Expected Float attribute"),
    }

    // 测试 1.0 - EPSILON
    let near_one = Attribute::Float(1.0 - f64::EPSILON);
    match near_one {
        Attribute::Float(val) => assert!(val < 1.0 && val > 0.9999),
        _ => panic!("Expected Float attribute"),
    }

    // 测试负的接近零的值
    let neg_tiny = Attribute::Float(-f64::EPSILON);
    match neg_tiny {
        Attribute::Float(val) => assert!(val < 0.0 && val > -1e-15),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 4: Module 包含多个具有相同输入的 Operation
#[test]
fn test_module_shared_inputs() {
    let mut module = Module::new("shared_inputs");

    // 创建共享的输入值
    let shared_input = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    };

    // 创建多个使用相同输入的操作
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_{}", i));
        // 克隆相同的输入（实际使用中可能使用引用）
        op.inputs.push(shared_input.clone());
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.inputs[0].name, "shared");
    }
}

/// Test 5: Operation 属性中包含零和负整数
#[test]
fn test_zero_and_negative_integer_attributes() {
    let mut op = Operation::new("boundary_ints");

    op.attributes.insert("zero".to_string(), Attribute::Int(0));
    op.attributes.insert("negative_one".to_string(), Attribute::Int(-1));
    op.attributes.insert("large_negative".to_string(), Attribute::Int(-1_000_000));
    op.attributes.insert("positive_one".to_string(), Attribute::Int(1));

    assert_eq!(op.attributes.len(), 4);

    match op.attributes.get("zero") {
        Some(Attribute::Int(0)) => (),
        _ => panic!("Expected Int(0)"),
    }

    match op.attributes.get("negative_one") {
        Some(Attribute::Int(-1)) => (),
        _ => panic!("Expected Int(-1)"),
    }
}

/// Test 6: Array Attribute 包含大量元素但类型单一
#[test]
fn test_large_uniform_array_attribute() {
    // 创建包含 1000 个相同整数的数组
    let large_uniform = Attribute::Array(
        (0..1000).map(|i| Attribute::Int(i)).collect()
    );

    match large_uniform {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1000);
            match &arr[0] {
                Attribute::Int(0) => (),
                _ => panic!("Expected Int(0)"),
            }
            match &arr[999] {
                Attribute::Int(999) => (),
                _ => panic!("Expected Int(999)"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: Tensor 嵌套层数为1（只有一层包装）
#[test]
fn test_single_level_nested_tensor() {
    let single_level = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![5, 5],
    };

    match single_level {
        Type::Tensor { ref element_type, ref shape } => {
            match **element_type {
                Type::F32 => (),
                _ => panic!("Expected F32 as element type"),
            }
            assert_eq!(shape, &vec![5, 5]);
        },
        _ => panic!("Expected Tensor type"),
    }

    // 验证这是合法的类型
    assert!(single_level.is_valid_type());
}

/// Test 8: Module 输入输出数量不匹配的边界情况
#[test]
fn test_module_unmatched_inputs_outputs() {
    let mut module = Module::new("unmatched");

    // 添加3个输入
    for i in 0..3 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
    }

    // 只添加1个输出
    module.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 1);
    assert_ne!(module.inputs.len(), module.outputs.len());
}

/// Test 9: Value 形状包含所有维度都是1的情况
#[test]
fn test_all_ones_shape() {
    let all_ones = Value {
        name: "all_ones".to_string(),
        ty: Type::F64,
        shape: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], // 10个1
    };

    assert_eq!(all_ones.shape.len(), 10);
    assert_eq!(all_ones.num_elements(), Some(1));
    for dim in &all_ones.shape {
        assert_eq!(*dim, 1);
    }
}

/// Test 10: Array Attribute 递归嵌套但只有2层
#[test]
fn test_two_level_nested_array() {
    let two_level = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
            Attribute::Int(3),
        ]),
        Attribute::Array(vec![
            Attribute::Int(4),
            Attribute::Int(5),
            Attribute::Int(6),
        ]),
    ]);

    match two_level {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 3);
                    assert_eq!(inner[0], Attribute::Int(1));
                    assert_eq!(inner[2], Attribute::Int(3));
                },
                _ => panic!("Expected Array"),
            }
            match &outer[1] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 3);
                    assert_eq!(inner[0], Attribute::Int(4));
                    assert_eq!(inner[2], Attribute::Int(6));
                },
                _ => panic!("Expected Array"),
            }
        },
        _ => panic!("Expected Array"),
    }
}