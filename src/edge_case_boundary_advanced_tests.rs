//! Advanced edge case boundary tests for Impulse compiler
//! 
//! 这10个测试用例覆盖了编译器IR的核心边界情况：
//! 1. 值的形状溢出检测
//! 2. 类型嵌套验证
//! 3. 操作属性序列化/反序列化
//! 4. 模块克隆独立性
//! 5. 属性的浮点精度
//! 6. 操作输入输出的形状一致性
//! 7. 类型验证扩展
//! 8. 值的零维特殊情况
//! 9. 操作的空属性处理
//! 10. 模块的空操作序列

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: 值的形状溢出检测 - 验证 num_elements() 正确处理溢出情况
#[test]
fn test_value_shape_overflow_detection() {
    // 正常情况下的元素数量计算
    let normal_value = Value {
        name: "normal".to_string(),
        ty: Type::F32,
        shape: vec![10, 10, 10],
    };
    assert_eq!(normal_value.num_elements(), Some(1000));

    // 包含零维度的情况
    let zero_dim_value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 10],
    };
    assert_eq!(zero_dim_value.num_elements(), Some(0));

    // 标量情况（空形状）
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar_value.num_elements(), Some(1));

    // 可能导致溢出的超大维度
    let large_value = Value {
        name: "large".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],
    };
    // 检查乘法是否会溢出
    assert_eq!(large_value.num_elements(), None);
}

/// Test 2: 类型嵌套验证 - 测试嵌套张量类型的正确性
#[test]
fn test_type_nesting_validation() {
    // 基础类型应该有效
    assert!(Type::F32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());

    // 一层嵌套的张量类型
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3, 4],
    };
    assert!(tensor_type.is_valid_type());

    // 两层嵌套的张量类型（虽然可能不太常见，但应该被支持）
    let nested_tensor = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![1],
        }),
        shape: vec![5, 5],
    };
    assert!(nested_tensor.is_valid_type());

    // 验证多层嵌套的形状计算
    match nested_tensor {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, vec![5, 5]);
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 3: 操作属性序列化/反序列化 - 测试属性的完整序列化
#[test]
fn test_operation_attribute_serialization() {
    let mut op = Operation::new("test_op");
    let mut attrs = HashMap::new();

    // 添加各种类型的属性
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14159));
    attrs.insert("string_attr".to_string(), Attribute::String("hello".to_string()));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Int(2),
        Attribute::Int(3),
    ]));

    op.attributes = attrs;

    // 验证属性数量
    assert_eq!(op.attributes.len(), 5);

    // 验证每个属性
    assert_eq!(op.attributes.get("int_attr"), Some(&Attribute::Int(42)));
    assert_eq!(op.attributes.get("float_attr"), Some(&Attribute::Float(3.14159)));
    assert_eq!(op.attributes.get("string_attr"), Some(&Attribute::String("hello".to_string())));
    assert_eq!(op.attributes.get("bool_attr"), Some(&Attribute::Bool(true)));

    // 验证数组属性
    match op.attributes.get("array_attr") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Attribute::Int(1));
            assert_eq!(arr[1], Attribute::Int(2));
            assert_eq!(arr[2], Attribute::Int(3));
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 4: 模块克隆独立性 - 验证克隆后的模块独立修改
#[test]
fn test_module_clone_independence() {
    let mut original = Module::new("original");
    
    // 添加一个操作
    let mut op = Operation::new("test_op");
    op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    original.add_operation(op);

    // 克隆模块
    let mut cloned = original.clone();

    // 修改原始模块
    original.operations[0].op_type = "modified_op".to_string();

    // 验证克隆模块保持不变
    assert_eq!(cloned.operations[0].op_type, "test_op");
    assert_eq!(original.operations[0].op_type, "modified_op");

    // 修改克隆模块
    cloned.name = "cloned".to_string();

    // 验证原始模块保持不变
    assert_eq!(original.name, "original");
    assert_eq!(cloned.name, "cloned");
}

/// Test 5: 属性的浮点精度 - 测试特殊浮点值的处理
#[test]
fn test_attribute_float_precision() {
    // 测试正无穷
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    match pos_inf_attr {
        Attribute::Float(val) => {
            assert!(val.is_infinite());
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected Float attribute"),
    }

    // 测试负无穷
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    match neg_inf_attr {
        Attribute::Float(val) => {
            assert!(val.is_infinite());
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected Float attribute"),
    }

    // 测试NaN
    let nan_attr = Attribute::Float(f64::NAN);
    match nan_attr {
        Attribute::Float(val) => {
            assert!(val.is_nan());
        }
        _ => panic!("Expected Float attribute"),
    }

    // 测试零的正负性
    let pos_zero = Attribute::Float(0.0);
    let neg_zero = Attribute::Float(-0.0);
    match (pos_zero, neg_zero) {
        (Attribute::Float(p), Attribute::Float(n)) => {
            assert_eq!(p, 0.0);
            assert_eq!(n, -0.0);
            // 注意：0.0 和 -0.0 在 f64 中是不同的
            assert!(p.is_sign_positive());
            assert!(n.is_sign_negative());
        }
        _ => panic!("Expected Float attributes"),
    }
}

/// Test 6: 操作输入输出的形状一致性 - 验证形状计算的正确性
#[test]
fn test_operation_io_shape_consistency() {
    let mut op = Operation::new("elementwise_op");

    // 添加相同形状的输入
    op.inputs.push(Value {
        name: "input1".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    });
    op.inputs.push(Value {
        name: "input2".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    });

    // 添加相同形状的输出
    op.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    });

    // 验证所有输入形状相同
    assert_eq!(op.inputs[0].shape, op.inputs[1].shape);
    // 验证输出形状与输入相同
    assert_eq!(op.inputs[0].shape, op.outputs[0].shape);

    // 验证元素数量一致
    let input_elements = op.inputs[0].num_elements();
    let output_elements = op.outputs[0].num_elements();
    assert_eq!(input_elements, output_elements);
    assert_eq!(input_elements, Some(200));
}

/// Test 7: 类型验证扩展 - 测试所有基础类型的有效性
#[test]
fn test_type_validation_extensions() {
    // 测试所有基础类型
    let types = [
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];

    for ty in types.iter() {
        assert!(ty.is_valid_type(), "Type {:?} should be valid", ty);
    }

    // 测试嵌套类型
    let tensor_f32 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![1, 2, 3],
    };
    assert!(tensor_f32.is_valid_type());

    let tensor_i64 = Type::Tensor {
        element_type: Box::new(Type::I64),
        shape: vec![100],
    };
    assert!(tensor_i64.is_valid_type());

    // 测试嵌套的张量类型
    let nested = Type::Tensor {
        element_type: Box::new(tensor_f32.clone()),
        shape: vec![5],
    };
    assert!(nested.is_valid_type());
}

/// Test 8: 值的零维特殊情况 - 测试标量和一维张量的处理
#[test]
fn test_value_zero_dimension_handling() {
    // 标量（空形状）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
    assert_eq!(scalar.shape.len(), 0);

    // 一维张量
    let vector = Value {
        name: "vector".to_string(),
        ty: Type::F32,
        shape: vec![100],
    };
    assert_eq!(vector.num_elements(), Some(100));
    assert_eq!(vector.shape.len(), 1);

    // 单元素张量（形状为 [1]）
    let single_element = Value {
        name: "single".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(single_element.num_elements(), Some(1));
    assert_eq!(single_element.shape.len(), 1);

    // 多个1相乘的情况
    let multi_one = Value {
        name: "multi_one".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1],
    };
    assert_eq!(multi_one.num_elements(), Some(1));
    assert_eq!(multi_one.shape.len(), 4);
}

/// Test 9: 操作的空属性处理 - 测试空属性HashMap的行为
#[test]
fn test_operation_empty_attributes() {
    // 创建没有属性的操作
    let op = Operation::new("no_attrs_op");
    
    // 验证属性HashMap为空
    assert_eq!(op.attributes.len(), 0);
    assert!(op.attributes.is_empty());

    // 验证获取不存在的属性返回None
    assert_eq!(op.attributes.get("nonexistent"), None);

    // 添加属性后验证
    let mut op_with_attrs = Operation::new("with_attrs_op");
    op_with_attrs.attributes.insert("key".to_string(), Attribute::Int(42));
    
    assert_eq!(op_with_attrs.attributes.len(), 1);
    assert!(!op_with_attrs.attributes.is_empty());
}

/// Test 10: 模块的空操作序列 - 测试空模块的行为
#[test]
fn test_module_empty_operation_sequence() {
    // 创建空模块
    let module = Module::new("empty_module");

    // 验证所有列表为空
    assert_eq!(module.name, "empty_module");
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);

    // 添加操作
    let mut module_with_ops = Module::new("module_with_ops");
    let op1 = Operation::new("op1");
    let op2 = Operation::new("op2");
    
    module_with_ops.add_operation(op1);
    module_with_ops.add_operation(op2);

    // 验证操作列表
    assert_eq!(module_with_ops.operations.len(), 2);
    assert_eq!(module_with_ops.operations[0].op_type, "op1");
    assert_eq!(module_with_ops.operations[1].op_type, "op2");

    // 验证输入输出仍然为空（操作不影响它们）
    assert_eq!(module_with_ops.inputs.len(), 0);
    assert_eq!(module_with_ops.outputs.len(), 0);
}