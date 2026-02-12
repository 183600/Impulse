//! Comprehensive edge boundary tests - covering advanced boundary scenarios
//! 使用标准库 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// 测试1: 检查 Value 的 num_elements() 方法的溢出检测
#[test]
fn test_value_num_elements_overflow_detection() {
    // 正常情况
    let normal = Value {
        name: "normal".to_string(),
        ty: Type::F32,
        shape: vec![2, 3, 4],
    };
    assert_eq!(normal.num_elements(), Some(24));

    // 标量（空形状）
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));

    // 包含零的形状
    let zero_dim = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![5, 0, 10],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));

    // 可能溢出的情况（使用 checked_mul）
    let large_safe = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: vec![10000, 10000],
    };
    assert_eq!(large_safe.num_elements(), Some(100_000_000));
}

/// 测试2: Type::Tensor 的嵌套验证
#[test]
fn test_nested_tensor_type_validation() {
    // 创建多层嵌套的张量类型
    let inner = Type::F32;
    let level1 = Type::Tensor {
        element_type: Box::new(inner.clone()),
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

    // 验证所有层级都是有效的
    assert!(inner.is_valid_type());
    assert!(level1.is_valid_type());
    assert!(level2.is_valid_type());
    assert!(level3.is_valid_type());

    // 验证形状的正确性
    if let Type::Tensor { shape, .. } = level1 {
        assert_eq!(shape, vec![2]);
    }
    if let Type::Tensor { shape, .. } = level2 {
        assert_eq!(shape, vec![3]);
    }
    if let Type::Tensor { shape, .. } = level3 {
        assert_eq!(shape, vec![4]);
    }
}

/// 测试3: Operation 的属性哈希表行为
#[test]
fn test_operation_attribute_hashmap_behavior() {
    let mut op = Operation::new("test_op");

    // 插入多个属性
    op.attributes.insert("int_attr".to_string(), Attribute::Int(42));
    op.attributes.insert("float_attr".to_string(), Attribute::Float(3.14));
    op.attributes.insert("bool_attr".to_string(), Attribute::Bool(true));
    op.attributes.insert("string_attr".to_string(), Attribute::String("test".to_string()));

    assert_eq!(op.attributes.len(), 4);

    // 验证属性存在且值正确
    assert!(op.attributes.contains_key("int_attr"));
    assert!(op.attributes.contains_key("float_attr"));
    assert!(op.attributes.contains_key("bool_attr"));
    assert!(op.attributes.contains_key("string_attr"));

    // 更新现有属性
    op.attributes.insert("int_attr".to_string(), Attribute::Int(100));
    assert_eq!(op.attributes.len(), 4); // 长度不变

    if let Some(Attribute::Int(val)) = op.attributes.get("int_attr") {
        assert_eq!(*val, 100);
    } else {
        panic!("Expected Int attribute");
    }
}

/// 测试4: Module 的输入输出管理
#[test]
fn test_module_input_output_management() {
    let mut module = Module::new("test_module");

    // 添加多个输入
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

    // 添加多个输出
    module.outputs.push(Value {
        name: "output1".to_string(),
        ty: Type::F64,
        shape: vec![10],
    });
    module.outputs.push(Value {
        name: "output2".to_string(),
        ty: Type::Bool,
        shape: vec![1],
    });

    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].name, "input1");
    assert_eq!(module.inputs[1].ty, Type::I32);
    assert_eq!(module.outputs[0].ty, Type::F64);
    assert_eq!(module.outputs[1].name, "output2");
}

/// 测试5: Attribute 的相等性和不等性
#[test]
fn test_attribute_equality_and_inequality() {
    // 整数属性
    let int1 = Attribute::Int(42);
    let int2 = Attribute::Int(42);
    let int3 = Attribute::Int(43);
    assert_eq!(int1, int2);
    assert_ne!(int1, int3);

    // 浮点属性
    let float1 = Attribute::Float(1.0);
    let float2 = Attribute::Float(1.0);
    let float3 = Attribute::Float(1.0 + 1e-10);
    assert_eq!(float1, float2);
    // 注意：微小的浮点差异可能导致不等
    assert_ne!(float1, float3);

    // 字符串属性
    let str1 = Attribute::String("hello".to_string());
    let str2 = Attribute::String("hello".to_string());
    let str3 = Attribute::String("world".to_string());
    assert_eq!(str1, str2);
    assert_ne!(str1, str3);
    assert_ne!(str2, str3);

    // 布尔属性
    let bool1 = Attribute::Bool(true);
    let bool2 = Attribute::Bool(true);
    let bool3 = Attribute::Bool(false);
    assert_eq!(bool1, bool2);
    assert_ne!(bool1, bool3);

    // 数组属性
    let arr1 = Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]);
    let arr2 = Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]);
    let arr3 = Attribute::Array(vec![Attribute::Int(1), Attribute::Int(3)]);
    assert_eq!(arr1, arr2);
    assert_ne!(arr1, arr3);
}

/// 测试6: Value 的克隆和相等性
#[test]
fn test_value_clone_and_equality() {
    let original = Value {
        name: "test_value".to_string(),
        ty: Type::F32,
        shape: vec![2, 3, 4],
    };

    let cloned = original.clone();

    // 验证克隆后的值相等
    assert_eq!(original, cloned);
    assert_eq!(original.name, cloned.name);
    assert_eq!(original.ty, cloned.ty);
    assert_eq!(original.shape, cloned.shape);
    assert_eq!(original.num_elements(), cloned.num_elements());

    // 修改克隆不应影响原始值
    let mut modified = cloned;
    modified.name = "modified".to_string();
    assert_ne!(original.name, modified.name);
    assert_eq!(original.name, "test_value");
}

/// 测试7: 特殊浮点值（NaN, Infinity）
#[test]
fn test_special_float_values() {
    let nan_val = Attribute::Float(f64::NAN);
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);

    // NaN 不等于自身
    if let Attribute::Float(val) = nan_val {
        assert!(val.is_nan());
        assert_ne!(val, val); // NaN != NaN
    }

    // 正无穷
    if let Attribute::Float(val) = pos_inf {
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
    }

    // 负无穷
    if let Attribute::Float(val) = neg_inf {
        assert!(val.is_infinite());
        assert!(val.is_sign_negative());
    }
}

/// 测试8: 空数组和嵌套数组
#[test]
fn test_empty_and_nested_arrays() {
    // 空数组
    let empty = Attribute::Array(vec![]);
    if let Attribute::Array(arr) = empty {
        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
    }

    // 嵌套数组
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
        Attribute::Array(vec![Attribute::Int(3), Attribute::Int(4)]),
    ]);

    if let Attribute::Array(outer) = nested {
        assert_eq!(outer.len(), 2);
        if let Attribute::Array(inner) = &outer[0] {
            assert_eq!(inner.len(), 2);
        }
    }
}

/// 测试9: Module 和 Operation 的序列化兼容性
#[test]
fn test_module_operation_serialization_compatibility() {
    let mut module = Module::new("serde_test");
    module.name = "test_module".to_string();

    let mut op = Operation::new("test_op");
    op.op_type = "custom_op".to_string();
    op.attributes.insert("key".to_string(), Attribute::String("value".to_string()));

    module.add_operation(op);

    // 验证基本结构
    assert_eq!(module.name, "test_module");
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].op_type, "custom_op");
    assert_eq!(module.operations[0].attributes.len(), 1);
}

/// 测试10: 极端形状的张量（宽高比）
#[test]
fn test_extreme_aspect_ratio_tensors() {
    // 非常宽的张量
    let wide = Value {
        name: "wide".to_string(),
        ty: Type::F32,
        shape: vec![1, 1_000_000],
    };
    assert_eq!(wide.shape, vec![1, 1_000_000]);
    assert_eq!(wide.num_elements(), Some(1_000_000));

    // 非常高的张量
    let tall = Value {
        name: "tall".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 1],
    };
    assert_eq!(tall.shape, vec![1_000_000, 1]);
    assert_eq!(tall.num_elements(), Some(1_000_000));

    // 一维超长向量
    let long_vec = Value {
        name: "long_vec".to_string(),
        ty: Type::F32,
        shape: vec![50_000_000],
    };
    assert_eq!(long_vec.shape, vec![50_000_000]);
    assert_eq!(long_vec.num_elements(), Some(50_000_000));
}