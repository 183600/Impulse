//! Comprehensive edge case boundary tests - 覆盖更多边界情况
//! 使用标准库 assert! 和 assert_eq! 进行验证

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: 值的形状包含 usize::MAX 时，num_elements 应返回 None（溢出检测）
#[test]
fn test_shape_with_usize_max() {
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // 会溢出
    };
    assert_eq!(value.num_elements(), None);
}

/// Test 2: 空字符串名称的值和操作
#[test]
fn test_empty_string_names() {
    let mut op = Operation::new("");
    op.inputs.push(Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    op.outputs.push(Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs[0].name, "");
    assert_eq!(op.outputs[0].name, "");
}

/// Test 3: 浮点属性包含负零和正零
#[test]
fn test_float_zero_sign() {
    let pos_zero = Attribute::Float(0.0);
    let neg_zero = Attribute::Float(-0.0);
    
    match pos_zero {
        Attribute::Float(val) => {
            assert!(val == 0.0);
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match neg_zero {
        Attribute::Float(val) => {
            assert!(val == 0.0);
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 4: 嵌套张量类型的深度验证
#[test]
fn test_deeply_nested_tensor_type() {
    // 创建 5 层嵌套的张量类型
    let level5 = Type::F32;
    let level4 = Type::Tensor {
        element_type: Box::new(level5),
        shape: vec![2],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level4),
        shape: vec![3],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![4],
    };
    let level1 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![5],
    };
    
    assert!(level1.is_valid_type());
}

/// Test 5: 属性数组包含所有类型混合
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(i64::MAX),
        Attribute::Float(f64::MIN),
        Attribute::String("mixed".to_string()),
        Attribute::Bool(false),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            match &arr[4] {
                Attribute::Array(nested) => assert_eq!(nested.len(), 2),
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 6: 单元素形状（标量）的张量
#[test]
fn test_single_element_shape() {
    let single_element = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(single_element.num_elements(), Some(1));
    
    let empty_shape = Value {
        name: "empty_shape_scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(empty_shape.num_elements(), Some(1));
}

/// Test 7: 属性 HashMap 包含大量键值对
#[test]
fn test_large_attribute_hashmap() {
    let mut op = Operation::new("large_attrs");
    let mut attrs = HashMap::new();
    
    // 添加 100 个属性
    for i in 0..100 {
        attrs.insert(format!("key_{}", i), Attribute::Int(i as i64));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 100);
    assert!(op.attributes.contains_key("key_0"));
    assert!(op.attributes.contains_key("key_99"));
}

/// Test 8: 张量形状包含 1 和 0 的交替模式
#[test]
fn test_alternating_one_zero_shape() {
    let patterns = vec![
        vec![1, 0, 1, 0],
        vec![0, 1, 0, 1],
        vec![1, 0, 1, 0, 1],
    ];
    
    for shape in patterns {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        // 包含 0 的形状应该返回 0 个元素
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 9: 值名称包含特殊字符和空格
#[test]
fn test_special_characters_in_names() {
    let special_names = vec![
        "tensor with spaces",
        "tab\tcharacter",
        "null\x00character",
        "backslash\\escape",
        "quote\"test\"quote",
        "emoji🔥special",
    ];
    
    for name in special_names {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![2, 2],
        };
        assert_eq!(value.name, name);
    }
}

/// Test 10: 模块包含大量操作但无输入输出
#[test]
fn test_module_with_many_operations_no_io() {
    let mut module = Module::new("no_io_many_ops");
    
    // 添加 50 个操作
    for i in 0..50 {
        let mut op = Operation::new(&format!("op_{}", i));
        // 每个操作有内部输入输出
        op.inputs.push(Value {
            name: format!("internal_input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: format!("internal_output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 50);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
    
    // 验证每个操作都有正确的输入输出
    for (i, op) in module.operations.iter().enumerate() {
        assert_eq!(op.op_type, format!("op_{}", i));
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
    }
}