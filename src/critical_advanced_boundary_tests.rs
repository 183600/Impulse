//! Critical advanced boundary tests - 覆盖更多边界情况
//! 使用标准库的 assert! 和 assert_eq! 宏

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::ir_utils::{calculate_tensor_size, get_element_type};
use std::collections::HashMap;

/// 测试1: 测试计算张量大小时的溢出保护
#[test]
fn test_calculate_tensor_size_overflow_protection() {
    // 测试可能导致溢出的超大形状
    let large_shape = vec![usize::MAX, 2]; // 这将导致乘法溢出
    
    // F32 类型，4字节
    let result = calculate_tensor_size(&Type::F32, &large_shape);
    assert!(result.is_err(), "Expected error for overflow in shape calculation");
    
    // 正常情况应该成功
    let normal_shape = vec![1000, 1000];
    let normal_result = calculate_tensor_size(&Type::F32, &normal_shape);
    assert_eq!(normal_result.unwrap(), 1_000_000 * 4);
}

/// 测试2: 测试深度嵌套张量类型的大小计算
#[test]
fn test_deeply_nested_tensor_size() {
    // 创建一个三层嵌套的张量类型
    let inner_type = Type::F32;
    let middle_type = Type::Tensor {
        element_type: Box::new(inner_type),
        shape: vec![2, 2],
    };
    let outer_type = Type::Tensor {
        element_type: Box::new(middle_type),
        shape: vec![3],
    };
    
    let shape = vec![2];
    let result = calculate_tensor_size(&outer_type, &shape);
    
    // 根据 calculate_tensor_size 函数逻辑:
    // outer_type 本身是一个 Tensor，内部包含 middle_type (也是 Tensor)
    // shape [2] 表示有 2 个这样的 outer_type 元素
    // 计算: 2 (outer shape) * (3 * 2 * 2 * 4) = 2 * 48 = 96
    assert_eq!(result.unwrap(), 96);
}

/// 测试3: 测试所有数据类型的张量大小计算
#[test]
fn test_tensor_size_for_all_types() {
    let test_cases = [
        (Type::F32, vec![1, 1, 1], 4),      // 1 element * 4 bytes
        (Type::F64, vec![1, 1, 1], 8),      // 1 element * 8 bytes
        (Type::I32, vec![1, 1, 1], 4),      // 1 element * 4 bytes
        (Type::I64, vec![1, 1, 1], 8),      // 1 element * 8 bytes
        (Type::Bool, vec![1, 1, 1], 1),     // 1 element * 1 byte
        (Type::F32, vec![2, 3], 24),        // 6 elements * 4 bytes
        (Type::I64, vec![5, 5], 200),       // 25 elements * 8 bytes
    ];
    
    for (dtype, shape, expected) in test_cases {
        let result = calculate_tensor_size(&dtype, &shape);
        assert_eq!(result.unwrap(), expected, 
                   "Failed for type {:?} with shape {:?}", dtype, shape);
    }
}

/// 测试4: 测试零维张量的处理
#[test]
fn test_zero_dimension_tensor_size() {
    let zero_shapes = [
        vec![0],
        vec![0, 10],
        vec![10, 0],
        vec![0, 0],
        vec![0, 5, 10],
    ];
    
    for shape in zero_shapes {
        for dtype in [Type::F32, Type::I32, Type::Bool] {
            let result = calculate_tensor_size(&dtype, &shape);
            assert_eq!(result.unwrap(), 0, 
                       "Zero dimension should result in 0 size for {:?} with shape {:?}", dtype, shape);
        }
    }
}

/// 测试5: 测试张量类型字符串表示的边界情况
#[test]
fn test_type_to_string_edge_cases() {
    // 空张量 (scalar)
    let scalar_str = crate::utils::ir_utils::type_to_string(&Type::F32);
    assert_eq!(scalar_str, "f32");
    
    // 多维张量
    let multi_dim = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3, 4, 5],
    };
    let multi_str = crate::utils::ir_utils::type_to_string(&multi_dim);
    assert!(multi_str.contains("tensor<f32, [2, 3, 4, 5]>"));
    
    // 嵌套张量
    let nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![10],
        }),
        shape: vec![5],
    };
    let nested_str = crate::utils::ir_utils::type_to_string(&nested);
    assert!(nested_str.contains("tensor<"));
}

/// 测试6: 测试包含多种属性类型的操作
#[test]
fn test_operation_with_comprehensive_attributes() {
    let mut op = Operation::new("comprehensive_op");
    let mut attrs = HashMap::new();
    
    // 添加各种类型的属性
    attrs.insert("int_attr".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("neg_int_attr".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("float_attr".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("neg_float_attr".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("string_attr".to_string(), Attribute::String("test_string".to_string()));
    attrs.insert("bool_true".to_string(), Attribute::Bool(true));
    attrs.insert("bool_false".to_string(), Attribute::Bool(false));
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    attrs.insert("mixed_array".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.5),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
    ]));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 9);
    assert_eq!(op.op_type, "comprehensive_op");
    
    // 验证关键属性
    if let Some(Attribute::Int(val)) = op.attributes.get("int_attr") {
        assert_eq!(*val, i64::MAX);
    } else {
        panic!("Expected Int attribute");
    }
    
    if let Some(Attribute::Array(arr)) = op.attributes.get("mixed_array") {
        assert_eq!(arr.len(), 4);
    } else {
        panic!("Expected Array attribute");
    }
}

/// 测试7: 测试模块操作计数和查找
#[test]
fn test_module_operation_counting() {
    let mut module = Module::new("count_test");
    
    // 添加多个操作，有些类型相同
    module.add_operation(Operation::new("conv"));
    module.add_operation(Operation::new("relu"));
    module.add_operation(Operation::new("conv"));
    module.add_operation(Operation::new("pool"));
    module.add_operation(Operation::new("relu"));
    module.add_operation(Operation::new("fc"));
    
    // 使用 count_operations_by_type
    let counts = crate::utils::ir_utils::count_operations_by_type(&module);
    assert_eq!(counts.get("conv"), Some(&2));
    assert_eq!(counts.get("relu"), Some(&2));
    assert_eq!(counts.get("pool"), Some(&1));
    assert_eq!(counts.get("fc"), Some(&1));
    
    // 使用 find_operations_by_type
    let conv_ops = crate::utils::ir_utils::find_operations_by_type(&module, "conv");
    assert_eq!(conv_ops.len(), 2);
    
    let relu_ops = crate::utils::ir_utils::find_operations_by_type(&module, "relu");
    assert_eq!(relu_ops.len(), 2);
    
    let non_existent = crate::utils::ir_utils::find_operations_by_type(&module, "nonexistent");
    assert_eq!(non_existent.len(), 0);
}

/// 测试8: 测试包含空输入和输出的操作链
#[test]
fn test_operation_chain_with_empty_io() {
    let mut module = Module::new("chain_test");
    
    // 创建一个操作链：input -> op1 -> op2 -> output
    let input_val = Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    let mut op1 = Operation::new("op1");
    op1.inputs.push(input_val.clone());
    op1.outputs.push(Value {
        name: "hidden".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(op1.outputs[0].clone());
    op2.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].inputs.len(), 1);
    assert_eq!(module.operations[0].outputs.len(), 1);
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[1].outputs.len(), 1);
}

/// 测试9: 测试嵌套张量的元素类型提取
#[test]
fn test_get_element_type_from_deeply_nested() {
    // 测试各种深度的嵌套张量
    let test_cases: Vec<(Type, Type)> = vec![
        // 非嵌套
        (Type::F32, Type::F32),
        (Type::I64, Type::I64),
        // 一层嵌套
        (
            Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            },
            Type::F32,
        ),
        // 两层嵌套
        (
            Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I32),
                    shape: vec![3, 3],
                }),
                shape: vec![5],
            },
            Type::I32,
        ),
        // 三层嵌套
        (
            Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Tensor {
                        element_type: Box::new(Type::F64),
                        shape: vec![10],
                    }),
                    shape: vec![7],
                }),
                shape: vec![3],
            },
            Type::F64,
        ),
    ];
    
    for (nested_type, expected) in test_cases {
        let extracted = get_element_type(&nested_type);
        assert_eq!(extracted, &expected, 
                   "Failed to extract element type from {:?}", nested_type);
    }
}

/// 测试10: 测试张量形状的有效性验证
#[test]
fn test_tensor_shape_validity() {
    // 创建各种形状的张量，验证它们能够正确创建
    let test_shapes = [
        vec![],                    // scalar
        vec![1],                   // single element vector
        vec![1, 1],                // single element matrix
        vec![2, 2],                // 2x2 matrix
        vec![1, 3, 224, 224],      // common CNN input
        vec![10, 3, 32, 32],       // batch of images
        vec![1024],                // large 1D tensor
        vec![256, 256],            // 2D tensor
        vec![64, 64, 64],          // 3D tensor
        vec![2, 3, 4, 5, 6],       // 5D tensor
    ];
    
    for shape in test_shapes {
        let value = Value {
            name: format!("tensor_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // 计算元素数量
        let num_elements = value.num_elements();
        assert!(num_elements.is_some(), "Shape {:?} should be valid", shape);
    }
    
    // 测试包含零的形状
    let zero_shapes = [
        vec![0],
        vec![0, 10],
        vec![10, 0],
        vec![0, 0, 10],
    ];
    
    for shape in zero_shapes {
        let value = Value {
            name: format!("zero_tensor_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.num_elements(), Some(0));
    }
}