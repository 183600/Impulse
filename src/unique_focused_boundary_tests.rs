//! Unique focused boundary tests - Additional edge cases for IR components
//! 使用标准库的 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Value.num_elements() 使用 checked_mul 防止溢出
    #[test]
    fn test_value_num_elements_overflow_protection() {
        // 测试会溢出的情况 - 应该返回 None
        let overflow_value = Value {
            name: "overflow_tensor".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2], // 会导致溢出
        };
        assert_eq!(overflow_value.num_elements(), None);

        // 测试接近边界但不会溢出的情况
        let safe_large_value = Value {
            name: "safe_large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 10],
        };
        assert_eq!(safe_large_value.num_elements(), Some(1_000_000));

        // 测试包含零维的情况 - 应该返回 0
        let zero_dim_value = Value {
            name: "zero_dim_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100, 0, 50],
        };
        assert_eq!(zero_dim_value.num_elements(), Some(0));

        // 测试标量（空形状）- 应该返回 1
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(scalar_value.num_elements(), Some(1));
    }

    /// Test 2: TypeExtensions trait 的 is_valid_type 方法
    #[test]
    fn test_type_extensions_validity() {
        // 测试基本类型的有效性
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());

        // 测试嵌套 Tensor 类型的有效性
        let valid_nested = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        assert!(valid_nested.is_valid_type());

        // 测试深层嵌套 Tensor 类型的有效性
        let deeply_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I64),
                    shape: vec![1],
                }),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        assert!(deeply_nested.is_valid_type());
    }

    /// Test 3: Module 的输入输出与操作数的关系
    #[test]
    fn test_module_io_operations_relationship() {
        let mut module = Module::new("test_module");

        // 添加输入
        let input1 = Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        };
        let input2 = Value {
            name: "input2".to_string(),
            ty: Type::I32,
            shape: vec![5],
        };
        module.inputs.push(input1.clone());
        module.inputs.push(input2.clone());

        // 添加操作
        let mut op1 = Operation::new("add");
        op1.inputs.push(input1.clone());
        op1.inputs.push(input2.clone());
        op1.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op1);

        // 添加输出
        module.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        // 验证关系
        assert_eq!(module.inputs.len(), 2);
        assert_eq!(module.outputs.len(), 1);
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].inputs.len(), 2);
        assert_eq!(module.operations[0].outputs.len(), 1);
    }

    /// Test 4: Operation 属性的哈希表操作
    #[test]
    fn test_operation_attributes_hashmap() {
        let mut op = Operation::new("test_op");
        let mut attrs = HashMap::new();

        // 添加各种类型的属性
        attrs.insert("int_attr".to_string(), Attribute::Int(42));
        attrs.insert("float_attr".to_string(), Attribute::Float(3.14));
        attrs.insert("string_attr".to_string(), Attribute::String("test".to_string()));
        attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
        attrs.insert("array_attr".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
            Attribute::Int(3),
        ]));

        op.attributes = attrs;

        // 验证属性数量
        assert_eq!(op.attributes.len(), 5);

        // 验证各种属性类型
        match op.attributes.get("int_attr") {
            Some(Attribute::Int(42)) => {},
            _ => panic!("Expected Int(42)"),
        }

        match op.attributes.get("float_attr") {
            Some(Attribute::Float(val)) if (val - 3.14).abs() < f64::EPSILON => {},
            _ => panic!("Expected Float(3.14)"),
        }

        match op.attributes.get("string_attr") {
            Some(Attribute::String(s)) if s == "test" => {},
            _ => panic!("Expected String(\"test\")"),
        }

        match op.attributes.get("bool_attr") {
            Some(Attribute::Bool(true)) => {},
            _ => panic!("Expected Bool(true)"),
        }

        match op.attributes.get("array_attr") {
            Some(Attribute::Array(arr)) => assert_eq!(arr.len(), 3),
            _ => panic!("Expected Array attribute"),
        }
    }

    /// Test 5: Value 的形状边界情况
    #[test]
    fn test_value_shape_boundary_cases() {
        // 测试单一维度为 1
        let single_dim_one = Value {
            name: "single_one".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(single_dim_one.num_elements(), Some(1));

        // 测试多个维度都为 1
        let all_dims_one = Value {
            name: "all_ones".to_string(),
            ty: Type::F32,
            shape: vec![1, 1, 1, 1],
        };
        assert_eq!(all_dims_one.num_elements(), Some(1));

        // 测试混合维度（包含 0）
        let mixed_with_zero = Value {
            name: "mixed_zero".to_string(),
            ty: Type::F32,
            shape: vec![1, 0, 1],
        };
        assert_eq!(mixed_with_zero.num_elements(), Some(0));

        // 测试交替模式
        let alternating = Value {
            name: "alternating".to_string(),
            ty: Type::F32,
            shape: vec![1, 0, 1, 0, 1],
        };
        assert_eq!(alternating.num_elements(), Some(0));
    }

    /// Test 6: Tensor 类型比较和相等性
    #[test]
    fn test_tensor_type_equality_and_comparison() {
        // 相同的 Tensor 类型应该相等
        let tensor1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        let tensor2 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        assert_eq!(tensor1, tensor2);

        // 不同的元素类型
        let tensor3 = Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![2, 3],
        };
        assert_ne!(tensor1, tensor3);

        // 不同的形状
        let tensor4 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![3, 2],
        };
        assert_ne!(tensor1, tensor4);

        // 嵌套 Tensor 比较
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

        // 不同深度的嵌套
        let nested3 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        };
        assert_ne!(nested1, nested3);
    }

    /// Test 7: Attribute 的各种边界值
    #[test]
    fn test_attribute_boundary_values() {
        // 整数边界值
        let max_int = Attribute::Int(i64::MAX);
        let min_int = Attribute::Int(i64::MIN);
        let zero_int = Attribute::Int(0);

        match max_int {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected i64::MAX"),
        }

        match min_int {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected i64::MIN"),
        }

        match zero_int {
            Attribute::Int(val) => assert_eq!(val, 0),
            _ => panic!("Expected 0"),
        }

        // 浮点数特殊值
        let nan_attr = Attribute::Float(f64::NAN);
        let pos_inf = Attribute::Float(f64::INFINITY);
        let neg_inf = Attribute::Float(f64::NEG_INFINITY);

        match nan_attr {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected NaN"),
        }

        match pos_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected positive infinity"),
        }

        match neg_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
            _ => panic!("Expected negative infinity"),
        }

        // 空字符串
        let empty_str = Attribute::String("".to_string());
        match empty_str {
            Attribute::String(s) => assert_eq!(s.len(), 0),
            _ => panic!("Expected empty string"),
        }

        // 空数组
        let empty_array = Attribute::Array(vec![]);
        match empty_array {
            Attribute::Array(arr) => assert_eq!(arr.len(), 0),
            _ => panic!("Expected empty array"),
        }
    }

    /// Test 8: Module、Operation 和 Value 的克隆行为
    #[test]
    fn test_clone_behavior() {
        // 测试 Module 克隆
        let mut original_module = Module::new("original");
        original_module.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        original_module.outputs.push(Value {
            name: "output".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        let mut op = Operation::new("add");
        op.attributes.insert("key".to_string(), Attribute::Int(42));
        original_module.add_operation(op);

        let cloned_module = original_module.clone();
        assert_eq!(original_module.name, cloned_module.name);
        assert_eq!(original_module.inputs.len(), cloned_module.inputs.len());
        assert_eq!(original_module.outputs.len(), cloned_module.outputs.len());
        assert_eq!(original_module.operations.len(), cloned_module.operations.len());

        // 测试 Operation 克隆
        let mut original_op = Operation::new("original_op");
        original_op.inputs.push(Value {
            name: "op_input".to_string(),
            ty: Type::I32,
            shape: vec![5],
        });
        original_op.outputs.push(Value {
            name: "op_output".to_string(),
            ty: Type::I32,
            shape: vec![5],
        });
        original_op.attributes.insert("attr".to_string(), Attribute::Float(1.5));

        let cloned_op = original_op.clone();
        assert_eq!(original_op.op_type, cloned_op.op_type);
        assert_eq!(original_op.inputs.len(), cloned_op.inputs.len());
        assert_eq!(original_op.outputs.len(), cloned_op.outputs.len());
        assert_eq!(original_op.attributes.len(), cloned_op.attributes.len());

        // 测试 Value 克隆
        let original_value = Value {
            name: "test_value".to_string(),
            ty: Type::F64,
            shape: vec![2, 2, 2],
        };
        let cloned_value = original_value.clone();
        assert_eq!(original_value, cloned_value);
    }

    /// Test 9: Module 的命名空间和操作唯一性
    #[test]
    fn test_module_namespace_operation_uniqueness() {
        let mut module = Module::new("namespace_test");

        // 添加多个相同类型的操作
        for i in 0..5 {
            let mut op = Operation::new("same_type_op");
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            module.add_operation(op);
        }

        assert_eq!(module.operations.len(), 5);

        // 验证所有操作的类型相同
        for op in &module.operations {
            assert_eq!(op.op_type, "same_type_op");
        }

        // 添加不同类型的操作
        let mut different_op = Operation::new("different_type_op");
        different_op.inputs.push(Value {
            name: "unique_input".to_string(),
            ty: Type::I64,
            shape: vec![2],
        });
        module.add_operation(different_op);

        assert_eq!(module.operations.len(), 6);
        assert_eq!(module.operations[5].op_type, "different_type_op");
    }

    /// Test 10: 嵌套属性的深度和复杂性
    #[test]
    fn test_nested_attributes_depth_and_complexity() {
        // 创建深度嵌套的数组属性
        let deeply_nested = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(1),
                    Attribute::Int(2),
                ]),
                Attribute::Array(vec![
                    Attribute::Int(3),
                    Attribute::Int(4),
                ]),
            ]),
            Attribute::Array(vec![
                Attribute::Float(1.0),
                Attribute::Float(2.0),
            ]),
        ]);

        match deeply_nested {
            Attribute::Array(level1) => {
                assert_eq!(level1.len(), 2);

                // 验证第一个元素是嵌套数组
                match &level1[0] {
                    Attribute::Array(level2) => {
                        assert_eq!(level2.len(), 2);

                        // 验证第二层嵌套
                        match &level2[0] {
                            Attribute::Array(level3) => {
                                assert_eq!(level3.len(), 2);
                                match &level3[0] {
                                    Attribute::Int(1) => {},
                                    _ => panic!("Expected Int(1)"),
                                }
                            },
                            _ => panic!("Expected third level array"),
                        }
                    },
                    _ => panic!("Expected second level array"),
                }

                // 验证第二个元素
                match &level1[1] {
                    Attribute::Array(arr) => {
                        assert_eq!(arr.len(), 2);
                        match arr[0] {
                            Attribute::Float(val) if (val - 1.0).abs() < f64::EPSILON => {},
                            _ => panic!("Expected Float(1.0)"),
                        }
                    },
                    _ => panic!("Expected array"),
                }
            },
            _ => panic!("Expected top-level array"),
        }
    }
}
