/// Focused critical boundary tests final - 10个关键边界情况测试，使用标准库 assert! 和 assert_eq!
#[cfg(test)]
mod focused_critical_boundary_tests_final {
    use super::*;
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;

    /// Test 1: 边界条件 - Value 包含全零形状，测试空张量
    #[test]
    fn test_value_all_zero_shape() {
        let zero_shape = vec![0, 0, 0];
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: zero_shape.clone(),
        };
        assert_eq!(value.shape, zero_shape);
        assert_eq!(value.num_elements(), Some(0));
    }

    /// Test 2: 边界条件 - Attribute 包含最大/最小 f64 值
    #[test]
    fn test_extreme_float_attributes() {
        let max_float = Attribute::Float(f64::MAX);
        let min_float = Attribute::Float(f64::MIN);
        let epsilon = Attribute::Float(f64::EPSILON);

        match max_float {
            Attribute::Float(val) => {
                assert!(val.is_finite());
                assert_eq!(val, f64::MAX);
            }
            _ => panic!("Expected Float attribute"),
        }

        match min_float {
            Attribute::Float(val) => {
                assert!(val.is_finite());
                assert_eq!(val, f64::MIN);
            }
            _ => panic!("Expected Float attribute"),
        }

        match epsilon {
            Attribute::Float(val) => {
                assert!(val.is_finite());
                assert_eq!(val, f64::EPSILON);
            }
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 3: 边界条件 - Value 包含极大维度（接近 usize::MAX）
    #[test]
    fn test_value_near_max_dimension() {
        // 使用两个相乘会溢出的值
        let value = Value {
            name: "large_dim".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2],
        };
        // 验证会溢出，返回 None
        assert_eq!(value.num_elements(), None);

        // 测试单极大维度不会溢出
        let safe_large = Value {
            name: "safe_large".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX],
        };
        assert_eq!(safe_large.num_elements(), Some(usize::MAX));
    }

    /// Test 4: 边界条件 - Operation 包含空属性 HashMap
    #[test]
    fn test_operation_empty_attributes() {
        let mut op = Operation::new("empty_attrs");
        op.attributes = HashMap::new();
        assert_eq!(op.attributes.len(), 0);
        assert!(op.attributes.is_empty());
    }

    /// Test 5: 边界条件 - Value 包含单维度形状（一维张量）
    #[test]
    fn test_value_single_dimension_shape() {
        let test_shapes = vec![
            vec![1],
            vec![100],
            vec![1_000_000],
            vec![usize::MAX / 10],
        ];

        for shape in test_shapes {
            let value = Value {
                name: "1d_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            assert_eq!(value.shape.len(), 1);
        }
    }

    /// Test 6: 边界条件 - Module 包含极大数量的操作
    #[test]
    fn test_module_many_operations() {
        let mut module = Module::new("many_ops");
        let op_count = 1000;

        for i in 0..op_count {
            let op = Operation::new(&format!("op_{}", i));
            module.add_operation(op);
        }

        assert_eq!(module.operations.len(), op_count);
    }

    /// Test 7: 边界条件 - Attribute Array 包含混合类型的嵌套结构
    #[test]
    fn test_mixed_type_nested_array() {
        let nested = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![Attribute::Int(0)]),
        ]);

        match nested {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 5);
                match &arr[4] {
                    Attribute::Array(inner) => assert_eq!(inner.len(), 1),
                    _ => panic!("Expected nested array"),
                }
            }
            _ => panic!("Expected Array attribute"),
        }
    }

    /// Test 8: 边界条件 - Value 包含重复维度的形状
    #[test]
    fn test_value_repeated_dimensions() {
        let repeated = vec![2, 2, 2, 2, 2];
        let value = Value {
            name: "repeated_dims".to_string(),
            ty: Type::F32,
            shape: repeated.clone(),
        };
        assert_eq!(value.shape, repeated);
        assert_eq!(value.num_elements(), Some(32));
    }

    /// Test 9: 边界条件 - Module 输入输出使用相同 Value 引用模式
    #[test]
    fn test_module_shared_value_pattern() {
        let mut module = Module::new("shared_vals");

        let shared_input = Value {
            name: "shared".to_string(),
            ty: Type::F32,
            shape: vec![10],
        };

        module.inputs.push(shared_input.clone());
        module.outputs.push(shared_input.clone());

        assert_eq!(module.inputs[0].name, "shared");
        assert_eq!(module.outputs[0].name, "shared");
        assert_eq!(module.inputs.len(), 1);
        assert_eq!(module.outputs.len(), 1);
    }

    /// Test 10: 边界条件 - Operation 包含特殊字符的操作类型名称
    #[test]
    fn test_operation_special_characters() {
        let special_names = vec![
            "op_with_underscore",
            "op-with-dash",
            "op.with.dot",
            "op:with:colon",
            "CamelCaseOp",
            "snake_case_op",
        ];

        for name in special_names {
            let op = Operation::new(name);
            assert_eq!(op.op_type, name);
        }
    }
}