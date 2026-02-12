/// Fundamental safety boundary tests - 基础安全边界测试
/// 覆盖编译器核心组件的关键安全边界情况
/// 使用标准库 assert! 和 assert_eq! 宏

#[cfg(test)]
mod tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};

    /// Test 1: 整数溢出保护 - Value::num_elements() 使用 checked_mul
    #[test]
    fn test_num_elements_overflow_protection() {
        // 创建一个会导致溢出的 shape
        let overflow_value = Value {
            name: "overflow_tensor".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2], // 这会触发溢出检测
        };
        
        // 应该返回 None 而不是 panic
        assert_eq!(overflow_value.num_elements(), None);
    }

    /// Test 2: 多维度零值处理
    #[test]
    fn test_multiple_zero_dimensions() {
        // 测试多个维度中包含零的情况
        let test_cases = vec![
            vec![0, 0, 0],
            vec![10, 0, 20, 0],
            vec![0, 5, 0],
            vec![1, 2, 0, 4, 5],
        ];

        for shape in test_cases {
            let value = Value {
                name: "zero_dim_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            // 任何包含零的维度都应该返回 0 个元素
            assert_eq!(value.num_elements(), Some(0));
        }
    }

    /// Test 3: 标量类型的处理
    #[test]
    fn test_scalar_type_handling() {
        // 空的 shape 代表标量
        let scalar = Value {
            name: "scalar_value".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        
        // 标量应该有 1 个元素
        assert_eq!(scalar.num_elements(), Some(1));
        assert!(scalar.shape.is_empty());
    }

    /// Test 4: 特殊浮点值 - NaN
    #[test]
    fn test_nan_float_attribute() {
        let nan_attr = Attribute::Float(f64::NAN);
        
        match nan_attr {
            Attribute::Float(val) => {
                // NaN 不等于任何值，包括它自己
                assert!(val.is_nan());
                assert_ne!(val, val);
            }
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 5: 特殊浮点值 - 无穷大
    #[test]
    fn test_infinity_float_attributes() {
        let pos_inf = Attribute::Float(f64::INFINITY);
        let neg_inf = Attribute::Float(f64::NEG_INFINITY);
        
        match pos_inf {
            Attribute::Float(val) => {
                assert!(val.is_infinite());
                assert!(val.is_sign_positive());
            }
            _ => panic!("Expected positive infinity"),
        }
        
        match neg_inf {
            Attribute::Float(val) => {
                assert!(val.is_infinite());
                assert!(val.is_sign_negative());
            }
            _ => panic!("Expected negative infinity"),
        }
    }

    /// Test 6: 最小正浮点数
    #[test]
    fn test_min_positive_float() {
        let min_pos = Attribute::Float(f64::MIN_POSITIVE);
        
        match min_pos {
            Attribute::Float(val) => {
                // 验证这是最小的正归一化浮点数
                assert!(val > 0.0);
                assert!(val / 2.0 > 0.0); // 仍然为正
                assert!(val / 2.0 < val);  // 比原值小
            }
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 7: 空模块的验证
    #[test]
    fn test_empty_module() {
        let module = Module::new("empty_module");
        
        assert_eq!(module.name, "empty_module");
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.inputs.len(), 0);
        assert_eq!(module.outputs.len(), 0);
    }

    /// Test 8: 操作属性的类型验证
    #[test]
    fn test_operation_attribute_types() {
        let mut op = Operation::new("test_op");
        
        // 插入各种类型的属性
        op.attributes.insert("int_attr".to_string(), Attribute::Int(42));
        op.attributes.insert("float_attr".to_string(), Attribute::Float(3.14159));
        op.attributes.insert("string_attr".to_string(), Attribute::String("hello".to_string()));
        op.attributes.insert("bool_attr".to_string(), Attribute::Bool(true));
        
        assert_eq!(op.attributes.len(), 4);
        
        // 验证每个属性的类型
        match op.attributes.get("int_attr") {
            Some(Attribute::Int(42)) => {}
            _ => panic!("Expected Int(42)"),
        }
        
        match op.attributes.get("float_attr") {
            Some(Attribute::Float(val)) if (val - 3.14159).abs() < 1e-6 => {}
            _ => panic!("Expected Float(3.14159)"),
        }
        
        match op.attributes.get("string_attr") {
            Some(Attribute::String(s)) if s == "hello" => {}
            _ => panic!("Expected String(\"hello\")"),
        }
        
        match op.attributes.get("bool_attr") {
            Some(Attribute::Bool(true)) => {}
            _ => panic!("Expected Bool(true)"),
        }
    }

    /// Test 9: 深度嵌套的 Tensor 类型
    #[test]
    fn test_deeply_nested_tensor_type() {
        // 创建三层嵌套的 Tensor 类型
        let innermost = Type::F32;
        let level1 = Type::Tensor {
            element_type: Box::new(innermost),
            shape: vec![2],
        };
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![3],
        };
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![4],
        };
        
        // 验证最内层类型是 F32
        match level3 {
            Type::Tensor { element_type: e1, shape: s1 } => {
                assert_eq!(*s1, vec![4]);
                match e1.as_ref() {
                    Type::Tensor { element_type: e2, shape: s2 } => {
                        assert_eq!(*s2, vec![3]);
                        match e2.as_ref() {
                            Type::Tensor { element_type: e3, shape: s3 } => {
                                assert_eq!(*s3, vec![2]);
                                assert_eq!(e3.as_ref(), &Type::F32);
                            }
                            _ => panic!("Expected Tensor at level 2"),
                        }
                    }
                    _ => panic!("Expected Tensor at level 1"),
                }
            }
            _ => panic!("Expected Tensor at outer level"),
        }
    }

    /// Test 10: 极值整数属性
    #[test]
    fn test_extreme_integer_attributes() {
        let max_int = Attribute::Int(i64::MAX);
        let min_int = Attribute::Int(i64::MIN);
        let zero = Attribute::Int(0);
        
        match max_int {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected i64::MAX"),
        }
        
        match min_int {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected i64::MIN"),
        }
        
        match zero {
            Attribute::Int(val) => assert_eq!(val, 0),
            _ => panic!("Expected 0"),
        }
    }
}
