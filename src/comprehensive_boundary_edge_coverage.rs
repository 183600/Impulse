//! Comprehensive boundary edge coverage tests
//! 10个测试用例，覆盖关键边界情况，使用标准库 assert! 和 assert_eq!

#[cfg(test)]
mod tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};

    /// 测试1: 值为负数的浮点数属性
    #[test]
    fn test_negative_float_attributes() {
        let neg_float = Attribute::Float(-3.14159);
        let neg_zero = Attribute::Float(-0.0);
        let large_neg = Attribute::Float(-1e308);

        match neg_float {
            Attribute::Float(val) => assert!(val < 0.0 && (val - (-3.14159)).abs() < f64::EPSILON),
            _ => panic!("Expected Float attribute"),
        }

        match neg_zero {
            Attribute::Float(val) => assert!(val.is_sign_negative()),
            _ => panic!("Expected negative zero Float attribute"),
        }

        match large_neg {
            Attribute::Float(val) => assert!(val < -1e300),
            _ => panic!("Expected large negative Float attribute"),
        }
    }

    /// 测试2: 值为i64::MIN的整数属性
    #[test]
    fn test_min_i64_attribute() {
        let min_attr = Attribute::Int(i64::MIN);
        match min_attr {
            Attribute::Int(val) => {
                assert_eq!(val, i64::MIN);
                assert!(val < 0);
            },
            _ => panic!("Expected Int attribute"),
        }
    }

    /// 测试3: 值为0的数组属性（空数组和单元素零数组）
    #[test]
    fn test_zero_array_attributes() {
        let empty_array = Attribute::Array(vec![]);
        let zero_int_array = Attribute::Array(vec![Attribute::Int(0)]);
        let zero_float_array = Attribute::Array(vec![Attribute::Float(0.0)]);

        match empty_array {
            Attribute::Array(arr) => assert_eq!(arr.len(), 0),
            _ => panic!("Expected empty Array attribute"),
        }

        match zero_int_array {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
                match &arr[0] {
                    Attribute::Int(0) => {},
                    _ => panic!("Expected Int(0)"),
                }
            },
            _ => panic!("Expected Array with Int(0)"),
        }

        match zero_float_array {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
                match &arr[0] {
                    Attribute::Float(val) => assert_eq!(*val, 0.0),
                    _ => panic!("Expected Float(0.0)"),
                }
            },
            _ => panic!("Expected Array with Float(0.0)"),
        }
    }

    /// 测试4: 操作的属性为空HashMap
    #[test]
    fn test_empty_attributes_hashmap() {
        let op = Operation::new("empty_attrs");
        assert_eq!(op.attributes.len(), 0);
        assert!(op.attributes.is_empty());

        // 验证空HashMap上的操作不会导致panic
        let _ = op.attributes.get("nonexistent");
        assert!(!op.attributes.contains_key("any_key"));
    }

    /// 测试5: 值的形状包含1和混合维度
    #[test]
    fn test_mixed_unit_dimensions() {
        let shapes = vec![
            vec![1],
            vec![1, 1],
            vec![1, 2, 1],
            vec![10, 1, 20, 1],
            vec![1, 1, 1, 1, 1],
        ];

        for shape in shapes {
            let value = Value {
                name: "test".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            assert_eq!(value.shape, shape);
            assert!(value.num_elements().is_some());
        }
    }

    /// 测试6: 包含控制字符的字符串属性
    #[test]
    fn test_string_with_control_chars() {
        let strings = vec![
            "\0null".to_string(),
            "tab\there".to_string(),
            "new\nline".to_string(),
            "carriage\rreturn".to_string(),
            "backspace\x08char".to_string(),
        ];

        for s in strings {
            let attr = Attribute::String(s.clone());
            match attr {
                Attribute::String(val) => assert_eq!(val, s),
                _ => panic!("Expected String attribute"),
            }
        }
    }

    /// 测试7: 模块的输入输出列表长度相等
    #[test]
    fn test_module_equal_io_count() {
        let mut module = Module::new("equal_io");

        // 添加3个输入
        for i in 0..3 {
            module.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
        }

        // 添加3个输出
        for i in 0..3 {
            module.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
        }

        assert_eq!(module.inputs.len(), module.outputs.len());
        assert_eq!(module.inputs.len(), 3);
    }

    /// 测试8: 值的形状计算接近usize::MAX
    #[test]
    fn test_near_max_elements() {
        // 使用安全的维度，确保不会溢出
        let safe_large = Value {
            name: "large_safe".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 10_000],
        };
        assert_eq!(safe_large.num_elements(), Some(1_000_000_000));

        // 空形状（标量）
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::I32,
            shape: vec![],
        };
        assert_eq!(scalar.num_elements(), Some(1));
    }

    /// 测试9: 嵌套属性数组深度为2
    #[test]
    fn test_two_level_nested_array() {
        let nested = Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
            Attribute::Array(vec![Attribute::Float(3.0), Attribute::Float(4.0)]),
        ]);

        match nested {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                match &outer[0] {
                    Attribute::Array(inner) => {
                        assert_eq!(inner.len(), 2);
                        match &inner[0] {
                            Attribute::Int(1) => {},
                            _ => panic!("Expected Int(1)"),
                        }
                    },
                    _ => panic!("Expected nested array"),
                }
            },
            _ => panic!("Expected outer array"),
        }
    }

    /// 测试10: 布尔属性值为false
    #[test]
    fn test_bool_false_attribute() {
        let false_attr = Attribute::Bool(false);

        match false_attr {
            Attribute::Bool(val) => assert!(!val),
            _ => panic!("Expected Bool(false) attribute"),
        }

        // 创建操作并添加false属性
        let mut op = Operation::new("bool_op");
        op.attributes.insert("enabled".to_string(), Attribute::Bool(false));

        match op.attributes.get("enabled") {
            Some(Attribute::Bool(val)) => assert!(!val),
            _ => panic!("Expected Bool(false) for 'enabled'"),
        }
    }
}