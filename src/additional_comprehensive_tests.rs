//! Additional comprehensive tests for edge cases in the Impulse compiler

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
    
    // Test 1: Empty collections and boundary conditions
    #[test]
    fn test_empty_collections() {
        let module = Module::new("");
        assert_eq!(module.name, "");
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
        
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    // Test 2: Boundary values for integers
    #[test]
    fn test_boundary_integer_values() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("boundary_test");
        let mut attrs = HashMap::new();
        
        // Test minimum and maximum integer values
        attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("max_int".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("zero_int".to_string(), Attribute::Int(0));
        attrs.insert("negative_one".to_string(), Attribute::Int(-1));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.get("min_int"), Some(&Attribute::Int(i64::MIN)));
        assert_eq!(op.attributes.get("max_int"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("zero_int"), Some(&Attribute::Int(0)));
        assert_eq!(op.attributes.get("negative_one"), Some(&Attribute::Int(-1)));
    }

    // Test 3: Edge cases for floating point values
    #[test]
    fn test_floating_point_edge_cases() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("float_test");
        let mut attrs = HashMap::new();
        
        // Test special floating point values
        attrs.insert("infinity".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("neg_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("zero".to_string(), Attribute::Float(0.0));
        attrs.insert("neg_zero".to_string(), Attribute::Float(-0.0));
        attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
        
        op.attributes = attrs;
        
        // Check non-NaN values
        if let Some(Attribute::Float(inf_val)) = op.attributes.get("infinity") {
            assert!(inf_val.is_infinite() && inf_val.is_sign_positive());
        }
        
        if let Some(Attribute::Float(neg_inf_val)) = op.attributes.get("neg_infinity") {
            assert!(neg_inf_val.is_infinite() && neg_inf_val.is_sign_negative());
        }
        
        if let Some(Attribute::Float(zero_val)) = op.attributes.get("zero") {
            assert!(*zero_val == 0.0);
        }
        
        if let Some(Attribute::Float(neg_zero_val)) = op.attributes.get("neg_zero") {
            assert!(*neg_zero_val == -0.0);
        }
        
        // Check for NaN specially
        if let Some(Attribute::Float(nan_val)) = op.attributes.get("nan") {
            assert!(nan_val.is_nan());
        }
    }

    // Test 4: Extremely long strings
    #[test]
    fn test_extremely_long_strings() {
        let long_name = "x".repeat(100_000); // 100k character string
        
        let value = Value {
            name: long_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name, long_name);
        assert_eq!(value.name.len(), 100_000);
        
        let long_op_name = "op_".repeat(50_000);
        let op = Operation::new(&long_op_name);
        assert_eq!(op.op_type, long_op_name);
    }

    // Test 5: Unicode and special character handling
    #[test]
    fn test_unicode_special_characters() {
        let unicode_names = [
            "αβγδε",           // Greek letters
            "测试数据",          // Chinese characters
            "тест",            // Cyrillic
            "🚀🔥✨",           // Emoji
            "São Paulo",       // Accented characters
            "Москва",          // Cyrillic with accents
            "مumbai",          // Arabic script
        ];
        
        for name in &unicode_names {
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, Type::F32);
            assert_eq!(value.shape, vec![1]);
        }
    }

    // Test 6: Zero-sized and empty tensors with different configurations
    #[test]
    fn test_zero_sized_tensor_configurations() {
        let test_cases = vec![
            (vec![], 1),           // Scalar: 0 dimensions = 1 element
            (vec![0], 0),          // One dim with 0 = 0 elements
            (vec![0, 5], 0),       // Contains 0 = 0 elements
            (vec![5, 0], 0),       // Contains 0 = 0 elements
            (vec![1, 0, 1], 0),    // Contains 0 in middle = 0 elements
            (vec![0, 0, 0], 0),    // All zeros = 0 elements
            (vec![1, 1, 1], 1),    // All ones = 1 element
            (vec![2, 3, 4], 24),   // Normal case = 24 elements
        ];
        
        for (shape, expected_count) in test_cases {
            let value = Value {
                name: "test_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            let actual_count: usize = value.shape.iter().product();
            assert_eq!(actual_count, expected_count, "Failed for shape {:?}", shape);
        }
    }

    // Test 7: Large but valid tensor shapes
    #[test]
    fn test_large_valid_tensor_shapes() {
        // Test tensor shapes that are large but still reasonable
        let value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000], // 1 million elements
        };
        
        assert_eq!(value.shape, vec![1000, 1000]);
        let element_count: usize = value.shape.iter().product();
        assert_eq!(element_count, 1_000_000);
        
        // Test very wide tensor
        let wide_value = Value {
            name: "wide_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, 10_000_000], // 1 x 10M
        };
        
        assert_eq!(wide_value.shape, vec![1, 10_000_000]);
        let wide_count: usize = wide_value.shape.iter().product();
        assert_eq!(wide_count, 10_000_000);
    }

    // Test 8: Complex nested structure with many elements
    #[test]
    fn test_complex_nested_structure() {
        use std::collections::HashMap;
        
        let mut module = Module::new("complex_module");
        
        // Create an operation with complex nested structure
        let mut op = Operation::new("complex_op");
        
        // Add multiple inputs with different shapes
        for i in 0..100 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![i + 1, i + 2],
            });
        }
        
        // Add multiple outputs
        for i in 0..50 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: if i % 3 == 0 { Type::F64 } else { Type::I64 },
                shape: vec![i + 1],
            });
        }
        
        // Add multiple attributes
        let mut attrs = HashMap::new();
        for i in 0..200 {
            attrs.insert(
                format!("attr_{}", i),
                if i % 3 == 0 {
                    Attribute::Int(i as i64)
                } else if i % 3 == 1 {
                    Attribute::Float(i as f64 * 0.5)
                } else {
                    Attribute::String(format!("str_{}", i))
                }
            );
        }
        op.attributes = attrs;
        
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].inputs.len(), 100);
        assert_eq!(module.operations[0].outputs.len(), 50);
        assert_eq!(module.operations[0].attributes.len(), 200);
    }

    // Test 9: Type validity checks
    #[test]
    fn test_type_validity_checks() {
        // Valid simple types
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());
        
        // Valid nested types
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![10, 20],
        };
        assert!(nested_type.is_valid_type());
        
        // Deeply nested valid type
        let deep_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![5],
            }),
            shape: vec![3, 3],
        };
        assert!(deep_type.is_valid_type());
    }

    // Test 10: Cloning and equality for complex nested types
    #[test]
    fn test_complex_type_clone_equality() {
        let complex_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![2, 3],
            }),
            shape: vec![4, 5, 6],
        };
        
        // Test cloning
        let cloned_type = complex_type.clone();
        assert_eq!(complex_type, cloned_type);
        
        // Test that different types are not equal
        let different_type = Type::Tensor {
            element_type: Box::new(Type::F64), // Different element type structure
            shape: vec![4, 5, 6],
        };
        
        assert_ne!(complex_type, different_type);
        
        // Test with same structure but different shape
        let different_shape = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![2, 3],
            }),
            shape: vec![7, 8, 9], // Different shape
        };
        
        assert_ne!(complex_type, different_shape);
    }

    // Test 11: Type conversion/serialization edge cases
    #[test]
    fn test_serialization_deserialization() {
        use serde_json;
        
        // Test serializing and deserializing complex structures
        let original_value = Value {
            name: "serialized_value".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        
        // Serialize to JSON
        let serialized = serde_json::to_string(&original_value).unwrap();
        
        // Deserialize back
        let deserialized: Value = serde_json::from_str(&serialized).unwrap();
        
        // Check that they're equal
        assert_eq!(original_value, deserialized);
        assert_eq!(original_value.name, deserialized.name);
        assert_eq!(original_value.ty, deserialized.ty);
        assert_eq!(original_value.shape, deserialized.shape);
        
        // Test with complex nested type
        let complex_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![2],
            }),
            shape: vec![3, 4],
        };
        
        let serialized_complex = serde_json::to_string(&complex_type).unwrap();
        let deserialized_complex: Type = serde_json::from_str(&serialized_complex).unwrap();
        
        assert_eq!(complex_type, deserialized_complex);
    }

    // Test 12: Module operations management edge cases
    #[test]
    fn test_module_operations_edge_cases() {
        let mut module = Module::new("operations_test");
        
        // Test adding operations in bulk
        for i in 0..1000 {
            let op = Operation::new(&format!("bulk_op_{}", i));
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 1000);
        
        // Test that first and last operations are as expected
        assert_eq!(module.operations[0].op_type, "bulk_op_0");
        assert_eq!(module.operations[999].op_type, "bulk_op_999");
        
        // Clear operations and test empty state
        let empty_module = Module::new("still_has_name");
        assert_eq!(empty_module.name, "still_has_name");
        assert!(empty_module.operations.is_empty());
    }

    // Test 13: Recursive type validation with maximum nesting
    #[test]
    fn test_recursive_type_validation_limits() {
        // Test creating a moderately deeply nested type to verify it works
        let mut current_type = Type::F32;
        
        // Create nested type with 50 levels (reasonable for testing without stack overflow)
        for i in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 5 + 1], // Cycle through shapes 1-5
            };
        }
        
        // Validate that the deeply nested type is still valid
        assert!(current_type.is_valid_type());
        
        // Test cloning of deeply nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
        
        // Validate the cloned one too
        assert!(cloned.is_valid_type());
    }
}

#[cfg(test)]
mod rstest_additional_tests {
    use rstest::rstest;
    use crate::ir::{Value, Type, Attribute};

    // Parametrized test for different basic types
    #[rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    fn test_basic_types(#[case] basic_type: Type) {
        let value = Value {
            name: "test_basic".to_string(),
            ty: basic_type.clone(),
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.ty, basic_type);
        assert_eq!(value.shape, vec![1, 2, 3]);
        assert_eq!(value.name, "test_basic");
    }

    // Parametrized test for different zero-dimension configurations
    #[rstest]
    #[case(vec![], 1)]      // Scalar
    #[case(vec![0], 0)]     // One zero dim
    #[case(vec![5, 0], 0)]  // Contains zero
    #[case(vec![10], 10)]   // Single dimension
    fn test_shape_products(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let value = Value {
            name: "shape_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let actual_product: usize = value.shape.iter().product();
        assert_eq!(actual_product, expected_product);
    }

    // Parametrized test for attribute types
    #[rstest]
    #[case(Attribute::Int(42))]
    #[case(Attribute::Float(3.14))]
    #[case(Attribute::String("test".to_string()))]
    #[case(Attribute::Bool(true))]
    #[case(Attribute::Bool(false))]
    fn test_attribute_types(#[case] original_attr: Attribute) {
        // Test that each attribute can be matched and compared properly
        match &original_attr {
            Attribute::Int(v) => assert!(*v == 42),  // Only the Int(42) case will pass
            Attribute::Float(v) => assert!((*v - 3.14).abs() < f64::EPSILON),  // Only the 3.14 case will pass
            Attribute::String(s) => assert!(s == "test"),  // Only the "test" case will pass
            Attribute::Bool(b) => assert!(*b == true || *b == false),  // Both true and false cases will pass
            Attribute::Array(_) => {}  // This case won't be triggered by our test cases
        }
        
        // More importantly, test that it can be cloned and compared to itself
        let cloned = original_attr.clone();
        assert_eq!(original_attr, cloned);
    }
}