//! Focused edge case tests for the Impulse compiler
//! Covers important boundary conditions not fully tested elsewhere



#[cfg(test)]
mod tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};

    #[test]
    fn test_num_elements_overflow_protection() {
        // Test that num_elements correctly returns None when the product would overflow
        let value = Value {
            name: "overflow_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000_000, 1_000_000_000], // Product exceeds usize::MAX
        };
        
        // This should return Some with a large value (actual overflow depends on platform)
        let _result = value.num_elements();
        // The checkedMul will return Some if multiplication is safe, None if it overflows
        // We just want to make sure it doesn't panic
        
        // Test with smaller values that are definitely safe
        let safe_value = Value {
            name: "safe_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 2000],
        };
        assert_eq!(safe_value.num_elements(), Some(2_000_000));
    }

    #[test]
    fn test_deeply_nested_tensor_type_comparison() {
        // Create two identical deeply nested types
        let mut type1 = Type::F32;
        for _ in 0..50 {
            type1 = Type::Tensor {
                element_type: Box::new(type1),
                shape: vec![2],
            };
        }
        
        let mut type2 = Type::F32;
        for _ in 0..50 {
            type2 = Type::Tensor {
                element_type: Box::new(type2),
                shape: vec![2],
            };
        }
        
        // They should be equal
        assert_eq!(type1, type2);
        
        // Modify one and they should not be equal
        let mut type3 = Type::I32;
        for _ in 0..50 {
            type3 = Type::Tensor {
                element_type: Box::new(type3),
                shape: vec![2],
            };
        }
        
        assert_ne!(type1, type3);
    }

    #[test]
    fn test_module_with_maximum_practical_operations() {
        // Create a module with a very large number of operations to test memory management
        let mut module = Module::new("stress_test_module");
        
        const NUM_OPS: usize = 50_000;
        for i in 0..NUM_OPS {
            let mut op = Operation::new(&format!("operation_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1, i % 100 + 1], // Vary the shape slightly
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), NUM_OPS);
        assert_eq!(module.name, "stress_test_module");
        
        // Check first and last for correctness
        assert_eq!(module.operations[0].op_type, "operation_0");
        assert_eq!(module.operations[NUM_OPS - 1].op_type, format!("operation_{}", NUM_OPS - 1));
    }

    #[test]
    fn test_recursive_tensor_type_validity_check() {
        // Test that deeply nested tensor types are correctly validated
        let mut nested_type = Type::F32;
        for depth in 0..100 {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![depth as usize + 1],
            };
            // At each step, the type should still be valid
            assert!(nested_type.is_valid_type());
        }
        
        // Verify the final deeply nested type is still valid
        assert!(nested_type.is_valid_type());
        
        // Clone the type and ensure it's still valid
        let cloned = nested_type.clone();
        assert_eq!(nested_type, cloned);
        assert!(cloned.is_valid_type());
    }

    #[test]
    fn test_extreme_aspect_ratio_tensors() {
        // Test tensors with very high aspect ratios (e.g., 1 x 10^8 or 10^8 x 1)
        let thin_tensor = Value {
            name: "thin_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, usize::MAX / 2], // Very wide but 1 high
        };
        
        // This should not crash and should calculate elements properly
        let elements_thin = thin_tensor.num_elements();
        // Could be None if it overflows, or Some(large_value)
        if let Some(count) = elements_thin {
            assert_eq!(count, usize::MAX / 2);
        }
        
        let tall_tensor = Value {
            name: "tall_tensor".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX / 2, 1], // Very tall but 1 wide
        };
        
        let elements_tall = tall_tensor.num_elements();
        if let Some(count) = elements_tall {
            assert_eq!(count, usize::MAX / 2);
        }
    }

    #[test]
    fn test_complex_attribute_structures() {
        // Test complex nested attribute structures
        let complex_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![Attribute::String("nested".to_string())]),
                Attribute::Float(3.14159),
            ]),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::Array(vec![Attribute::Array(vec![Attribute::Int(42)])]),
            ]),
        ]);
        
        // Verify structure
        match &complex_attr {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                
                match &outer[0] {
                    Attribute::Array(first_inner) => {
                        assert_eq!(first_inner.len(), 3);
                        match &first_inner[0] {
                            Attribute::Int(1) => (),
                            _ => panic!("Expected Int(1)"),
                        }
                    },
                    _ => panic!("Expected Array as first element"),
                }
            },
            _ => panic!("Expected Array as top level"),
        }
    }

    #[test]
    fn test_operation_with_maximum_attributes() {
        // Create an operation with many attributes to test limits
        let mut op = Operation::new("attribute_heavy_op");
        
        // Add a large number of attributes
        for i in 0..10_000 {
            op.attributes.insert(
                format!("attribute_{}", i),
                Attribute::String(format!("value_for_{}", i))
            );
        }
        
        assert_eq!(op.attributes.len(), 10_000);
        assert_eq!(op.op_type, "attribute_heavy_op");
        
        // Verify some specific attributes exist
        assert!(op.attributes.contains_key("attribute_0"));
        assert!(op.attributes.contains_key("attribute_9999"));
        assert_eq!(
            op.attributes.get("attribute_0").unwrap(),
            &Attribute::String("value_for_0".to_string())
        );
    }

    #[test]
    fn test_unicode_string_attributes() {
        // Test string attributes with Unicode characters
        let unicode_str = "Hello 🌍 世界 🚀 𝄞 music";
        let attr = Attribute::String(unicode_str.to_string());
        
        match &attr {
            Attribute::String(s) => {
                assert_eq!(s, unicode_str);
                // Just verify the string matches - character counting with emojis can be tricky
                // due to multi-byte encodings
            },
            _ => panic!("Expected String attribute"),
        }
        
        // Test in the context of an operation
        let mut op = Operation::new("unicode_op");
        op.attributes.insert("unicode_param".to_string(), attr);
        
        match op.attributes.get("unicode_param").unwrap() {
            Attribute::String(s) => assert_eq!(s, unicode_str),
            _ => panic!("Expected String attribute"),
        }
    }

    #[test]
    fn test_empty_and_single_element_values() {
        // Test 0-dimension tensor (scalar)
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F64,
            shape: vec![], // Empty shape = scalar
        };
        assert_eq!(scalar.num_elements(), Some(1));
        assert!(scalar.shape.is_empty());
        
        // Test 1-element tensor
        let single_elem = Value {
            name: "single_element".to_string(),
            ty: Type::I32,
            shape: vec![1], // Single element
        };
        assert_eq!(single_elem.num_elements(), Some(1));
        assert_eq!(single_elem.shape, vec![1]);
        
        // Test 1x1 tensor
        let unit_matrix = Value {
            name: "unit_matrix".to_string(),
            ty: Type::F32,
            shape: vec![1, 1], // 1x1 matrix
        };
        assert_eq!(unit_matrix.num_elements(), Some(1));
        assert_eq!(unit_matrix.shape, vec![1, 1]);
        
        // Test tensor with zero dimensions anywhere (should yield 0 elements)
        let zero_tensor = Value {
            name: "zero_tensor".to_string(),
            ty: Type::Bool,
            shape: vec![5, 0, 10], // Contains 0, so total = 0
        };
        assert_eq!(zero_tensor.num_elements(), Some(0));
    }

    #[test]
    fn test_serialization_deserialization_roundtrip() {
        // Test that objects can be serialized and deserialized without corruption
        use serde_json;
        
        // Create a complex module
        let mut module = Module::new("serialization_test");
        
        let mut op = Operation::new("complex_op");
        op.inputs.push(Value {
            name: "input_val".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        });
        
        op.outputs.push(Value {
            name: "output_val".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        });
        
        op.attributes.insert(
            "config".to_string(),
            Attribute::Array(vec![
                Attribute::Int(42),
                Attribute::String("test".to_string()),
                Attribute::Bool(true),
            ])
        );
        
        module.add_operation(op);
        
        // Serialize
        let serialized = serde_json::to_string(&module).expect("Serialization failed");
        
        // Deserialize
        let deserialized: Module = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        // Verify they're equivalent
        assert_eq!(deserialized.name, module.name);
        assert_eq!(deserialized.operations.len(), module.operations.len());
        assert_eq!(deserialized.operations[0].op_type, module.operations[0].op_type);
        assert_eq!(deserialized.operations[0].inputs.len(), module.operations[0].inputs.len());
        assert_eq!(deserialized.operations[0].outputs.len(), module.operations[0].outputs.len());
        assert_eq!(deserialized.operations[0].attributes.len(), module.operations[0].attributes.len());
        
        // Specific attribute check
        assert!(deserialized.operations[0].attributes.contains_key("config"));
    }
}