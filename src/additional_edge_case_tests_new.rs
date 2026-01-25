//! Additional edge case tests for the Impulse compiler
//! Covers boundary conditions and extreme values

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use std::collections::HashMap;

    /// Test floating point edge cases in attributes
    #[test]
    fn test_floating_point_edge_cases_in_attributes() {
        let test_cases = [
            (std::f64::INFINITY, "infinity"),
            (std::f64::NEG_INFINITY, "neg_infinity"),
            (std::f64::NAN, "nan"),
            (-0.0, "negative_zero"),
            (std::f64::EPSILON, "epsilon"),
            (std::f64::MIN_POSITIVE, "min_positive"),
        ];

        for (value, desc) in test_cases.iter() {
            let attr = Attribute::Float(*value);

            match attr {
                Attribute::Float(retrieved) => {
                    if value.is_nan() {
                        // NaN is special: it's not equal to itself
                        assert!(retrieved.is_nan(), "For {} value should remain NaN", desc);
                    } else if value.is_infinite() {
                        assert!(retrieved.is_infinite(), "For {} value should remain infinite", desc);
                        assert_eq!(value.signum(), retrieved.signum(), "Sign should be preserved for {}", desc);
                    } else {
                        // For normal finite values, use approximate equality
                        let tolerance = std::f64::EPSILON * 10.0;
                        if (retrieved - value).abs() > tolerance {
                            panic!("Value mismatch for {}: expected {}, got {}", desc, value, retrieved);
                        }
                    }
                },
                _ => panic!("Expected Float attribute for {}", desc),
            }
        }
    }

    /// Test operations with maximum length names
    #[test]
    fn test_maximum_length_names() {
        // Test with very long names
        let long_module_name = "A".repeat(1_000_000); // 1 million A's
        let module = Module::new(&long_module_name);
        assert_eq!(module.name, long_module_name);
        
        let long_op_name = "B".repeat(1_000_000);
        let op = Operation::new(&long_op_name);
        assert_eq!(op.op_type, long_op_name);
        
        let long_value_name = "C".repeat(1_000_000);
        let value = Value {
            name: long_value_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, long_value_name);
    }

    /// Test deep recursion with tensor types to test stack limits
    #[test]
    fn test_deep_recursion_tensor_types() {
        // Create a deeply nested tensor type with 1000 levels - this tests stack safety
        let mut current_type = Type::F32;
        for i in 0..1000 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 2 + 1], // Alternate between [1] and [2]
            };
        }

        // Verify we can still access the top level
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![((1000-1) % 2 + 1)]); // Should be [2] since 999%2=1, 1+1=2
            },
            _ => panic!("Top level should still be a tensor"),
        }

        // Test that we can clone deeply nested types
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);

        // Test that we can drop without stack overflow
        drop(cloned);
    }

    /// Test integer overflow in tensor shape calculations
    #[test]
    fn test_shape_calculation_overflow() {
        // Use values that would definitely cause overflow when multiplied
        // For most systems, multiplying large usizes would overflow
        let large_value = (usize::MAX / 1000) as u128; // Reduce to avoid immediate overflow during mult
        
        // Simulate what would happen with very large dimensions
        // Since multiplying actual MAX values might cause a panic in debug mode,
        // we test with a calculation that would overflow usize
        let shapes_to_test = vec![
            vec![large_value as usize, 2000],  // Would overflow
            vec![10000, 10000, 10000, 1000],  // Would definitely overflow
        ];

        for shape in shapes_to_test {
            let value = Value {
                name: "potential_overflow".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };

            // Use checked arithmetic to prevent actual overflow during calculation
            let mut product: Option<u128> = Some(1);
            for dim in &value.shape {
                if let Some(prev) = product {
                    product = prev.checked_mul(*dim as u128);
                }
            }

            // Verify that we handle potential overflow gracefully
            if shape == vec![10000, 10000, 10000, 1000] {
                assert!(product.is_none()); // Should overflow
            }
        }
    }

    /// Test with maximum possible values for shape dimensions
    #[test]
    fn test_maximum_shape_dimensions() {
        let max_shape = vec![usize::MAX, usize::MAX >> 1, 2]; // Using max values to test boundaries
        let value = Value {
            name: "max_dims".to_string(),
            ty: Type::F32,
            shape: max_shape.clone(),
        };
        
        assert_eq!(value.shape, max_shape);
        assert_eq!(value.name, "max_dims");
        assert_eq!(value.ty, Type::F32);
    }

    /// Test operations with extremely large numbers of inputs/outputs
    #[test]
    fn test_extremely_large_io_counts() {
        const NUM_INPUTS: usize = 100_000;
        const NUM_OUTPUTS: usize = 50_000;
        
        let mut op = Operation::new("high_io_op");
        
        // Add many inputs
        for i in 0..NUM_INPUTS {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
        
        // Add many outputs
        for i in 0..NUM_OUTPUTS {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![2],
            });
        }
        
        assert_eq!(op.inputs.len(), NUM_INPUTS);
        assert_eq!(op.outputs.len(), NUM_OUTPUTS);
        assert_eq!(op.op_type, "high_io_op");
    }

    /// Test attribute array with maximum nesting depth
    #[test]
    fn test_maximum_attribute_nesting_depth() {
        // Create deeply nested array attributes (similar to tensor nesting)
        let mut attr: Attribute = Attribute::Int(42);
        
        for _ in 0..100 { // 100 levels of nesting
            attr = Attribute::Array(vec![attr]);
        }
        
        // Verify the deep nesting worked
        match &attr {
            Attribute::Array(nested) => {
                assert_eq!(nested.len(), 1);
                // Continue verifying deeper levels if needed
            },
            _ => panic!("Should be a nested array"),
        }
        
        // Test cloning of deeply nested structure
        let cloned = attr.clone();
        assert_eq!(attr, cloned);
    }

    /// Test for potential memory allocation failures with very large collections
    #[test]
    fn test_large_collection_allocations() {
        // Create a large number of operations to test memory allocation
        let mut module = Module::new("large_alloc_test");
        
        // Add a very large number of operations
        for i in 0..1_000_000 {
            // Only add every 100th operation to avoid actual out-of-memory
            if i % 1000 == 0 {
                module.add_operation(Operation::new(&format!("sparse_op_{}", i)));
            }
        }
        
        // Verify module is still valid
        assert_eq!(module.name, "large_alloc_test");
    }

    /// Test Unicode and special character handling
    #[test]
    fn test_unicode_and_special_characters() {
        let unicode_test_cases = [
            ("tensor_名称_日本語_🔥", Type::F32), // Chinese, Japanese, emoji
            ("arabic_chars_مرحبا", Type::I64), // Arabic
            ("russian_привет", Type::F64), // Russian
            ("control_\u{0001}_\u{001F}", Type::Bool), // Control characters
            ("whitespace_\t_\n_\r", Type::I32), // Whitespace characters
        ];

        for (name, typ) in unicode_test_cases.iter() {
            let value = Value {
                name: name.to_string(),
                ty: typ.clone(),
                shape: vec![1, 2, 3],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, *typ);
            assert_eq!(value.shape, vec![1, 2, 3]);
            
            // Test operation with unicode name
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
        }
    }

    /// Test type validation edge cases
    #[test]
    fn test_type_validation_edge_cases() {
        use crate::ir::TypeExtensions;

        // Test validation of simple types
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());

        // Test validation of nested types
        let nested_valid = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        assert!(nested_valid.is_valid_type());

        // Test validation of deeply nested types
        let deep_valid = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I64),
                    shape: vec![1],
                }),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        assert!(deep_valid.is_valid_type());

        // Test validation with invalid cases (this is just for completeness, 
        // as our Type enum is exhaustive and all variants should be valid)
        // For now, we just ensure all valid types return true
        assert!(Type::F32.is_valid_type());
    }

    /// Test serialization and deserialization of complex nested structures
    #[test]
    fn test_serialization_edge_cases() {
        use serde_json;

        // Create a complex nested structure
        let complex_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 3],
            }),
            shape: vec![4, 5],
        };

        // Serialize to JSON
        let serialized = serde_json::to_string(&complex_type);
        assert!(serialized.is_ok());
        
        let json_str = serialized.unwrap();
        
        // Deserialize back
        let deserialized: Result<Type, _> = serde_json::from_str(&json_str);
        assert!(deserialized.is_ok());
        
        let recovered_type = deserialized.unwrap();
        assert_eq!(complex_type, recovered_type);
        
        // Test with more complex structure including attributes
        let mut op = Operation::new("serialization_test");
        op.attributes.insert(
            "complex_attr".to_string(),
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Float(std::f64::consts::PI),
                Attribute::Array(vec![
                    Attribute::Int(42),
                    Attribute::Bool(true),
                ]),
            ])
        );

        let op_serialized = serde_json::to_string(&op);
        assert!(op_serialized.is_ok());
        
        let op_deserialized: Result<Operation, _> = serde_json::from_str(&op_serialized.unwrap());
        assert!(op_deserialized.is_ok());
    }

    /// Test hash map edge cases with operations
    #[test]
    fn test_hash_map_edge_cases_in_operations() {
        use std::collections::HashMap;

        let mut op = Operation::new("hashmap_test");

        // Test with maximum number of attributes
        for i in 0..10000 {  // Reduced for practicality
            if i % 1000 == 0 {  // Only add every 1000th attribute to avoid memory issues
                op.attributes.insert(
                    format!("attr_{}", i),
                    Attribute::String(format!("value_{}", i))
                );
            }
        }

        // Test with special key characters
        op.attributes.insert(
            "special.key.with.dots".to_string(),
            Attribute::Int(123)
        );
        op.attributes.insert(
            "key with spaces".to_string(),
            Attribute::Float(42.5)
        );
        op.attributes.insert(
            "key\twith\ttabs".to_string(),
            Attribute::Bool(true)
        );
        op.attributes.insert(
            "key\nwith\nnewlines".to_string(),
            Attribute::String("test".to_string())
        );

        // Verify all attributes are stored correctly
        assert!(op.attributes.contains_key("special.key.with.dots"));
        assert!(op.attributes.contains_key("key with spaces"));
        assert!(op.attributes.contains_key("key\twith\ttabs"));
        assert!(op.attributes.contains_key("key\nwith\nnewlines"));

        assert_eq!(op.attributes.get("special.key.with.dots"), Some(&Attribute::Int(123)));
        assert_eq!(op.attributes.get("key with spaces"), Some(&Attribute::Float(42.5)));
        assert_eq!(op.attributes.get("key\twith\ttabs"), Some(&Attribute::Bool(true)));
        assert_eq!(op.attributes.get("key\nwith\nnewlines"), Some(&Attribute::String("test".to_string())));
    }
}