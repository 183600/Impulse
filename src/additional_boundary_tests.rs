#[cfg(test)]
mod additional_boundary_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    
    // Test 1: Empty operations with no inputs, outputs, or attributes
    #[test]
    fn test_empty_operation() {
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
        
        // Test with operation that has empty string name
        let op_named = Operation::new(" ");
        assert_eq!(op_named.op_type, " ");
    }

    // Test 2: Maximum usize values in tensor shapes to test boundary conditions
    #[test]
    fn test_max_usize_tensor_shape() {
        let value = Value {
            name: "max_shape".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 1],  // Testing boundary values
        };
        
        assert_eq!(value.shape, vec![usize::MAX, 1]);
        
        // This should not cause overflow when calculating product (but actual product would overflow)
        // Just verify the values are stored correctly
        assert_eq!(value.shape[0], usize::MAX);
        assert_eq!(value.shape[1], 1);
    }

    // Test 3: Nested tensor with maximum depth, testing recursion limits
    #[test]
    fn test_max_depth_nested_tensor() {
        let mut current_type = Type::F32;
        // Create nested type with moderate depth to avoid stack overflow
        for _ in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],
            };
        }
        
        // Verify the type can be created and cloned
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Pattern match to verify structure
        match &current_type {
            Type::Tensor { element_type: _, shape } => {
                assert_eq!(shape, &vec![1]);
            },
            _ => panic!("Expected nested tensor"),
        }
    }

    // Test 4: Operations with maximum possible attribute count
    #[test]
    fn test_operation_with_many_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("many_attrs");
        let mut attrs = HashMap::new();
        
        // Add many attributes to test map behavior
        for i in 0..100_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 100_000);
        
        // Test that a few specific attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_50000"));
        assert!(op.attributes.contains_key("attr_99999"));
        
        // Verify some values
        match op.attributes.get("attr_0") {
            Some(Attribute::String(s)) => assert_eq!(s, "value_0"),
            _ => panic!("Expected string attribute"),
        }
    }

    // Test 5: Parameterized test for different tensor shape extremes
    #[cfg(test)]
    use rstest::rstest;

    #[rstest]
    #[case(vec![], 1)]  // scalar
    #[case(vec![0], 0)] // zero-size
    #[case(vec![1], 1)] // 1-element
    #[case(vec![1, 1, 1, 1], 1)] // multi-dim unit
    #[case(vec![2, 3, 4], 24)] // multi-dim
    #[case(vec![1000, 1000], 1_000_000)] // large but safe
    fn test_shape_product_calculation(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let value = Value {
            name: "test_shape".to_string(),
            ty: Type::F32,
            shape,
        };
        
        let actual_product: usize = value.shape.iter().product();
        assert_eq!(actual_product, expected_product);
    }

    // Test 6: Test floating-point special values in attributes
    #[test]
    fn test_special_float_attributes() {
        let special_attrs = [
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(f64::NAN),
            Attribute::Float(-0.0), // negative zero
            Attribute::Float(f64::EPSILON), // smallest positive
        ];
        
        // Infinity values
        match special_attrs[0] {
            Attribute::Float(f) if f.is_infinite() && f.is_sign_positive() => (),
            _ => panic!("Expected positive infinity"),
        }
        
        match special_attrs[1] {
            Attribute::Float(f) if f.is_infinite() && f.is_sign_negative() => (),
            _ => panic!("Expected negative infinity"),
        }
        
        // NaN value (NaN != NaN, so need special test)
        match special_attrs[2] {
            Attribute::Float(f) if f.is_nan() => (),
            _ => panic!("Expected NaN"),
        }
        
        // Negative zero (equals positive zero in value but different bit representation)
        match special_attrs[3] {
            Attribute::Float(f) if f == 0.0 && f.is_sign_negative() => (),
            _ => panic!("Expected negative zero"),
        }
        
        // Epsilon
        match special_attrs[4] {
            Attribute::Float(f) if (f - f64::EPSILON).abs() < f64::EPSILON => (),
            _ => panic!("Expected epsilon"),
        }
    }

    // Test 7: Test very large string names for entities
    #[test]
    fn test_extremely_long_names() {
        let long_module_name = "M".repeat(1_000_000); // 1 million 'M's
        let module = Module::new(&long_module_name);
        
        assert_eq!(module.name.len(), 1_000_000);
        assert_eq!(module.name, long_module_name);
        
        // Test with operation
        let long_op_name = "O".repeat(500_000);
        let op = Operation::new(&long_op_name);
        assert_eq!(op.op_type.len(), 500_000);
        
        // Test with value
        let long_value_name = "V".repeat(500_000);
        let value = Value {
            name: long_value_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name.len(), 500_000);
    }

    // Test 8: Test operations with mixed data types and complex nesting
    #[test]
    fn test_complex_type_nesting() {
        // Create a complex nested type: tensor of tensors of tensors of F32
        let complex_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2, 2],
                }),
                shape: vec![3, 3],
            }),
            shape: vec![4, 4],
        };
        
        // Verify the structure exists
        match &complex_type {
            Type::Tensor { shape: outer_shape, element_type: outer_elem } => {
                assert_eq!(outer_shape, &vec![4, 4]);
                
                match outer_elem.as_ref() {
                    Type::Tensor { shape: mid_shape, element_type: mid_elem } => {
                        assert_eq!(mid_shape, &vec![3, 3]);
                        
                        match mid_elem.as_ref() {
                            Type::Tensor { shape: inner_shape, element_type: inner_elem } => {
                                assert_eq!(inner_shape, &vec![2, 2]);
                                
                                match inner_elem.as_ref() {
                                    Type::F32 => (), // Expected
                                    _ => panic!("Expected F32 at innermost level"),
                                }
                            },
                            _ => panic!("Expected Tensor at inner level"),
                        }
                    },
                    _ => panic!("Expected Tensor at middle level"),
                }
            },
            _ => panic!("Expected Tensor at outer level"),
        }
        
        // Verify equality works for identical complex types
        let complex_type2 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2, 2],
                }),
                shape: vec![3, 3],
            }),
            shape: vec![4, 4],
        };
        
        assert_eq!(complex_type, complex_type2);
    }

    // Test 9: Module with maximum possible components
    #[test]
    fn test_module_with_maximum_components() {
        let mut module = Module::new("max_module");
        
        // Add many operations with complex structures
        for i in 0..10_000 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Add inputs
            for j in 0..5 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
            }
            
            // Add outputs  
            for j in 0..3 {
                op.outputs.push(Value {
                    name: format!("output_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1, j + 2],
                });
            }
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 10_000);
        assert_eq!(module.name, "max_module");
        
        // Verify structure of first and last operations
        let first_op = &module.operations[0];
        assert_eq!(first_op.inputs.len(), 5);
        assert_eq!(first_op.outputs.len(), 3);
        assert_eq!(first_op.op_type, "op_0");
        
        let last_op = &module.operations[9999];
        assert_eq!(last_op.inputs.len(), 5);
        assert_eq!(last_op.outputs.len(), 3);
        assert_eq!(last_op.op_type, "op_9999");
    }

    // Test 10: Test tensor type validation edge cases
    #[test]
    fn test_tensor_type_validation() {
        use crate::ir::TypeExtensions;
        
        // Valid simple type
        assert!(Type::F32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        
        // Valid tensor
        let valid_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 4],
        };
        assert!(valid_tensor.is_valid_type());
        
        // Valid deeply nested tensor
        let nested_valid = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![1, 1],
            }),
            shape: vec![5],
        };
        assert!(nested_valid.is_valid_type());
        
        // Zero in shape should still be valid
        let zero_shape = Type::Tensor {
            element_type: Box::new(Type::Bool),
            shape: vec![100, 0, 50],  // Contains zero
        };
        assert!(zero_shape.is_valid_type());
        
        // Empty shape in tensor (would be unusual but shouldn't crash)
        let empty_shape = Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![],  // Empty shape
        };
        assert!(empty_shape.is_valid_type());
    }
}