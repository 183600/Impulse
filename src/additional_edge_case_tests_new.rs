//! Additional edge case tests for the Impulse compiler
//! These tests cover boundary conditions and unusual scenarios

#[cfg(test)]
mod additional_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use crate::ImpulseCompiler;
    use std::collections::HashMap;
    use rstest::rstest;

    /// Test creating operations with empty string names
    #[test]
    fn test_operation_with_empty_name() {
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    /// Test creating modules with empty names
    #[test]
    fn test_module_with_empty_name() {
        let module = Module::new("");
        assert_eq!(module.name, "");
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }

    /// Test value with maximum possible dimensions
    #[test]
    fn test_value_with_max_dimensions() {
        let value = Value {
            name: "max_dims".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 1],
        };
        assert_eq!(value.name, "max_dims");
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![usize::MAX, 1]);
        
        // Test shape product calculation (this should not overflow in normal cases)
        let product: usize = value.shape.iter().map(|&x| x as usize).fold(1, |acc, x| acc.saturating_mul(x));
        assert_eq!(product, usize::MAX); // Multiplication by 1 doesn't change the value
    }

    /// Test deeply nested tensor types
    #[test]
    fn test_deeply_nested_tensor_types() {
        let mut current_type = Type::F32;
        
        // Create 50 levels of nesting - deep enough to test but not excessive
        for _ in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the structure
        match current_type {
            Type::Tensor { element_type: _, ref shape } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a tensor type"),
        }
        
        // Test cloning of deeply nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test with extremely large tensor shapes
    #[test]
    fn test_extremely_large_tensor_shapes() {
        // Test with a shape that has one very large dimension
        let large_value = Value {
            name: "large_shape".to_string(),
            ty: Type::F32,
            shape: vec![10_000_000, 10], // 100 million elements
        };
        
        assert_eq!(large_value.shape, vec![10_000_000, 10]);
        let total_elements: usize = large_value.shape.iter().product();
        assert_eq!(total_elements, 100_000_000);
    }

    /// Test operations with maximum number of attributes
    #[test]
    fn test_operation_with_many_attributes() {
        let mut op = Operation::new("many_attrs");
        let mut attrs = HashMap::new();
        
        // Add 10,000 attributes to test memory handling
        for i in 0..10_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 10_000);
        
        // Verify a few attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_9999"));
        assert_eq!(op.attributes.get("attr_5000"), Some(&Attribute::String("value_5000".to_string())));
    }

    /// Test operations with extremely long input/output names
    #[test]
    fn test_operations_with_extremely_long_names() {
        let mut op = Operation::new("normal_op");
        
        // Add an input with an extremely long name
        let long_name_input = Value {
            name: "input_".repeat(1000) + "end",
            ty: Type::F32,
            shape: vec![1],
        };
        op.inputs.push(long_name_input);
        
        // Add an output with an extremely long name 
        let long_name_output = Value {
            name: "output_".repeat(1000) + "end",
            ty: Type::F32,
            shape: vec![1],
        };
        op.outputs.push(long_name_output);
        
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
        assert!(op.inputs[0].name.len() > 5000); // Should be much longer than 5000 chars
        assert!(op.outputs[0].name.len() > 5000); // Should be much longer than 5000 chars
    }

    /// Parameterized test using rstest for different primitive types
    #[rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    fn test_primitive_type_properties(#[case] primitive_type: Type) {
        // Test that primitive types are valid
        match &primitive_type {
            Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Bool => {
                // These are all valid primitive types
                assert!(match primitive_type {
                    Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Bool => true,
                    _ => false,
                });
            },
            Type::Tensor { .. } => panic!("Expected a primitive type, got tensor"),
        }
        
        // Test cloning
        let cloned_type = primitive_type.clone();
        assert_eq!(primitive_type, cloned_type);
    }

    /// Parameterized test for special tensor shapes
    #[rstest]
    #[case(vec![], 1)] // scalar
    #[case(vec![0], 0)] // zero-sized tensor
    #[case(vec![1], 1)] // size 1 tensor
    #[case(vec![1, 1, 1], 1)] // multi-dimensional size 1 tensor
    #[case(vec![2, 3, 4], 24)] // normal multi-dimensional tensor
    #[case(vec![100, 100], 10000)] // larger tensor
    fn test_tensor_shape_products(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let value = Value {
            name: "test_tensor".to_string(),
            ty: Type::F32,
            shape,
        };
        
        let calculated_product: usize = value.shape.iter().product();
        assert_eq!(calculated_product, expected_product);
    }

    /// Test special floating point values in attributes
    #[test]
    fn test_special_float_values_in_attributes() {
        let special_attrs = [
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(f64::NAN),
            Attribute::Float(-0.0),
            Attribute::Float(f64::EPSILON),
            Attribute::Float(std::f64::consts::PI),
        ];

        // Check that we can create and store these values
        assert_eq!(special_attrs.len(), 6);

        // Test specific properties of each
        match special_attrs[0] {  // INFINITY
            Attribute::Float(f) => assert!(f.is_infinite() && f.is_sign_positive()),
            _ => panic!("Expected Float attribute"),
        }

        match special_attrs[1] {  // NEG_INFINITY
            Attribute::Float(f) => assert!(f.is_infinite() && f.is_sign_negative()),
            _ => panic!("Expected Float attribute"),
        }

        match special_attrs[2] {  // NAN
            Attribute::Float(f) => assert!(f.is_nan()),
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test compiler with invalid inputs
    #[test]
    fn test_compiler_with_invalid_inputs() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with empty byte array
        let empty_input = vec![];
        let _result = compiler.compile(&empty_input, "cpu");
        // Result might be error or success depending on implementation, 
        // but should not panic
        
        // Test with very large input
        let large_input = vec![0u8; 100_000_000]; // 100 MB
        let result2 = compiler.compile(&large_input, "cpu");
        // Should not panic regardless of result
        assert!(result2.is_ok() || result2.is_err());
        
        // Test with random bytes
        let random_bytes: Vec<u8> = (0..1000).map(|i| i as u8).collect();
        let result3 = compiler.compile(&random_bytes, "cpu");
        assert!(result3.is_ok() || result3.is_err());
    }

    /// Test value with many zero dimensions
    #[test]
    fn test_tensor_with_multiple_zeros() {
        let test_cases = vec![
            vec![0],
            vec![0, 5],
            vec![5, 0],
            vec![2, 0, 3],
            vec![0, 0, 0],
            vec![1, 0, 1, 0],
        ];
        
        for shape in test_cases {
            let value = Value {
                name: "zero_test".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            // Any tensor with a zero dimension should have 0 total elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements", shape);
        }
    }

    /// Test operations with unicode names
    #[test]
    fn test_operations_with_unicode_names() {
        let unicode_names = [
            "op_名称_日本語",
            "operation_🚀_⚡",
            "opération_émojis_🔥",
            "операция_кириллица",
        ];
        
        for name in &unicode_names {
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
        }
    }

    /// Test deeply nested arrays in attributes
    #[test]
    fn test_deeply_nested_arrays_in_attributes() {
        // Create nested arrays: [[1, 2], [3, 4]]
        let nested_array = Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
            Attribute::Array(vec![Attribute::Int(3), Attribute::Int(4)]),
        ]);
        
        match &nested_array {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                
                match &outer[0] {
                    Attribute::Array(inner) => {
                        assert_eq!(inner.len(), 2);
                        assert_eq!(inner[0], Attribute::Int(1));
                        assert_eq!(inner[1], Attribute::Int(2));
                    }
                    _ => panic!("Expected inner array"),
                }
                
                match &outer[1] {
                    Attribute::Array(inner) => {
                        assert_eq!(inner.len(), 2);
                        assert_eq!(inner[0], Attribute::Int(3));
                        assert_eq!(inner[1], Attribute::Int(4));
                    }
                    _ => panic!("Expected inner array"),
                }
            }
            _ => panic!("Expected outer array"),
        }
    }

    /// Test very deep recursion in tensor types
    #[test]
    fn test_very_deep_tensor_recursion() {
        let mut current_type = Type::Bool;
        
        // Create 100 levels of nesting - testing stack space
        for i in 0..100 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 5 + 1], // Varying shapes
            };
        }
        
        // Verify the structure still holds
        match &current_type {
            Type::Tensor { shape, .. } => {
                // The loop ran 100 times, with i from 0 to 99
                // The last iteration used i = 99, so shape = vec![99 % 5 + 1] = vec[4 + 1] = vec![5]
                assert_eq!(shape, &vec![5]); 
            }
            _ => panic!("Expected tensor at top level"),
        }
        
        // Test equality comparison
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }
}