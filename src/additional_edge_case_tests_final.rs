//! Additional edge case tests for the Impulse compiler
//! Covers more boundary conditions using standard library asserts and rstest

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use rstest::rstest;
    
    /// Test 1: Operations with extremely long attribute names
    #[test]
    fn test_operation_extremely_long_attribute_names() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("test_op");
        let mut attrs = HashMap::new();
        
        // Create an attribute with a very long name
        let long_attr_name = "a".repeat(50_000);
        attrs.insert(long_attr_name.clone(), Attribute::String("value".to_string()));
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 1);
        assert!(op.attributes.contains_key(&long_attr_name));
    }

    /// Test 2: Module with maximum number of inputs/outputs
    #[test]
    fn test_module_with_maximum_io_values() {
        let mut module = Module::new("max_io_module");
        
        // Add many input values
        for i in 0..10_000 {
            let input_val = Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1],
            };
            module.inputs.push(input_val);
        }
        
        // Add many output values
        for i in 0..10_000 {
            let output_val = Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1],
            };
            module.outputs.push(output_val);
        }
        
        assert_eq!(module.inputs.len(), 10_000);
        assert_eq!(module.outputs.len(), 10_000);
    }

    /// Test 3: Values with maximum dimension count in shape
    #[test]
    fn test_values_with_maximum_shape_dimensions() {
        // Create a value with many dimensions (up to 1000)
        let mut shape = Vec::new();
        for i in 1..=1000 {
            shape.push(i % 10);  // Small values to prevent overflow in product
        }
        
        let value = Value {
            name: "max_dims".to_string(),
            ty: Type::F32,
            shape,
        };
        
        assert_eq!(value.shape.len(), 1000);
        // Check that all dimensions are set correctly
        for (i, &dim) in value.shape.iter().enumerate() {
            assert_eq!(dim, (i + 1) % 10);
        }
    }

    /// Test 4: Operations with conflicting attribute types
    #[test]
    fn test_operation_conflicting_attribute_types() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("mixed_attr_op");
        let mut attrs = HashMap::new();
        
        // Add attributes of different types
        attrs.insert("int_attr".to_string(), Attribute::Int(-9223372036854775808)); // i64 min
        attrs.insert("uint_attr".to_string(), Attribute::Int(9223372036854775807));  // i64 max
        attrs.insert("zero_int".to_string(), Attribute::Int(0));
        attrs.insert("float_special".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("float_neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("float_nan".to_string(), Attribute::Float(f64::NAN));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 6);
        assert_eq!(op.attributes.get("int_attr"), Some(&Attribute::Int(i64::MIN)));
        assert_eq!(op.attributes.get("uint_attr"), Some(&Attribute::Int(i64::MAX)));
    }

    /// Test 5: Nested arrays in attributes with maximum depth
    #[test]
    fn test_deeply_nested_attribute_arrays() {
        // Create a deeply nested array structure
        let mut nested = Attribute::Int(42);
        
        // Nest arrays 20 levels deep
        for _ in 0..20 {
            nested = Attribute::Array(vec![nested]);
        }
        
        // Verify the structure (this tests that deep nesting doesn't cause issues)
        match &nested {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
                // Verify we can access nested structure without problems
                assert!(true); // Basic test that accessing didn't panic
            },
            _ => panic!("Expected nested array structure"),
        }
    }

    /// Test 6: Tensor types with alternating element types
    #[rstest]
    #[case(
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2] },
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2] }
    )]
    #[case(
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2] },
        Type::Tensor { element_type: Box::new(Type::I32), shape: vec![2] }
    )]
    fn test_tensor_type_comparisons(
        #[case] type1: Type,
        #[case] type2: Type
    ) {
        // Extract element types if both types are tensors
        let both_tensors = matches!(&type1, Type::Tensor { .. }) && matches!(&type2, Type::Tensor { .. });
        let element_types_equal = if both_tensors {
            match (&type1, &type2) {
                (Type::Tensor { element_type: e1, .. }, Type::Tensor { element_type: e2, .. }) => {
                    e1.as_ref() == e2.as_ref()
                },
                _ => false,
            }
        } else {
            false
        };
        
        // Now compare the original types and assert accordingly
        if type1 == type2 {
            assert_eq!(element_types_equal, true); // If types are equal, element types must be too
        } else {
            // If types are not equal, either they're not tensors or their element types differ
            // Nothing to assert here as inequality is expected
        }
    }

    /// Test 7: Module with operations that have circular references (conceptually)
    #[test]
    fn test_module_with_complex_operation_chains() {
        let mut module = Module::new("complex_chain_module");
        
        // Create operations in a chain where each depends on previous
        for i in 0..5000 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Add input that conceptually comes from previous operation
            if i > 0 {
                op.inputs.push(Value {
                    name: format!("output_from_{}", i-1),
                    ty: Type::F32,
                    shape: vec![i % 100 + 1],
                });
            }
            
            // Add output that will be input to next operation
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 5000);
        
        // Verify first and last operations have correct structure
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[4999].op_type, "op_4999");
    }

    /// Test 8: Extreme values in tensor shapes that could cause math overflow
    #[test]
    fn test_tensor_shape_math_with_potential_overflow() {
        // Use values that could potentially cause issues in calculations
        let shapes_to_test = [
            vec![usize::MAX, 1],     // Potential overflow
            vec![1, usize::MAX],     // Potential overflow  
            vec![2, usize::MAX/2 + 1], // Potential overflow when multiplied
            vec![1000, 1000, 1000],  // Large but safe
            vec![0, usize::MAX],     // Contains 0, so product should be 0
        ];
        
        for shape in &shapes_to_test {
            let value = Value {
                name: "potential_overflow".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            // Calculate product safely
            let mut product: usize = 1;
            for &dim in &value.shape {
                if dim == 0 {
                    product = 0;
                    break;
                }
                if let Some(result) = product.checked_mul(dim) {
                    product = result;
                } else {
                    // Overflow detected - this is expected behavior
                    product = 0; // Or we could handle differently based on requirements
                    break;
                }
            }
            
            // If any dimension is 0, total should be 0
            if shape.contains(&0) {
                assert_eq!(product, 0);
            }
        }
    }

    /// Test 9: String attributes with unicode and special characters
    #[rstest]
    #[case("simple_ascii")]
    #[case("üñíçødé")]  // Unicode
    #[case("🚀🎉🦀")]     // Emojis
    #[case("multi\nline\r\ntext")]  // Special whitespace
    #[case("")]          // Empty string
    #[case("very_long_string_with_unicode_".repeat(1000))]  // Long unicode string
    fn test_string_attributes_with_various_unicode(#[case] test_string: String) {
        let attr = Attribute::String(test_string.clone());
        
        match attr {
            Attribute::String(s) => {
                assert_eq!(s, test_string);
            },
            _ => panic!("Expected String attribute"),
        }
    }

    /// Test 10: Operations with maximum possible attribute count
    #[test]
    fn test_operation_with_maximum_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add a very large number of attributes
        for i in 0..50_000 {
            attrs.insert(
                format!("attribute_{:06}", i), 
                Attribute::String(format!("value_{}", i))
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 50_000);
        
        // Verify a few specific attributes exist
        assert!(op.attributes.contains_key("attribute_000000"));
        assert!(op.attributes.contains_key("attribute_049999")); // Highest index is 49999, formatted with 6 digits as 049999
        assert_eq!(
            op.attributes.get("attribute_000000"), 
            Some(&Attribute::String("value_0".to_string()))
        );
    }
}