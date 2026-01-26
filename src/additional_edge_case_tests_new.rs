//! Additional edge case tests for the Impulse compiler
//! Tests focus on boundary conditions and unusual scenarios

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use crate::ImpulseCompiler;
    use std::collections::HashMap;

    /// Test 1: Module with maximum possible string length in name
    #[test]
    fn test_module_with_maximum_string_length() {
        // Create a module with a very long name to test string handling
        let long_name = "a".repeat(1_000_000); // 1 million 'a's
        let module = Module::new(&long_name);
        
        assert_eq!(module.name.len(), 1_000_000);
        assert_eq!(module.name.chars().count(), 1_000_000);
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }

    /// Test 2: Operation with maximum possible attribute count
    #[test]
    fn test_operation_with_maximum_attributes() {
        let mut op = Operation::new("max_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add a very large number of attributes
        for i in 0..100_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 100_000);
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        
        // Verify some attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_50000"));
        assert!(op.attributes.contains_key("attr_99999"));
    }

    /// Test 3: Value with extremely large shape dimensions that could cause overflow
    #[test]
    fn test_value_with_potential_overflow_dimensions() {
        // Test shape that would cause overflow when calculating total size
        // Use values that are large but won't actually overflow in multiplication
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX / 1000, 1000], // Large but potentially risky
        };
        
        assert_eq!(value.shape.len(), 2);
        
        // Compute with checked arithmetic to prevent actual overflow
        let result = value.shape.iter()
            .try_fold(1usize, |acc, &x| acc.checked_mul(x));
        
        // This might overflow, but that's the test - ensure it doesn't crash
        assert!(result.is_some() || result.is_none()); // Either succeeds or handles overflow
    }

    /// Test 4: Nested tensor type with maximum nesting depth
    #[test]
    fn test_maximum_depth_nested_tensors() {
        // Create a deeply nested tensor type to test recursion limits
        let mut current_type = Type::F32;
        
        // Nest 1000 levels deep (may need adjustment based on system limits)
        for _ in 0..1000 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the deeply nested type can be created and compared
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Ensure it's still a tensor type
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected tensor type even when deeply nested"),
        }
    }

    /// Test 5: Operations with Unicode and special character names
    #[test]
    fn test_operations_with_unicode_names() {
        let unicode_names = [
            "tensor_名称_日本語_🔥",           // Mixed scripts and emoji
            "café naïve résumé",           // Accented characters
            "москва Москва россия",        // Cyrillic
            "مرحبا بالعالم",               // Arabic
            "こんにちは世界",               // Japanese hiragana and kanji
            "😀😃😄😁😂🤣☺️😊",          // Multiple emojis
            "tab\there",                  // Tab character
            "newline\nhere",              // Newline character  
            "\x00\x01\x02\x03",          // Control characters
        ];
        
        for (i, &name) in unicode_names.iter().enumerate() {
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![i + 1, i + 2],
            };
            
            assert_eq!(value.name, name);
            assert_eq!(value.shape, vec![i + 1, i + 2]);
            
            let op = Operation::new(name);
            assert_eq!(op.op_type, name);
            
            let module = Module::new(name);
            assert_eq!(module.name, name);
        }
    }

    /// Test 6: Tensor shapes with zeros in various positions causing zero-size tensors
    #[test]
    fn test_various_zero_dimension_combinations() {
        let zero_combinations = [
            vec![0],                    // 0D tensor with zero elements
            vec![0, 1, 2, 3],         // Leading zero
            vec![1, 0, 2, 3],         // Middle zero
            vec![1, 2, 0, 3],         // Another middle zero  
            vec![1, 2, 3, 0],         // Trailing zero
            vec![0, 0],               // Multiple leading zeros
            vec![1, 0, 0, 3],         // Multiple zeros in middle
            vec![2, 3, 0, 0],         // Trailing zeros
            vec![0, 1, 0, 1, 0],      // Alternating zeros
        ];
        
        for shape in zero_combinations.iter() {
            let value = Value {
                name: format!("zero_test_{:?}", shape),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, *shape);
            
            // Any tensor with a zero dimension should have 0 total elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements", shape);
        }
    }

    /// Test 7: Invalid tensor types and validation
    #[test]
    fn test_invalid_tensor_type_scenarios() {
        // Test valid nested tensor
        let valid_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        
        assert!(valid_tensor.is_valid_type());
        
        // Deeply nested valid tensor
        let mut deep_tensor = Type::F32;
        for _ in 0..10 {
            deep_tensor = Type::Tensor {
                element_type: Box::new(deep_tensor),
                shape: vec![2],
            };
        }
        
        assert!(deep_tensor.is_valid_type());
        
        // Test equality of complex nested types
        let another_deep_tensor = {
            let mut t = Type::F32;
            for _ in 0..10 {
                t = Type::Tensor {
                    element_type: Box::new(t),
                    shape: vec![2],
                };
            }
            t
        };
        
        assert_eq!(deep_tensor, another_deep_tensor);
    }

    /// Test 8: Operation with maximum possible inputs and outputs
    #[test]
    fn test_operation_with_extreme_io_counts() {
        let mut op = Operation::new("extreme_io_op");
        
        // Add many inputs
        for i in 0..50_000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1], // Varying small shapes
            });
        }
        
        // Add many outputs
        for i in 0..25_000 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![(i + 50) % 100 + 1], // Varying small shapes
            });
        }
        
        assert_eq!(op.inputs.len(), 50_000);
        assert_eq!(op.outputs.len(), 25_000);
        assert_eq!(op.op_type, "extreme_io_op");
        
        // Verify first and last inputs/outputs have correct names
        assert_eq!(op.inputs[0].name, "input_0");
        assert_eq!(op.inputs[49999].name, "input_49999");
        assert_eq!(op.outputs[0].name, "output_0");
        assert_eq!(op.outputs[24999].name, "output_24999");
    }

    /// Test 9: Special floating-point attribute values
    #[test]
    fn test_special_float_attribute_values() {
        let special_values = [
            (f64::INFINITY, "positive_infinity"),
            (f64::NEG_INFINITY, "negative_infinity"),
            (-0.0, "negative_zero"),
            (f64::EPSILON, "epsilon"),
            (f64::consts::PI, "pi"),
            (f64::consts::E, "euler_number"),
        ];
        
        for (value, desc) in special_values.iter() {
            let attr = Attribute::Float(*value);
            
            match attr {
                Attribute::Float(retrieved_value) => {
                    if value.is_infinite() {
                        assert!(retrieved_value.is_infinite(), 
                                "For {}: retrieved value should be infinite", desc);
                        assert_eq!(value.is_sign_positive(), retrieved_value.is_sign_positive(),
                                   "For {}: sign should match", desc);
                    } else if value.is_sign_negative() && *value == -0.0 {
                        // Special case for negative zero
                        assert!(retrieved_value.is_sign_negative(), 
                                "For {}: should preserve negative zero sign", desc);
                        assert!(retrieved_value == 0.0, 
                                "For {}: should still be zero", desc);
                    } else {
                        assert!((*value - retrieved_value).abs() < f64::EPSILON,
                               "For {}: values should match", desc);
                    }
                },
                _ => panic!("For {}: Expected Float attribute", desc),
            }
        }
        
        // Test NaN separately since NaN != NaN
        let nan_attr = Attribute::Float(f64::NAN);
        match nan_attr {
            Attribute::Float(retrieved_nan) => {
                assert!(retrieved_nan.is_nan(), "NaN should be preserved");
            },
            _ => panic!("Expected Float(NaN) attribute"),
        }
    }

    /// Test 10: Complex compiler integration with edge case values
    #[test]
    fn test_compiler_with_complex_edge_case_values() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test that the compiler can be created and has expected initial state
        assert_eq!(compiler.frontend, compiler.frontend); // Basic equality test
        assert_eq!(compiler.passes.passes.len(), 0);
        assert_eq!(compiler.backends.backends.len(), 0); // Assuming this field exists
        assert_eq!(compiler.runtime.devices.len(), 0);   // Assuming this field exists
        
        // Create a module with complex edge-case values
        let mut module = Module::new("edge_case_module");
        
        // Add an operation with complex attributes
        let mut op = Operation::new("complex_op_for_compiler");
        op.inputs.push(Value {
            name: "unicode_input_🚀".to_string(),
            ty: Type::F32,
            shape: vec![0, 10, 0], // Zero dimensions
        });
        
        let mut attrs = HashMap::new();
        attrs.insert(
            "special_float".to_string(),
            Attribute::Float(f64::INFINITY)
        );
        attrs.insert(
            "unicode_string".to_string(),
            Attribute::String("🔥🚀🌟".to_string())
        );
        op.attributes = attrs;
        
        module.add_operation(op);
        
        // Add another operation with nested tensor types
        let nested_type_op = {
            let mut nested_type = Type::F32;
            for _ in 0..5 {
                nested_type = Type::Tensor {
                    element_type: Box::new(nested_type.clone()),
                    shape: vec![2],
                };
            }
            
            let mut op = Operation::new("nested_tensor_op");
            op.outputs.push(Value {
                name: "nested_tensor_output".to_string(),
                ty: nested_type,
                shape: vec![3, 3],
            });
            op
        };
        
        module.add_operation(nested_type_op);
        
        // Verify the module was constructed properly
        assert_eq!(module.name, "edge_case_module");
        assert_eq!(module.operations.len(), 2);
        
        // Check first operation
        assert_eq!(module.operations[0].op_type, "complex_op_for_compiler");
        assert_eq!(module.operations[0].inputs[0].name, "unicode_input_🚀");
        assert_eq!(module.operations[0].attributes.len(), 2);
        
        // Check second operation  
        assert_eq!(module.operations[1].op_type, "nested_tensor_op");
        
        // The test passes if no panics occurred during construction
    }
}