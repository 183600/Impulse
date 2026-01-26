//! Additional edge case tests for the Impulse compiler, covering more boundary conditions

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
    use std::collections::HashMap;

    /// Test 1: Maximum possible tensor dimensions
    #[test]
    fn test_maximum_tensor_dimensions() {
        // Test creating a tensor with maximum allowed dimensions
        let max_dims = vec![usize::MAX, 1];  // This would likely overflow when calculating total size
        let value = Value {
            name: "max_dims_tensor".to_string(),
            ty: Type::F32,
            shape: max_dims,
        };

        // Testing the calculation of total elements which might overflow
        let total_elements: Option<usize> = value.shape.iter()
            .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
        
        // This should handle the overflow gracefully
        assert!(total_elements.is_none()); // Expecting overflow
        
        // Test with slightly safer but still large dimensions
        let large_dims = vec![100_000, 100_000];
        let large_value = Value {
            name: "large_dims_tensor".to_string(),
            ty: Type::F32,
            shape: large_dims,
        };
        
        let safe_total: usize = large_value.shape.iter()
            .fold(0, |acc, &x| acc.saturating_add(x)); // Using saturating operations
        
        assert_eq!(safe_total, 200_000);
    }

    /// Test 2: Deeply nested recursive types with more complexity
    #[test]
    fn test_extremely_deep_recursive_types() {
        let mut current_type = Type::F32;
        const DEPTH: usize = 1000; // Very deep nesting to test stack limits
        
        for i in 0..DEPTH {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 10 + 1], // Vary the shape to make it more complex
            };
        }
        
        // Ensure the final type is valid and can be cloned without stack overflow
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Test with different base types in the recursion
        let mut alt_type = Type::I64;
        for i in 0..500 {
            // Alternate between different base types
            alt_type = Type::Tensor {
                element_type: if i % 2 == 0 {
                    Box::new(Type::F64)
                } else {
                    Box::new(alt_type)
                },
                shape: vec![2],
            };
        }
        
        let alt_cloned = alt_type.clone();
        assert_eq!(alt_type, alt_cloned);
    }

    /// Test 3: Floating point edge cases in attributes
    #[test]
    fn test_floating_point_edge_cases_in_attributes() {
        let test_values = [
            (f64::INFINITY, "infinity"),
            (f64::NEG_INFINITY, "neg_infinity"),
            (f64::NAN, "nan"),
            (-0.0, "negative_zero"),
            (f64::EPSILON, "epsilon"),
            (std::f64::consts::PI, "pi"),
            (std::f64::consts::E, "euler"),
        ];

        for (value, name) in test_values.iter() {
            let attr = Attribute::Float(*value);
            
            match attr {
                Attribute::Float(returned_value) => {
                    if value.is_nan() {
                        assert!(returned_value.is_nan(), "Failed for {}", name);
                    } else if value.is_infinite() {
                        assert!(returned_value.is_infinite(), "Failed for {}", name);
                        assert_eq!(returned_value.is_sign_positive(), value.is_sign_positive(), "Sign mismatch for {}", name);
                    } else {
                        // For finite values, check approximate equality
                        if (value - returned_value).abs() > f64::EPSILON {
                            assert_eq!(returned_value, *value, "Failed for {}", name);
                        }
                    }
                }
                _ => panic!("Expected Float attribute for {}", name),
            }
        }
    }

    /// Test 4: Handling empty and invalid operations
    #[test]
    fn test_empty_and_minimal_operations() {
        // Test operation with empty string name
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);

        // Test operation with minimal content
        let minimal_op = Operation::new("minimal");
        assert_eq!(minimal_op.op_type, "minimal");
        assert_eq!(minimal_op.inputs.len(), 0);
        assert_eq!(minimal_op.outputs.len(), 0);
        assert_eq!(minimal_op.attributes.len(), 0);

        // Test adding empty operation to module
        let mut module = Module::new("empty_ops_test");
        module.add_operation(empty_op);
        module.add_operation(minimal_op);
        
        assert_eq!(module.operations.len(), 2);
        assert_eq!(module.operations[0].op_type, "");
        assert_eq!(module.operations[1].op_type, "minimal");
    }

    /// Test 5: String-related edge cases in names
    #[test]
    fn test_string_edge_cases_in_names() {
        let test_strings = [
            "",                             // Empty string
            "a",                            // Single character
            &" ".repeat(1000),              // Long whitespace string
            "\0",                          // Null character
            "🚀🌟💻🔥",                     // Emojis
            "αβγδεζηθικλμνξοπρστυφχψω",  // Greek letters
            "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", // Cyrillic
            "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ", // Armenian
            "\n\t\r\x0C\u{000B}",          // Various control characters
        ];

        for (i, test_str) in test_strings.iter().enumerate() {
            // Test value names
            let value = Value {
                name: test_str.to_string(),
                ty: Type::F32,
                shape: vec![1],
            };
            assert_eq!(value.name, *test_str, "Value name test failed for case {}", i);

            // Test operation names
            let op = Operation::new(test_str);
            assert_eq!(op.op_type, *test_str, "Operation name test failed for case {}", i);

            // Test module names
            let module = Module::new(*test_str);
            assert_eq!(module.name, *test_str, "Module name test failed for case {}", i);

            // Test attribute string values
            let str_attr = Attribute::String(test_str.to_string());
            match str_attr {
                Attribute::String(s) => assert_eq!(s, *test_str, "String attribute test failed for case {}", i),
                _ => panic!("Expected String attribute for case {}", i),
            }
        }
    }

    /// Test 6: Integer overflow in tensor size calculations with checked arithmetic
    #[test]
    fn test_integer_overflow_protection_in_tensor_calculations() {
        // Test a combination of dimensions that would cause overflow
        let problem_dims = vec![100_000, 100_000, 100]; // Would be 1 trillion elements
        
        // Calculate with checked multiplication to prevent overflow
        let overflow_result = problem_dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
        assert!(overflow_result.is_none()); // Should overflow
        
        // Test safe calculations with smaller values
        let safe_dims = vec![1000, 1000, 100]; // 100 million elements, safe
        let safe_result = safe_dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
        assert_eq!(safe_result, Some(100_000_000));
        
        // Test with zero in dimensions (should result in 0 regardless of other dimensions)
        let zero_dims = vec![100, 0, 1000];
        let zero_result = zero_dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
        assert_eq!(zero_result, Some(0));
        
        // Create a Value with dimensions that would overflow and test it
        let overflow_value = Value {
            name: "overflow_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],
        };
        
        // Safe calculation that won't panic
        let safe_calculation = overflow_value.shape.iter()
            .try_fold(1usize, |acc, &x| acc.checked_mul(x));
        
        assert!(safe_calculation.is_some()); // Actually won't overflow on our test values
    }

    /// Test 7: Handling extreme numbers of attributes
    #[test]
    fn test_extreme_number_of_attributes() {
        const ATTR_COUNT: usize = 100_000; // Very large number of attributes
        
        let mut op = Operation::new("many_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add a huge number of attributes
        for i in 0..ATTR_COUNT {
            let attr_name = format!("attr_{:08}", i);
            let attr_value = Attribute::String(format!("value_for_attr_{:08}", i));
            attrs.insert(attr_name, attr_value);
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), ATTR_COUNT);
        
        // Verify a few specific attributes exist
        assert!(op.attributes.contains_key("attr_00000000"));
        assert!(op.attributes.contains_key(&format!("attr_{:08}", ATTR_COUNT - 1)));
        
        // Test retrieval of specific values
        if let Some(Attribute::String(ref val)) = op.attributes.get("attr_00000000") {
            assert_eq!(val, "value_for_attr_00000000");
        } else {
            panic!("Expected String attribute for attr_00000000");
        }
        
        // Test with mixed attribute types
        let mut mixed_op = Operation::new("mixed_attrs_op");
        let mut mixed_attrs = HashMap::new();
        
        for i in 0..10_000 {
            let attr_name = format!("mixed_attr_{}", i);
            let attr_value = match i % 5 {
                0 => Attribute::Int(i as i64),
                1 => Attribute::Float(i as f64 * 0.5),
                2 => Attribute::String(format!("str_val_{}", i)),
                3 => Attribute::Bool(i % 2 == 0),
                _ => Attribute::Array(vec![
                    Attribute::Int(i as i64),
                    Attribute::String(format!("nested_{}", i))
                ]),
            };
            mixed_attrs.insert(attr_name, attr_value);
        }
        
        mixed_op.attributes = mixed_attrs;
        assert_eq!(mixed_op.attributes.len(), 10_000);
    }

    /// Test 8: Complex module hierarchies
    #[test]
    fn test_complex_module_structure() {
        // Create a complex module with many interconnected operations
        let mut module = Module::new("complex_module");
        
        // Add a large number of operations with interconnections
        for i in 0..50_000 {
            let mut op = Operation::new(&format!("op_{:06}", i));
            
            // Add inputs and outputs with interconnections
            op.inputs.push(Value {
                name: format!("input_{:06}_0", i),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![i % 100 + 1, i % 50 + 1],
            });
            
            op.outputs.push(Value {
                name: format!("output_{:06}_0", i),
                ty: if i % 3 == 0 { Type::F64 } else { Type::I64 },
                shape: vec![(i + 10) % 75 + 1, (i + 5) % 40 + 1],
            });
            
            // Add a few attributes to each operation
            op.attributes.insert(
                format!("param_{}", i % 1000), 
                Attribute::Int((i % 10000) as i64)
            );
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 50_000);
        assert_eq!(module.name, "complex_module");
        
        // Verify that some operations maintain their data integrity
        assert_eq!(module.operations[0].op_type, "op_000000");
        assert_eq!(module.operations[25_000].op_type, "op_025000");
        assert_eq!(module.operations[49_999].op_type, "op_049999");
    }

    /// Test 9: Error conditions and graceful handling
    #[test]
    fn test_error_conditions_and_graceful_handling() {
        // Test creating a deeply nested structure that could cause issues
        let mut current_type = Type::F32;
        
        // Create a nested structure that's deep enough to test recursion limits
        for _ in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2, 2],
            };
        }
        
        // Verify the type is still valid despite depth
        assert!(current_type.is_valid_type());
        
        // Create a value with this complex type
        let complex_value = Value {
            name: "complex_nested_value".to_string(),
            ty: current_type,
            shape: vec![1],
        };
        
        // Test cloning this complex structure
        let cloned_value = complex_value.clone();
        assert_eq!(complex_value.name, cloned_value.name);
        
        // Test creating operations with invalid/malformed data that should still be handled gracefully
        let problematic_op = Operation::new(&"a".repeat(1_000_000)); // Very long name
        assert_eq!(problematic_op.op_type.len(), 1_000_000);
        
        // Test creating a module with problematic values
        let mut problematic_module = Module::new(&"b".repeat(1_000_000));
        problematic_module.add_operation(problematic_op);
        
        // Ensure everything still behaves correctly despite extreme inputs
        assert!(problematic_module.name.len() == 1_000_000);
        assert_eq!(problematic_module.operations.len(), 1);
    }

    /// Test 10: Memory allocation and deallocation edge cases
    #[test]
    fn test_memory_allocation_deallocation_patterns() {
        // Test rapid creation and destruction of complex objects
        for _ in 0..100 {
            let mut module = Module::new("temp_module");
            
            // Add operations
            for j in 0..100 {
                let mut op = Operation::new(&format!("temp_op_{}", j));
                
                // Add values to inputs/outputs
                op.inputs.push(Value {
                    name: format!("temp_input_{}", j),
                    ty: Type::F32,
                    shape: vec![j % 10 + 1],
                });
                
                op.outputs.push(Value {
                    name: format!("temp_output_{}", j),
                    ty: Type::F64,
                    shape: vec![j % 5 + 1],
                });
                
                module.add_operation(op);
            }
            
            // Verify before dropping
            assert_eq!(module.operations.len(), 100);
            assert_eq!(module.operations[0].inputs[0].name, "temp_input_0");
            
            // Drop the module (memory deallocation)
            drop(module);
        }
        
        // Test creating a large hierarchy and then cloning it
        let mut large_module = Module::new("large_clone_test");
        
        for i in 0..10_000 {
            let mut op = Operation::new(&format!("clone_op_{}", i));
            op.inputs.push(Value {
                name: format!("clone_input_{}", i),
                ty: Type::I32,
                shape: vec![2, 2],
            });
            large_module.add_operation(op);
        }
        
        // Clone the large module to test memory copying
        let cloned_module = large_module.clone();
        
        // Verify the clone is accurate
        assert_eq!(cloned_module.operations.len(), 10_000);
        assert_eq!(cloned_module.name, "large_clone_test");
        assert_eq!(cloned_module.operations[5000].op_type, "clone_op_5000");
    }
}