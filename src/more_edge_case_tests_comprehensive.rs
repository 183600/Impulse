//! Additional comprehensive edge case tests for the Impulse compiler
//! This file includes tests for boundary conditions, special values, and complex scenarios

#[cfg(test)]
mod comprehensive_edge_case_tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;
    use rstest::*;

    /// Test 1: Zero-size tensors with various configurations
    #[test]
    fn test_zero_size_tensor_configurations() {
        let zero_configs = [
            vec![0],
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![0, 1, 0],
            vec![5, 0, 10],
            vec![0, 0, 0, 0],
        ];

        for config in &zero_configs {
            let value = Value {
                name: "zero_tensor".to_string(),
                ty: Type::F32,
                shape: config.to_vec(),
            };

            // Any tensor with a zero dimension should have 0 total elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, 0, "Shape {:?} should result in 0 elements", config);
            assert_eq!(value.shape, *config);
        }
    }

    /// Test 2: Extremely large tensor dimensions that might approach memory limits
    #[test]
    fn test_extremely_large_tensor_dimensions() {
        // Test with dimensions that would be huge but potentially manageable
        let huge_shape = vec![50_000, 50_000];
        let value = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: huge_shape,
        };

        assert_eq!(value.shape[0], 50_000);
        assert_eq!(value.shape[1], 50_000);

        // Calculate product safely (might be a huge number but should not overflow usize on 64-bit systems)
        let product: usize = value.shape.iter().product();
        assert_eq!(product, 50_000 * 50_000);
        
        // Test with maximum reasonable dimensions without causing overflow
        let max_reasonable = vec![100_000, 100_000];  // 10 billion elements
        let large_value = Value {
            name: "max_reasonable_tensor".to_string(),
            ty: Type::F32,
            shape: max_reasonable,
        };
        
        assert_eq!(large_value.shape, vec![100_000, 100_000]);
    }

    /// Test 3: Nested recursive types at maximum depth to test stack limits
    #[test]
    fn test_deeply_nested_recursive_types() {
        let mut current_type = Type::F32;
        
        // Create 500 levels of nesting (should not cause stack overflow in reasonable implementations)
        for level in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
            
            // Occasionally check that we can still clone and compare
            if level % 100 == 0 {
                let cloned = current_type.clone();
                assert_eq!(current_type, cloned);
            }
        }

        // Verify the final nested structure
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a Tensor type after deep nesting"),
        }
        
        // Ensure cloning works for deeply nested types
        let cloned_final = current_type.clone();
        assert_eq!(current_type, cloned_final);
    }

    /// Test 4: Operations with maximum number of attributes to test hash map limits
    #[test]
    fn test_operations_with_maximum_attributes() {
        let mut op = Operation::new("max_attr_op");
        let mut attrs = HashMap::new();
        
        // Add a large number of attributes to test hash map behavior
        // Using a more reasonable number to avoid potential collisions
        for i in 0..10_000 {
            attrs.insert(
                format!("attribute_{:05}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 10_000);
        
        // Verify some specific attributes exist
        // For i=0: format!("{:05}", 0) -> "00000" -> "attribute_00000", value -> "value_0"
        // For i=9999: format!("{:05}", 9999) -> "09999" -> "attribute_09999", value -> "value_9999"
        assert!(op.attributes.contains_key("attribute_00000"));  // First attribute
        assert!(op.attributes.contains_key("attribute_09999"));  // Last attribute
        
        // Check a couple of specific attributes to ensure they exist
        if let Some(attr) = op.attributes.get("attribute_00000") {
            match attr {
                Attribute::String(val) => assert_eq!(val, "value_0"),
                _ => panic!("Expected String attribute"),
            }
        } else {
            panic!("Attribute not found");
        }
        
        if let Some(attr) = op.attributes.get("attribute_09999") {
            match attr {
                Attribute::String(val) => assert_eq!(val, "value_9999"),
                _ => panic!("Expected String attribute"),
            }
        } else {
            panic!("Attribute not found");
        }
    }

    /// Test 5: Tensors with extreme dimensional ratios (very wide or very tall)
    #[test]
    fn test_tensors_with_extreme_aspect_ratios() {
        let extreme_cases = [
            (vec![1, 100_000_000], "very wide"),      // 1 row, 100M columns
            (vec![100_000_000, 1], "very tall"),      // 100M rows, 1 column  
            (vec![10_000, 10_000, 10_000], "cube"),   // Cube-like structure
            (vec![2, 2, 2, 2, 2, 25_000_000], "hyper"), // High-dimensional with one large dim
        ];
        
        for (shape, description) in &extreme_cases {
            let tensor = Value {
                name: format!("{}_tensor", description),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(tensor.shape, *shape, "Failed for {}", description);
            
            // Calculate total elements for each case
            let total_elements: usize = tensor.shape.iter().product();
            match description as &str {
                "very wide" | "very tall" => assert_eq!(total_elements, 100_000_000),
                "cube" => assert_eq!(total_elements, 10_000 * 10_000 * 10_000),
                "hyper" => assert_eq!(total_elements, 2 * 2 * 2 * 2 * 2 * 25_000_000),
                _ => panic!("Unknown test case"),
            }
        }
    }

    /// Test 6: Special floating-point values in tensor calculations
    #[rstest]
    #[case(std::f64::INFINITY)]
    #[case(std::f64::NEG_INFINITY)]
    #[case(std::f64::NAN)]
    #[case(-0.0)]
    #[case(std::f64::EPSILON)]
    #[case(std::f64::consts::PI)]
    #[case(std::f64::consts::E)]
    fn test_special_floating_point_attributes(#[case] special_value: f64) {
        let attr = Attribute::Float(special_value);
        
        if special_value.is_nan() {
            // Special case for NaN since NaN != NaN
            if let Attribute::Float(retrieved) = attr {
                assert!(retrieved.is_nan());
            } else {
                panic!("Expected Float attribute");
            }
        } else {
            // For other special values
            match attr {
                Attribute::Float(retrieved) => {
                    assert!(
                        (retrieved - special_value).abs() < f64::EPSILON || 
                        (retrieved.is_infinite() && special_value.is_infinite()),
                        "Values should be approximately equal or both infinite"
                    );
                },
                _ => panic!("Expected Float attribute"),
            }
        }
    }

    /// Test 7: Unicode and special character handling in identifiers
    #[test]
    fn test_unicode_identifiers_in_values_operations_modules() {
        let unicode_names = [
            "tensor_名称_日本語_🔥",
            "operation_🚀_unicode_🚀",
            "module_中文_العربية_🐍",
            "var_αβγδε_∑∏∯∮",
            "func_∀∃∄∅_∞",
        ];
        
        for name in &unicode_names {
            // Test Value with unicode name
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1, 2, 3],
            };
            assert_eq!(value.name, *name);
            
            // Test Operation with unicode name
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
            
            // Test Module with unicode name
            let module = Module::new(*name);
            assert_eq!(module.name, *name);
        }
    }

    /// Test 8: Empty collections and boundary size conditions
    #[test]
    fn test_empty_collections_and_boundary_conditions() {
        // Test empty module
        let empty_module = Module::new("");
        assert_eq!(empty_module.name, "");
        assert_eq!(empty_module.operations.len(), 0);
        assert_eq!(empty_module.inputs.len(), 0);
        assert_eq!(empty_module.outputs.len(), 0);
        
        // Test operation with empty components
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);
        
        // Test value with empty shape (scalar)
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![], // Empty shape = scalar
        };
        assert_eq!(scalar_value.shape.len(), 0);
        assert!(scalar_value.shape.is_empty());
        
        // Scalar should have 1 element
        let scalar_elements: usize = scalar_value.shape.iter().product();
        assert_eq!(scalar_elements, 1);
        
        // Test empty string attributes
        let empty_string_attr = Attribute::String("".to_string());
        if let Attribute::String(s) = empty_string_attr {
            assert_eq!(s, "");
        } else {
            panic!("Expected empty string attribute");
        }
        
        // Test empty array attributes
        let empty_array_attr = Attribute::Array(vec![]);
        if let Attribute::Array(v) = empty_array_attr {
            assert_eq!(v.len(), 0);
        } else {
            panic!("Expected empty array attribute");
        }
    }

    /// Test 9: Array attributes with various nesting levels
    #[test]
    fn test_complex_nested_array_attributes() {
        // Create a complex nested structure: [[1, 2], [3, [4, 5]], 6]
        let complex_nested = Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),  // [1, 2]
            Attribute::Array(vec![
                Attribute::Int(3),
                Attribute::Array(vec![Attribute::Int(4), Attribute::Int(5)])  // [4, 5]
            ]),  // [3, [4, 5]]
            Attribute::Int(6),  // 6
        ]);

        if let Attribute::Array(first_level) = complex_nested {
            assert_eq!(first_level.len(), 3);
            
            // Check [1, 2]
            if let Attribute::Array(second_level_0) = &first_level[0] {
                assert_eq!(second_level_0.len(), 2);
                assert_eq!(second_level_0[0], Attribute::Int(1));
                assert_eq!(second_level_0[1], Attribute::Int(2));
            } else {
                panic!("Expected array at index 0");
            }
            
            // Check [3, [4, 5]]
            if let Attribute::Array(second_level_1) = &first_level[1] {
                assert_eq!(second_level_1.len(), 2);
                assert_eq!(second_level_1[0], Attribute::Int(3));
                
                // Check [4, 5]
                if let Attribute::Array(third_level) = &second_level_1[1] {
                    assert_eq!(third_level.len(), 2);
                    assert_eq!(third_level[0], Attribute::Int(4));
                    assert_eq!(third_level[1], Attribute::Int(5));
                } else {
                    panic!("Expected nested array at third level");
                }
            } else {
                panic!("Expected array at index 1");
            }
            
            // Check 6
            assert_eq!(first_level[2], Attribute::Int(6));
        } else {
            panic!("Expected top-level array");
        }
    }

    /// Test 10: Memory stress test with many objects
    #[test]
    fn test_memory_stress_multiple_objects() {
        const COUNT: usize = 10_000;
        
        // Create many small objects to stress memory management
        let mut modules = Vec::with_capacity(COUNT);
        
        for i in 0..COUNT {
            let mut module = Module::new(&format!("stress_module_{}", i));
            
            // Add a few operations to each module
            for j in 0..5 {
                let mut op = Operation::new(&format!("stress_op_{}_{}", i, j));
                
                // Add inputs and outputs
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1, j + 1],
                });
                
                op.outputs.push(Value {
                    name: format!("output_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1, j + 2],
                });
                
                module.add_operation(op);
            }
            
            modules.push(module);
        }
        
        // Verify we created the expected number of modules
        assert_eq!(modules.len(), COUNT);
        
        // Check that some random modules have the expected structure
        assert_eq!(modules[0].name, "stress_module_0");
        assert_eq!(modules[0].operations.len(), 5);
        assert_eq!(modules[modules.len()/2].name, format!("stress_module_{}", modules.len()/2));
        assert_eq!(modules[COUNT-1].operations.len(), 5);
        
        // Explicitly drop to ensure cleanup
        drop(modules);
        
        // Test passes if no memory issues occurred during creation/destruction
        assert!(true);
    }
}