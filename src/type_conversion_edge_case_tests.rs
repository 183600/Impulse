//! Additional edge case tests for type conversion and validation in the Impulse compiler
//! This file contains tests covering type conversion edge cases and validation scenarios

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::{
        validation_utils,
        ir_utils,
        math_utils,
    },
};

#[cfg(test)]
mod additional_edge_case_conversion_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_type_to_string_conversions() {
        // Test type-to-string conversion for all basic types
        assert_eq!(ir_utils::type_to_string(&Type::F32), "f32");
        assert_eq!(ir_utils::type_to_string(&Type::F64), "f64");
        assert_eq!(ir_utils::type_to_string(&Type::I32), "i32");
        assert_eq!(ir_utils::type_to_string(&Type::I64), "i64");
        assert_eq!(ir_utils::type_to_string(&Type::Bool), "bool");

        // Test tensor type string representation
        let tensor_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 4],
        };
        
        let tensor_str = ir_utils::type_to_string(&tensor_type);
        // The string should contain f32 and the shape information
        assert!(tensor_str.contains("f32"));
        assert!(tensor_str.contains("[2, 3, 4]"));

        // Test nested tensor representation
        let nested_tensor = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![1, 2],
            }),
            shape: vec![3],
        };
        
        let nested_str = ir_utils::type_to_string(&nested_tensor);
        assert!(nested_str.contains("i64"));
        assert!(nested_str.contains("[1, 2]"));
        assert!(nested_str.contains("[3]"));

        // Test deeply nested tensor
        let deep_tensor = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Bool),
                    shape: vec![4, 4, 4],
                }),
                shape: vec![2, 2],
            }),
            shape: vec![1, 1, 1],
        };
        
        let deep_str = ir_utils::type_to_string(&deep_tensor);
        assert!(deep_str.contains("bool")); // The innermost type should be visible
    }

    #[test]
    fn test_module_validation_edge_cases() {
        // Test validation of a properly constructed module
        let mut valid_module = Module::new("valid_module");
        
        // Add inputs with unique names
        valid_module.inputs.push(Value {
            name: "input_a".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        valid_module.inputs.push(Value {
            name: "input_b".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // Add outputs with unique names
        valid_module.outputs.push(Value {
            name: "output_result".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // Add operations
        let mut op = Operation::new("add_op");
        op.inputs.push(Value {
            name: "op_input".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        op.outputs.push(Value {
            name: "op_output".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        valid_module.add_operation(op);
        
        // This should pass validation
        assert!(validation_utils::validate_module(&valid_module).is_ok());

        // Test module validation with duplicate input names (should fail)
        let mut duplicate_input_module = Module::new("duplicate_input");
        duplicate_input_module.inputs.push(Value {
            name: "same_name".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        duplicate_input_module.inputs.push(Value {
            name: "same_name".to_string(),  // Duplicate name
            ty: Type::F32,
            shape: vec![10],
        });
        
        let validation_result = validation_utils::validate_module(&duplicate_input_module);
        assert!(validation_result.is_err());

        // Test module validation with duplicate output names (should fail)
        let mut duplicate_output_module = Module::new("duplicate_output");
        duplicate_output_module.outputs.push(Value {
            name: "same_output".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        duplicate_output_module.outputs.push(Value {
            name: "same_output".to_string(),  // Duplicate name
            ty: Type::F32,
            shape: vec![10],
        });
        
        let validation_result2 = validation_utils::validate_module(&duplicate_output_module);
        assert!(validation_result2.is_err());
    }

    #[test]
    fn test_operation_validation_edge_cases() {
        // Test operation with unique input/output names (should pass)
        let mut valid_op = Operation::new("valid_op");
        valid_op.inputs.push(Value {
            name: "input_x".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        valid_op.outputs.push(Value {
            name: "output_y".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        assert!(validation_utils::validate_operation(&valid_op).is_ok());

        // Test operation with duplicate names in inputs/outputs (should fail)
        let mut conflicting_op = Operation::new("conflicting_op");
        conflicting_op.inputs.push(Value {
            name: "same_name".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        conflicting_op.outputs.push(Value {
            name: "same_name".to_string(),  // Same name as input (this should conflict)
            ty: Type::F32,
            shape: vec![5],
        });
        
        // This should fail validation
        let result = validation_utils::validate_operation(&conflicting_op);
        assert!(result.is_err());
        
        // Test operation with multiple identical input names (should fail)
        let mut dup_inputs_op = Operation::new("dup_inputs_op");
        dup_inputs_op.inputs.push(Value {
            name: "dupe_name".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        dup_inputs_op.inputs.push(Value {
            name: "dupe_name".to_string(),  // Duplicate in inputs
            ty: Type::F32,
            shape: vec![5],
        });
        
        let dup_input_result = validation_utils::validate_operation(&dup_inputs_op);
        assert!(dup_input_result.is_err());

        // Test operation with multiple identical output names (should fail)
        let mut dup_outputs_op = Operation::new("dup_outputs_op");
        dup_outputs_op.outputs.push(Value {
            name: "dupe_output".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        dup_outputs_op.outputs.push(Value {
            name: "dupe_output".to_string(),  // Duplicate in outputs
            ty: Type::F32,
            shape: vec![5],
        });
        
        let dup_output_result = validation_utils::validate_operation(&dup_outputs_op);
        assert!(dup_output_result.is_err());
    }

    #[test]
    fn test_math_utils_edge_cases() {
        // Test GCD with edge cases
        assert_eq!(math_utils::gcd(0, 0), 0);
        assert_eq!(math_utils::gcd(0, 5), 5);
        assert_eq!(math_utils::gcd(5, 0), 5);
        assert_eq!(math_utils::gcd(1, 1000000), 1);
        assert_eq!(math_utils::gcd(17, 17), 17);
        assert_eq!(math_utils::gcd(48, 18), 6);

        // Test LCM with edge cases
        assert_eq!(math_utils::lcm(0, 0), 0);
        assert_eq!(math_utils::lcm(0, 5), 0);
        assert_eq!(math_utils::lcm(5, 0), 0);
        assert_eq!(math_utils::lcm(1, 100), 100);
        assert_eq!(math_utils::lcm(4, 6), 12);
        assert_eq!(math_utils::lcm(7, 11), 77); // Prime numbers

        // Test round_up_to_multiple with edge cases
        assert_eq!(math_utils::round_up_to_multiple(0, 5), 0);
        assert_eq!(math_utils::round_up_to_multiple(5, 0), 5); // When multiple is 0, return original
        assert_eq!(math_utils::round_up_to_multiple(7, 1), 7); // Round to 1 is identity
        assert_eq!(math_utils::round_up_to_multiple(7, 7), 7); // Exact match
        assert_eq!(math_utils::round_up_to_multiple(8, 7), 14); // Round up to next multiple
        assert_eq!(math_utils::round_up_to_multiple(1, 10), 10); // Round up to larger multiple
        assert_eq!(math_utils::round_up_to_multiple(10, 10), 10); // Exact match at higher number

        // Test next_power_of_2 with edge cases
        assert_eq!(math_utils::next_power_of_2(0), 1);  // Convention for 0
        assert_eq!(math_utils::next_power_of_2(1), 1);  // 1 is a power of 2
        assert_eq!(math_utils::next_power_of_2(2), 2);  // 2 is a power of 2
        assert_eq!(math_utils::next_power_of_2(3), 4);  // Next after 3 is 4
        assert_eq!(math_utils::next_power_of_2(4), 4);  // 4 is a power of 2
        assert_eq!(math_utils::next_power_of_2(5), 8);  // Next after 5 is 8
        assert_eq!(math_utils::next_power_of_2(8), 8);  // 8 is a power of 2
        assert_eq!(math_utils::next_power_of_2(9), 16); // Next after 9 is 16
        assert_eq!(math_utils::next_power_of_2(15), 16); // Next after 15 is 16
        assert_eq!(math_utils::next_power_of_2(16), 16); // 16 is a power of 2
        assert_eq!(math_utils::next_power_of_2(17), 32); // Next after 17 is 32
    }

    #[test]
    fn test_math_utils_property_checks() {
        // Test the mathematical property that gcd(a,b) * lcm(a,b) = a * b
        // (when neither a nor b is 0)
        let test_pairs = vec![(12, 18), (15, 25), (7, 11), (21, 14)];
        
        for (a, b) in test_pairs {
            if a != 0 && b != 0 {
                let gcd_val = math_utils::gcd(a, b);
                let lcm_val = math_utils::lcm(a, b);
                assert_eq!(gcd_val * lcm_val, a * b, "Property failed for ({}, {})", a, b);
            }
        }

        // Test that a number is a power of 2 if and only if it equals its next power of 2
        let powers_of_2 = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        for pow in &powers_of_2 {
            assert_eq!(math_utils::next_power_of_2(*pow), *pow, "{} should equal its next power of 2", pow);
        }

        // Test that non-powers of 2 are not equal to their next power of 2
        let non_powers = vec![3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15];
        for &non_pow in &non_powers {
            assert!(math_utils::next_power_of_2(non_pow) > non_pow, "{} should not equal its next power of 2", non_pow);
        }
    }

    #[test]
    fn test_attribute_equality_deeply_nested() {
        // Test equality for deeply nested attributes
        
        // Create two identical nested structures
        let attr1 = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.5),
                Attribute::Array(vec![
                    Attribute::String("nested".to_string()),
                    Attribute::Bool(true),
                ])
            ])
        ]);

        let attr2 = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.5),
                Attribute::Array(vec![
                    Attribute::String("nested".to_string()),
                    Attribute::Bool(true),
                ])
            ])
        ]);

        assert_eq!(attr1, attr2);

        // Create a slightly different nested structure
        let attr3 = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.5),
                Attribute::Array(vec![
                    Attribute::String("different".to_string()), // Changed this
                    Attribute::Bool(true),
                ])
            ])
        ]);

        assert_ne!(attr1, attr3);
        
        // Create another different structure
        let attr4 = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.5),
                Attribute::Array(vec![
                    Attribute::String("nested".to_string()),
                    Attribute::Bool(false), // Changed this
                ])
            ])
        ]);

        assert_ne!(attr1, attr4);
    }

    #[test]
    fn test_complex_validation_scenarios() {
        // Test validation of a complex module with many interconnected operations
        let mut complex_module = Module::new("complex_module");
        
        // Add inputs
        complex_module.inputs.push(Value {
            name: "input_image".to_string(),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224],
        });
        
        complex_module.inputs.push(Value {
            name: "conv_weights".to_string(),
            ty: Type::F32,
            shape: vec![64, 3, 7, 7],
        });
        
        // Add outputs
        complex_module.outputs.push(Value {
            name: "final_output".to_string(),
            ty: Type::F32,
            shape: vec![1, 64, 112, 112],
        });
        
        // Add multiple operations
        for i in 0..100 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Add inputs and outputs with unique names
            op.inputs.push(Value {
                name: format!("input_op_{}", i),
                ty: Type::F32,
                shape: vec![i + 1, i + 1],
            });
            
            op.outputs.push(Value {
                name: format!("output_op_{}", i),
                ty: Type::F32,
                shape: vec![i + 1, i + 1],
            });
            
            // Add some attributes
            let mut attrs = HashMap::new();
            attrs.insert(format!("attr_{}", i), Attribute::Int(i as i64));
            op.attributes = attrs;
            
            complex_module.add_operation(op);
        }
        
        // This complex module should validate successfully
        let result = validation_utils::validate_module(&complex_module);
        assert!(result.is_ok(), "Complex module should validate successfully, error: {:?}", result.err());

        // Check that all operations were added
        assert_eq!(complex_module.operations.len(), 100);
        
        // Check that inputs and outputs maintain their integrity
        assert_eq!(complex_module.inputs.len(), 2);
        assert_eq!(complex_module.outputs.len(), 1);
        
        // Verify the first and last operations have correct names
        assert_eq!(complex_module.operations[0].op_type, "op_0");
        assert_eq!(complex_module.operations[99].op_type, "op_99");
    }

    #[test]
    fn test_tensor_size_with_extreme_dimensions() {
        // Test calculating tensor sizes with extreme but valid dimensions
        // These won't cause overflow in our implementation but test edge cases
        
        // Test tensor with very large but valid single dimension
        let large_single_dim = Value {
            name: "large_single".to_string(),
            ty: Type::F32,
            shape: vec![10_000_000],
        };
        
        let size_result = ir_utils::calculate_tensor_size(&large_single_dim.ty, &large_single_dim.shape);
        assert!(size_result.is_ok());
        if let Ok(size) = size_result {
            assert_eq!(size, 10_000_000 * 4); // 4 bytes per F32
        }
        
        // Test mixed large and small dimensions
        let mixed_dims = Value {
            name: "mixed_dims".to_string(),
            ty: Type::I64,
            shape: vec![1000_000, 10, 10], // 100 million elements
        };
        
        let mixed_size_result = ir_utils::calculate_tensor_size(&mixed_dims.ty, &mixed_dims.shape);
        assert!(mixed_size_result.is_ok());
        if let Ok(size) = mixed_size_result {
            assert_eq!(size, 1000_000 * 10 * 10 * 8); // 8 bytes per I64
        }
        
        // Test with 4D tensor of common dimensions in neural networks
        let cnn_tensor = Value {
            name: "cnn_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, 512, 7, 7], // Common in CNNs: [batch, channels, height, width]
        };
        
        let cnn_size_result = ir_utils::calculate_tensor_size(&cnn_tensor.ty, &cnn_tensor.shape);
        assert!(cnn_size_result.is_ok());
        if let Ok(size) = cnn_size_result {
            assert_eq!(size, 1 * 512 * 7 * 7 * 4); // 4 bytes per F32
        }
    }

    #[test]
    fn test_module_operation_counters() {
        // Test the utility functions for counting operations by type
        
        let mut module = Module::new("counter_test");
        
        // Add operations of different types
        for _ in 0..10 {
            module.add_operation(Operation::new("matmul"));
        }
        
        for _ in 0..5 {
            module.add_operation(Operation::new("add"));
        }
        
        for _ in 0..3 {
            module.add_operation(Operation::new("relu"));
        }
        
        // Add some more matmul operations
        for _ in 0..7 {
            module.add_operation(Operation::new("matmul"));
        }
        
        // Count operations by type
        let counts = ir_utils::count_operations_by_type(&module);
        
        assert_eq!(counts.get("matmul"), Some(&17)); // 10 + 7
        assert_eq!(counts.get("add"), Some(&5));
        assert_eq!(counts.get("relu"), Some(&3));
        
        // Verify total count
        let total_ops: usize = counts.values().sum();
        assert_eq!(total_ops, 25); // 17 + 5 + 3
        assert_eq!(module.operations.len(), 25);
        
        // Test finding operations by type
        let matmul_ops = ir_utils::find_operations_by_type(&module, "matmul");
        assert_eq!(matmul_ops.len(), 17);
        
        let add_ops = ir_utils::find_operations_by_type(&module, "add");
        assert_eq!(add_ops.len(), 5);
        
        let relu_ops = ir_utils::find_operations_by_type(&module, "relu");
        assert_eq!(relu_ops.len(), 3);
        
        // Test finding non-existent operations
        let nonexistent_ops = ir_utils::find_operations_by_type(&module, "conv2d");
        assert_eq!(nonexistent_ops.len(), 0);
    }

    #[test]
    fn test_type_extensions_with_complex_nested_types() {
        // Test the TypeExtensions trait with complex nested types
        use impulse::ir::TypeExtensions;
        
        // Create a deeply nested type
        let mut current_type = Type::F32;
        for i in 0..50 { // Create 50 levels of nesting
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 3 + 1], // Varying shapes: [1], [2], [3], [1], [2], ...
            };
        }
        
        // The deeply nested type should still be valid
        assert!(current_type.is_valid_type());
        
        // Clone and test equality
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        assert!(cloned_type.is_valid_type());
        
        // Modify the original and test inequality
        let modified_type = Type::Tensor {
            element_type: Box::new(cloned_type),
            shape: vec![5], // Different shape
        };
        
        assert_ne!(current_type, modified_type);
        assert!(modified_type.is_valid_type());
    }

    #[test]
    fn test_operation_with_extreme_attribute_structures() {
        // Test operation with complex attribute structures
        let mut op = Operation::new("complex_attr_op");
        
        // Create a complex nested attribute structure
        let complex_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(3.14159),
                    Attribute::Array(vec![
                        Attribute::String("deeply_nested".to_string()),
                        Attribute::Bool(true),
                        Attribute::Array(vec![
                            Attribute::Int(42),
                            Attribute::Array(vec![
                                Attribute::String("even_deeper".to_string())
                            ])
                        ])
                    ])
                ])
            ]),
            Attribute::Int(999),
            Attribute::Array(vec![
                Attribute::Bool(false),
                Attribute::Float(2.71828),
            ])
        ]);
        
        op.attributes.insert("complex_structure".to_string(), complex_attr);
        
        // Add a few more attributes
        op.attributes.insert("simple_int".to_string(), Attribute::Int(123));
        op.attributes.insert("simple_string".to_string(), Attribute::String("hello".to_string()));
        
        // Verify the operation has the right number of attributes
        assert_eq!(op.attributes.len(), 3);
        
        // Check that we can access the complex structure
        let complex_retrieved = op.attributes.get("complex_structure").unwrap();
        
        match complex_retrieved {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 3);
                
                // Verify the complex nested structure hasn't corrupted
                match &outer[0] {
                    Attribute::Array(_) => {}, // Expected
                    _ => panic!("Expected nested array"),
                }
                
                match &outer[1] {
                    Attribute::Int(999) => {}, // Expected
                    _ => panic!("Expected Int(999)"),
                }
            },
            _ => panic!("Expected complex structure to be an Array"),
        }
    }
}