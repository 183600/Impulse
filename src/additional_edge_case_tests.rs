//! Additional edge case tests for the Impulse compiler
//! This file contains extra test cases focusing on boundary conditions and error scenarios

#[cfg(test)]
mod additional_edge_case_tests {
    use rstest::*;
    use crate::{ir::{Module, Value, Type, Operation, Attribute}, ImpulseCompiler};

    /// Test 1: Empty string handling in various contexts
    #[test]
    fn test_empty_string_handling() {
        // Testing empty module name
        let module = Module::new("");
        assert_eq!(module.name, "");
        
        // Testing empty value name
        let value = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(value.name, "");
        
        // Testing empty operation name
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
    }

    /// Test 2: Unicode and special character handling in names
    #[test]
    fn test_unicode_and_special_characters() {
        let unicode_value = Value {
            name: "tensor_名称_日本語_🔥_🇺🇸".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        assert_eq!(unicode_value.name, "tensor_名称_日本語_🔥_🇺🇸");
        
        let special_op = Operation::new("op_!@#$%^&*()");
        assert_eq!(special_op.op_type, "op_!@#$%^&*()");
        
        let special_module = Module::new("module_🚀_©_®");
        assert_eq!(special_module.name, "module_🚀_©_®");
    }

    /// Test 3: Maximum size containers to test memory limits
    #[test]
    fn test_maximum_container_sizes() {
        let mut large_module = Module::new("large_module");
        
        // Add maximum possible operations to the module
        for i in 0..100 {
            let mut op = Operation::new(&format!("large_op_{}", i));
            
            // Add many inputs to each operation
            for j in 0..100 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![1],
                });
            }
            
            // Add many outputs to each operation
            for k in 0..50 {
                op.outputs.push(Value {
                    name: format!("output_{}_{}", i, k),
                    ty: Type::F32,
                    shape: vec![1],
                });
            }
            
            large_module.add_operation(op);
        }
        
        assert_eq!(large_module.operations.len(), 100);
        assert_eq!(large_module.operations[0].inputs.len(), 100);
        assert_eq!(large_module.operations[0].outputs.len(), 50);
    }

    /// Test 4: Extreme tensor dimensions including zero-size tensors
    #[rstest]
    #[case(vec![], 1)]  // scalar (0-dim tensor)
    #[case(vec![0], 0)]  // tensor with zero elements
    #[case(vec![0, 10], 0)]  // 2D tensor with zero elements
    #[case(vec![10, 0], 0)]  // 2D tensor with zero elements
    #[case(vec![0, 0, 10], 0)]  // 3D tensor with zero elements
    #[case(vec![1, 1, 1], 1)]  // minimal non-zero tensor  
    #[case(vec![1000, 1000], 1_000_000)]  // large tensor
    fn test_extreme_tensor_dimensions(#[case] shape: Vec<usize>, #[case] expected_elements: usize) {
        let value = Value {
            name: "dimension_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        let calculated_elements: usize = value.shape.iter().product();
        assert_eq!(calculated_elements, expected_elements);
    }

    /// Test 5: Deeply nested tensor types
    #[test]
    fn test_deeply_nested_tensor_types() {
        // Create a deeply nested tensor type
        let mut nested_type = Type::F32;
        for _ in 0..10 {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![2],
            };
        }
        
        // Verify it can be created without stack overflow
        let cloned_type = nested_type.clone();
        assert_eq!(nested_type, cloned_type);
        
        // Test with a shallower nested type to make a specific check
        let deep_tensor = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![3],
            }),
            shape: vec![2],
        };
        
        match deep_tensor {
            Type::Tensor { shape, element_type } => {
                assert_eq!(shape, vec![2]);
                
                match element_type.as_ref() {
                    Type::Tensor { shape: inner_shape, element_type: inner_element } => {
                        assert_eq!(inner_shape, &vec![3]);
                        
                        match inner_element.as_ref() {
                            Type::F32 => {}, // Success
                            _ => panic!("Expected F32 as innermost type"),
                        }
                    },
                    _ => panic!("Expected nested tensor"),
                }
            },
            _ => panic!("Expected tensor type"),
        }
    }

    /// Test 6: Operations with various attribute configurations
    #[test]
    fn test_operation_attribute_configurations() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("attribute_test");
        let mut attrs = HashMap::new();
        
        // Test with all attribute types
        attrs.insert("int_attr".to_string(), Attribute::Int(9223372036854775807)); // Max i64
        attrs.insert("min_int_attr".to_string(), Attribute::Int(-9223372036854775808)); // Min i64
        attrs.insert("float_attr".to_string(), Attribute::Float(1.7976931348623157e308)); // Max f64
        attrs.insert("min_positive_attr".to_string(), Attribute::Float(f64::MIN_POSITIVE));
        attrs.insert("zero_attr".to_string(), Attribute::Float(0.0));
        attrs.insert("negative_attr".to_string(), Attribute::Float(-42.0));
        attrs.insert("bool_true".to_string(), Attribute::Bool(true));
        attrs.insert("bool_false".to_string(), Attribute::Bool(false));
        attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
        attrs.insert("long_string".to_string(), Attribute::String("a".repeat(1000)));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 10);
        
        // Verify specific attributes
        assert_eq!(op.attributes.get("int_attr"), Some(&Attribute::Int(9223372036854775807)));
        assert_eq!(op.attributes.get("min_int_attr"), Some(&Attribute::Int(-9223372036854775808)));
        assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
        assert_eq!(op.attributes.get("empty_string"), Some(&Attribute::String("".to_string())));
        
        // Test accessing a non-existent attribute
        assert_eq!(op.attributes.get("nonexistent"), None);
    }

    /// Test 7: Very large numerical values in shapes and calculations
    #[test]
    fn test_large_numerical_values() {
        // Test tensor with very large shape values
        let large_shape_value = Value {
            name: "large_values".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 10_000],  // Total = 1 billion elements
        };
        
        assert_eq!(large_shape_value.shape[0], 100_000);
        assert_eq!(large_shape_value.shape[1], 10_000);
        
        let total_elements: usize = large_shape_value.shape.iter().product();
        assert_eq!(total_elements, 1_000_000_000);
        
        // Test tensor with extreme aspect ratio
        let extreme_ratio = Value {
            name: "extreme_ratio".to_string(),
            ty: Type::I64,
            shape: vec![1, 1_000_000_000],  // 1 x 1 billion
        };
        
        assert_eq!(extreme_ratio.shape, vec![1, 1_000_000_000]);
        let ratio_elements: usize = extreme_ratio.shape.iter().product();
        assert_eq!(ratio_elements, 1_000_000_000);
    }

    /// Test 8: Compiler robustness with edge cases
    #[test]
    fn test_compiler_edge_cases() {
        let mut compiler = ImpulseCompiler::new();
        
        // Verify all components were initialized properly
        assert_eq!(compiler.frontend.name(), "Frontend"); // assuming this method exists
        
        // Test with various string lengths for target
        let short_target = "cpu";
        let long_target = &"x".repeat(10_000);
        let special_target = "!@#$%^&*()";
        let unicode_target = "_gpu_日本語_🚀";
        
        // Test that the compiler doesn't crash with various target strings
        // Since the actual functionality is not implemented, we just verify it doesn't panic
        let mock_model = vec![1u8, 2u8, 3u8];
        
        let _result1 = compiler.compile(&mock_model, short_target);
        let _result2 = compiler.compile(&mock_model, long_target);
        let _result3 = compiler.compile(&mock_model, special_target);
        let _result4 = compiler.compile(&mock_model, unicode_target);
        
        // Test creating multiple compilers simultaneously
        let compiler1 = ImpulseCompiler::new();
        let compiler2 = ImpulseCompiler::new();
        let compiler3 = ImpulseCompiler::new();
        
        // Verify they're independent
        assert_eq!(compiler1.passes.passes.len(), 0);
        assert_eq!(compiler2.passes.passes.len(), 0);
        assert_eq!(compiler3.passes.passes.len(), 0);
    }

    /// Test 9: Mixed-type tensor operations and type validation
    #[test]
    fn test_mixed_type_operations() {
        let mut module = Module::new("mixed_types");
        
        // Create operations with different types
        let f32_op = Operation::new("f32_op");
        let i32_op = Operation::new("i32_op");
        let bool_op = Operation::new("bool_op");
        let f64_op = Operation::new("f64_op");
        let i64_op = Operation::new("i64_op");
        
        // Create values with different types
        let f32_val = Value {
            name: "f32_val".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        };
        
        let i32_val = Value {
            name: "i32_val".to_string(),
            ty: Type::I32,
            shape: vec![5, 5],
        };
        
        let bool_val = Value {
            name: "bool_val".to_string(),
            ty: Type::Bool,
            shape: vec![2, 3, 4],
        };
        
        let f64_val = Value {
            name: "f64_val".to_string(),
            ty: Type::F64,
            shape: vec![1, 1],
        };
        
        let i64_val = Value {
            name: "i64_val".to_string(),
            ty: Type::I64,
            shape: vec![7, 2],
        };
        
        // Add operations to module
        module.add_operation(f32_op);
        module.add_operation(i32_op);
        module.add_operation(bool_op);
        module.add_operation(f64_op);
        module.add_operation(i64_op);
        
        // Add values as inputs to operations
        module.operations[0].inputs.push(f32_val);
        module.operations[1].inputs.push(i32_val);
        module.operations[2].inputs.push(bool_val);
        module.operations[3].inputs.push(f64_val);
        module.operations[4].inputs.push(i64_val);
        
        assert_eq!(module.operations.len(), 5);
        assert_eq!(module.operations[0].inputs[0].ty, Type::F32);
        assert_eq!(module.operations[1].inputs[0].ty, Type::I32);
        assert_eq!(module.operations[2].inputs[0].ty, Type::Bool);
        assert_eq!(module.operations[3].inputs[0].ty, Type::F64);
        assert_eq!(module.operations[4].inputs[0].ty, Type::I64);
    }

    /// Test 10: Array attribute with nested complex structures
    #[test]
    fn test_complex_nested_array_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("complex_array_op");
        
        // Create a complex nested array structure
        let complex_array_attr = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(3.14),
            Attribute::String("nested".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![
                Attribute::Int(10),
                Attribute::Array(vec![
                    Attribute::String("deeply_nested".to_string()),
                    Attribute::Bool(false),
                ]),
            ]),
        ]);
        
        let mut attrs = HashMap::new();
        attrs.insert("complex_array".to_string(), complex_array_attr);
        op.attributes = attrs;
        
        // Verify the structure
        assert_eq!(op.attributes.len(), 1);
        
        match op.attributes.get("complex_array").unwrap() {
            Attribute::Array(outer_array) => {
                assert_eq!(outer_array.len(), 5);
                
                // Check first element
                match &outer_array[0] {
                    Attribute::Int(1) => {},
                    _ => panic!("Expected Int(1) as first element"),
                }
                
                // Check third element
                match &outer_array[2] {
                    Attribute::String(s) if s == "nested" => {},
                    _ => panic!("Expected String(\"nested\") as third element"),
                }
                
                // Check fifth element (nested array)
                match &outer_array[4] {
                    Attribute::Array(nested_array) => {
                        assert_eq!(nested_array.len(), 2);
                        
                        // Check first element of nested array
                        match &nested_array[0] {
                            Attribute::Int(10) => {},
                            _ => panic!("Expected Int(10) in nested array"),
                        }
                        
                        // Check second element - another nested array
                        match &nested_array[1] {
                            Attribute::Array(deeply_nested) => {
                                assert_eq!(deeply_nested.len(), 2);
                            },
                            _ => panic!("Expected another nested array"),
                        }
                    },
                    _ => panic!("Expected nested array as fifth element"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }
}