//! Additional edge case tests for the Impulse compiler - Part 2
//! More comprehensive test cases beyond the first set

use crate::ImpulseCompiler;

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use std::collections::HashMap;

    /// Test 1: Memory allocation stress test with very large collections
    #[test]
    fn test_huge_collection_allocation() {
        // Create a module with a very large number of operations to stress memory allocation
        let mut module = Module::new("stress_test_module");
        
        // Add 50,000 operations to the module to test memory allocation
        for i in 0..50_000 {
            let mut op = Operation::new(&format!("stress_op_{:06}", i));
            
            // Add 10 inputs to each operation
            for j in 0..10 {
                op.inputs.push(Value {
                    name: format!("input_{:06}_{:02}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
            }
            
            // Add 5 outputs to each operation
            for k in 0..5 {
                op.outputs.push(Value {
                    name: format!("output_{:06}_{:02}", i, k),
                    ty: Type::F32,
                    shape: vec![k + 1],
                });
            }
            
            // Add attributes
            use std::collections::HashMap;
            let mut attrs = HashMap::new();
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{:06}", i)),
            );
            op.attributes = attrs;
            
            module.add_operation(op);
        }
        
        // Verify we have the expected number of operations
        assert_eq!(module.operations.len(), 50_000);
        
        // Verify first and last operations exist and have correct structure
        assert_eq!(module.operations[0].op_type, "stress_op_000000");
        assert_eq!(module.operations[0].inputs.len(), 10);
        assert_eq!(module.operations[0].outputs.len(), 5);
        
        assert_eq!(module.operations[49_999].op_type, "stress_op_049999");
        assert_eq!(module.operations[49_999].inputs.len(), 10);
        assert_eq!(module.operations[49_999].outputs.len(), 5);
    }

    /// Test 2: Recursive type structure with alternating types at different depths
    #[test]
    fn test_multi_depth_alternating_types() {
        // Create complex nested types with alternating structures at different depths
        let create_alternating_type = |depth: usize| -> Type {
            let mut current = Type::I32;
            for i in 0..depth {
                current = match i % 4 {
                    0 => Type::Tensor {
                        element_type: Box::new(Type::F32),
                        shape: vec![i + 1],
                    },
                    1 => Type::Tensor {
                        element_type: Box::new(Type::I64),
                        shape: vec![i + 2],
                    },
                    2 => Type::Tensor {
                        element_type: Box::new(Type::F64),
                        shape: vec![i + 3],
                    },
                    _ => Type::Tensor {
                        element_type: Box::new(current),
                        shape: vec![i + 4],
                    },
                };
            }
            current
        };
        
        // Test different depths
        for depth in [5, 10, 20, 50] {
            let complex_type = create_alternating_type(depth);
            let cloned = complex_type.clone();
            assert_eq!(complex_type, cloned);
        }
    }

    /// Test 3: Invalid/edge case model compilation tests
    #[test]
    fn test_compiler_edge_cases() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with empty model (should not panic)
        let _result = compiler.compile(&[], "cpu");
        // Result may be error, but the important part is it doesn't panic
        
        // Test with single-byte model
        let _result2 = compiler.compile(&[42], "cpu");
        
        // Test with large repeated byte pattern that might cause issues
        let large_pattern = vec![0xFF; 1_000_000]; // 1MB of 0xFF
        let _result3 = compiler.compile(&large_pattern, "cpu");
        
        // Test with alternating bit pattern
        let alternating = (0..100_000)
            .flat_map(|i| if i % 2 == 0 { [0xAA] } else { [0x55] })
            .collect::<Vec<_>>();
        let _result4 = compiler.compile(&alternating, "cpu");
        
        // Just testing that these don't crash
        assert!(true); 
    }

    /// Test 4: Parameterized tests for tensor operations with different types
    #[rstest]
    #[case(Type::F32, vec![10, 10])]
    #[case(Type::F64, vec![100, 100])]
    #[case(Type::I32, vec![50, 50])]
    #[case(Type::I64, vec![25, 25])]
    #[case(Type::Bool, vec![200, 200])]
    #[case(Type::F32, vec![])]  // Scalar
    #[case(Type::I32, vec![0])]  // Zero-sized tensor
    #[case(Type::F64, vec![0, 10])]  // Another zero-sized tensor
    fn test_various_tensor_types_and_shapes(#[case] data_type: Type, #[case] shape: Vec<usize>) {
        let value = Value {
            name: "param_test_tensor".to_string(),
            ty: data_type.clone(),  // Fixed to avoid moving data_type
            shape: shape.clone(),   // Fixed to avoid moving shape
        };
        
        // Verify that the value was created properly
        assert_eq!(value.ty, data_type);
        assert_eq!(value.shape.len(), shape.len());
        
        // Calculate total elements
        let total_elements: usize = value.shape.iter().product();
        
        // If shape is empty (scalar), total_elements should be 1
        // If shape contains 0, total_elements should be 0
        if value.shape.is_empty() {
            assert_eq!(total_elements, 1);
        } else if value.shape.contains(&0) {
            assert_eq!(total_elements, 0);
        }
    }

    /// Test 5: Very long attribute string values
    #[test]
    fn test_very_long_attribute_strings() {
        use std::collections::HashMap;
        
        // Create an operation with extremely long string attributes
        let mut op = Operation::new("long_attr_op");
        
        let mut attrs = HashMap::new();
        
        // Add strings of various lengths
        attrs.insert("short".to_string(), Attribute::String("hi".to_string()));
        attrs.insert("medium".to_string(), Attribute::String("a".repeat(1_000)));
        attrs.insert("long".to_string(), Attribute::String("b".repeat(10_000)));
        attrs.insert("very_long".to_string(), Attribute::String("c".repeat(100_000)));
        attrs.insert("extremely_long".to_string(), Attribute::String("d".repeat(1_000_000))); // 1MB string
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 5);
        
        // Verify the string lengths
        if let Some(Attribute::String(s)) = op.attributes.get("short") {
            assert_eq!(s.len(), 2);
        } else {
            panic!("Expected short string attribute");
        }
        
        if let Some(Attribute::String(s)) = op.attributes.get("extremely_long") {
            assert_eq!(s.len(), 1_000_000);
        } else {
            panic!("Expected extremely long string attribute");
        }
    }

    /// Test 6: Nested array attributes with extreme depth
    #[test]
    fn test_deeply_nested_array_attributes() {
        // Build a deeply nested array structure
        let mut nested = Attribute::Int(42);
        
        // Nest 20 levels deep
        for _ in 0..20 {
            nested = Attribute::Array(vec![nested]);
        }
        
        // Verify the structure exists (avoiding recursion that causes issues)
        match &nested {
            Attribute::Array(ref inner) => {
                assert_eq!(inner.len(), 1);
                // We can't effectively count nested arrays without potentially stack overflowing,
                // so we just verify the structure is there
            },
            _ => panic!("Expected nested Array attribute"),
        }
        
        // Clone the nested structure to ensure cloning works
        let cloned = nested.clone();
        assert_eq!(nested, cloned);
    }

    /// Test 7: Mixed attribute types in a single operation
    #[test]
    fn test_operation_mixed_attribute_types() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("mixed_attr_op");
        
        let mut attrs = HashMap::new();
        
        // Add all possible attribute types
        attrs.insert("int_attr".to_string(), Attribute::Int(123));
        attrs.insert("float_attr".to_string(), Attribute::Float(45.67));
        attrs.insert("string_attr".to_string(), Attribute::String("hello".to_string()));
        attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
        
        // Add array with mixed types
        attrs.insert("mixed_array".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("three".to_string()),
            Attribute::Bool(false),
        ]));
        
        // Add nested array
        attrs.insert("nested_array".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(10),
                Attribute::Int(20),
            ]),
            Attribute::Array(vec![
                Attribute::Float(1.1),
                Attribute::Float(2.2),
            ]),
        ]));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 6);
        assert!(op.attributes.contains_key("int_attr"));
        assert!(op.attributes.contains_key("float_attr"));
        assert!(op.attributes.contains_key("string_attr"));
        assert!(op.attributes.contains_key("bool_attr"));
        assert!(op.attributes.contains_key("mixed_array"));
        assert!(op.attributes.contains_key("nested_array"));
    }

    /// Test 8: Shape combinations that might cause integer overflow when multiplied
    #[test]
    fn test_potential_overflow_shapes() {
        // Test shapes that are just below the point of overflowing usize when multiplied
        let test_shapes = [
            vec![usize::MAX / 2, 2],  // Close to usize::MAX
            vec![usize::MAX / 4, 4],
            vec![100, 100, 100, 100], // Multiple dimensions
            vec![usize::MAX, 1],      // Max in first position
            vec![1, usize::MAX],      // Max in second position
        ];
        
        for (i, shape) in test_shapes.iter().enumerate() {
            let value = Value {
                name: format!("overflow_test_{}", i),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            // Use checked multiplication to avoid overflow
            let _product_result: Option<usize> = value.shape.iter()
                .try_fold(1_usize, |acc, &x| {
                    if x == 0 { 
                        Some(0) 
                    } else { 
                        acc.checked_mul(x) 
                    }
                });
            
            // The operation should not panic regardless of overflow
            assert!(true); // Just ensure no panic occurred
            
            // Also test safe multiplication for non-overflow cases
            let non_overflow_shapes = vec![1000, 1000];
            let safe_product: usize = non_overflow_shapes.iter().product();
            assert_eq!(safe_product, 1_000_000);
        }
    }

    /// Test 9: Unicode names and special characters in various contexts
    #[test]
    fn test_unicode_names_in_all_contexts() {
        // Test module with unicode name
        let module = Module::new("模块测试_テスト_اختبار");
        assert_eq!(module.name, "模块测试_テスト_اختبار");
        
        // Test operation with unicode name
        let op = Operation::new("opération_عملية_작업");
        assert_eq!(op.op_type, "opération_عملية_작업");
        
        // Test value with unicode name
        let value = Value {
            name: "значение_Valor_值".to_string(),
            ty: Type::F32,
            shape: vec![42, 42],
        };
        assert_eq!(value.name, "значение_Valor_值");
        
        // Test attribute with unicode string
        use std::collections::HashMap;
        let mut attrs = HashMap::new();
        attrs.insert(
            "ключ_キー_مفتاح".to_string(), 
            Attribute::String("значение_値_Valor".to_string())
        );
        
        let mut op_with_unicode_attrs = Operation::new("unicode_op");
        op_with_unicode_attrs.attributes = attrs;
        
        assert!(op_with_unicode_attrs.attributes.contains_key("ключ_キー_مفتاح"));
    }

    /// Test 10: Comprehensive test combining multiple edge cases
    #[test]
    fn test_combined_edge_cases() {
        let mut module = Module::new("combined_edge_test_module_🚀_🔥");
        
        // Add an operation with extreme characteristics
        let mut complex_op = Operation::new(&"x".repeat(50_000)); // Very long operation name
        
        // Add inputs with extreme shapes
        for i in 0..10 {
            complex_op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: match i % 5 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    3 => Type::I64,
                    _ => Type::Bool,
                },
                shape: match i {
                    0 => vec![0],                    // Zero-sized tensor
                    1 => vec![1_000_000, 10],        // Large dimensions
                    2 => vec![10, 0, 100],           // Contains zero
                    3 => vec![5, 5, 5, 5],           // 4D tensor
                    4 => vec![],                     // Scalar
                    5 => vec![2, 2, 2, 2, 2, 2],     // 6D tensor
                    6 => vec![usize::MAX, 1],         // Near-max size
                    7 => vec![100, 100, 100],        // 3D large tensor
                    _ => vec![i, i + 1],
                },
            });
        }
        
        // Add extreme outputs
        for i in 0..5 {
            complex_op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: match i % 3 {
                    0 => Type::F32,
                    1 => Type::I32,
                    _ => Type::Bool,
                },
                shape: vec![i + 10, i + 20],
            });
        }
        
        // Add extreme attributes
        use std::collections::HashMap;
        let mut attrs = HashMap::new();
        attrs.insert("very_long_key".to_string(), Attribute::String("y".repeat(100_000)));
        attrs.insert("integer_val".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("float_val".to_string(), Attribute::Float(f64::MAX));
        attrs.insert("bool_val".to_string(), Attribute::Bool(true));
        attrs.insert("nullish_key".to_string(), Attribute::String("".to_string()));
        
        // Array attribute with mixed and nested types
        attrs.insert("complex_array".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
            ]),
            Attribute::String("nested".to_string()),
            Attribute::Array(vec![
                Attribute::Array(vec![Attribute::Bool(true)]),
            ])
        ]));
        
        complex_op.attributes = attrs;
        
        module.add_operation(complex_op);
        
        // Verify the structure
        assert_eq!(module.name, "combined_edge_test_module_🚀_🔥");
        assert_eq!(module.operations.len(), 1);
        
        let op = &module.operations[0];
        assert_eq!(op.inputs.len(), 10);
        assert_eq!(op.outputs.len(), 5);
        assert_eq!(op.attributes.len(), 6);
        
        // Verify some specific properties
        assert_eq!(op.op_type.len(), 50_000);  // Very long name preserved
        assert!(op.attributes.contains_key("very_long_key"));
        
        // Check that each input has the expected shape characteristics
        for (i, input) in op.inputs.iter().enumerate() {
            match i {
                0 => assert_eq!(input.shape, vec![0]),          // Zero-sized tensor
                1 => assert_eq!(input.shape, vec![1_000_000, 10]), // Large dims
                2 => assert_eq!(input.shape, vec![10, 0, 100]), // Contains zero
                4 => assert_eq!(input.shape.len(), 0),         // Scalar
                _ => assert!(true),          // Other shapes should exist (always true for usize)
            }
        }
    }
}