//! More edge case tests for the Impulse compiler
//! This file contains additional test cases focusing on complex boundary conditions

#[cfg(test)]
mod more_edge_case_tests {
    use rstest::*;
    use crate::{ir::{Module, Value, Type, Operation, Attribute}, ImpulseCompiler};
    #[test]
    fn test_extremely_large_attribute_values() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("large_values_test");
        let mut attrs = HashMap::new();
        
        // Test with maximum possible integer value
        attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
        
        // Test with special floating-point values
        attrs.insert("infinity".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("neg_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
        
        // Test with extremely large string
        let huge_string = "x".repeat(1_000_000); // 1MB string
        attrs.insert("huge_string".to_string(), Attribute::String(huge_string));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
        
        // Test NaN handling
        if let Some(Attribute::Float(val)) = op.attributes.get("nan_value") {
            assert!(val.is_nan());
        } else {
            panic!("Expected NaN value");
        }
    }

    /// Test 2: Module with maximum possible nested operations
    #[test]
    fn test_deeply_nested_module_operations() {
        let mut module = Module::new("nested_module");
        
        // Create a chain of operations where each connects to the next
        for i in 0..1000 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Add preceding operation as input if not first
            if i > 0 {
                op.inputs.push(Value {
                    name: format!("output_from_op_{}", i-1),
                    ty: Type::F32,
                    shape: vec![1],
                });
            }
            
            // Add output that can be used by next operation
            op.outputs.push(Value {
                name: format!("output_from_op_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 1000);
        assert_eq!(module.operations[0].outputs[0].name, "output_from_op_0");
        assert_eq!(module.operations[999].outputs[0].name, "output_from_op_999");
    }

    /// Test 3: Recursive type definitions that could cause infinite loops
    #[test]
    fn test_potentially_recursive_type_definitions() {
        // Create a recursive type structure that's actually acyclic but looks complex
        let complex_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),  // Changed from Tuple to F64
                shape: vec![3, 3],
            }),
            shape: vec![5, 5],
        };
        
        // Ensure we can clone and compare this complex type
        let cloned_type = complex_type.clone();
        assert_eq!(complex_type, cloned_type);
        
        // Check that the structure is as expected
        match &complex_type {
            Type::Tensor { shape, element_type } => {
                assert_eq!(shape, &vec![5, 5]);
                
                match element_type.as_ref() {
                    Type::Tensor { shape: inner_shape, element_type: inner_elem } => {
                        assert_eq!(inner_shape, &vec![3, 3]);
                        
                        match inner_elem.as_ref() {
                            Type::F64 => {},  // Changed from Tuple check to F64
                            _ => panic!("Expected F64 type"),
                        }
                    },
                    _ => panic!("Expected nested Tensor"),
                }
            },
            _ => panic!("Expected Tensor type"),
        }
    }

    /// Test 4: Edge cases with tensor shape calculations that could overflow
    #[rstest]
    #[case(vec![usize::MAX, 1], false)] // May overflow but shouldn't panic
    #[case(vec![100_000, 100_000], true)] // Should multiply to a reasonable number
    #[case(vec![1_000_000, 1_000_000], false)] // Likely to overflow
    fn test_tensor_shape_multiplication_edge_cases(#[case] shape: Vec<usize>, #[case] should_not_overflow: bool) {
        let value = Value {
            name: "shape_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Perform checked multiplication to avoid panic
        let mut product: Option<usize> = Some(1);
        for dim in &value.shape {
            if let Some(current) = product {
                product = current.checked_mul(*dim);
                if product.is_none() {
                    break; // Overflow detected
                }
            } else {
                break; // Already overflowed
            }
        }
        
        if should_not_overflow {
            assert!(product.is_some(), "Expected calculation to not overflow for shape {:?}", shape);
        }
    }

    /// Test 5: Operations with maximum attribute complexity
    #[test]
    fn test_maximally_complex_operation_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("complex_attr_op");
        let mut attrs = HashMap::new();
        
        // Add a highly complex nested attribute structure
        let deeply_nested = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(1),
                    Attribute::Float(2.5),
                    Attribute::String("deep".to_string()),
                    Attribute::Array(vec![Attribute::Bool(true)])
                ]),
                Attribute::Array(vec![
                    Attribute::String("nested".to_string()),
                    Attribute::Float(std::f64::consts::PI)
                ])
            ]),
            Attribute::Array(vec![
                Attribute::Bool(false),
                Attribute::Int(-999)
            ])
        ]);
        
        attrs.insert("deeply_nested".to_string(), deeply_nested);
        
        // Add many different primitive attributes
        for i in 0..100 {
            attrs.insert(format!("attr_{}", i), Attribute::Int(i as i64));
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 101); // 100 numbered + 1 complex
        
        // Verify the complex attribute exists
        assert!(op.attributes.contains_key("deeply_nested"));
    }

    /// Test 6: Boundary conditions with empty collections
    #[test]
    fn test_empty_collection_boundaries() {
        // Test empty module
        let empty_module = Module::new("");
        assert_eq!(empty_module.operations.len(), 0);
        assert_eq!(empty_module.inputs.len(), 0);
        assert_eq!(empty_module.outputs.len(), 0);
        
        // Test operation with empty collections
        let empty_op = Operation::new("empty_op");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);
        
        // Test value with empty shape (scalar)
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(scalar_value.shape.len(), 0);
        
        // Test operations with single-element collections
        let mut single_op = Operation::new("single_op");
        single_op.inputs.push(Value {
            name: "single_input".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        assert_eq!(single_op.inputs.len(), 1);
        
        single_op.outputs.push(Value {
            name: "single_output".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        assert_eq!(single_op.outputs.len(), 1);
    }

    /// Test 7: High-memory usage scenarios without allocation failures
    #[test]
    fn test_high_memory_usage_scenarios() {
        let mut module = Module::new("memory_intensive");
        
        // Create many operations with large names to test memory handling
        for i in 0..10_000 {
            let mut op = Operation::new(&format!("op_{}", "x".repeat(10))); // Repeated name pattern
            
            // Each operation has multiple inputs and outputs
            for j in 0..5 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
                
                op.outputs.push(Value {
                    name: format!("output_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
            }
            
            module.add_operation(op);
        }
        
        // Just verify the structure was created without crashing
        assert_eq!(module.operations.len(), 10_000);
        assert_eq!(module.operations[0].inputs.len(), 5);
    }

    /// Test 8: Concurrency and parallel access edge cases
    #[test]
    fn test_concurrent_access_patterns() {
        // Although Rust prevents real data races at compile time,
        // we can test creating multiple resources simultaneously
        use std::thread;

        // Create multiple compiler instances in different threads
        let handles: Vec<_> = (0..4).map(|_| {
            thread::spawn(|| {
                let compiler = ImpulseCompiler::new();
                // Verify basic creation works in thread
                assert_eq!(compiler.passes.passes.len(), 0);
                compiler
            })
        }).collect();

        let compilers: Vec<_> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();

        assert_eq!(compilers.len(), 4);
        
        // Verify all created properly
        for compiler in compilers {
            assert_eq!(compiler.passes.passes.len(), 0);
        }
    }

    /// Test 9: Invalid UTF-8 and encoding edge cases
    #[test]
    fn test_encoding_edge_cases() {
        // Create values with names containing special Unicode
        let unicode_value = Value {
            name: "test_🚀_unicode_✓_null\x00_char".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        // Ensure null character is preserved
        assert!(unicode_value.name.contains('\x00'));
        
        // Create operations with special names
        let special_op = Operation::new("operation_with_\x07_bell\x1B_escape");
        assert!(special_op.op_type.contains('\x07'));  // Bell character
        assert!(special_op.op_type.contains('\x1B'));  // Escape character
        
        // Create module with special characters
        let special_module = Module::new("module_with_emoji_🔥_and_symbols_@#$%");
        assert!(special_module.name.contains('🔥'));
        assert!(special_module.name.contains('#'));
    }

    /// Test 10: Arithmetic edge cases with tensor shapes
    #[rstest]
    #[case(vec![], 1)]          // Scalar has 1 element
    #[case(vec![0], 0)]         // Contains 0, product is 0
    #[case(vec![1, 1, 1], 1)]   // Multiple ones
    #[case(vec![2, 3, 4], 24)]  // Normal multiplication
    #[case(vec![10, 10, 10], 1000)]  // Larger multiplication
    #[case(vec![5, 0, 20], 0)]  // Zero in middle
    fn test_tensor_shape_arithmetic(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let value = Value {
            name: "arithmetic_test".to_string(),
            ty: Type::F32,
            shape: shape,
        };
        
        let calculated: usize = value.shape.iter().product();
        assert_eq!(calculated, expected_product);
        
        // Test that we can calculate this multiple times consistently
        let calculated_again: usize = value.shape.iter().product();
        assert_eq!(calculated, calculated_again);
    }
}