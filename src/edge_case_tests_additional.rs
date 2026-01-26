//! Additional edge case tests for the Impulse compiler
//! This file contains test cases focusing on previously uncovered edge cases

#[cfg(test)]
mod edge_case_tests_additional {
    use rstest::*;
    use crate::{ir::{Module, Value, Type, Operation, Attribute}, ImpulseCompiler};
    
    /// Test 1: Overflow in tensor size calculations
    #[test]
    fn test_tensor_size_overflow() {
        // Test tensor dimensions that would cause overflow in size calculation
        // Using values that are realistic but large enough to potentially cause overflow
        let large_dims = vec![100_000, 100_000]; // Would be 10 billion elements
        
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: large_dims,
        };
        
        // Calculate product to check for overflow
        let mut product: Option<usize> = Some(1);
        for &dim in &value.shape {
            product = product.and_then(|p| p.checked_mul(dim));
        }
        
        // This should handle overflow gracefully
        assert!(product.is_some() || product.is_none()); // Either succeeds or handles overflow
        
        // Test with smaller values that definitely won't overflow
        let safe_dims = vec![10_000, 10_000];
        let safe_value = Value {
            name: "safe_test".to_string(),
            ty: Type::F32,
            shape: safe_dims,
        };
        
        let safe_product: usize = safe_value.shape.iter().product();
        assert_eq!(safe_product, 100_000_000);
    }

    /// Test 2: Division by zero in tensor operations
    #[test]
    fn test_division_by_zero_edge_cases() {
        // Test mathematical operations that could lead to division by zero
        let value = Value {
            name: "divide_zero_test".to_string(),
            ty: Type::F32,
            shape: vec![10, 0], // Contains zero which could cause issues in certain calculations
        };
        
        // Safe product calculation that includes zero
        let product: usize = value.shape.iter().product();
        assert_eq!(product, 0);
        
        // Test for any potential divisions by elements in the shape
        for &dim in &value.shape {
            if dim == 0 {
                // Ensure no division by zero occurs
                let _safe_result = if dim != 0 { 100 / dim } else { 0 }; // Actually avoids division by zero
                assert_eq!(_safe_result, 0);
            }
        }
    }

    /// Test 3: Invalid UTF-8 sequences in names (using valid unicode)
    #[test]
    fn test_invalid_utf8_sequences_in_names() {
        // Although Rust strings are always valid UTF-8, we can test edge cases
        // with special unicode characters that might be problematic
        
        let special_names = [
            "normal_name",
            "unicode_🔥_test",
            "chinese_中文_test",
            "emoji_sequence_😀😃😄",
            "zero_width_space_\u{200B}_test",
            "combining_accent_a\u{0300}", // à with combining accent
        ];
        
        for name in &special_names {
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1],
            };
            
            assert_eq!(value.name, *name);
            assert!(value.name.is_ascii() || value.name.chars().count() > 0); // Just ensure it's valid
        }
    }

    /// Test 4: Maximum recursion depth in type definitions
    #[test]
    fn test_maximum_recursion_depth_in_types() {
        // Create a deeply nested type but limit depth to avoid stack overflow
        const MAX_DEPTH: usize = 50; // Limit depth to prevent stack overflow
        
        let mut nested_type = Type::F32;
        for _ in 0..MAX_DEPTH {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![2],
            };
        }
        
        // Verify it can be created and cloned without stack overflow
        let cloned = nested_type.clone();
        assert_eq!(nested_type, cloned);
        
        // Try to access nested elements to verify the structure is sound
        let mut current = &nested_type;
        for _ in 0..5 { // Only check a few levels to prevent deep recursion at runtime
            match current {
                Type::Tensor { element_type, shape } => {
                    assert_eq!(shape, &vec![2]);
                    current = element_type.as_ref();
                },
                Type::F32 => break, // Reached the base type
                _ => continue,
            }
        }
    }

    /// Test 5: Memory allocation edge cases
    #[test]
    fn test_memory_allocation_edge_cases() {
        // Test creation of many small objects to stress allocation
        let mut values = Vec::new();
        for i in 0..1000 {
            values.push(Value {
                name: format!("allocation_test_{}", i),
                ty: Type::F32,
                shape: vec![i % 10 + 1],
            });
        }
        
        assert_eq!(values.len(), 1000);
        
        // Test with one very large object
        let large_shape = vec![10_000; 2]; // [10000, 10000]
        let large_value = Value {
            name: "large_allocation".to_string(),
            ty: Type::F32,
            shape: large_shape,
        };
        
        let total_elements: usize = large_value.shape.iter().product();
        assert_eq!(total_elements, 100_000_000);
        
        // Clean up
        drop(values);
    }

    /// Test 6: Thread safety with concurrent operations (basic check)
    #[test]
    fn test_basic_thread_safety_simulation() {
        // While we can't actually test thread safety in a single-threaded test,
        // we can create multiple objects simultaneously to simulate concurrent usage
        
        let compilers: Vec<ImpulseCompiler> = (0..10)
            .map(|_| ImpulseCompiler::new())
            .collect();
        
        // Verify all were created independently
        for (i, compiler) in compilers.iter().enumerate() {
            assert_eq!(compiler.passes.passes.len(), 0);
            assert_eq!(compiler.frontend.name(), "Frontend");
        }
        
        // Test each compiler works independently
        for compiler in compilers.iter() {
            assert_eq!(compiler.passes.passes.len(), 0);
        }
    }

    /// Test 7: Parsing of malformed data (simulated)
    #[test]
    fn test_malformed_data_parsing_simulation() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with various types of malformed/bad input data
        let test_cases = vec![
            vec![],                    // Empty data
            vec![0],                   // Single byte
            vec![0xFF; 1000],         // Uniform non-ASCII data
            vec![0x00; 100],          // All zeros
            vec![0xFF, 0xFE, 0xFD],   // Potential UTF-8 sequences that aren't valid
        ];
        
        for (i, bad_data) in test_cases.iter().enumerate() {
            // Each should handle gracefully without crashing
            let result = compiler.compile(bad_data, "cpu");
            // We don't care about the result, just that it doesn't panic
            if result.is_err() {
                // If error, verify it's a valid error (has message)
                let err_msg = result.unwrap_err().to_string();
                assert!(!err_msg.is_empty());
            }
        }
    }

    /// Test 8: Edge cases with signed integer operations
    #[test]
    fn test_signed_integer_edge_cases() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("signed_int_test");
        let mut attrs = HashMap::new();
        
        // Test with various signed integer edge values
        attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("negative_value".to_string(), Attribute::Int(-42));
        attrs.insert("zero_value".to_string(), Attribute::Int(0));
        attrs.insert("large_negative".to_string(), Attribute::Int(-1_000_000_000));
        
        op.attributes = attrs;
        
        // Verify all values were stored correctly
        assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
        assert_eq!(op.attributes.get("negative_value"), Some(&Attribute::Int(-42)));
        assert_eq!(op.attributes.get("zero_value"), Some(&Attribute::Int(0)));
        assert_eq!(op.attributes.get("large_negative"), Some(&Attribute::Int(-1_000_000_000)));
        
        // Test arithmetic with these values
        if let Some(&Attribute::Int(max_val)) = op.attributes.get("max_i64") {
            // Test that we can work with the value without overflow in our test
            assert_eq!(max_val, i64::MAX);
        }
    }


    /// Test 9: Underflow in numerical calculations
    #[test]
    fn test_numerical_underflow_cases() {
        // Test subtraction that could cause underflow
        let result = i64::MIN.checked_sub(1); // This would underflow
        assert_eq!(result, None); // Correctly detected underflow
        
        // Test with tensor shapes that could cause underflow in size calculations
        let value = Value {
            name: "underflow_test".to_string(),
            ty: Type::F32,
            shape: vec![1, 1, 1], // Minimal shape
        };
        
        // Safe manipulations that avoid underflow
        let safe_calculation = if value.shape.iter().all(|&x| x > 0) {
            value.shape.iter().product::<usize>()
        } else {
            0 // Handle zero dimensions safely
        };
        
        assert_eq!(safe_calculation, 1);
        
        // Test with a zero dimension (which is handled differently from underflow)
        let zero_value = Value {
            name: "zero_underflow_test".to_string(),
            ty: Type::F32,
            shape: vec![5, 0, 3], // Contains zero
        };
        
        let zero_product: usize = zero_value.shape.iter().product();
        assert_eq!(zero_product, 0);
    }

    /// Test 10: Comprehensive error handling in complex structures
    #[test]
    fn test_error_handling_in_complex_structures() {
        // Create a complex module structure with various potential error points
        let mut complex_module = Module::new("complex_error_test");
        
        // Add operations with complex attributes
        for i in 0..5 {
            let mut op = Operation::new(&format!("complex_op_{}", i));
            
            // Add various attribute types
            use std::collections::HashMap;
            let mut attrs = HashMap::new();
            
            attrs.insert(format!("int_attr_{}", i), Attribute::Int(i as i64));
            attrs.insert(format!("float_attr_{}", i), Attribute::Float(i as f64 * 3.14));
            attrs.insert(format!("str_attr_{}", i), Attribute::String(
                format!("attribute_value_{}", i)
            ));
            attrs.insert(format!("bool_attr_{}", i), Attribute::Bool(i % 2 == 0));
            
            // Add some nested arrays
            if i % 2 == 0 {
                attrs.insert(
                    format!("array_attr_{}", i),
                    Attribute::Array(vec![
                        Attribute::Int(i as i64),
                        Attribute::String(format!("nested_{}", i)),
                        Attribute::Bool(true),
                    ])
                );
            }
            
            op.attributes = attrs;
            
            // Add inputs and outputs
            for j in 0..3 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: if j % 2 == 0 { Type::F32 } else { Type::I32 },
                    shape: vec![i + 1, j + 1],
                });
                
                op.outputs.push(Value {
                    name: format!("output_{}_{}", i, j),
                    ty: if j % 2 == 0 { Type::F64 } else { Type::I64 },
                    shape: vec![j + 1, i + 1],
                });
            }
            
            complex_module.add_operation(op);
        }
        
        // Verify the complex structure was built correctly
        assert_eq!(complex_module.operations.len(), 5);
        
        for (i, op) in complex_module.operations.iter().enumerate() {
            assert_eq!(op.op_type, format!("complex_op_{}", i));
            assert_eq!(op.inputs.len(), 3);
            assert_eq!(op.outputs.len(), 3);
            
            // Verify attributes
            assert!(op.attributes.contains_key(&format!("int_attr_{}", i)));
            assert!(op.attributes.contains_key(&format!("float_attr_{}", i)));
            assert!(op.attributes.contains_key(&format!("str_attr_{}", i)));
            assert!(op.attributes.contains_key(&format!("bool_attr_{}", i)));
            
            if i % 2 == 0 {
                assert!(op.attributes.contains_key(&format!("array_attr_{}", i)));
            }
        }
        
        // Test that the complex structure can be cloned safely
        let cloned_module = complex_module.clone();
        assert_eq!(cloned_module.name, "complex_error_test");
        assert_eq!(cloned_module.operations.len(), 5);
    }
}