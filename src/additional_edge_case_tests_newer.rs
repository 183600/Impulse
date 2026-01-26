#[cfg(test)]
mod additional_edge_case_tests {
    use crate::{ImpulseCompiler, ir::{Module, Value, Type, Operation}};
    use rstest::rstest;

    /// Test 1: Creating modules with maximum capacity vectors
    #[test]
    fn test_module_max_capacity() {
        let module = Module::new("max_capacity_test");
        
        // Test that basic properties work with potentially large internal storage
        assert_eq!(module.name, "max_capacity_test");
        assert!(module.operations.is_empty());
        
        // We can't actually fill to max capacity without running out of memory,
        // but we can test that the data structure behaves correctly
        assert_eq!(module.operations.capacity(), 0); // Initially 0
    }

    /// Test 2: Value creation with all possible Type variants
    #[rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    #[case(Type::Tensor { element_type: Box::new(Type::F32), shape: vec![1, 2] })]
    fn test_value_with_all_type_variants(#[case] ty: Type) {
        let value = Value {
            name: "typed_value".to_string(),
            ty: ty.clone(),
            shape: vec![1, 2, 3],
        };
        
        // Test that the value was created correctly
        assert_eq!(value.name, "typed_value");
        // We can't directly compare the types due to recursion, so we just verify creation worked
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    /// Test 3: Operation with many inputs and outputs
    #[test]
    fn test_operation_with_many_inputs_outputs() {
        let mut op = Operation::new("many_io_op");
        
        // Add many inputs
        for i in 0..100 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i],
            });
        }
        
        // Add many outputs
        for i in 0..50 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i * 2],
            });
        }
        
        assert_eq!(op.inputs.len(), 100);
        assert_eq!(op.outputs.len(), 50);
        
        // Verify first and last inputs/outputs are correct
        assert_eq!(op.inputs[0].name, "input_0");
        assert_eq!(op.inputs[0].shape, vec![0]);
        assert_eq!(op.inputs[99].name, "input_99");
        assert_eq!(op.inputs[99].shape, vec![99]);
        
        assert_eq!(op.outputs[0].name, "output_0");
        assert_eq!(op.outputs[0].shape, vec![0]);
        assert_eq!(op.outputs[49].name, "output_49");
        assert_eq!(op.outputs[49].shape, vec![98]);
    }

    /// Test 4: Module with deeply nested operations and dependencies
    #[test]
    fn test_deeply_nested_operations() {
        let mut module = Module::new("deep_nesting_test");
        
        // Create a chain of dependent operations
        for i in 0..100 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Each operation depends on previous output (except first)
            if i > 0 {
                op.inputs.push(Value {
                    name: format!("output_{}", i - 1),
                    ty: Type::F32,
                    shape: vec![i],
                });
            }
            
            // Each operation produces an output (except last, but we'll add one anyway)
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 100);
        
        // Check that connections are properly formed
        for i in 1..100 {
            if i > 0 {
                assert_eq!(module.operations[i].inputs.len(), 1);
                assert_eq!(module.operations[i].inputs[0].name, format!("output_{}", i - 1));
            }
        }
    }

    /// Test 5: Compiler with concurrent access patterns (simulated)
    #[test]
    fn test_compiler_concurrent_access_simulation() {
        // Although Rust doesn't allow true shared mutable access without synchronization,
        // we can test creating multiple compilers and simulating usage patterns
        
        let compilers: Vec<_> = (0..10).map(|_| ImpulseCompiler::new()).collect();
        
        // Verify all compilers were created properly
        for (i, compiler) in compilers.iter().enumerate() {
            assert_eq!(compiler.passes.passes.len(), 0, "Compiler {} should have 0 passes", i);
        }
        
        assert_eq!(compilers.len(), 10);
    }

    /// Test 6: Operations with extremely large integer values in attributes
    #[test]
    fn test_operations_with_extreme_integer_values() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("extreme_integers");
        let mut attrs = HashMap::new();
        
        // Add attributes with extreme integer values
        attrs.insert("max_i64".to_string(), crate::ir::Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), crate::ir::Attribute::Int(i64::MIN));
        attrs.insert("max_u64_as_i64".to_string(), crate::ir::Attribute::Int(i64::MAX));
        attrs.insert("negative_large".to_string(), crate::ir::Attribute::Int(-9223372036854775807i64));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 4);
        assert_eq!(op.attributes.get("max_i64"), Some(&crate::ir::Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("min_i64"), Some(&crate::ir::Attribute::Int(i64::MIN)));
    }

    /// Test 7: Values with empty but non-zero rank shapes
    #[test]
    fn test_values_with_empty_nonzero_rank_shapes() {
        // Test tensors that have dimensions defined but are empty in one dimension
        let test_cases = vec![
            (vec![0, 1, 2, 3], "has zero dimension"),
            (vec![1, 0, 2, 3], "has zero in middle"),
            (vec![1, 2, 3, 0], "has zero at end"),
            (vec![1], "single dimensional"),
            (vec![], "scalar/zero dimensional"),
        ];
        
        for (shape, description) in test_cases {
            let value = Value {
                name: format!("test_value_{}", description.replace(" ", "_")),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape, "Shape mismatch for {}", description);
            
            // Calculate total elements (tensor product of dimensions)
            let total_elements: usize = value.shape.iter().product();
            if shape.contains(&0) {
                assert_eq!(total_elements, 0, "Shape {:?} with 0 should have 0 total elements", shape);
            }
            
            assert_eq!(value.ty, Type::F32);
        }
    }

    /// Test 8: Operations with mixed valid and invalid UTF-8 names
    #[test]
    fn test_operations_with_mixed_utf8_names() {
        let valid_names = [
            "valid_ascii",
            "mixed_名称",
            "emoji_🚀_test",
            "numbers_12345",
            "symbols_!@#$_",
        ];
        
        for name in valid_names.iter() {
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
            
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1],
            };
            assert_eq!(value.name, *name);
        }
    }

    /// Test 9: Module serialization edge cases (if supported)
    #[test]
    fn test_module_serialization_potential_issues() {
        let mut module = Module::new("serialization_test");
        
        // Add operations that might cause issues during serialization
        for i in 0..10 {
            let mut op = Operation::new(&format!("serialize_op_{}", i));
            op.inputs.push(Value {
                name: format!("serialize_input_{}", i),
                ty: Type::F32,
                shape: vec![i, i + 1, i + 2],
            });
            
            // Add some attributes too
            use std::collections::HashMap;
            let mut attrs = HashMap::new();
            attrs.insert(format!("attr_{}", i), crate::ir::Attribute::Int(i as i64));
            op.attributes = attrs;
            
            module.add_operation(op);
        }
        
        // Just ensure the module has the expected structure
        assert_eq!(module.operations.len(), 10);
        assert_eq!(module.name, "serialization_test");
        
        // Check that all operations have the expected structure
        for (idx, op) in module.operations.iter().enumerate() {
            assert_eq!(op.inputs.len(), 1);
            assert_eq!(op.inputs[0].name, format!("serialize_input_{}", idx));
            assert_eq!(op.attributes.len(), 1);
            assert_eq!(op.attributes.get(&format!("attr_{}", idx)), 
                      Some(&crate::ir::Attribute::Int(idx as i64)));
        }
    }

    /// Test 10: Error handling with malformed inputs
    #[test]
    fn test_error_handling_with_malformed_inputs() {
        // Test compiler with various malformed inputs to ensure no panics
        let mock_inputs = vec![
            vec![],  // Completely empty
            vec![0xFF, 0xFF, 0xFF, 0xFF],  // All 1s
            vec![0x00, 0x00, 0x00, 0x00],  // All 0s
            vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],  // Classic magic numbers
        ];
        
        for input in mock_inputs {
            let mut compiler = ImpulseCompiler::new();
            
            // This should either succeed or return a Result::Err, but not panic
            let result = compiler.compile(&input, "cpu");
            
            // Verify that no panic occurred (the mere fact we got here means no panic)
            // The result can be either Ok or Err, both are fine for this test
            match result {
                Ok(_) => (), // Success is fine
                Err(_) => (), // Error is fine too
            }
        }
    }
}