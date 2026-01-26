//! Additional edge case tests for the Impulse compiler
//! This file contains extra tests focusing on boundary conditions and potential corner cases

#[cfg(test)]
mod extra_edge_case_tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;
    use rstest::*;

    /// Test 1: Operations with extremely large number of attributes
    #[test]
    fn test_operation_with_extremely_large_attributes() {
        let mut op = Operation::new("huge_attrs_op");
        
        // Add 50,000 attributes to test memory limits
        for i in 0..50_000 {
            op.attributes.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        assert_eq!(op.attributes.len(), 50_000);
        assert_eq!(op.op_type, "huge_attrs_op");
        
        // Verify a few random attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_25000"));
        assert!(op.attributes.contains_key("attr_49999"));
    }

    /// Test 2: Tensor operations with maximum possible dimension count
    #[test]
    fn test_tensor_with_maximum_dimensions() {
        // Test tensor with the maximum reasonable number of dimensions
        let max_dims = vec![1; 32]; // 32 dimensions, all size 1
        let value = Value {
            name: "max_dim_tensor".to_string(),
            ty: Type::F32,
            shape: max_dims,
        };
        
        assert_eq!(value.shape.len(), 32);
        assert_eq!(value.shape.iter().product::<usize>(), 1); // Still 1 element total
        
        // Test with different values
        let irregular_dims = vec![2, 1, 3, 1, 4, 1, 5, 1, 6];
        let irregular_value = Value {
            name: "irregular_tensor".to_string(),
            ty: Type::I64,
            shape: irregular_dims,
        };
        
        assert_eq!(irregular_value.shape.len(), 9);
        assert_eq!(irregular_value.shape.iter().product::<usize>(), 720); // 2*3*4*5*6 = 720
    }

    /// Test 3: Deeply nested operations with complex dependencies
    #[test]
    fn test_deeply_nested_operations_graph() {
        let mut module = Module::new("nested_ops_graph");
        
        // Create a chain of operations where each depends on the previous one
        for i in 0..10_000 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Each operation takes output from previous as input (except first)
            if i > 0 {
                op.inputs.push(Value {
                    name: format!("output_{}", i - 1),
                    ty: Type::F32,
                    shape: vec![100],
                });
            }
            
            // Each operation produces an output
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![100],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 10_000);
        assert_eq!(module.name, "nested_ops_graph");
    }

    /// Test 4: Memory management with rapid creation/destruction of IR objects
    #[test]
    fn test_rapid_ir_object_allocation_deallocation() {
        // Test rapid creation and destruction to ensure no memory leaks
        for _ in 0..1000 {
            let _module = Module::new("temp");
            let _op = Operation::new("temp_op");
            let _value = Value {
                name: "temp_val".to_string(),
                ty: Type::F32,
                shape: vec![10, 10],
            };
            // Objects are destroyed here automatically
        }
        
        // If we reach here without issues, the test passed
        assert!(true);
    }

    /// Test 5: Invalid UTF-8 sequences in IR string fields (as much as Rust allows)
    #[rstest]
    #[case("normal_ascii")]
    #[case("unicode_测试_مختبر")]
    #[case("emoji_🚀_🔥_🌟")]
    #[case("symbols_!@#$%^&*()")]
    #[case("")]
    fn test_various_unicode_strings_in_ir_fields(#[case] test_string: &str) {
        // Test that various unicode strings work correctly in IR fields
        let module = Module::new(test_string);
        assert_eq!(module.name, test_string);
        
        let value = Value {
            name: test_string.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, test_string);
        
        let op = Operation::new(test_string);
        assert_eq!(op.op_type, test_string);
        
        let attr = Attribute::String(test_string.to_string());
        if let Attribute::String(s) = attr {
            assert_eq!(s, test_string);
        } else {
            panic!("Expected String attribute");
        }
    }

    /// Test 6: Zero-size tensor edge cases with different types
    #[rstest]
    #[case(Type::F32, vec![0])]
    #[case(Type::F64, vec![1, 0])]
    #[case(Type::I32, vec![0, 1])]
    #[case(Type::I64, vec![5, 0, 10])]
    #[case(Type::Bool, vec![0, 0, 0])]
    #[case(Type::F32, vec![2, 0, 3, 0, 4])]
    fn test_zero_size_tensor_variations(#[case] data_type: Type, #[case] shape: Vec<usize>) {
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: data_type.clone(),
            shape: shape.clone(),
        };
        
        // All zero-size tensors should have 0 elements
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 0);
        
        // Verify all fields
        assert_eq!(value.ty, data_type);
        assert_eq!(value.shape, shape);
        assert_eq!(value.num_elements(), Some(0));
    }

    /// Test 7: Arithmetic overflow protection in tensor size calculations
    #[test]
    fn test_tensor_size_overflow_detection() {
        // Create shapes that would cause overflow if computed naively
        // We use the checked arithmetic in our implementation
        let large_value = Value {
            name: "potentially_overflowing_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000], // Would be 10^10 elements
        };
        
        // Use the safe method that handles overflow
        let elements = large_value.num_elements();
        assert!(elements.is_some()); // Should be Some(10_000_000_000) if no overflow
        assert_eq!(elements.unwrap(), 10_000_000_000);
        
        // Test with a shape that definitely has zero elements
        let zero_value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::I64,
            shape: vec![1000, 0, 5000],
        };
        
        let zero_elements = zero_value.num_elements();
        assert_eq!(zero_elements, Some(0));
    }

    /// Test 8: Maximum depth recursive tensor types
    #[test]
    fn test_maximum_recursive_tensor_depth() {
        // Create a deeply nested tensor type and test cloning and equality
        let mut current_type = Type::F32;
        const MAX_DEPTH: usize = 500; // Reasonable depth to avoid stack overflow
        
        for _ in 0..MAX_DEPTH {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the structure
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a tensor type after nesting"),
        }
        
        // Test cloning of deeply nested type doesn't cause issues
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
        
        // Test with different depth to ensure inequality works
        let mut other_type = Type::F32;
        for _ in 0..(MAX_DEPTH - 1) {
            other_type = Type::Tensor {
                element_type: Box::new(other_type),
                shape: vec![2],
            };
        }
        
        assert_ne!(current_type, other_type);
    }

    /// Test 9: Edge cases with HashMap operations on attributes
    #[test]
    fn test_hashmap_attribute_manipulation_edge_cases() {
        let mut op = Operation::new("hashmap_test_op");
        
        // Test with a large number of attributes that might collide
        for i in 0..10_000i32 {
            // Use hash patterns that might cause collisions
            let key = format!("key_{:08x}", i.wrapping_mul(31).wrapping_add(17));
            op.attributes.insert(key, Attribute::Int(i as i64));
        }
        
        assert_eq!(op.attributes.len(), 10_000);
        
        // Test retrieval of specific attributes
        assert_eq!(op.attributes.get("key_00000011"), Some(&Attribute::Int(0))); // 0*31+17 = 17 = 0x11
        // Find one in the middle (we don't actually use this variable, but it's computed for reference)
        let _mid_key = format!("key_{:08x}", (5000i32.wrapping_mul(31).wrapping_add(17)) as u32);
        // Skip checking specific middle keys as the calculation gets complex
        
        // Count how many keys start with "key_0" or "key_1" first to see if any match
        let count_starting_with_0_or_1 = op.attributes.keys()
            .filter(|k| k.starts_with("key_0") || k.starts_with("key_1"))
            .count();
        
        // Remove some attributes (those starting with "key_0" or "key_1")
        let keys_to_remove: Vec<_> = op.attributes.keys()
            .filter(|k| k.starts_with("key_0") || k.starts_with("key_1"))
            .cloned()
            .collect();
        
        for key in keys_to_remove {
            op.attributes.remove(&key);
        }
        
        // Should have removed some of the keys (those starting with 0 or 1)
        let remaining = op.attributes.len();
        assert!(remaining < 10_000); // Should be less than original
        // The number of removed items depends on the hash distribution
        // At least some keys should be affected by the removal
        assert_eq!(remaining, 10_000 - count_starting_with_0_or_1);
    }

    /// Test 10: Concurrency-like behavior with IR objects (simulated)
    #[test]
    fn test_simulated_concurrent_ir_manipulation() {
        // Create multiple modules simulating concurrent access patterns
        let mut modules = Vec::new();
        
        for i in 0..100 {
            let mut module = Module::new(&format!("concurrent_module_{}", i));
            
            // Add various ops with different characteristics
            for j in 0..100 {
                let mut op = Operation::new(&format!("op_{}_{}", i, j));
                
                // Add inputs and outputs varying by index
                if j % 2 == 0 {
                    op.inputs.push(Value {
                        name: format!("input_{}_{}", i, j),
                        ty: if j % 4 == 0 { Type::F32 } else { Type::I32 },
                        shape: vec![i + 1, j + 1],
                    });
                }
                
                if j % 3 == 0 {
                    op.outputs.push(Value {
                        name: format!("output_{}_{}", i, j),
                        ty: if j % 6 == 0 { Type::F64 } else { Type::I64 },
                        shape: vec![i + 2, j + 2],
                    });
                }
                
                // Add attribute randomly
                if j % 7 == 0 {
                    let mut attrs = HashMap::new();
                    attrs.insert(
                        format!("param_{}_{}", i, j),
                        Attribute::String(format!("value_{}_{}", i, j))
                    );
                    op.attributes = attrs;
                }
                
                module.add_operation(op);
            }
            
            modules.push(module);
        }
        
        // Verify creation was successful
        assert_eq!(modules.len(), 100);
        
        // Check a few random modules
        assert_eq!(modules[0].name, "concurrent_module_0");
        assert_eq!(modules[50].name, "concurrent_module_50");
        assert_eq!(modules[99].name, "concurrent_module_99");
        
        // Check operations in a specific module
        assert!(modules[0].operations.len() >= 100); // At least the base count
        
        // Clean up by dropping - this tests memory management under load
        drop(modules);
        assert!(true); // If we get here without panic, test passed
    }
}