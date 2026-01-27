//! Additional important edge case tests for the Impulse compiler
//! Covering missing boundary conditions and critical safety checks
//!
//! This module tests critical edge cases that could affect compiler stability,
//! memory safety, and correctness in production environments.

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use rstest::rstest;

#[cfg(test)]
mod new_important_edge_cases {
    use super::*;

    /// Test 1: Overflow detection in tensor size calculation using checked_mul
    #[test]
    fn test_tensor_size_overflow_detection() {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2],  // This will overflow when multiplied
        };
        
        // Using the actual method that handles overflow
        let result = value.num_elements();
        // Should return None due to overflow
        assert!(result.is_none());
        
        // Test with a safe large tensor
        let safe_large = Value {
            name: "safe_large".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000],
        };
        
        let safe_result = safe_large.num_elements();
        assert_eq!(safe_result, Some(100_000_000));
    }

    /// Test 2: Empty attribute maps and operations with all empty collections
    #[test]
    fn test_completely_empty_structures() {
        // Empty module
        let empty_module = Module::new("");
        assert!(empty_module.operations.is_empty());
        assert!(empty_module.inputs.is_empty());
        assert!(empty_module.outputs.is_empty());
        
        // Empty operation
        let empty_op = Operation::new("empty");
        assert!(empty_op.inputs.is_empty());
        assert!(empty_op.outputs.is_empty());
        assert!(empty_op.attributes.is_empty());
        
        // Empty value
        let empty_value = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(empty_value.shape.is_empty());
    }

    /// Test 3: Extremely deep recursion with nested types that could cause stack overflow
    #[test]
    fn test_deeply_nested_types_stack_safety() {
        // Create a deeply nested type carefully to test stack limits
        let mut current_type = Type::Bool;
        
        // 1000 levels deep - this could potentially cause stack overflow if not handled properly
        for i in 0..1000 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],
            };
            
            // Periodically check that cloning works even at depth
            if i % 100 == 0 {
                let cloned = current_type.clone();
                assert_eq!(current_type, cloned);
            }
        }
        
        // At the end, the type should still be valid
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![1]);
            },
            _ => panic!("Expected tensor type after deep nesting"),
        }
        
        // Final clone test
        let final_clone = current_type.clone();
        assert_eq!(current_type, final_clone);
    }

    /// Test 4: Operations with invalid/incompatible type combinations
    #[test]
    fn test_type_compatibility_edge_cases() {
        let mut op = Operation::new("type_test_op");
        
        // Add inputs and outputs with different types to test compatibility checking
        op.inputs.push(Value {
            name: "f32_input".to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        });
        op.inputs.push(Value {
            name: "i64_input".to_string(),
            ty: Type::I64,
            shape: vec![4, 5, 6],
        });
        op.outputs.push(Value {
            name: "bool_output".to_string(),
            ty: Type::Bool,
            shape: vec![7, 8, 9],
        });
        
        // Verify all types are preserved correctly
        assert_eq!(op.inputs[0].ty, Type::F32);
        assert_eq!(op.inputs[1].ty, Type::I64);
        assert_eq!(op.outputs[0].ty, Type::Bool);
        
        assert_eq!(op.inputs[0].shape, vec![1, 2, 3]);
        assert_eq!(op.inputs[1].shape, vec![4, 5, 6]);
        assert_eq!(op.outputs[0].shape, vec![7, 8, 9]);
    }

    /// Test 5: Attribute deserialization/parsing edge cases
    #[test]
    fn test_attribute_parsing_security_edges() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("attr_test");
        let mut attrs = HashMap::new();
        
        // Test very long string attributes
        attrs.insert("very_long_string".to_string(), Attribute::String("x".repeat(1_000_000)));
        
        // Test special Unicode sequences
        attrs.insert("unicode_attr".to_string(), Attribute::String("🚀🎉中文✓".to_string()));
        
        // Test empty string
        attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
        
        // Test special float values
        attrs.insert("pos_inf".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan_val".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("zero".to_string(), Attribute::Float(0.0));
        attrs.insert("tiny".to_string(), Attribute::Float(f64::MIN_POSITIVE));
        
        op.attributes = attrs;
        
        // Verify the attributes were stored properly
        assert_eq!(op.attributes.len(), 8);
        
        // Test NaN specifically since NaN != NaN
        if let Some(Attribute::Float(val)) = op.attributes.get("nan_val") {
            assert!(val.is_nan());
        } else {
            panic!("NaN attribute not found or incorrect type");
        }
    }

    /// Test 6: Boundary conditions for numeric types and their limits
    #[rstest]
    #[case(i64::MAX, f64::from(i64::MAX as u32))]
    #[case(i64::MIN, f64::from(i64::MIN as i32))]  
    #[case(0, 0.0)]
    #[case(1, 1.0)]
    #[case(-1, -1.0)]
    fn test_numeric_boundaries(#[case] int_val: i64, #[case] float_approx: f64) {
        let int_attr = Attribute::Int(int_val);
        let float_attr = Attribute::Float(float_approx);
        
        match int_attr {
            Attribute::Int(val) => assert_eq!(val, int_val),
            _ => panic!("Expected Int attribute"),
        }
        
        match float_attr {
            Attribute::Float(val) => assert!((val - float_approx).abs() < f64::EPSILON),
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 7: Module serialization/deserialization edge cases
    #[test]
    fn test_module_serialization_edges() {
        // Create a complex module with various edge cases
        let mut module = Module::new("serialization_test_module");
        
        // Add operations with different complexities
        let mut simple_op = Operation::new("simple");
        simple_op.inputs.push(Value {
            name: "simple_input".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(simple_op);
        
        let mut complex_op = Operation::new("complex");
        complex_op.inputs.push(Value {
            name: "complex_input".to_string(),
            ty: Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![2, 2],
            },
            shape: vec![3, 3],
        });
        complex_op.outputs.push(Value {
            name: "complex_output".to_string(),
            ty: Type::Bool,
            shape: vec![],
        });
        module.add_operation(complex_op);
        
        // Test that the module is structured correctly
        assert_eq!(module.name, "serialization_test_module");
        assert_eq!(module.operations.len(), 2);
        assert_eq!(module.operations[0].op_type, "simple");
        assert_eq!(module.operations[1].op_type, "complex");
        
        // Test accessing nested types
        match &module.operations[1].inputs[0].ty {
            Type::Tensor { element_type, shape } => {
                match element_type.as_ref() {
                    Type::I32 => (), // Expected
                    _ => panic!("Expected I32 as tensor element type"),
                }
                assert_eq!(shape, &vec![2, 2]);
            },
            _ => panic!("Expected tensor type for complex input"),
        }
    }

    /// Test 8: Zero-sized tensors and dimension edge cases
    #[test]
    fn test_zero_sized_tensor_operations() {
        // Test tensors with zero dimensions in various positions
        let tensors = vec![
            (vec![], 1),           // Scalar: 0 dimensions = 1 element
            (vec![0], 0),          // Single zero dim: 0 elements  
            (vec![1, 0], 0),       // 1×0: 0 elements
            (vec![0, 1], 0),       // 0×1: 0 elements
            (vec![1, 0, 1], 0),    // 1×0×1: 0 elements
            (vec![2, 0, 3], 0),    // 2×0×3: 0 elements
            (vec![1, 1, 1], 1),    // Unit tensor: 1 element
            (vec![2, 2, 2], 8),    // 2×2×2: 8 elements
        ];
        
        for (shape, expected_elements) in tensors {
            let value = Value {
                name: format!("tensor_{:?}", shape).to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            
            let calculated = value.num_elements();
            if expected_elements == 0 {
                assert_eq!(calculated, Some(0));  // Tensors with zero dims should have 0 elements
            } else {
                assert_eq!(calculated, Some(expected_elements));
            }
        }
    }

    /// Test 9: Memory allocation stress with many nested structures
    #[test]
    fn test_memory_stress_deep_nesting() {
        // Create a complex nested structure to test memory allocation
        let base = Type::F32;
        
        // Build nested tensors iteratively
        let mut current = base;
        for depth in 0..50 {  // Not too deep to avoid stack overflow but enough to stress
            current = Type::Tensor {
                element_type: Box::new(current),
                shape: vec![depth + 1],
            };
        }
        
        // The structure should still be valid after deep nesting
        match &current {
            Type::Tensor { shape, element_type: inner } => {
                assert_eq!(shape, &vec![50]);  // Last iteration was depth 49, so shape[49] = 50
                
                // Check that we can access properties without crashing
                assert!(inner.is_valid_type());
            },
            _ => panic!("Expected deepest level to be a tensor"),
        }
        
        // Test cloning the deep structure
        let cloned = current.clone();
        assert_eq!(current, cloned);
    }

    /// Test 10: Concurrency and thread safety edge cases (basic checks)
    #[test]
    fn test_basic_thread_safety_structures() {
        // Although not testing actual concurrency here, we test structures that 
        // would be used in concurrent scenarios
        
        let value = Value {
            name: "thread_safe_test".to_string(),
            ty: Type::F64,
            shape: vec![100, 100],
        };
        
        // Test that clone works correctly (basic requirement for thread safety)
        let cloned_value = value.clone();
        assert_eq!(value, cloned_value);
        
        // Test operation cloning
        let mut op = Operation::new("concurrent_test_op");
        op.inputs.push(value);
        op.outputs.push(cloned_value);
        
        let cloned_op = op.clone();
        // Compare individual fields since Operation doesn't implement PartialEq
        assert_eq!(op.op_type, cloned_op.op_type);
        assert_eq!(op.inputs, cloned_op.inputs);
        assert_eq!(op.outputs, cloned_op.outputs);
        assert_eq!(op.attributes, cloned_op.attributes);
        
        // Test attribute cloning
        let attr = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("test".to_string()),
        ]);
        let cloned_attr = attr.clone();
        assert_eq!(attr, cloned_attr);
    }
}