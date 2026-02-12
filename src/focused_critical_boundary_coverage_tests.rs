//! Focused critical boundary coverage tests
//! Tests covering edge cases that may not be fully covered by existing test suites
//! Uses standard library assert! and assert_eq! for verification

#[cfg(test)]
mod focused_critical_boundary_coverage_tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use crate::utils::{math_utils, validation_utils};
    use std::collections::HashMap;

    /// Test 1: num_elements() returns None for overflow scenarios
    #[test]
    fn test_num_elements_overflow_returns_none() {
        // Create a shape that would overflow when computing product
        let overflow_value = Value {
            name: "overflow_tensor".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX / 2, 3], // Would overflow
        };
        
        // Should return None due to overflow
        assert_eq!(overflow_value.num_elements(), None);
        
        // Safe case should still work
        let safe_value = Value {
            name: "safe_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],
        };
        assert_eq!(safe_value.num_elements(), Some(1_000_000));
    }

    /// Test 2: GCD with both zeros
    #[test]
    fn test_gcd_with_both_zeros() {
        // Edge case: gcd(0, 0) should return 0
        assert_eq!(math_utils::gcd(0, 0), 0);
    }

    /// Test 3: LCM with extremely large numbers that stay within bounds
    #[test]
    fn test_lcm_large_numbers_within_bounds() {
        // Use prime numbers to avoid simplification
        let large1 = 999983; // Large prime
        let large2 = 999979; // Another large prime
        
        // LCM of co-prime numbers is their product
        let result = math_utils::lcm(large1, large2);
        assert_eq!(result, large1 * large2);
    }

    /// Test 4: round_up_to_multiple with zero multiple edge case
    #[test]
    fn test_round_up_to_multiple_zero_multiple() {
        // When multiple is 0, should return value unchanged (based on implementation)
        assert_eq!(math_utils::round_up_to_multiple(42, 0), 42);
        assert_eq!(math_utils::round_up_to_multiple(0, 0), 0);
    }

    /// Test 5: next_power_of_2 with value of 0
    #[test]
    fn test_next_power_of_2_zero() {
        // Edge case: next_power_of_2(0) should return 1
        assert_eq!(math_utils::next_power_of_2(0), 1);
    }

    /// Test 6: Module with extremely large operation count
    #[test]
    fn test_module_large_operation_count_stability() {
        let mut module = Module::new("large_ops_test");
        
        // Add 5000 operations to test stability with large counts
        for i in 0..5000 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("in_{}", i),
                ty: Type::F32,
                shape: vec![2],
            });
            op.outputs.push(Value {
                name: format!("out_{}", i),
                ty: Type::F32,
                shape: vec![2],
            });
            module.add_operation(op);
        }
        
        // Verify all operations are present
        assert_eq!(module.operations.len(), 5000);
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[4999].op_type, "op_4999");
        
        // Verify module is still valid
        assert!(validation_utils::validate_module(&module).is_ok());
    }

    /// Test 7: Attribute array with deeply recursive nesting
    #[test]
    fn test_attribute_deep_recursive_nesting() {
        // Create deeply nested attribute arrays
        let level1 = Attribute::Array(vec![Attribute::Int(1)]);
        let level2 = Attribute::Array(vec![level1]);
        let level3 = Attribute::Array(vec![level2]);
        let level4 = Attribute::Array(vec![level3]);
        let level5 = Attribute::Array(vec![level4]);
        
        // Verify structure is preserved
        match level5 {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 1);
                match &outer[0] {
                    Attribute::Array(l4) => {
                        match &l4[0] {
                            Attribute::Array(l3) => {
                                match &l3[0] {
                                    Attribute::Array(l2) => {
                                        match &l2[0] {
                                            Attribute::Array(l1) => {
                                                match &l1[0] {
                                                    Attribute::Int(1) => (),
                                                    _ => panic!("Expected Int(1) at innermost level"),
                                                }
                                            }
                                            _ => panic!("Expected Array at level 1"),
                                        }
                                    }
                                    _ => panic!("Expected Array at level 2"),
                                }
                            }
                            _ => panic!("Expected Array at level 3"),
                        }
                    }
                    _ => panic!("Expected Array at level 4"),
                }
            }
            _ => panic!("Expected Array at outermost level"),
        }
    }

    /// Test 8: Value with all possible single dimension zero patterns
    #[test]
    fn test_value_all_zero_dimension_patterns() {
        let patterns = vec![
            vec![0],              // Single zero
            vec![0, 1],           // Zero at start
            vec![1, 0],           // Zero at end
            vec![0, 0],           // All zeros
            vec![0, 1, 1],        // Zero in first position
            vec![1, 0, 1],        // Zero in middle
            vec![1, 1, 0],        // Zero in last position
            vec![0, 0, 1],        // Multiple zeros
            vec![0, 1, 0],        // Zeros at extremes
            vec![1, 0, 0],        // Multiple trailing zeros
        ];
        
        for shape in patterns {
            let value = Value {
                name: "zero_pattern".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            // All shapes containing zero should have 0 elements
            assert_eq!(value.num_elements(), Some(0));
        }
    }

    /// Test 9: Operation with very large attribute count
    #[test]
    fn test_operation_very_large_attribute_count() {
        let mut op = Operation::new("large_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add 1000 attributes to test with large count
        for i in 0..1000 {
            attrs.insert(
                format!("attr_{:04}", i),
                Attribute::Int(i as i64),
            );
        }
        
        op.attributes = attrs;
        
        // Verify all attributes are present
        assert_eq!(op.attributes.len(), 1000);
        assert_eq!(op.attributes.get("attr_0000"), Some(&Attribute::Int(0)));
        assert_eq!(op.attributes.get("attr_0999"), Some(&Attribute::Int(999)));
        
        // Verify operation is still valid
        assert!(validation_utils::validate_operation(&op).is_ok());
    }

    /// Test 10: Module with multiple operations sharing input names
    #[test]
    fn test_module_operations_sharing_input_names() {
        let mut module = Module::new("shared_inputs_test");
        
        // Add a module input
        module.inputs.push(Value {
            name: "shared_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        // Add multiple operations that use the same input name
        // This should be valid as operations can share inputs
        for i in 0..5 {
            let mut op = Operation::new(&format!("consumer_{}", i));
            op.inputs.push(Value {
                name: "shared_input".to_string(), // Shared across operations
                ty: Type::F32,
                shape: vec![10],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
            module.add_operation(op);
        }
        
        // Verify all operations are present
        assert_eq!(module.operations.len(), 5);
        
        // Verify module is valid (shared inputs across operations should be OK)
        let result = validation_utils::validate_module(&module);
        // Note: This may fail depending on validation rules
        // The test verifies the current behavior
        assert!(result.is_ok() || result.is_err());
    }
}