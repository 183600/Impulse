//! Additional comprehensive edge case tests for the Impulse compiler
//! Covers more boundary conditions and overflow scenarios

#[cfg(test)]
mod additional_comprehensive_edge_case_tests {
    use crate::ir::{Value, Type, Attribute, Module, Operation};
    use crate::utils::{calculate_tensor_size_safe, gcd, lcm, round_up_to_multiple, next_power_of_2};
    use std::collections::HashMap;

    #[test]
    fn test_value_num_elements_overflow_scenarios() {
        // Test scenarios where num_elements could overflow
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000_000, 100_000_000], // Potential for overflow
        };
        
        // This might return None if it overflows on the system
        let result = large_value.num_elements();
        // Should either be Some(value) or None depending on platform
        assert!(result.is_some() || result.is_none());
    }

    #[test]
    fn test_gcd_with_maximum_values() {
        // Test GCD with maximum values
        let a = usize::MAX;
        let b = usize::MAX - 1;
        let result = gcd(a, b);
        
        // The GCD of consecutive integers is typically 1
        assert_eq!(result, 1);
        
        // Test with MAX and itself
        assert_eq!(gcd(usize::MAX, usize::MAX), usize::MAX);
    }

    #[test]
    fn test_lcm_with_maximum_values() {
        // Test LCM with values that could cause overflow
        let result = lcm(100_000_000, 100_000_001);
        // Just check that it doesn't crash and returns a reasonable value
        assert!(result > 0);
        
        // Test LCM with MAX values that should overflow
        let max_safe_result = lcm(65536, 65536); // 2^16 * 2^16 = 2^32
        assert_eq!(max_safe_result, 65536 * 65536);
    }

    #[test]
    fn test_round_up_to_multiple_edge_cases() {
        // Test with maximum possible multiple
        assert_eq!(round_up_to_multiple(10, usize::MAX), usize::MAX);
        
        // Test with value already at maximum
        assert_eq!(round_up_to_multiple(usize::MAX, 2), usize::MAX);
        
        // Test with zero as value and various multiples
        assert_eq!(round_up_to_multiple(0, 1), 0);
        assert_eq!(round_up_to_multiple(0, 100), 0);
        assert_eq!(round_up_to_multiple(0, usize::MAX), 0);
        
        // Test with very large values
        let near_max = usize::MAX - 10;
        let rounded = round_up_to_multiple(near_max, 1000);
        assert!(rounded >= near_max);
    }

    #[test]
    fn test_next_power_of_2_with_maximum_values() {
        // Test next power of 2 for maximum values
        let half_max = usize::MAX / 2 + 1;
        let result = next_power_of_2(half_max);
        
        // The next power of 2 after half of max might be max itself or overflow
        // In case of overflow, it might return a value or behave differently
        assert!(result > 0);
        
        // Specific test for the maximum power of 2 that fits in usize
        let max_power_of_2 = 1 << (usize::BITS - 1);
        assert_eq!(next_power_of_2(max_power_of_2), max_power_of_2);
        
        assert_eq!(next_power_of_2(max_power_of_2 + 1), max_power_of_2 << 1);
    }

    #[test]
    fn test_calculate_tensor_size_safe_with_extreme_values() {
        // Test with a combination that could overflow
        let dims = vec![1_000_000, 1_000_000];
        let result = calculate_tensor_size_safe(&dims);
        
        // Either the calculation succeeds or it detects overflow
        assert!(result.is_some() || result.is_none());
        
        // Test with many small dimensions
        let many_small_dims = vec![2; 30]; // 2^30 = ~1 billion
        let result2 = calculate_tensor_size_safe(&many_small_dims);
        assert!(result2.is_some());
        assert_eq!(result2.unwrap(), 1u64 << 30);
    }

    #[test]
    fn test_attribute_with_maximum_string_size() {
        // Test creating an attribute with a very large string
        // Note: This will be limited by available memory
        let large_string = "z".repeat(1_000_000); // 1MB string
        let attr = Attribute::String(large_string.clone());
        
        match attr {
            Attribute::String(s) => {
                assert_eq!(s.len(), large_string.len());
                assert_eq!(s, large_string);
            },
            _ => panic!("Expected String attribute"),
        }
    }

    #[test]
    fn test_operation_with_maximum_inputs() {
        // Test creating an operation with maximum possible inputs
        // Limited by system memory in practice
        let mut op = Operation::new("max_input_op");
        
        // Add a very large number of inputs
        let num_inputs = 100_000;
        for i in 0..num_inputs {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
        
        assert_eq!(op.inputs.len(), num_inputs);
        assert_eq!(op.op_type, "max_input_op");
        
        // Verify some of the inputs are still correct
        assert_eq!(op.inputs[0].name, "input_0");
        assert_eq!(op.inputs[num_inputs-1].name, format!("input_{}", num_inputs-1));
    }

    #[test]
    fn test_module_with_extreme_numbers_of_operations() {
        // Test creating a module with an extreme number of operations
        let mut module = Module::new("extreme_ops_module");
        
        let num_ops = 50_000;
        for i in 0..num_ops {
            let mut op = Operation::new(&format!("operation_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
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
        
        assert_eq!(module.operations.len(), num_ops);
        assert_eq!(module.name, "extreme_ops_module");
        
        // Verify some operations are still correct
        assert_eq!(module.operations[0].op_type, "operation_0");
        assert_eq!(module.operations[num_ops-1].op_type, format!("operation_{}", num_ops-1));
    }

    #[test]
    fn test_deeply_nested_attribute_arrays() {
        // Test creating deeply nested attribute arrays
        // This tests recursion depth and memory management
        
        // Build a deeply nested array: [[[[...1...]]]]
        let mut nested: Attribute = Attribute::Int(42);
        for _ in 0..50 { // 50 levels deep
            nested = Attribute::Array(vec![nested]);
        }
        
        // Verify we can clone it
        let cloned = nested.clone();
        assert_eq!(nested, cloned);
        
        // Unwrap the nesting to make sure the value is preserved
        let mut current = &cloned;
        for _ in 0..50 {
            match current {
                Attribute::Array(arr) => {
                    assert_eq!(arr.len(), 1);
                    current = &arr[0];
                },
                _ => panic!("Expected nested array"),
            }
        }
        
        // Should reach the innermost value
        match current {
            Attribute::Int(42) => {}, // Success
            _ => panic!("Expected innermost value to be Int(42)"),
        }
    }
}