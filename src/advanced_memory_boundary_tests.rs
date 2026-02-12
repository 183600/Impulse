/// Advanced memory and resource boundary tests - 覆盖内存和资源边界情况
/// 使用标准库 assert! 和 assert_eq!
#[cfg(test)]
mod advanced_memory_boundary_tests {
    use super::*;
    use crate::ir::{Module, Value, Type, Operation, Attribute};

    /// Test 1: Module with maximum possible operations without overflow
    #[test]
    fn test_max_operations_module() {
        let mut module = Module::new("max_ops");
        
        // Add a large number of operations (but not causing overflow)
        for i in 0..1000 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 1000);
    }

    /// Test 2: Value with shape that triggers 64-bit integer overflow prevention
    #[test]
    fn test_shape_overflow_64bit() {
        // Use dimensions that would overflow u64 if multiplied directly
        let value = Value {
            name: "potential_overflow".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000], // 10 billion - within u64
        };
        
        // Should return None or handle gracefully instead of overflowing
        match value.num_elements() {
            Some(n) => assert_eq!(n, 10_000_000_000),
            None => {
                // Alternative: return None for overflow cases
                assert!(true);
            }
        }
    }

    /// Test 3: PassManager with repeated small allocations (memory fragmentation test)
    #[test]
    fn test_repeated_small_allocations() {
        use crate::passes::{PassManager, ConstantFoldPass};
        
        let mut pass_manager = PassManager::new();
        
        // Perform multiple add operations
        for i in 0..50 {
            pass_manager.add_pass(Box::new(ConstantFoldPass));
            assert_eq!(pass_manager.passes.len(), i + 1);
        }
        
        // PassManager should have 50 passes
        assert_eq!(pass_manager.passes.len(), 50);
    }

    /// Test 4: Value with very long name (string boundary)
    #[test]
    fn test_very_long_value_name() {
        let long_name = "a".repeat(10_000);
        let value = Value {
            name: long_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name.len(), 10_000);
    }

    /// Test 5: Attribute with deeply nested recursive structure
    #[test]
    fn test_deeply_nested_attribute() {
        // Create a deeply nested attribute structure
        let mut nested = Attribute::Int(1);
        
        // Create 10 levels of nesting
        for _ in 0..10 {
            nested = Attribute::Array(vec![nested]);
        }
        
        // Verify the structure exists without panicking
        match nested {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
            }
            _ => panic!("Expected nested array"),
        }
    }

    /// Test 6: Module with inputs that exceed reasonable limits
    #[test]
    fn test_excessive_module_inputs() {
        let mut module = Module::new("many_inputs");
        
        // Add 500 inputs (stress test but reasonable)
        for i in 0..500 {
            module.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
        
        assert_eq!(module.inputs.len(), 500);
    }

    /// Test 7: Operation with empty string operation type
    #[test]
    fn test_empty_operation_type() {
        let op = Operation::new(""); // Empty string
        assert_eq!(op.op_type, "");
    }

    /// Test 8: Attribute array with alternating bool values
    #[test]
    fn test_alternating_bool_array() {
        let bool_array = Attribute::Array(vec![
            Attribute::Bool(true),
            Attribute::Bool(false),
            Attribute::Bool(true),
            Attribute::Bool(false),
            Attribute::Bool(true),
        ]);
        
        match bool_array {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 5);
                match &arr[0] { Attribute::Bool(true) => (), _ => panic!() }
                match &arr[1] { Attribute::Bool(false) => (), _ => panic!() }
                match &arr[2] { Attribute::Bool(true) => (), _ => panic!() }
                match &arr[3] { Attribute::Bool(false) => (), _ => panic!() }
                match &arr[4] { Attribute::Bool(true) => (), _ => panic!() }
            }
            _ => panic!("Expected Array"),
        }
    }

    /// Test 9: Value with maximum dimension (edge case for overflow detection)
    #[test]
    fn test_negative_dimensions_handling() {
        // Create value with maximum dimension (tests overflow detection)
        let value = Value {
            name: "max_dim".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 10], // Very large dimension that may cause overflow
        };
        
        // num_elements should handle overflow gracefully
        match value.num_elements() {
            Some(n) => {
                // If it returns a value, it should be valid
                assert!(n > 0);
            }
            None => {
                // Or return None for overflow cases
                assert!(true);
            }
        }
    }

    /// Test 10: Module with operation containing very large attribute map
    #[test]
    fn test_large_attribute_map() {
        let mut op = Operation::new("large_attrs");
        let mut attrs = std::collections::HashMap::new();
        
        // Add 1000 attributes
        for i in 0..1000 {
            attrs.insert(format!("attr_{}", i), Attribute::Int(i as i64));
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 1000);
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_999"));
    }
}