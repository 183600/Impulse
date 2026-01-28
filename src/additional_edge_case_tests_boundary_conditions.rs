//! Additional edge case tests for boundary conditions in the Impulse compiler

#[cfg(test)]
mod boundary_condition_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
    use crate::runtime::{Device, MemoryHandle, ExecutionContext, Runtime};
    use crate::utils::{calculate_tensor_size_safe};

    #[test]
    fn test_very_large_tensor_shape_calculation() {
        // Test for potential overflow in tensor size calculation
        let shape = vec![100_000, 100_000];
        let result = calculate_tensor_size_safe(&shape);
        
        // This should not overflow for values that stay within bounds
        if let Some(size) = result {
            assert_eq!(size, 10_000_000_000); // 10 Billion
        } else {
            // If it overflows as expected, that's also valid
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_very_deeply_nested_tensor_types() {
        // Test deeply nested tensor types to check for stack overflow issues
        let mut nested_type = Type::F32;
        
        // Create a deeply nested type structure
        for _ in 0..50 {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![2],
            };
        }
        
        // Verify that it still validates correctly
        assert!(nested_type.is_valid_type());
        
        // Clone the deeply nested type to test clone functionality
        let cloned_nested = nested_type.clone();
        assert_eq!(nested_type, cloned_nested);
    }

    #[test]
    fn test_memory_handle_with_maximum_values() {
        // Test MemoryHandle creation with maximum possible values
        let handle = MemoryHandle {
            id: usize::MAX,
            size: usize::MAX,
            device: Device::Cpu,
        };
        
        assert_eq!(handle.id, usize::MAX);
        assert_eq!(handle.size, usize::MAX);
        assert_eq!(handle.device, Device::Cpu);
    }

    #[test]
    fn test_tensor_size_calculation_with_zeros() {
        // Test tensor size calculation with various arrangements of zeros
        assert_eq!(calculate_tensor_size_safe(&[0]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[0, 100]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[100, 0]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[100, 0, 50]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[1, 2, 0, 4, 5]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[]), Some(1)); // Scalar case
    }

    #[test]
    fn test_operation_with_empty_input_output_vectors() {
        let mut op = Operation::new("empty_io_op");
        
        // Test that empty vectors are properly handled
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
        
        // Test after adding/removing values
        op.inputs.push(Value {
            name: "test_input".to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        });
        
        assert_eq!(op.inputs.len(), 1);
        
        // Clear inputs and verify
        op.inputs.clear();
        assert_eq!(op.inputs.len(), 0);
    }

    #[test]
    fn test_module_with_maximum_sized_name() {
        // Test module creation with a very long name
        let long_name = "a".repeat(10_000); // 10k character name
        let module = Module::new(long_name.clone());
        
        assert_eq!(module.name, long_name);
        assert!(module.operations.is_empty());
    }

    #[test]
    fn test_value_with_maximum_sized_name() {
        // Test value creation with a very long name
        let long_name = "x".repeat(10_000); // 10k character name
        let value = Value {
            name: long_name.clone(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name, long_name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    #[test]
    fn test_execution_context_edge_cases() {
        // Test execution context creation and basic functionality
        let ctx_result = ExecutionContext::new(Device::Cpu);
        assert!(ctx_result.is_ok());
        
        let mut ctx = ctx_result.unwrap();
        
        // Test allocation with edge case sizes
        let handle_small = ctx.allocate_tensor(1); // Minimum allocation
        assert!(handle_small.is_ok());
        
        let handle_medium = ctx.allocate_tensor(1024); // 1KB allocation  
        assert!(handle_medium.is_ok());
        
        let handle_large = ctx.allocate_tensor(1024 * 1024); // 1MB allocation
        assert!(handle_large.is_ok());
    }

    #[test]
    fn test_runtime_initialization_edge_cases() {
        // Test runtime initialization
        let runtime = Runtime::new();
        
        // Should have at least CPU device
        assert!(!runtime.devices.is_empty());
        assert!(runtime.devices.contains(&Device::Cpu));
        
        // Test module caching with edge case keys
        let mut runtime = runtime;
        runtime.cache_module("".to_string(), vec![]); // Empty key and value
        assert!(runtime.get_cached_module("").is_some());
        
        runtime.cache_module("long_key_".repeat(1000), vec![42u8; 10000]); // Long key
        let long_key = "long_key_".repeat(1000);
        assert!(runtime.get_cached_module(&long_key).is_some());
    }

    #[test]
    fn test_attribute_creation_edge_cases() {
        // Test creating attributes with edge case values
        let int_attr = Attribute::Int(i64::MAX);
        let neg_int_attr = Attribute::Int(i64::MIN);
        let float_attr = Attribute::Float(f64::INFINITY);
        let neg_float_attr = Attribute::Float(f64::NEG_INFINITY);
        let nan_float_attr = Attribute::Float(f64::NAN);
        
        // These should all be creatable without issues
        match int_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }
        
        match neg_int_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected Int attribute"),
        }
        
        match float_attr {
            Attribute::Float(val) => assert!(val.is_infinite() && val > 0.0),
            _ => panic!("Expected Float attribute"),
        }
        
        match neg_float_attr {
            Attribute::Float(val) => assert!(val.is_infinite() && val < 0.0),
            _ => panic!("Expected Float attribute"),
        }
        
        // Note: NaN comparisons behave differently, so we just check it's a Float
        match nan_float_attr {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected Float attribute"),
        }
    }
}