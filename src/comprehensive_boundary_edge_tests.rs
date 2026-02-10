//! Comprehensive boundary edge tests for Impulse compiler
//! Tests covering various boundary conditions and edge cases across modules

use crate::ir::{Module, Operation, Value, Type, Attribute};
use crate::runtime::{Device, Runtime, ExecutionContext};
use crate::utils::{gcd, lcm, round_up_to_multiple, next_power_of_2};

/// Test 1: GCD and LCM with extreme values
#[test]
fn test_gcd_lcm_extreme_values() {
    // Test GCD with same values
    assert_eq!(gcd(123456789, 123456789), 123456789);
    
    // Test GCD where one number is a multiple of the other
    assert_eq!(gcd(1000, 100), 100);
    assert_eq!(gcd(100, 1000), 100);
    
    // Test GCD with consecutive Fibonacci numbers (should be 1)
    assert_eq!(gcd(1836311903, 2971215073 % 1836311903), 1);
    
    // Test LCM with same values
    assert_eq!(lcm(999983, 999983), 999983);
    
    // Test LCM with prime numbers
    assert_eq!(lcm(2, 3), 6);
    assert_eq!(lcm(5, 7), 35);
    assert_eq!(lcm(13, 17), 221);
}

/// Test 2: Zero-size allocation handling in memory pool
#[test]
fn test_zero_size_allocation() {
    let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
    
    // Test zero-byte allocation (edge case)
    let zero_handle = ctx.allocate_tensor(0).unwrap();
    assert_eq!(zero_handle.size, 0);
    
    // Deallocation should work
    assert!(ctx.memory_pool.deallocate(zero_handle).is_ok());
}

/// Test 3: Round up to multiple with boundary conditions
#[test]
fn test_round_up_to_multiple_boundaries() {
    // Test with value exactly at the multiple boundary
    assert_eq!(round_up_to_multiple(16, 16), 16);
    assert_eq!(round_up_to_multiple(256, 256), 256);
    
    // Test with value just below the multiple
    assert_eq!(round_up_to_multiple(15, 16), 16);
    assert_eq!(round_up_to_multiple(255, 256), 256);
    
    // Test with value just above the multiple
    assert_eq!(round_up_to_multiple(17, 16), 32);
    assert_eq!(round_up_to_multiple(257, 256), 512);
    
    // Test with large values (1000000 % 1024 = 576, so round_up = 1000000 + (1024 - 576) = 1000000 + 448 = 1000448)
    assert_eq!(round_up_to_multiple(1000000, 1024), 1000448);
    assert_eq!(round_up_to_multiple(usize::MAX, 1), usize::MAX);
}

/// Test 4: Device cloning and comparison
#[test]
fn test_device_clone_and_comparison() {
    // Test Device::Cpu cloning
    let cpu1 = Device::Cpu;
    let cpu2 = cpu1.clone();
    assert_eq!(cpu1, cpu2);
    
    // Test Device::Cuda cloning
    let cuda1 = Device::Cuda { device_id: 5 };
    let cuda2 = cuda1.clone();
    assert_eq!(cuda1, cuda2);
    
    // Test Device::Npu cloning
    let npu1 = Device::Npu { vendor: "Qualcomm".to_string(), device_id: 0 };
    let npu2 = npu1.clone();
    assert_eq!(npu1, npu2);
    
    // Test that different devices are not equal
    assert_ne!(Device::Cpu, Device::Cuda { device_id: 0 });
    assert_ne!(Device::Cpu, Device::Npu { vendor: "Nvidia".to_string(), device_id: 0 });
    
    // Test that different CUDA devices are not equal
    assert_ne!(Device::Cuda { device_id: 0 }, Device::Cuda { device_id: 1 });
    
    // Test that different NPU vendors are not equal
    assert_ne!(
        Device::Npu { vendor: "Qualcomm".to_string(), device_id: 0 },
        Device::Npu { vendor: "Nvidia".to_string(), device_id: 0 }
    );
}

/// Test 5: MemoryHandle uniqueness
#[test]
fn test_memory_handle_uniqueness() {
    let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
    
    // Allocate multiple handles and verify they all have unique IDs
    let mut handles = Vec::new();
    let mut ids = std::collections::HashSet::new();
    
    for i in 0..50 {
        let handle = ctx.allocate_tensor(1024 * (i + 1)).unwrap();
        ids.insert(handle.id);
        handles.push(handle);
    }
    
    // All IDs should be unique
    assert_eq!(ids.len(), 50);
    
    // Verify each handle has correct size
    for (i, handle) in handles.iter().enumerate() {
        assert_eq!(handle.size, 1024 * (i + 1));
    }
}

/// Test 6: Next power of 2 with large values
#[test]
fn test_next_power_of_2_large_values() {
    // Test with powers of 2 (should return themselves)
    assert_eq!(next_power_of_2(1), 1);
    assert_eq!(next_power_of_2(2), 2);
    assert_eq!(next_power_of_2(1024), 1024);
    assert_eq!(next_power_of_2(65536), 65536);
    assert_eq!(next_power_of_2(1048576), 1048576);
    
    // Test with values just below powers of 2
    assert_eq!(next_power_of_2(1023), 1024);
    assert_eq!(next_power_of_2(65535), 65536);
    assert_eq!(next_power_of_2(1048575), 1048576);
    
    // Test with values just above powers of 2
    assert_eq!(next_power_of_2(1025), 2048);
    assert_eq!(next_power_of_2(65537), 131072);
    assert_eq!(next_power_of_2(1048577), 2097152);
    
    // Test with larger values
    assert_eq!(next_power_of_2(1000000), 1048576);
    assert_eq!(next_power_of_2(10000000), 16777216);
}

/// Test 7: Module with empty and zero-sized operations
#[test]
fn test_module_with_edge_case_operations() {
    let mut module = Module::new("edge_case_module");
    
    // Add operation with empty name
    let op_empty_name = Operation::new("");
    module.add_operation(op_empty_name);
    
    // Add operation with no inputs and no outputs
    let op_no_io = Operation::new("no_io");
    module.add_operation(op_no_io);
    
    // Add operation with zero-sized input tensors
    let mut op_zero_input = Operation::new("zero_input");
    op_zero_input.inputs.push(Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![0],
    });
    module.add_operation(op_zero_input);
    
    // Add operation with scalar output (empty shape)
    let mut op_scalar_output = Operation::new("scalar_output");
    op_scalar_output.outputs.push(Value {
        name: "scalar".to_string(),
        ty: Type::I32,
        shape: vec![],
    });
    module.add_operation(op_scalar_output);
    
    assert_eq!(module.operations.len(), 4);
}

/// Test 8: Runtime with cache key edge cases
#[test]
fn test_runtime_cache_edge_cases() {
    let mut runtime = Runtime::new();
    
    // Test with empty key
    runtime.cache_module("".to_string(), vec![1u8, 2, 3]);
    assert!(runtime.get_cached_module("").is_some());
    
    // Test with very long key
    let long_key = "a".repeat(10000);
    runtime.cache_module(long_key.clone(), vec![4u8, 5, 6]);
    assert!(runtime.get_cached_module(&long_key).is_some());
    
    // Test with special characters in key
    let special_key = "key_with_special_!@#$%^&*()_+-=[]{}|;':\",./<>?~`";
    runtime.cache_module(special_key.to_string(), vec![7u8, 8, 9]);
    assert!(runtime.get_cached_module(special_key).is_some());
    
    // Test with empty value
    let empty_vec: Vec<u8> = vec![];
    runtime.cache_module("empty_value".to_string(), empty_vec.clone());
    assert_eq!(runtime.get_cached_module("empty_value").unwrap(), &empty_vec);
}

/// Test 9: Operation attributes with empty and special values
#[test]
fn test_operation_attributes_edge_cases() {
    let mut op = Operation::new("test_attrs");
    
    // Add attribute with empty string
    op.attributes.insert("empty_str".to_string(), Attribute::String("".to_string()));
    
    // Add attribute with zero value
    op.attributes.insert("zero_int".to_string(), Attribute::Int(0));
    op.attributes.insert("zero_float".to_string(), Attribute::Float(0.0));
    
    // Add attribute with boolean false (edge case)
    op.attributes.insert("false_bool".to_string(), Attribute::Bool(false));
    
    // Add attribute with empty array
    op.attributes.insert("empty_array".to_string(), Attribute::Array(vec![]));
    
    // Add attribute with very long string
    let long_str = "x".repeat(1000);
    op.attributes.insert("long_str".to_string(), Attribute::String(long_str));
    
    assert_eq!(op.attributes.len(), 6);
}

/// Test 10: Execution context with single-byte allocations
#[test]
fn test_execution_context_single_byte_allocations() {
    let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
    
    // Allocate many single-byte tensors to test fragmentation handling
    let mut handles = Vec::new();
    
    for _ in 0..100 {
        let handle = ctx.allocate_tensor(1).unwrap();
        handles.push(handle);
    }
    
    // Verify all allocations succeeded
    assert_eq!(handles.len(), 100);
    
    // Verify all handles have size at least 1
    for handle in &handles {
        assert!(handle.size >= 1);
    }
    
    // Deallocate all handles
    for handle in handles {
        ctx.memory_pool.deallocate(handle).unwrap();
    }
    
    // All deallocations should succeed without panics
    assert!(true);
}
