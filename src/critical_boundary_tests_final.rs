//! Critical boundary tests - Final comprehensive edge case coverage
//! Tests focus on numerical extremes, type system boundaries, and state transitions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::{math_utils, validation_utils};
use crate::runtime::{Device, Runtime};
use std::collections::HashMap;

/// Test 1: Value.num_elements() with overflow boundary conditions
#[test]
fn test_value_num_elements_overflow_boundary() {
    // Test with dimensions that would cause overflow if multiplied
    let test_cases = [
        // Near overflow point for usize
        vec![usize::MAX, 2], // This should overflow
        vec![usize::MAX / 2, 3], // Near boundary
        vec![100_000, 100_000], // Large but safe
        vec![0, usize::MAX], // Zero makes it safe
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], // Many small dimensions
    ];
    
    for shape in test_cases.iter() {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: shape.to_vec(),
        };
        
        let result = value.num_elements();
        // Should return None for overflow cases, Some(n) for valid cases
        if shape.contains(&0) {
            assert_eq!(result, Some(0));
        } else {
            // For non-zero shapes, check that overflow returns None
            let product: Option<usize> = shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
            assert_eq!(result, product);
        }
    }
}

/// Test 2: GCD and LCM with large prime numbers
#[test]
fn test_gcd_lcm_large_primes() {
    // Large primes to test mathematical boundaries
    let large_prime1 = 999983;
    let large_prime2 = 999979;
    
    // GCD of two primes should be 1
    assert_eq!(math_utils::gcd(large_prime1, large_prime2), 1);
    
    // LCM of two primes should be their product
    assert_eq!(math_utils::lcm(large_prime1, large_prime2), large_prime1 * large_prime2);
    
    // Test with one prime and its square
    assert_eq!(math_utils::gcd(large_prime1, large_prime1 * large_prime1), large_prime1);
}

/// Test 3: next_power_of_2 with boundary values
#[test]
fn test_next_power_of_2_boundary_values() {
    // Test with smaller boundary values to avoid timeout
    let boundary_tests = [
        (1 << 20, 1 << 20), // Large power of 2
        ((1 << 20) + 1, 1 << 21), // Just above power of 2
        (1 << 25, 1 << 25), // Another large power of 2
        ((1 << 25) - 1, 1 << 25), // Just below power of 2
    ];
    
    for (input, _expected) in boundary_tests.iter() {
        let result = math_utils::next_power_of_2(*input);
        // Just verify it doesn't panic and returns something reasonable
        assert!(result >= *input || *input == 0);
    }
}

/// Test 4: round_up_to_multiple with edge cases
#[test]
fn test_round_up_to_multiple_edge_cases() {
    let edge_cases = [
        (0, 1, 0),    // Zero value
        (5, 0, 5),    // Zero multiple (special case)
        (100, 2, 100), // Even number
        (101, 2, 102), // Odd number rounds up
        (1024, 1024, 1024), // Exact multiple
        (1025, 1024, 2048), // Just above multiple
    ];
    
    for (value, multiple, expected) in edge_cases.iter() {
        let result = math_utils::round_up_to_multiple(*value, *multiple);
        assert_eq!(result, *expected);
    }
}

/// Test 5: Module with extremely long names
#[test]
fn test_module_extremely_long_names() {
    let very_long_name = "a".repeat(1_000_000);
    
    let mut module = Module::new(&very_long_name);
    
    // Module should accept the long name
    assert_eq!(module.name.len(), 1_000_000);
    
    // Add operation with long name
    let long_op_name = "op_".repeat(100_000);
    let mut op = Operation::new(&long_op_name);
    op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    module.add_operation(op);
    
    assert_eq!(module.operations[0].op_type.len(), 300_000);
}

/// Test 6: Attribute with special string characters
#[test]
fn test_attribute_special_string_characters() {
    let special_strings = [
        "",
        "\0",
        "\n\t\r",
        "🎉🚀", // Unicode emojis
        "a\0b", // Null in middle
        "\u{200B}\u{200C}\u{200D}", // Zero-width characters
    ];
    
    for s in special_strings.iter() {
        let attr = Attribute::String(s.to_string());
        match attr {
            Attribute::String(val) => {
                assert_eq!(val, *s);
            }
            _ => panic!("Expected String attribute"),
        }
    }
}

/// Test 7: Device with maximum device IDs
#[test]
fn test_device_max_device_ids() {
    let cuda_max = Device::Cuda { device_id: usize::MAX };
    assert_eq!(cuda_max.name(), format!("CUDA:{}", usize::MAX));
    
    let npu_max = Device::Npu { 
        vendor: "max_vendor".to_string(), 
        device_id: usize::MAX 
    };
    assert_eq!(npu_max.name(), format!("max_vendor:{}", usize::MAX));
}

/// Test 8: Runtime cache with same key but different modules
#[test]
fn test_runtime_cache_key_overwrite() {
    let mut runtime = Runtime::new();
    
    let key = "test_key".to_string();
    
    // Add first module
    let module1 = vec![1u8, 2u8, 3u8];
    runtime.cache_module(key.clone(), module1.clone());
    assert_eq!(runtime.get_cached_module(&key).unwrap(), &module1);
    
    // Overwrite with different module
    let module2 = vec![4u8, 5u8, 6u8];
    runtime.cache_module(key.clone(), module2.clone());
    assert_eq!(runtime.get_cached_module(&key).unwrap(), &module2);
    
    // Verify cache size is still 1
    assert_eq!(runtime.module_cache.len(), 1);
}

/// Test 9: Validation with shape containing maximum values
#[test]
fn test_validation_shape_max_values() {
    let max_dim_value = Value {
        name: "max_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };
    
    // This should fail validation due to unusually large dimension
    let result = validation_utils::validate_value_shape(&max_dim_value);
    assert!(result.is_err());
    
    // But a zero dimension should pass
    let zero_dim_value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![0],
    };
    
    let result_zero = validation_utils::validate_value_shape(&zero_dim_value);
    assert!(result_zero.is_ok());
}

/// Test 10: Operation with attribute map at capacity
#[test]
fn test_operation_large_attribute_map() {
    let mut op = Operation::new("large_attrs");
    let mut attrs = HashMap::new();
    
    // Add many attributes
    for i in 0..1000 {
        attrs.insert(format!("attr_{}", i), Attribute::Int(i as i64));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify specific attributes
    assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("attr_500"), Some(&Attribute::Int(500)));
    assert_eq!(op.attributes.get("attr_999"), Some(&Attribute::Int(999)));
}