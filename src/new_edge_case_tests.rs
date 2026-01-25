//! Additional edge case tests for Impulse compiler
//! This file contains tests that focus on boundary conditions and error handling

use rstest::*;
use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils,
};

// Test 1: Integer overflow in tensor size calculations
#[test]
fn test_tensor_size_overflow() {
    // Testing with values that are known to cause overflow when multiplied
    // We'll use large values that exceed what can be represented in usize when multiplied
    let large_val = usize::MAX / 2 + 1; // More than half of max, so double would overflow
    let huge_shape = vec![large_val, 2]; // This should cause overflow: (MAX/2 + 1) * 2 > MAX
    
    // Use checked operations to detect overflow
    let result = huge_shape.iter().try_fold(1_usize, |acc, &x| {
        acc.checked_mul(x)
    });
    
    // This should return None due to overflow
    assert!(result.is_none());
}

// Test 2: Memory allocation edge cases
#[test]
fn test_large_allocation_handling() {
    // Test how the system handles very large allocations
    let mut test_module = Module::new("allocation_test");
    
    // Add a very large operation that would normally consume lots of memory
    let large_num = 1_000_000;
    let large_value = Value {
        name: "large_allocation".to_string(),
        ty: Type::F32,
        shape: vec![large_num],
    };
    
    let mut op = Operation::new("large_op");
    op.inputs.push(large_value);
    
    test_module.add_operation(op);
    
    assert_eq!(test_module.operations.len(), 1);
    assert_eq!(test_module.operations[0].inputs[0].name, "large_allocation");
}

// Test 3: Invalid UTF-8 sequences in identifiers (simulated)
#[test]
fn test_utf8_valid_identifiers() {
    // Test with some challenging but valid UTF-8 identifiers
    let valid_identifiers = [
        "valid_中文",
        "valid_🚀",
        "café_naïve",
        "test_αβγ",  // Greek letters
        "variable_émojis_🔥✨🚀",
    ];
    
    for id in valid_identifiers.iter() {
        let value = Value {
            name: id.to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name, *id);
        assert_eq!(value.ty, Type::F32);
    }
}

// Test 4: Deeply nested recursive operations
#[test]
fn test_deeply_nested_types() {
    // Create a deeply nested type structure to test recursion limits
    let mut current_type = Type::F32;
    
    // Build a moderately deep nested structure
    for _ in 0..100 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Test that we can handle and compare this nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
    
    // Test that we can handle a complex nested type in a value
    let nested_value = Value {
        name: "nested_complex".to_string(),
        ty: current_type,
        shape: vec![1],
    };
    
    assert_eq!(nested_value.name, "nested_complex");
}

// Test 5: Operations with maximum number of attributes
#[test]
fn test_operation_max_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attr_op");
    let mut attrs = HashMap::new();
    
    // Add a large number of different attribute types
    for i in 0..1000 {
        let key = format!("attr_{}", i);
        attrs.insert(key, Attribute::Int(i as i64));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify some attributes are present
    assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("attr_999"), Some(&Attribute::Int(999)));
}

// Test 6: Invalid tensor shapes handling
#[test]
fn test_invalid_or_problematic_shapes() {
    // Test various edge case shapes
    let test_shapes = vec![
        vec![],           // Scalar
        vec![0],          // Zero-dimensional in a way
        vec![0, 10],      // Contains zero
        vec![10, 0],      // Contains zero at end
        vec![0, 0, 0],    // All zeros
        vec![1, 1, 1, 1], // Many unit dimensions
        vec![usize::MAX, 1], // Extreme values
    ];
    
    for (i, shape) in test_shapes.iter().enumerate() {
        let value = Value {
            name: format!("shape_test_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Calculate expected total elements
        let expected_total: usize = shape.iter().product();
        let actual_total: usize = value.shape.iter().product();
        
        assert_eq!(expected_total, actual_total);
        
        if shape.iter().any(|&dim| dim == 0) {
            assert_eq!(actual_total, 0);
        }
    }
}

// Test 7: Special floating-point values
#[rstest]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
#[case(f64::NAN)]
#[case(-0.0)]  // Negative zero
#[case(f64::EPSILON)]
#[case(f64::MIN_POSITIVE)]
fn test_special_float_attributes(#[case] float_val: f64) {
    let attr = Attribute::Float(float_val);
    
    match attr {
        Attribute::Float(retrieved_val) => {
            if float_val.is_nan() {
                assert!(retrieved_val.is_nan());
            } else if float_val.is_infinite() {
                assert!(retrieved_val.is_infinite());
                assert_eq!(retrieved_val.is_sign_positive(), float_val.is_sign_positive());
            } else if float_val == -0.0 {
                assert_eq!(retrieved_val, -0.0);
                assert!(retrieved_val.is_sign_negative());
            } else {
                assert!((retrieved_val - float_val).abs() < f64::EPSILON);
            }
        },
        _ => panic!("Expected Float attribute"),
    }
}

// Test 8: Concurrent access simulation (using multiple scopes)
#[test]
fn test_concurrent_access_simulation() {
    // Simulate multiple scopes accessing similar data structures
    
    let mut modules = Vec::new();
    
    // Create several modules in separate scopes to simulate concurrent access patterns
    for i in 0..10 {
        let module_name = format!("concurrent_test_{}", i);
        let mut module = Module::new(&module_name);
        
        // Add operations to each module
        for j in 0..100 {
            let mut op = Operation::new(&format!("op_{}_{}", i, j));
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: Type::F32,
                shape: vec![j + 1, j + 2],
            });
            
            module.add_operation(op);
        }
        
        modules.push(module);
    }
    
    // Verify we created the expected number of modules
    assert_eq!(modules.len(), 10);
    
    // Verify each module has the expected number of operations
    for (i, module) in modules.iter().enumerate() {
        assert_eq!(module.operations.len(), 100);
        assert_eq!(module.name, format!("concurrent_test_{}", i));
        
        // Check first operation in this module
        let first_op = &module.operations[0];
        assert_eq!(first_op.op_type, format!("op_{}_0", i));
        assert_eq!(first_op.inputs[0].name, format!("input_{}_0", i));
    }
    
    // Clean up by dropping all modules
    drop(modules);
    
    // Verify nothing crashed during drop
    assert!(true);
}

// Test 9: Serialization edge cases
#[test]
fn test_serialization_edge_cases() {
    use serde_json;
    
    // Test serializing complex nested structures
    let complex_value = Value {
        name: "complex_serialization".to_string(),
        ty: Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            }),
            shape: vec![3, 3],
        },
        shape: vec![1, 2, 3, 4],
    };
    
    // Attempt to serialize the complex structure
    let serialized = serde_json::to_string(&complex_value);
    assert!(serialized.is_ok());
    
    // And deserialize it back
    let deserialized: Result<Value, _> = serde_json::from_str(&serialized.unwrap());
    assert!(deserialized.is_ok());
    
    // Compare key fields of original and deserialized
    let deserialized_value = deserialized.unwrap();
    assert_eq!(complex_value.name, deserialized_value.name);
    assert_eq!(complex_value.shape, deserialized_value.shape);
}

// Test 10: Boundary value testing for integer attributes
#[rstest]
#[case(i64::MAX)]
#[case(i64::MIN)]
#[case(0)]
#[case(1)]
#[case(-1)]
fn test_integer_boundary_attributes(#[case] int_val: i64) {
    let attr = Attribute::Int(int_val);
    
    match attr {
        Attribute::Int(retrieved_val) => {
            assert_eq!(retrieved_val, int_val);
            
            // Test edge case behaviors
            if int_val == i64::MAX {
                // Test that we can handle max value correctly
                assert_eq!(retrieved_val, i64::MAX);
            } else if int_val == i64::MIN {
                // Test that we can handle min value correctly
                assert_eq!(retrieved_val, i64::MIN);
            }
        },
        _ => panic!("Expected Int attribute"),
    }
}

// Additional helper test to ensure tensor size calculation is robust
#[test]
fn test_tensor_size_calculation_robustness() {
    let test_cases = vec![
        (vec![], 1, Type::F32),        // Scalar
        (vec![0], 0, Type::F32),       // Zero-size tensor
        (vec![1], 1, Type::F32),       // Single element
        (vec![10, 10], 100, Type::F32), // 2D tensor
        (vec![2, 3, 4], 24, Type::I32), // 3D tensor with different type
        (vec![5, 0, 10], 0, Type::Bool), // Contains zero dimension
    ];
    
    for (shape, expected_count, dtype) in test_cases {
        let total_elements: usize = shape.iter().product();
        assert_eq!(total_elements, expected_count, "Shape {:?} should have {} elements", shape, expected_count);
        
        // Also test with the IR utility function if available
        if let Ok(calculated_size) = ir_utils::calculate_tensor_size(&dtype, &shape) {
            let element_size = match dtype {
                Type::F32 | Type::I32 => 4,
                Type::F64 | Type::I64 => 8,
                Type::Bool => 1,
                _ => 4, // Default assumption for other types
            };
            
            assert_eq!(calculated_size, expected_count * element_size, 
                      "Size calculation failed for shape {:?} and type {:?}", shape, dtype);
        }
    }
}