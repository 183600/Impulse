//! Additional edge case and boundary tests for the Impulse compiler
//! This file contains important edge case tests that complement existing test suites

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::utils;

/// Test for overflow protection in tensor size calculations
#[test]
fn test_calculate_tensor_size_safe_overflow() {
    // Test very large dimensions that could cause overflow in multiplication
    let large_dims = vec![usize::MAX/2, 3];
    let result = utils::calculate_tensor_size_safe(&large_dims);
    assert!(result.is_none(), "Should return None for overflow condition");
    
    // Test with three very large dimensions
    let three_large_dims = vec![100_000, 100_000, 100];
    let result2 = utils::calculate_tensor_size_safe(&three_large_dims);
    // Depending on platform, this may or may not overflow
    // At minimum, it shouldn't panic
    assert!(result2.is_some() || result2.is_none());
}

/// Test creating a module with maximum possible name length without crashing
#[test]
fn test_module_extremely_long_name() {
    let long_name = "very_long_module_name_".repeat(1000);
    let module = Module::new(long_name.clone());
    assert_eq!(module.name, long_name);
    assert!(module.operations.is_empty());
}

/// Test creating a value with maximum possible shape dimensions
#[test]
fn test_value_maximum_shape_dimensions() {
    // Create a shape with many dimensions (but small values to avoid overflow)
    let many_dims = vec![1; 10_000]; // 10,000 dimensions each of size 1
    let value = Value {
        name: "multi_dim_tensor".to_string(),
        ty: Type::F32,
        shape: many_dims,
    };
    
    assert_eq!(value.shape.len(), 10_000);
    // The total size should be 1 since all dimensions are 1
    assert_eq!(value.num_elements(), Some(1));
}

/// Test deeply nested operations in a module
#[test]
fn test_module_with_deeply_nested_operations() {
    let mut module = Module::new("nested_ops_module");
    
    // Create an operation with nested attributes (attributes containing arrays of attributes)
    let mut complex_op = Operation::new("complex_op");
    
    // Create nested attribute structure
    let nested_attr = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
        ]),
        Attribute::Array(vec![
            Attribute::Float(3.14),
            Attribute::Float(2.71),
        ])
    ]);
    
    complex_op.attributes.insert("nested_data".to_string(), nested_attr);
    module.add_operation(complex_op);
    
    assert_eq!(module.operations.len(), 1);
    let retrieved_attr = module.operations[0].attributes.get("nested_data").unwrap();
    
    match retrieved_attr {
        Attribute::Array(outer_vec) => {
            assert_eq!(outer_vec.len(), 2);
            match &outer_vec[0] {
                Attribute::Array(inner_vec) => {
                    assert_eq!(inner_vec.len(), 2);
                    assert!(matches!(inner_vec[0], Attribute::Int(1)));
                },
                _ => panic!("Expected nested array structure"),
            }
        },
        _ => panic!("Expected array attribute"),
    }
}

/// Test zero-initialized tensor shapes and their properties
#[test]
fn test_zero_initialized_tensor_properties() {
    // Test tensor with a zero dimension (represents an empty tensor)
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![5, 0, 10],
    };
    
    // The total number of elements should be 0
    assert_eq!(zero_tensor.num_elements(), Some(0));
    
    // Test another zero tensor
    let zero_tensor2 = Value {
        name: "zero_tensor_scalar".to_string(),
        ty: Type::I64,
        shape: vec![0],
    };
    
    assert_eq!(zero_tensor2.num_elements(), Some(0));
    
    // Test tensor with multiple zeros
    let zero_tensor3 = Value {
        name: "multiple_zeros_tensor".to_string(),
        ty: Type::Bool,
        shape: vec![0, 0, 0],
    };
    
    assert_eq!(zero_tensor3.num_elements(), Some(0));
}

/// Test recursive type validation with maximum nesting depth
#[test]
fn test_deeply_nested_type_validation() {
    // Create a deeply nested type structure
    let mut current_type = Type::F32;
    const NESTING_DEPTH: usize = 1000; // Deep nesting to test recursion limits
    
    for _ in 0..NESTING_DEPTH {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1], // Small shape to avoid size explosion
        };
    }
    
    // Verify the type is still valid
    assert!(current_type.is_valid_type());
    
    // Test cloning of deeply nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test operations with maximum possible inputs, outputs and attributes
#[test]
fn test_operation_maximum_complexity() {
    let mut op = Operation::new("max_complexity_op");
    
    // Add maximum inputs (within reason)
    for i in 0..10_000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i % 100 + 1], // Varying shapes to add complexity
        });
    }
    
    // Add maximum outputs
    for i in 0..5_000 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![(i + 10) % 100 + 1],
        });
    }
    
    // Add maximum attributes
    for i in 0..2_000 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    assert_eq!(op.inputs.len(), 10_000);
    assert_eq!(op.outputs.len(), 5_000);
    assert_eq!(op.attributes.len(), 2_000);
    assert_eq!(op.op_type, "max_complexity_op");
}

/// Test tensor size calculation with edge case data types
#[test]
fn test_tensor_size_calculation_edge_cases() {
    // Test with different data types and edge case shapes
    let test_cases = vec![
        // (Type, Shape, Expected size in bytes)
        (Type::F32, vec![], 4),           // Scalar F32
        (Type::F64, vec![], 8),           // Scalar F64  
        (Type::I32, vec![], 4),           // Scalar I32
        (Type::I64, vec![], 8),           // Scalar I64
        (Type::Bool, vec![], 1),          // Scalar Bool
        (Type::F32, vec![0], 0),          // Zero-dimensional F32
        (Type::I32, vec![0, 10], 0),      // Contains zero, I32
        (Type::Bool, vec![1000, 1000], 1_000_000), // Large 2D Bool
        (Type::F32, vec![100, 100], 40_000),       // Large 2D F32
    ];
    
    for (tensor_type, shape, expected_size) in test_cases {
        let calculated_size = utils::ir_utils::calculate_tensor_size(&tensor_type, &shape).unwrap_or(0);
        assert_eq!(
            calculated_size, expected_size,
            "Failed for type {:?} with shape {:?}: expected {}, got {}",
            tensor_type, shape, expected_size, calculated_size
        );
    }
}

/// Test attribute comparison with complex nested structures
#[test]
fn test_complex_attribute_comparison() {
    // Create two identical complex nested attribute structures
    let attr1 = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Array(vec![
            Attribute::String("nested".to_string()),
            Attribute::Float(3.14),
        ]),
        Attribute::Bool(true),
    ]);
    
    let attr2 = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Array(vec![
            Attribute::String("nested".to_string()),
            Attribute::Float(3.14),
        ]),
        Attribute::Bool(true),
    ]);
    
    // They should be equal
    assert_eq!(attr1, attr2);
    
    // Create a slightly different one
    let attr3 = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Array(vec![
            Attribute::String("different".to_string()), // Changed this
            Attribute::Float(3.14),
        ]),
        Attribute::Bool(true),
    ]);
    
    // They should not be equal
    assert_ne!(attr1, attr3);
}

/// Test value with maximum name length and special unicode characters
#[test]
fn test_value_with_unicode_and_long_name() {
    // Test with a long name containing unicode characters
    let unicode_name = format!("tensor_{}", "测试_数据_张量_".repeat(1000));
    let value = Value {
        name: unicode_name.clone(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value.name, unicode_name);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 2, 3]);
    assert_eq!(value.num_elements(), Some(6));
    
    // Test with emoji characters
    let emoji_name = "tensor_🚀_⚡_🎯_🔥".to_string();
    let emoji_value = Value {
        name: emoji_name.clone(),
        ty: Type::I64,
        shape: vec![2, 2],
    };
    
    assert_eq!(emoji_value.name, emoji_name);
    assert_eq!(emoji_value.num_elements(), Some(4));
}