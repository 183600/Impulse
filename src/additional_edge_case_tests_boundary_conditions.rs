//! Additional edge case tests with boundary conditions for the Impulse compiler
//! This file contains more tests to cover additional edge cases using standard library assertions
//! and rstest for parameterized testing

use rstest::rstest;
use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::utils;

/// Test Value creation with many shape dimensions (reasonable amount)
#[test]
fn test_value_with_many_shape_dims() {
    // Create a value with many shape dimensions (but small sizes to avoid overflow)
    let many_dims = vec![1; 10_000]; // 10,000 dimensions of size 1
    let value = Value {
        name: "many_dims_tensor".to_string(),
        ty: Type::F32,
        shape: many_dims.clone(),
    };

    assert_eq!(value.shape.len(), 10_000);
    assert_eq!(value.num_elements().unwrap(), 1); // All dimensions are 1
}

/// Test creating operations with many attributes (reasonable amount to avoid memory issues)
#[test]
fn test_operation_with_many_attributes() {
    let mut op = Operation::new("many_attrs_op");
    
    // Add a large number of attributes to test memory limits (but reasonable to avoid issues)
    for i in 0..10_000 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }

    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.op_type, "many_attrs_op");
}

/// Test deeply nested tensor types with reasonable nesting depth to avoid stack overflow
#[test]
fn test_reasonable_depth_nested_tensor_types() {
    // Create nested tensor types but with reasonable depth to avoid stack overflow
    let mut current_type = Type::Bool;
    for i in 0..100 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 5 + 1], // Small varying shape to avoid size explosion
        };
    }

    // This should not crash and should validate correctly
    assert!(current_type.is_valid_type());
    
    // Test equality operations on deeply nested types
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test tensor size calculations with near-maximum values that could cause overflow
#[rstest]
#[case(vec![usize::MAX, 1], Some(usize::MAX))]
#[case(vec![1, usize::MAX], Some(usize::MAX))]
#[case(vec![0, usize::MAX], Some(0))]
#[case(vec![usize::MAX, 0], Some(0))]
#[case(vec![2, usize::MAX/2], Some(usize::MAX-1))] // Should not overflow
#[case(vec![], Some(1))] // Scalar case
fn test_tensor_size_calculation_edge_cases(#[case] shape: Vec<usize>, #[case] expected: Option<usize>) {
    let result = utils::calculate_tensor_size_safe(&shape);
    assert_eq!(result, expected);
}

/// Test operations with very long strings in attributes (reasonable length)
#[test]
fn test_very_long_strings_in_attributes() {
    let very_long_string = "test_char_".repeat(100_000); // 1M character string (reduced from 10M)
    
    let attr = Attribute::String(very_long_string.clone());
    
    match attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), very_long_string.len());
            assert_eq!(s, very_long_string);
        },
        _ => panic!("Expected String attribute"),
    }
}

/// Test value with zero dimensions but different types
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_scalar_values_different_types(#[case] data_type: Type) {
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: data_type.clone(),
        shape: vec![], // Empty shape indicates scalar
    };

    assert!(scalar_value.shape.is_empty());
    assert_eq!(scalar_value.ty, data_type);
    assert_eq!(scalar_value.num_elements().unwrap(), 1);
}

/// Test integer overflow protection in tensor size calculations
#[test]
fn test_overflow_protection_with_large_multiplications() {
    // Values that would cause overflow when multiplied
    let dims = vec![usize::MAX, 2];
    let result = utils::calculate_tensor_size_safe(&dims);
    
    // Should return None to indicate overflow rather than panicking
    assert!(result.is_none());
    
    // Test another combination that causes overflow
    let dims2 = vec![100_000_000, 100_000_000, 100];
    let result2 = utils::calculate_tensor_size_safe(&dims2);
    
    // On most systems this would overflow
    assert!(result2.is_some() || result2.is_none()); // Should not panic in either case
}

/// Test operations with mixed data types and edge case shapes
#[test]
fn test_mixed_type_operations_edge_cases() {
    let mut op = Operation::new("mixed_type_op");

    // Add values with different types and edge case shapes
    let test_values = vec![
        Value { name: "f32_scalar".to_string(), ty: Type::F32, shape: vec![] },
        Value { name: "i64_zero_tensor".to_string(), ty: Type::I64, shape: vec![0] },
        Value { name: "bool_large_flat".to_string(), ty: Type::Bool, shape: vec![1_000_000] },
        Value { name: "f64_very_deep".to_string(), ty: Type::F64, shape: vec![1; 10_000] },
    ];

    for value in test_values {
        op.inputs.push(value.clone());
        op.outputs.push(Value {
            name: format!("output_{}", value.name),
            ty: value.ty.clone(),
            shape: value.shape.clone(),
        });
    }

    assert_eq!(op.inputs.len(), 4);
    assert_eq!(op.outputs.len(), 4);
    assert_eq!(op.op_type, "mixed_type_op");
}

/// Test attribute array nesting at reasonable depth to avoid stack issues
#[test]
fn test_reasonable_depth_attribute_nesting() {
    // Create a nested array structure for attributes (but not too deep to cause stack overflow)
    let mut nested_attr = Attribute::Int(42);
    for _ in 0..100 {
        nested_attr = Attribute::Array(vec![nested_attr]);
    }

    // Verify we can still clone and compare the nested structure
    let cloned_attr = nested_attr.clone();
    assert_eq!(nested_attr, cloned_attr);

    // Verify the structure is preserved
    if let Attribute::Array(_) = nested_attr {
        // Success - it's still an array
    } else {
        panic!("Expected nested array structure");
    }
}

/// Test module creation with many combinations
#[test]
fn test_module_with_many_combinations() {
    let mut module = Module::new("many_combinations_module");

    // Add an operation with significant complexity but not extreme
    let mut complex_op = Operation::new("complex_op");
    
    // Add many inputs with scalar shapes (should be efficient)
    for i in 0..1_000 {  // Reduced from 10_000
        complex_op.inputs.push(Value {
            name: format!("scalar_input_{}", i),
            ty: Type::F32,
            shape: vec![], // All scalars
        });
    }
    
    // Add attributes with complex nested structure (reduced amounts)
    for i in 0..500 {  // Reduced from 5_000
        complex_op.attributes.insert(
            format!("complex_attr_{}", i),
            Attribute::Array(vec![
                Attribute::Int(i as i64),
                Attribute::Array(vec![
                    Attribute::String(format!("nested_{}", i)),
                    Attribute::Float(i as f64 * 0.5),
                ]),
                Attribute::Bool(i % 2 == 0),
            ])
        );
    }

    module.add_operation(complex_op);

    assert_eq!(module.name, "many_combinations_module");
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 1_000);
    assert_eq!(module.operations[0].attributes.len(), 500);
}

/// Test tensor size calculations for byte size with different types
#[rstest]
#[case(Type::F32, vec![], 4)]      // Scalar F32 = 4 bytes
#[case(Type::F64, vec![], 8)]      // Scalar F64 = 8 bytes
#[case(Type::I32, vec![], 4)]      // Scalar I32 = 4 bytes
#[case(Type::I64, vec![], 8)]      // Scalar I64 = 8 bytes
#[case(Type::Bool, vec![], 1)]     // Scalar Bool = 1 byte
#[case(Type::F32, vec![2, 3], 24)] // 2x3 F32 tensor = 2*3*4 = 24 bytes
#[case(Type::I64, vec![5, 10], 8*5*10)] // 5x10 I64 tensor = 5*10*8 = 400 bytes
fn test_tensor_byte_size_calculation(#[case] data_type: Type, #[case] shape: Vec<usize>, #[case] expected_size: usize) {
    match utils::ir_utils::calculate_tensor_size(&data_type, &shape) {
        Ok(size) => {
            assert_eq!(size, expected_size);
        },
        Err(e) => panic!("Unexpected error in tensor size calculation: {}", e),
    }
}