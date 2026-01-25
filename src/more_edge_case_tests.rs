//! Additional edge case tests for the Impulse compiler
//! Focusing on boundary conditions not covered in existing tests

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module, TypeExtensions};

/// Test 1: Value with maximum possible shape dimensions
#[test]
fn test_value_max_shape_dimensions() {
    // Create a value with an extremely large number of dimensions
    let max_shape = vec![1; 1000]; // 1000 dimensions, each with size 1
    let value = Value {
        name: "max_dims".to_string(),
        ty: Type::F32,
        shape: max_shape,
    };
    
    assert_eq!(value.shape.len(), 1000);
    // Total elements should be 1 (1^1000)
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 1);
}

/// Test 2: Operations with empty string names (boundary case)
#[test]
fn test_operation_empty_name() {
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
}

/// Test 3: Value with special Unicode names
#[test]
fn test_special_unicode_names() {
    let special_names = [
        "тест",           // Cyrillic
        "テスト",         // Japanese
        "اختبار",         // Arabic
        "🚀_tensor",     // Emoji
        "a̐e̊i̊o̊ů",       // Combining characters
    ];
    
    for name in &special_names {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, *name);
    }
}

/// Test 4: Tensor shape overflow scenarios with checked arithmetic
#[rstest]
#[case(vec![usize::MAX, 2], None)]  // Would overflow
#[case(vec![usize::MAX / 2, 2], Some((usize::MAX / 2) * 2))]  // Should not overflow  
#[case(vec![0, usize::MAX], Some(0))]  // Contains 0, so product is 0
fn test_tensor_shape_overflow(#[case] shape: Vec<usize>, #[case] expected_result: Option<usize>) {
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let product_result: Option<usize> = value.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    assert_eq!(product_result, expected_result);
}

/// Test 5: Attribute equality with complex nested structures
#[test]
fn test_complex_attribute_equality() {
    let attr1 = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![Attribute::Float(2.5), Attribute::Bool(true)]),
        Attribute::String("test".to_string()),
    ]);
    
    let attr2 = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![Attribute::Float(2.5), Attribute::Bool(true)]),
        Attribute::String("test".to_string()),
    ]);
    
    let attr3 = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![Attribute::Float(2.6), Attribute::Bool(true)]),  // Different float
        Attribute::String("test".to_string()),
    ]);
    
    assert_eq!(attr1, attr2);  // Should be equal
    assert_ne!(attr1, attr3);  // Should not be equal
}

/// Test 6: Deep nesting with different primitive types
#[test]
fn test_deep_nesting_with_different_types() {
    let base_types = [Type::F32, Type::I64, Type::Bool];
    
    for base_type in &base_types {
        let mut nested_type = base_type.clone();
        
        // Create 10 levels of nesting
        for _ in 0..10 {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![2],
            };
        }
        
        // Verify that the nested type is valid
        assert!(nested_type.is_valid_type());
        
        // Clone and verify equality
        let cloned = nested_type.clone();
        assert_eq!(nested_type, cloned);
    }
}

/// Test 7: Operation with extremely long attribute names
#[test]
fn test_extremely_long_attribute_names() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("test_op");
    let mut attrs = HashMap::new();
    
    // Add an attribute with a very long name
    let long_name = "a".repeat(10_000);
    attrs.insert(long_name.clone(), Attribute::Int(123));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1);
    assert!(op.attributes.contains_key(&long_name));
    match op.attributes.get(&long_name) {
        Some(Attribute::Int(123)) => assert!(true),  // Success
        _ => panic!("Expected Int(123)"),
    }
}

/// Test 8: Boolean tensor type edge cases
#[test]
fn test_boolean_tensor_edge_cases() {
    let bool_tensor = Value {
        name: "bool_tensor".to_string(),
        ty: Type::Bool,
        shape: vec![100_000],  // Large boolean tensor
    };
    
    assert_eq!(bool_tensor.ty, Type::Bool);
    assert_eq!(bool_tensor.shape, vec![100_000]);
    
    // Calculate expected size: 100_000 elements, 1 byte each
    let total_elements: usize = bool_tensor.shape.iter().product();
    assert_eq!(total_elements, 100_000);
}

/// Test 9: Empty collections and null-like values
#[test]
fn test_empty_collections() {
    let mut op = Operation::new("empty_test");
    
    // Test initially empty collections
    assert!(op.inputs.is_empty());
    assert!(op.outputs.is_empty());
    assert!(op.attributes.is_empty());
    
    // Add a single input, then clear
    op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    assert_eq!(op.inputs.len(), 1);
    
    // Clear inputs
    op.inputs.clear();
    assert!(op.inputs.is_empty());
    
    // Create a module and test empty state
    let module = Module::new("empty_module");
    assert_eq!(module.name, "empty_module");
    assert!(module.operations.is_empty());
    assert!(module.inputs.is_empty());
    assert!(module.outputs.is_empty());
}

/// Test 10: Recursive type validation with mixed nesting
#[test]
fn test_mixed_nested_type_validation() {
    // Create a complex nested structure: tensor<tensor<F32, [2]>, tensor<I32, [3]>> (sort of)
    let complex_type = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    
    // Verify it's valid
    assert!(complex_type.is_valid_type());
    
    // Create a potentially invalid structure (theoretical - current definition allows it)
    let valid_type = Type::Tensor {
        element_type: Box::new(Type::I32),
        shape: vec![5, 0, 10],  // Contains 0 but still valid
    };
    
    assert!(valid_type.is_valid_type());
    
    // Test shape with multiple zeros
    let zero_shape = Value {
        name: "multi_zero".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 20, 0],  // Multiple zeros
    };
    
    let calculated_size: usize = zero_shape.shape.iter().product();
    assert_eq!(calculated_size, 0);  // Should be 0 due to zeros
}

/// Test 11: Error-prone numeric edge cases in tensor operations
#[rstest]
#[case(vec![usize::MAX, 1], usize::MAX)]  // Maximum usize
#[case(vec![100, 100, 100], 1_000_000)]  // 3D tensor
#[case(vec![2; 10], 1024)]  // 2^10
fn test_tensor_numeric_boundaries(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let value = Value {
        name: "numeric_boundary".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let calculated_size: usize = value.shape.iter().product();
    assert_eq!(calculated_size, expected_size);
}

/// Test 12: Module serialization edge cases
#[test]
fn test_module_serialization_prep() {
    let mut module = Module::new("serialization_test");
    
    // Add operations that might cause issues during serialization
    let mut op = Operation::new("complex_op");
    op.inputs.push(Value {
        name: "input_with_special_chars".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    });
    
    // Add various attribute types
    use std::collections::HashMap;
    let mut attrs = HashMap::new();
    attrs.insert("int".to_string(), Attribute::Int(42));
    attrs.insert("float".to_string(), Attribute::Float(3.14159));
    attrs.insert("string".to_string(), Attribute::String("special chars: \n\t\r".to_string()));
    attrs.insert("bool".to_string(), Attribute::Bool(true));
    attrs.insert("array".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.0),
        Attribute::String("nested".to_string()),
    ]));
    
    op.attributes = attrs;
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.name, "serialization_test");
}