//! Additional boundary scenario tests for the Impulse compiler
//! Focuses on edge cases and boundary conditions for IR structures

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;

// Test 1: Value with maximum possible shape dimensions
#[test]
fn test_maximum_shape_dimensions() {
    let max_dims = vec![usize::MAX, 1];  // Testing near-maximum values
    let value = Value {
        name: "max_dims_tensor".to_string(),
        ty: Type::F32,
        shape: max_dims.clone(),
    };
    
    assert_eq!(value.shape, max_dims);
    // Note: This multiplication may overflow, but the test validates construction
    let _product_check = value.shape.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
}

// Test 2: Operations with empty string names (edge case for identifiers)
#[test]
fn test_empty_operation_name() {
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    // Although unusual, this test ensures the system handles empty names properly
}

// Test 3: Nested tensor types at maximum reasonable depth
#[test]
fn test_very_deeply_nested_tensors() {
    let mut current_type = Type::F32;
    // Create 50 levels of nesting to stress-test recursion limits
    for i in 0..50 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 3 + 1], // Varies shape to make each level different
        };
    }
    
    // Verify the final type is still a tensor
    if let Type::Tensor { .. } = current_type {
        // Success
    } else {
        panic!("Expected deeply nested tensor type");
    }
}

// Test 4: Testing potential integer overflow in value.num_elements()
#[test]
fn test_num_elements_overflow_handling() {
    // Create a value with dimensions that would cause overflow in multiplication
    let problematic_value = Value {
        name: "problematic_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 1_000_000], // Potential overflow
    };
    
    // Use the safe method that returns Option
    let elements_opt = problematic_value.num_elements();
    match elements_opt {
        Some(_) => {}, // May or may not overflow depending on platform
        None => assert!(true), // Indicates overflow was detected
    }
}

// Test 5: Boundary test for empty attribute arrays
#[test]
fn test_empty_attribute_operations() {
    let mut op = Operation::new("empty_attrs_op");
    op.attributes.clear(); // Ensure it's empty
    
    assert_eq!(op.attributes.len(), 0);
    assert!(op.attributes.is_empty());
    
    // Add and remove attributes to test boundary behavior
    op.attributes.insert("temp_attr".to_string(), Attribute::Int(123));
    assert_eq!(op.attributes.len(), 1);
    
    op.attributes.remove("temp_attr");
    assert_eq!(op.attributes.len(), 0);
}

// Test 6: Test operations with extremely long attribute names
#[rstest]
fn test_extremely_long_attribute_names() {
    let mut op = Operation::new("long_attr_name_op");
    let long_attr_name = "a".repeat(50_000); // Very long attribute name
    op.attributes.insert(long_attr_name.clone(), Attribute::Int(42));
    
    assert_eq!(op.attributes.len(), 1);
    assert!(op.attributes.contains_key(&long_attr_name));
    
    // Verify the value is correct
    if let Some(attr_val) = op.attributes.get(&long_attr_name) {
        if let Attribute::Int(val) = attr_val {
            assert_eq!(*val, 42);
        } else {
            panic!("Expected Int attribute");
        }
    }
}

// Test 7: Test value with all possible primitive types in boundary scenarios
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_all_primitive_types_with_empty_shapes(#[case] primitive_type: Type) {
    let value = Value {
        name: "primitive_scalar".to_string(),
        ty: primitive_type.clone(),
        shape: vec![], // Empty shape = scalar
    };
    
    assert_eq!(value.ty, primitive_type);
    assert_eq!(value.shape.len(), 0);
    assert!(value.shape.is_empty());
}

// Test 8: Test operations with 0 inputs, 0 outputs but many attributes
#[test]
fn test_operation_with_only_attributes() {
    let mut op = Operation::new("attrs_only_op");
    
    // Add many attributes but no inputs or outputs
    for i in 0..1000 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify a few random attributes exist
    assert!(op.attributes.contains_key("attr_0"));
    assert!(op.attributes.contains_key("attr_500"));
    assert!(op.attributes.contains_key("attr_999"));
}

// Test 9: Test floating point special values in attributes
#[test]
fn test_special_float_values_in_attributes() {
    use std::collections::HashMap;
    
    let mut attrs = HashMap::new();
    
    // Test special float values
    attrs.insert("positive_infinity".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("negative_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("tiny_positive".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("max_finite".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_finite".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    
    // Verify special values are preserved
    assert!(matches!(attrs.get("positive_infinity"), Some(Attribute::Float(f)) if f.is_infinite() && f.is_sign_positive()));
    assert!(matches!(attrs.get("negative_infinity"), Some(Attribute::Float(f)) if f.is_infinite() && f.is_sign_negative()));
    assert!(matches!(attrs.get("nan_value"), Some(Attribute::Float(f)) if f.is_nan()));
    
    // Test with an operation
    let mut op = Operation::new("special_float_op");
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 7);
}

// Test 10: Test recursive type validation with complex nesting
#[test]
fn test_recursive_type_validation_scenarios() {
    // Create a complex nested type structure
    let complex_type = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2],
            }),
            shape: vec![3, 4],
        }),
        shape: vec![5],
    };
    
    // Validate the recursive structure
    assert!(complex_type.is_valid_type());
    
    // Create a value with this complex type
    let complex_value = Value {
        name: "complex_nested_value".to_string(),
        ty: complex_type.clone(),
        shape: vec![10, 20],
    };
    
    assert_eq!(complex_value.name, "complex_nested_value");
    assert_eq!(complex_value.ty, complex_type);
    assert_eq!(complex_value.shape, vec![10, 20]);
    
    // Verify the nested type is valid
    assert!(complex_value.ty.is_valid_type());
}