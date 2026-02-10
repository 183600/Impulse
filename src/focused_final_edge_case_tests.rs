//! Focused final edge case tests for the Impulse compiler
//! Tests cover edge cases that may not be covered by other test files

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils::{self, get_element_type},
};

/// Test 1: Value.num_elements() returns None for overflow cases
#[test]
fn test_value_num_elements_overflow() {
    // Test with dimensions that would overflow when multiplied
    // Using values that would exceed usize::MAX when multiplied together
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // This would overflow
    };
    
    // num_elements should return None for overflow cases
    let result = overflow_value.num_elements();
    assert!(result.is_none(), "num_elements should return None for overflow cases");
}

/// Test 2: calculate_tensor_size with nested tensor overflow detection
#[test]
fn test_calculate_tensor_size_nested_overflow() {
    // Test nested tensor that would cause overflow in size calculation
    // Use very large inner tensor to trigger overflow
    let inner_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![100_000_000, 100_000_000], // Very large inner tensor
    };
    
    let outer_shape = vec![100_000_000]; // Very large outer dimension
    
    // This should detect overflow and return an error
    let result = ir_utils::calculate_tensor_size(&inner_tensor, &outer_shape);
    assert!(result.is_err(), "Should detect overflow in nested tensor size calculation");
}

/// Test 3: Attribute with NaN and infinity float values
#[test]
fn test_attribute_nan_infinity_floats() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    // NaN is not equal to itself
    assert_ne!(nan_attr, nan_attr);
    
    // Positive infinity equals itself
    assert_eq!(pos_inf_attr, pos_inf_attr);
    
    // Negative infinity equals itself
    assert_eq!(neg_inf_attr, neg_inf_attr);
    
    // Positive and negative infinity are not equal
    assert_ne!(pos_inf_attr, neg_inf_attr);
}

/// Test 4: Module with empty string name and operation type
#[test]
fn test_module_empty_strings() {
    let mut module = Module::new("");
    assert_eq!(module.name, "");
    
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    
    // Add operation to module
    module.add_operation(op);
    assert_eq!(module.operations[0].op_type, "");
}

/// Test 5: get_element_type with deeply nested tensor types
#[test]
fn test_get_element_type_deeply_nested() {
    // Create a tensor with 10 levels of nesting
    let mut current_type = Type::F32;
    for _ in 0..10 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // get_element_type should correctly extract the base type
    let base_type = get_element_type(&current_type);
    assert_eq!(base_type, &Type::F32);
}

/// Test 6: Operation with duplicate attribute keys (last wins)
#[test]
fn test_operation_duplicate_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("test_op");
    let mut attrs = HashMap::new();
    
    // Insert same key multiple times (last value wins)
    attrs.insert("key".to_string(), Attribute::Int(1));
    attrs.insert("key".to_string(), Attribute::Int(2)); // Overwrites previous
    attrs.insert("key".to_string(), Attribute::Int(3)); // Overwrites previous
    
    op.attributes = attrs;
    
    // Only one entry should exist
    assert_eq!(op.attributes.len(), 1);
    assert_eq!(op.attributes.get("key"), Some(&Attribute::Int(3)));
}

/// Test 7: Array attribute with all same elements (intensive cloning test)
#[test]
fn test_array_attribute_same_elements() {
    // Create array with 1000 identical elements
    let elem = Attribute::Int(42);
    let large_array = Attribute::Array(vec![elem.clone(); 1000]);
    
    match large_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1000);
            // All elements should be equal
            for item in &arr {
                assert_eq!(item, &elem);
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 8: type_to_string with empty tensor shape
#[test]
fn test_type_to_string_empty_tensor_shape() {
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![], // Empty shape (scalar tensor)
    };
    
    let result = ir_utils::type_to_string(&tensor_type);
    assert_eq!(result, "tensor<f32, []>");
}

/// Test 9: count_operations_by_type with empty module
#[test]
fn test_count_operations_empty_module() {
    let module = Module::new("empty");
    let counts = ir_utils::count_operations_by_type(&module);
    
    // Should return empty hash map
    assert!(counts.is_empty());
    assert_eq!(counts.len(), 0);
}

/// Test 10: find_operations_by_type with non-existent operation type
#[test]
fn test_find_operations_non_existent() {
    let mut module = Module::new("test");
    module.add_operation(Operation::new("add"));
    module.add_operation(Operation::new("multiply"));
    
    // Try to find non-existent operation
    let results = ir_utils::find_operations_by_type(&module, "non_existent");
    assert!(results.is_empty());
    assert_eq!(results.len(), 0);
}