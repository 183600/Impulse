//! Comprehensive edge case boundary tests - 覆盖更多边界情况
//! 使用标准库 assert! 和 assert_eq! 测试独特的边界场景

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Module with maximum valid dimensions in a single tensor
#[test]
fn test_max_valid_dimensions_tensor() {
    // Test tensor with many dimensions that are all valid
    let value = Value {
        name: "high_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], // 10 dimensions
    };
    
    assert_eq!(value.shape.len(), 10);
    assert_eq!(value.num_elements(), Some(1024)); // 2^10
}

/// Test 2: Value with num_elements returning None for overflow
#[test]
fn test_overflow_in_num_elements() {
    // Create a shape that would cause overflow when multiplying
    // Using dimensions that would exceed usize::MAX when multiplied
    // On 64-bit systems, usize::MAX is 18446744073709551615
    // So we need dimensions that multiply to more than that
    let large_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100_000_000_000, 100_000_000_000], // Would overflow on 64-bit systems
    };
    
    // num_elements should return None due to overflow
    assert_eq!(large_value.num_elements(), None);
}

/// Test 3: Operation with attribute keys containing special characters
#[test]
fn test_special_attribute_keys() {
    let mut op = Operation::new("special_keys");
    let mut attrs = HashMap::new();
    
    // Test attribute keys with various special characters
    let special_keys = vec![
        "key.with.dots",
        "key-with-dashes",
        "key_with_underscores",
        "key:with:colons",
        "key/with/slashes",
        "key@with@ats",
    ];
    
    for key in special_keys {
        attrs.insert(key.to_string(), Attribute::Int(1));
    }
    
    op.attributes = attrs;
    
    // Verify all keys are present
    assert_eq!(op.attributes.len(), 6);
    for key in &["key.with.dots", "key-with-dashes", "key_with_underscores",
                  "key:with:colons", "key/with/slashes", "key@with@ats"] {
        assert!(op.attributes.contains_key(*key));
    }
}

/// Test 4: Value with all dimensions equal to 1 (scalar-like but not scalar)
#[test]
fn test_all_ones_shape() {
    let value = Value {
        name: "all_ones".to_string(),
        ty: Type::F64,
        shape: vec![1, 1, 1, 1, 1], // 5 dimensions all with size 1
    };
    
    assert_eq!(value.shape.len(), 5);
    assert_eq!(value.num_elements(), Some(1));
}

/// Test 5: Module with cyclic name pattern
#[test]
fn test_cyclic_module_names() {
    let base_names = vec!["module_a", "module_b", "module_c"];
    
    for base_name in base_names {
        let module = Module::new(base_name);
        assert_eq!(module.name, base_name);
        assert!(module.operations.is_empty());
    }
}

/// Test 6: Attribute array with single element (edge case between scalar and array)
#[test]
fn test_single_element_array() {
    let single_array = Attribute::Array(vec![Attribute::Int(42)]);
    
    match single_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: Module with operations that share inputs
#[test]
fn test_shared_inputs_across_operations() {
    let mut module = Module::new("shared_inputs");
    
    // Create a shared input
    let shared_input = Value {
        name: "shared_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    };
    
    // Create two operations using the same input
    let mut op1 = Operation::new("op1");
    op1.inputs.push(shared_input.clone());
    op1.outputs.push(Value {
        name: "output1".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(shared_input);
    op2.outputs.push(Value {
        name: "output2".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].inputs[0].name, "shared_input");
    assert_eq!(module.operations[1].inputs[0].name, "shared_input");
}

/// Test 8: Type validation for deep nesting
#[test]
fn test_deep_nesting_type_validation() {
    // Create a deeply nested tensor type
    let mut nested_type: Type = Type::F32;
    for _ in 0..20 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![1],
        };
    }
    
    // Validate the nested type
    assert!(nested_type.is_valid_type());
}

/// Test 9: Operation with empty string in attributes
#[test]
fn test_empty_string_attribute() {
    let mut op = Operation::new("empty_strings");
    let mut attrs = HashMap::new();
    
    attrs.insert("empty_key".to_string(), Attribute::String("".to_string()));
    attrs.insert("normal_key".to_string(), Attribute::String("value".to_string()));
    
    op.attributes = attrs;
    
    match op.attributes.get("empty_key") {
        Some(Attribute::String(s)) => assert_eq!(s.len(), 0),
        _ => panic!("Expected empty String attribute"),
    }
    
    match op.attributes.get("normal_key") {
        Some(Attribute::String(s)) => assert_eq!(s, "value"),
        _ => panic!("Expected normal String attribute"),
    }
}

/// Test 10: Module with operations having no attributes
#[test]
fn test_operations_without_attributes() {
    let mut module = Module::new("no_attrs");
    
    for i in 0..5 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert!(op.attributes.is_empty());
    }
}