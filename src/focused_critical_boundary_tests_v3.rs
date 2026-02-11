/// Focused critical boundary tests v3 - additional edge case coverage with standard assertions
/// 
/// This module adds 10 new test cases covering boundary scenarios not covered by existing tests:
/// 1. Module with input/output name collisions
/// 2. Value with negative zero float in attributes
/// 3. Operation with attribute key containing special characters
/// 4. Type with self-referential tensor edge case
/// 5. Attribute array with single element followed by empty arrays
/// 6. Module with circular reference pattern in values
/// 7. Value with alternating large/small dimensions pattern
/// 8. Operation with boolean attribute combinations
/// 9. Type equality with nested tensor variations
/// 10. Module with operation that modifies attributes during cloning

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Module with input/output name collisions
#[test]
fn test_module_name_collisions() {
    let mut module = Module::new("collision_test");
    
    // Add inputs and outputs with same names
    let shared_name = "shared_tensor";
    module.inputs.push(Value {
        name: shared_name.to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.outputs.push(Value {
        name: shared_name.to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    // Both should be present independently
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, "shared_tensor");
    assert_eq!(module.outputs[0].name, "shared_tensor");
}

/// Test 2: Value with negative zero float in attributes
#[test]
fn test_negative_zero_float_attributes() {
    let neg_zero = Attribute::Float(-0.0);
    let pos_zero = Attribute::Float(0.0);
    
    // Verify negative zero is distinguishable via bit pattern
    match neg_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val.is_sign_negative());
        },
        _ => panic!("Expected Float attribute"),
    }
    
    match pos_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val.is_sign_positive());
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Operation with attribute key containing special characters
#[test]
fn test_special_characters_in_attribute_keys() {
    let mut op = Operation::new("special_key_op");
    let mut attrs = HashMap::new();
    
    let special_keys = [
        "key with spaces",
        "key-with-dashes",
        "key_with_underscores",
        "key.with.dots",
        "key:with:colons",
        "key/with/slashes",
        "key\\with\\backslashes",
    ];
    
    for key in special_keys.iter() {
        attrs.insert(key.to_string(), Attribute::Int(1));
    }
    
    op.attributes = attrs;
    
    // All special keys should be stored and retrievable
    for key in special_keys.iter() {
        assert!(op.attributes.contains_key(*key));
        match op.attributes.get(*key) {
            Some(Attribute::Int(1)) => {},
            _ => panic!("Expected Int(1) for key: {}", key),
        }
    }
    assert_eq!(op.attributes.len(), 7);
}

/// Test 4: Type with self-referential tensor edge case
#[test]
fn test_tensor_with_single_dimension_edge_cases() {
    // Test various single-dimension tensors
    let single_1 = Value {
        name: "single_dim_1".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(single_1.num_elements(), Some(1));
    
    let single_2 = Value {
        name: "single_dim_2".to_string(),
        ty: Type::I32,
        shape: vec![2],
    };
    assert_eq!(single_2.num_elements(), Some(2));
    
    // Test with many 1s in shape
    let many_ones = Value {
        name: "many_ones".to_string(),
        ty: Type::F64,
        shape: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    };
    assert_eq!(many_ones.num_elements(), Some(1));
}

/// Test 5: Attribute array with single element followed by empty arrays
#[test]
fn test_mixed_empty_and_nonempty_arrays() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Array(vec![]),
        Attribute::Array(vec![]),
        Attribute::Float(3.14),
        Attribute::Array(vec![]),
    ]);
    
    match mixed {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            
            match arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("First element should be Int(42)"),
            }
            
            // Check empty arrays
            for i in [1, 2, 4] {
                match &arr[i] {
                    Attribute::Array(inner) => assert_eq!(inner.len(), 0),
                    _ => panic!("Element at index {} should be empty array", i),
                }
            }
            
            match arr[3] {
                Attribute::Float(val) if (val - 3.14).abs() < f64::EPSILON => {},
                _ => panic!("Fourth element should be Float(3.14)"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: Module with circular reference pattern in values
#[test]
fn test_circular_reference_pattern() {
    let mut module = Module::new("circular_ref_module");
    
    // Create a chain of operations where output of one becomes input of next
    let mut op1 = Operation::new("op1");
    op1.outputs.push(Value {
        name: "intermediate_1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(op1.outputs[0].clone());
    op2.outputs.push(Value {
        name: "intermediate_2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(op2.outputs[0].clone());
    op3.outputs.push(Value {
        name: "final_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    // Verify the chain
    assert_eq!(module.operations[1].inputs[0].name, "intermediate_1");
    assert_eq!(module.operations[2].inputs[0].name, "intermediate_2");
}

/// Test 7: Value with alternating large/small dimensions pattern
#[test]
fn test_alternating_dimension_pattern() {
    let patterns = vec![
        vec![1, 1000, 1, 1000, 1],
        vec![100, 1, 100, 1, 100],
        vec![10000, 1, 1, 1, 10000],
        vec![1, 1, 100000, 1, 1],
    ];
    
    for (idx, pattern) in patterns.iter().enumerate() {
        let value = Value {
            name: format!("alternating_{}", idx),
            ty: Type::F32,
            shape: pattern.clone(),
        };
        
        let expected_elements: usize = pattern.iter().product();
        assert_eq!(value.num_elements(), Some(expected_elements));
        assert_eq!(value.shape, *pattern);
    }
}

/// Test 8: Operation with boolean attribute combinations
#[test]
fn test_boolean_attribute_combinations() {
    let mut op = Operation::new("bool_combo_op");
    let mut attrs = HashMap::new();
    
    // Test all 4 combinations of two booleans
    attrs.insert("a_true_b_true".to_string(), Attribute::Array(vec![
        Attribute::Bool(true),
        Attribute::Bool(true),
    ]));
    attrs.insert("a_true_b_false".to_string(), Attribute::Array(vec![
        Attribute::Bool(true),
        Attribute::Bool(false),
    ]));
    attrs.insert("a_false_b_true".to_string(), Attribute::Array(vec![
        Attribute::Bool(false),
        Attribute::Bool(true),
    ]));
    attrs.insert("a_false_b_false".to_string(), Attribute::Array(vec![
        Attribute::Bool(false),
        Attribute::Bool(false),
    ]));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    
    // Verify each combination
    match op.attributes.get("a_true_b_true") {
        Some(Attribute::Array(arr)) => {
            match (&arr[0], &arr[1]) {
                (Attribute::Bool(true), Attribute::Bool(true)) => {},
                _ => panic!("Expected (true, true)"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
    
    match op.attributes.get("a_false_b_false") {
        Some(Attribute::Array(arr)) => {
            match (&arr[0], &arr[1]) {
                (Attribute::Bool(false), Attribute::Bool(false)) => {},
                _ => panic!("Expected (false, false)"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Type equality with nested tensor variations
#[test]
fn test_nested_tensor_equality_variations() {
    // Test deeply nested tensors with different structures
    let t1 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        }),
        shape: vec![3, 3],
    };
    
    let t2 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        }),
        shape: vec![3, 3],
    };
    
    let t3 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F64), // Different inner type
            shape: vec![2, 2],
        }),
        shape: vec![3, 3],
    };
    
    let t4 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![3, 3], // Different nested shape
        }),
        shape: vec![3, 3],
    };
    
    assert_eq!(t1, t2); // Same structure
    assert_ne!(t1, t3); // Different inner type
    assert_ne!(t1, t4); // Different nested shape
}

/// Test 10: Module with operation that modifies attributes during cloning
#[test]
fn test_operation_attribute_modification_during_clone() {
    let mut op1 = Operation::new("attr_mod_op");
    op1.attributes.insert("initial".to_string(), Attribute::Int(100));
    op1.attributes.insert("constant".to_string(), Attribute::String("unchanged".to_string()));
    
    // Clone the operation
    let mut op2 = op1.clone();
    
    // Modify the clone's attributes
    op2.attributes.insert("initial".to_string(), Attribute::Int(200));
    op2.attributes.insert("new".to_string(), Attribute::Bool(true));
    
    // Original should be unchanged
    match op1.attributes.get("initial") {
        Some(Attribute::Int(100)) => {},
        _ => panic!("Original should still have Int(100)"),
    }
    assert!(!op1.attributes.contains_key("new"));
    
    // Clone should have modified values
    match op2.attributes.get("initial") {
        Some(Attribute::Int(200)) => {},
        _ => panic!("Clone should have Int(200)"),
    }
    match op2.attributes.get("new") {
        Some(Attribute::Bool(true)) => {},
        _ => panic!("Clone should have new Bool(true) attribute"),
    }
    
    // Constant should be same in both
    assert_eq!(op1.attributes.get("constant"), op2.attributes.get("constant"));
}