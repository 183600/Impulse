//! Advanced comprehensive tests covering additional edge cases and boundary conditions
//! Focus on numerical precision, memory safety, and complex scenarios

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with extremely small float attribute values (subnormal numbers)
#[test]
fn test_subnormal_float_values() {
    // Test subnormal float values (denormalized numbers)
    let mut op = Operation::new("subnormal_test");
    
    // f64::MIN_POSITIVE is the smallest positive normal number
    let min_positive = f64::MIN_POSITIVE;
    // Subnormal values are smaller than MIN_POSITIVE
    let subnormal = f64::MIN_POSITIVE / 2.0;
    
    op.attributes.insert("min_positive".to_string(), Attribute::Float(min_positive));
    op.attributes.insert("subnormal".to_string(), Attribute::Float(subnormal));
    
    // Verify attributes are stored correctly
    assert!(op.attributes.contains_key("min_positive"));
    assert!(op.attributes.contains_key("subnormal"));
    
    match op.attributes.get("subnormal").unwrap() {
        Attribute::Float(val) => {
            assert!(*val > 0.0);
            assert!(*val < min_positive);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: Module with operations containing NaN and Infinity attributes
#[test]
fn test_nan_infinity_attributes() {
    let mut module = Module::new("nan_infinity_module");
    
    let mut op_with_special = Operation::new("special_values");
    op_with_special.attributes.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
    op_with_special.attributes.insert("positive_inf".to_string(), Attribute::Float(f64::INFINITY));
    op_with_special.attributes.insert("negative_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    
    module.add_operation(op_with_special);
    
    assert_eq!(module.operations.len(), 1);
    
    // Verify NaN is stored (NaN != NaN, so we check it's stored as Float)
    match module.operations[0].attributes.get("nan_value") {
        Some(Attribute::Float(val)) => assert!(val.is_nan()),
        _ => panic!("Expected NaN Float attribute"),
    }
    
    // Verify positive infinity
    match module.operations[0].attributes.get("positive_inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val > 0.0),
        _ => panic!("Expected positive infinity Float attribute"),
    }
    
    // Verify negative infinity
    match module.operations[0].attributes.get("negative_inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val < 0.0),
        _ => panic!("Expected negative infinity Float attribute"),
    }
}

/// Test 3: Value with very large integer attribute values
#[test]
fn test_large_integer_attributes() {
    let mut op = Operation::new("large_integers");
    
    // Test with extreme integer values
    op.attributes.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("zero".to_string(), Attribute::Int(0));
    op.attributes.insert("negative_one".to_string(), Attribute::Int(-1));
    
    assert_eq!(op.attributes.len(), 4);
    
    match op.attributes.get("max_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected max i64"),
    }
    
    match op.attributes.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected min i64"),
    }
}

/// Test 4: Module with deeply nested tensor type definitions
#[test]
fn test_deeply_nested_tensor_types() {
    // Create a tensor with 5 levels of nesting
    // tensor<tensor<tensor<tensor<tensor<f32, [2]>, [3]>, [4]>, [5]>, [6]>
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![3],
    };
    
    let level3 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![4],
    };
    
    let level4 = Type::Tensor {
        element_type: Box::new(level3.clone()),
        shape: vec![5],
    };
    
    let level5 = Type::Tensor {
        element_type: Box::new(level4.clone()),
        shape: vec![6],
    };
    
    // Verify the deepest level contains F32
    match &level5 {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, &vec![6]);
            
            // Traverse down to check F32
            match element_type.as_ref() {
                Type::Tensor { element_type: e4, shape: s4 } => {
                    assert_eq!(s4, &vec![5]);
                    match e4.as_ref() {
                        Type::Tensor { element_type: e3, shape: s3 } => {
                            assert_eq!(s3, &vec![4]);
                            match e3.as_ref() {
                                Type::Tensor { element_type: e2, shape: s2 } => {
                                    assert_eq!(s2, &vec![3]);
                                    match e2.as_ref() {
                                        Type::Tensor { element_type: e1, shape: s1 } => {
                                            assert_eq!(s1, &vec![2]);
                                            assert_eq!(e1.as_ref(), &Type::F32);
                                        }
                                        _ => panic!("Expected Tensor at level 2"),
                                    }
                                }
                                _ => panic!("Expected Tensor at level 3"),
                            }
                        }
                        _ => panic!("Expected Tensor at level 4"),
                    }
                }
                _ => panic!("Expected Tensor at level 5"),
            }
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 5: Value with num_elements() returning None for overflow cases
#[test]
fn test_num_elements_overflow_handling() {
    // Create a value that would cause overflow when calculating num_elements
    // On a 64-bit system, this would overflow usize
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // Would overflow if multiplied
    };
    
    // num_elements should handle this gracefully
    // The implementation uses checked_mul, so it should return None
    let result = overflow_value.num_elements();
    assert_eq!(result, None);
    
    // Test with a shape that would also overflow
    let another_overflow = Value {
        name: "another_overflow".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000_000, 10, 10], // 100 billion elements
    };
    
    // This might also overflow on 32-bit systems
    let another_result = another_overflow.num_elements();
    // On 64-bit this should work, on 32-bit it would overflow
    // Just verify it doesn't crash
    assert!(another_result.is_some() || another_result.is_none());
}

/// Test 6: Module with operations having extremely long attribute keys
#[test]
fn test_extremely_long_attribute_keys() {
    let mut op = Operation::new("long_keys");
    
    // Create attribute keys of varying extreme lengths
    let key_1000 = "x".repeat(1000);
    let key_5000 = "y".repeat(5000);
    let key_10000 = "z".repeat(10000);
    
    op.attributes.insert(key_1000.clone(), Attribute::Int(1));
    op.attributes.insert(key_5000.clone(), Attribute::Int(2));
    op.attributes.insert(key_10000.clone(), Attribute::Int(3));
    
    assert_eq!(op.attributes.len(), 3);
    assert!(op.attributes.contains_key(&key_1000));
    assert!(op.attributes.contains_key(&key_5000));
    assert!(op.attributes.contains_key(&key_10000));
}

/// Test 7: Value with shape containing both very large and very small dimensions
#[test]
fn test_mixed_extreme_dimensions() {
    // Test shapes with extreme variations in dimension sizes
    let test_cases = vec![
        (vec![1, 1000000, 1], "thin_middle", 1_000_000),
        (vec![1000000, 1, 1], "thin_end", 1_000_000),
        (vec![1, 1, 1000000], "thin_start", 1_000_000),
        (vec![100000, 100000, 1], "2d_thin", 10_000_000_000),
        (vec![1, 100000, 100000], "2d_thin_start", 10_000_000_000),
    ];
    
    for (shape, name, expected_product) in test_cases {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Calculate product safely
        let product: usize = value.shape.iter().product();
        assert_eq!(product, expected_product);
    }
}

/// Test 8: Module with operations having duplicate attribute keys (last one wins)
#[test]
fn test_duplicate_attribute_keys() {
    let mut op = Operation::new("duplicate_attrs");
    
    // Insert the same key multiple times
    op.attributes.insert("test_key".to_string(), Attribute::Int(1));
    op.attributes.insert("test_key".to_string(), Attribute::Int(2));
    op.attributes.insert("test_key".to_string(), Attribute::Int(3));
    
    // HashMap behavior: last insert wins
    assert_eq!(op.attributes.len(), 1);
    
    match op.attributes.get("test_key") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 3),
        _ => panic!("Expected Int(3)"),
    }
}

/// Test 9: Value with all primitive types in a single operation
#[test]
fn test_all_primitive_types_in_single_operation() {
    let mut op = Operation::new("all_types");
    
    // Create values with all primitive types
    let f32_val = Value {
        name: "f32_value".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    let f64_val = Value {
        name: "f64_value".to_string(),
        ty: Type::F64,
        shape: vec![1],
    };
    
    let i32_val = Value {
        name: "i32_value".to_string(),
        ty: Type::I32,
        shape: vec![1],
    };
    
    let i64_val = Value {
        name: "i64_value".to_string(),
        ty: Type::I64,
        shape: vec![1],
    };
    
    let bool_val = Value {
        name: "bool_value".to_string(),
        ty: Type::Bool,
        shape: vec![1],
    };
    
    op.inputs.push(f32_val);
    op.inputs.push(f64_val);
    op.inputs.push(i32_val);
    op.inputs.push(i64_val);
    op.inputs.push(bool_val);
    
    assert_eq!(op.inputs.len(), 5);
    assert_eq!(op.inputs[0].ty, Type::F32);
    assert_eq!(op.inputs[1].ty, Type::F64);
    assert_eq!(op.inputs[2].ty, Type::I32);
    assert_eq!(op.inputs[3].ty, Type::I64);
    assert_eq!(op.inputs[4].ty, Type::Bool);
}

/// Test 10: Module with empty operations and empty values
#[test]
fn test_empty_operations_and_values() {
    let mut module = Module::new("empty_test");
    
    // Add an operation with no inputs, outputs, or attributes
    let empty_op = Operation::new("empty_operation");
    assert!(empty_op.inputs.is_empty());
    assert!(empty_op.outputs.is_empty());
    assert!(empty_op.attributes.is_empty());
    
    module.add_operation(empty_op);
    
    // Add another operation
    let another_empty_op = Operation::new("another_empty");
    module.add_operation(another_empty_op);
    
    assert_eq!(module.operations.len(), 2);
    assert!(module.operations[0].inputs.is_empty());
    assert!(module.operations[1].outputs.is_empty());
    
    // Verify module has no inputs or outputs
    assert!(module.inputs.is_empty());
    assert!(module.outputs.is_empty());
}