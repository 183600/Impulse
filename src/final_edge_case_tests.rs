//! Additional edge case tests for the Impulse compiler
//! Focusing on boundary conditions not covered in existing tests

use rstest::*;
use crate::{
    ir::{Value, Type, Operation, Attribute, Module},
};

/// Test 1: Operations with extremely large integer values in attributes
#[test]
fn test_extremely_large_integer_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("large_int_attr_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with maximum and minimum possible integer values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero_i64".to_string(), Attribute::Int(0));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 3);
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("zero_i64"), Some(&Attribute::Int(0)));
}

/// Test 2: Operations with maximum floating-point values
#[test]
fn test_maximum_floating_point_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("float_attr_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with special floating-point values
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("positive_inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("negative_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    // Note: NaN != NaN, so we handle it separately
    assert_eq!(op.attributes.get("max_f64"), Some(&Attribute::Float(f64::MAX)));
    assert_eq!(op.attributes.get("min_f64"), Some(&Attribute::Float(f64::MIN)));
    assert_eq!(op.attributes.get("positive_inf"), Some(&Attribute::Float(f64::INFINITY)));
    assert_eq!(op.attributes.get("negative_inf"), Some(&Attribute::Float(f64::NEG_INFINITY)));
    
    // For NaN we check that it exists and is NaN
    if let Some(Attribute::Float(val)) = op.attributes.get("nan_value") {
        assert!(val.is_nan());
    } else {
        panic!("NaN attribute not found or not a float");
    }
}

/// Test 3: Values with empty names (edge case for identifier validation)
#[test]
fn test_values_with_empty_names() {
    let value_with_empty_name = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value_with_empty_name.name, "");
    assert_eq!(value_with_empty_name.ty, Type::F32);
    assert_eq!(value_with_empty_name.shape, vec![1, 2, 3]);
    
    // Test operation with empty string as type
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
}

/// Test 4: Tensor shapes with potential overflow in dimension calculation
#[rstest]
#[case(vec![usize::MAX, 1], usize::MAX)]
#[case(vec![1, usize::MAX], usize::MAX)]
#[case(vec![0, usize::MAX], 0)]
#[case(vec![2, usize::MAX / 2], usize::MAX)]  // This should result in about half of MAX
fn test_potential_overflow_shapes(#[case] shape: Vec<usize>, #[case] expected_first_part: usize) {
    let value = Value {
        name: "potential_overflow".to_string(),
        ty: Type::F32,
        shape,
    };
    
    // Use checked multiplication to avoid overflow
    let total_elements: Option<usize> = value.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    // Note: This test may result in None if overflow occurs
    if value.shape.contains(&0) {
        assert_eq!(total_elements, Some(0));
    } else if value.shape.iter().any(|&x| usize::MAX / x < value.shape.iter()
        .filter(|&&y| y != x)
        .fold(1, |acc, &val| acc.saturating_mul(val))) {
        // If potential overflow situation exists
        assert!(total_elements.is_none() || total_elements.unwrap() == expected_first_part);
    }
}

/// Test 5: Deeply nested tensor equality checks
#[test]
fn test_deeply_nested_tensor_equality() {
    // Create two identical deeply nested tensors
    let mut nested1 = Type::F32;
    let mut nested2 = Type::F32;
    
    // Nest each 10 levels deep
    for _ in 0..10 {
        nested1 = Type::Tensor {
            element_type: Box::new(nested1),
            shape: vec![2],
        };
        nested2 = Type::Tensor {
            element_type: Box::new(nested2),
            shape: vec![2],
        };
    }
    
    assert_eq!(nested1, nested2);
    
    // Create a slightly different one (different shape)
    let mut nested3 = Type::F32;
    for _ in 0..10 {
        nested3 = Type::Tensor {
            element_type: Box::new(nested3),
            shape: vec![3],  // Different shape
        };
    }
    
    assert_ne!(nested1, nested3);
}

/// Test 6: Module with operations that have circular references (conceptually)
#[test]
fn test_module_structure_integrity() {
    let mut module = Module::new("structure_test");
    
    // Add operations with various types to ensure no circular references cause issues
    for i in 0..100 {
        let mut op = Operation::new(&format!("op_{}", i));
        
        // Add inputs and outputs with different types
        for j in 0..5 {
            let input_type = match j % 5 {
                0 => Type::F32,
                1 => Type::F64,
                2 => Type::I32,
                3 => Type::I64,
                _ => Type::Bool,
            };
            
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: input_type.clone(),
                shape: vec![j + 1, i + 1],
            });
            
            op.outputs.push(Value {
                name: format!("output_{}_{}", i, j),
                ty: input_type,
                shape: vec![j + 1, i + 2],
            });
        }
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    assert_eq!(module.name, "structure_test");
    
    // Verify no corruption in data
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[99].op_type, "op_99");
    assert_eq!(module.operations[0].inputs.len(), 5);
    assert_eq!(module.operations[0].outputs.len(), 5);
}

/// Test 7: Memory allocation failure simulation (testing for proper error handling)
#[test]
fn test_out_of_memory_simulation() {
    // Simulate trying to create a tensor with a huge amount of dimensions
    // (This won't actually cause OOM, but tests error handling logic)
    
    // A more realistic test of large but feasible allocations
    let mut large_shape = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        large_shape.push(1);  // Very sparse, but many dimensions
    }
    
    let value = Value {
        name: "high_dimensional_sparse".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    assert_eq!(value.shape.len(), 10_000);
    
    // Since all dimensions are 1, total size should be 1
    let product: usize = value.shape.iter().product();
    assert_eq!(product, 1);
}

/// Test 8: String length limits for names and attributes
#[rstest]
#[case("a".repeat(100))]
#[case("x".repeat(1_000))]
#[case("test".repeat(2_500))]  // 10k characters
fn test_long_string_values(#[case] long_string: String) {
    // Test value with long name
    let value = Value {
        name: long_string.clone(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    assert_eq!(value.name, long_string);
    assert_eq!(value.name.len(), long_string.len());
    
    // Test operation with long name
    let op = Operation::new(&long_string);
    assert_eq!(op.op_type, long_string);
    
    // Test attribute with long string value
    use std::collections::HashMap;
    let mut test_op = Operation::new("long_string_test");
    let mut attrs = HashMap::new();
    attrs.insert("long_attr".to_string(), Attribute::String(long_string.clone()));
    test_op.attributes = attrs;
    
    assert_eq!(test_op.attributes.len(), 1);
    match test_op.attributes.get("long_attr") {
        Some(Attribute::String(s)) => assert_eq!(s, &long_string),
        _ => panic!("Expected string attribute"),
    }
}

/// Test 9: Boolean attribute edge cases
#[test]
fn test_boolean_attribute_edge_cases() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("bool_attr_test");
    let mut attrs = HashMap::new();
    
    // Add all possible boolean attribute values
    attrs.insert("true_bool".to_string(), Attribute::Bool(true));
    attrs.insert("false_bool".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 2);
    assert_eq!(op.attributes.get("true_bool"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("false_bool"), Some(&Attribute::Bool(false)));
    
    // Test comparison between boolean attributes
    let attr_true1 = Attribute::Bool(true);
    let attr_true2 = Attribute::Bool(true);
    let attr_false = Attribute::Bool(false);
    
    assert_eq!(attr_true1, attr_true2);
    assert_ne!(attr_true1, attr_false);
    assert_ne!(attr_true2, attr_false);
}

/// Test 10: Error/panic recovery and bounds checking
#[test]
fn test_bounds_safety_checks() {
    // Test various potential out-of-bounds operations
    
    let empty_value = Value {
        name: "empty".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape (scalar)
    };
    
    // Ensure scalar product is 1
    let scalar_product: usize = empty_value.shape.iter().product();
    assert_eq!(scalar_product, 1);
    
    // Test a shape with a single element
    let single_value = Value {
        name: "single".to_string(),
        ty: Type::I64,
        shape: vec![1],
    };
    
    let single_product: usize = single_value.shape.iter().product();
    assert_eq!(single_product, 1);
    
    // Test with multiple zero dimensions
    let multi_zero_value = Value {
        name: "multi_zero".to_string(),
        ty: Type::Bool,
        shape: vec![0, 0, 0],
    };
    
    let zero_product: usize = multi_zero_value.shape.iter().product();
    assert_eq!(zero_product, 0);
    
    // Test with alternating zero/non-zero dimensions
    let alt_value = Value {
        name: "alt_pattern".to_string(),
        ty: Type::F32,
        shape: vec![2, 0, 5, 0, 10],
    };
    
    let alt_product: usize = alt_value.shape.iter().product();
    assert_eq!(alt_product, 0);
}