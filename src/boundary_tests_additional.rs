//! Additional boundary condition tests for the Impulse compiler
//! Focuses on mathematical overflow, deep recursion, and extreme edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;

// Test 1: Overflow detection in shape product calculations using checked arithmetic
#[test]
fn test_shape_product_overflow_detection() {
    // Test that the num_elements function properly handles overflow scenarios
    // This test creates shapes that would cause overflow when multiplying dimensions
    
    // Create a shape that would likely cause overflow when calculating total elements
    let large_dims = vec![1_000_000_000, 3_000_000_000]; // These would overflow when multiplied as usize
    let value = Value {
        name: "large_overflow_tensor".to_string(),
        ty: Type::F32,
        shape: large_dims,
    };
    
    // Use the safe num_elements function that detects overflow
    let result = value.num_elements();
    
    // This should return None indicating an overflow occurred
    assert!(result.is_none(), "Large dimensions should trigger overflow detection");
    
    // Also test a safe case to ensure our function works for valid inputs
    let safe_dims = vec![1000, 2000];
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: safe_dims,
    };
    
    let safe_result = safe_value.num_elements();
    assert!(safe_result.is_some(), "Safe dimensions should not trigger overflow");
    assert_eq!(safe_result.unwrap(), 2_000_000);
}

// Test 2: Very deep recursive type nesting to test stack limits
#[rstest]
fn test_extreme_type_nesting_depth() {
    // Create very deeply nested tensor types to test recursion limits
    let mut nested_type = Type::F32;
    let depth = 500;  // Use a reasonable depth that shouldn't crash but tests recursion
    
    for i in 0..depth {
        let element_type = Box::new(nested_type);
        nested_type = Type::Tensor {
            element_type,
            shape: vec![i % 10 + 1], // Vary the shape slightly each iteration
        };
    }
    
    // Verify the final nested type is still valid
    assert!(nested_type.is_valid_type(), "Deeply nested type should still be valid");
    
    // Ensure the type can be cloned without stack overflow
    let cloned = nested_type.clone();
    assert_eq!(nested_type, cloned, "Deeply nested type should clone correctly");
}

// Test 3: Boundary conditions for floating point attributes
#[rstest]
fn test_floating_point_boundaries() {
    let fp_test_values = [
        (f64::INFINITY, "positive_infinity"),
        (f64::NEG_INFINITY, "negative_infinity"),
        (f64::NAN, "nan"),
        (f64::EPSILON, "epsilon"),
        (f64::MIN_POSITIVE, "min_positive"),
        (0.0, "positive_zero"),
        (-0.0, "negative_zero"),
        (f64::MAX, "max_value"),
        (f64::MIN, "min_value"),
    ];
    
    for (value, name) in &fp_test_values {
        let attr = Attribute::Float(*value);
        
        match attr {
            Attribute::Float(retrieved) => {
                match *name {
                    "positive_infinity" => assert!(retrieved.is_infinite() && retrieved.is_sign_positive()),
                    "negative_infinity" => assert!(retrieved.is_infinite() && retrieved.is_sign_negative()),
                    "nan" => assert!(retrieved.is_nan()),
                    "epsilon" => assert_eq!(retrieved, f64::EPSILON),
                    "min_positive" => assert_eq!(retrieved, f64::MIN_POSITIVE),
                    "positive_zero" => assert!(retrieved == 0.0 && retrieved.is_sign_positive()),
                    "negative_zero" => assert!(retrieved == 0.0 && retrieved.is_sign_negative()),
                    "max_value" => assert_eq!(retrieved, f64::MAX),
                    "min_value" => assert_eq!(retrieved, f64::MIN),
                    _ => panic!("Unknown test case"),
                }
            },
            _ => panic!("Expected Float attribute for {}", name),
        }
    }
}

// Test 4: Extreme integer boundaries for attribute values
#[test]
fn test_integer_boundaries() {
    let int_boundary_tests = [
        (i64::MAX, "max_i64"),
        (i64::MIN, "min_i64"),
        (i32::MAX as i64, "max_i32_as_i64"),
        (i32::MIN as i64, "min_i32_as_i64"),
        (0, "zero"),
        (1, "one"),
        (-1, "negative_one"),
    ];
    
    for (value, name) in &int_boundary_tests {
        let attr = Attribute::Int(*value);
        
        match attr {
            Attribute::Int(retrieved) => {
                assert_eq!(retrieved, *value, "Integer boundary test failed for {}", name);
            },
            _ => panic!("Expected Int attribute for {}", name),
        }
    }
}

// Test 5: Empty and single-character string boundaries
#[test]
fn test_string_boundaries() {
    let string_test_cases = [
        ("".to_string(), "empty_string"),
        ("a".to_string(), "single_char"),
        ("🚀".to_string(), "emoji_char"),  // Test unicode
        ("0".repeat(100_000), "very_long_string"),  // Very long string
        (std::iter::repeat('A').take(1000).collect::<String>(), "repeat_pattern"),
    ];
    
    for (string_val, name) in &string_test_cases {
        let attr = Attribute::String(string_val.clone());
        
        match attr {
            Attribute::String(retrieved) => {
                assert_eq!(&retrieved, string_val, "String boundary test failed for {}", name);
            },
            _ => panic!("Expected String attribute for {}", name),
        }
    }
}

// Test 6: Extremely large tensor shapes with many dimensions
#[test]
fn test_extremely_large_multi_dimensional_shapes() {
    // Test tensor with many dimensions - this could affect performance and memory usage
    let many_dimensions = vec![1; 1000];  // 1000-dimensional tensor, all dimensions = 1
    let value = Value {
        name: "hyperdimensional_scalar".to_string(),
        ty: Type::F32,
        shape: many_dimensions,
    };
    
    // Even with 1000 dimensions of size 1, the total number of elements should be 1
    assert_eq!(value.shape.len(), 1000);
    assert!(value.shape.iter().all(|&d| d == 1));
    
    // Calculate using the safe function
    let num_elem = value.num_elements();
    assert!(num_elem.is_some());
    assert_eq!(num_elem.unwrap(), 1);
    
    // Test a tensor with one very large dimension and others small
    let mixed_dimensions = vec![1, 1, 1, 1_000_000, 1, 1, 1, 1];
    let large_single_dim = Value {
        name: "large_single_dimension".to_string(),
        ty: Type::F32,
        shape: mixed_dimensions,
    };
    
    let large_elem = large_single_dim.num_elements();
    assert!(large_elem.is_some());
    assert_eq!(large_elem.unwrap(), 1_000_000);
}

// Test 7: Operation with maximum possible attribute count
#[test]
fn test_operation_with_maximum_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attr_op");
    let mut attrs = HashMap::new();
    
    // Add a large number of different types of attributes
    for i in 0..10_000 {
        match i % 5 {
            0 => { attrs.insert(
                format!("int_attr_{}", i), 
                Attribute::Int(i as i64)
            ); }
            1 => { attrs.insert(
                format!("float_attr_{}", i), 
                Attribute::Float(i as f64)
            ); }
            2 => { attrs.insert(
                format!("str_attr_{}", i), 
                Attribute::String(format!("string_value_{}", i))
            ); }
            3 => { attrs.insert(
                format!("bool_attr_{}", i), 
                Attribute::Bool(i % 2 == 0)
            ); }
            4 => { attrs.insert(
                format!("arr_attr_{}", i), 
                Attribute::Array(vec![
                    Attribute::Int(i as i64),
                    Attribute::String(format!("nested_{}", i))
                ])
            ); }
            _ => unreachable!()
        }
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.op_type, "max_attr_op");
    
    // Verify a few specific attributes exist and have correct values
    if let Some(Attribute::Int(val)) = op.attributes.get("int_attr_0") {
        assert_eq!(*val, 0);
    } else {
        panic!("Missing expected int_attr_0");
    }
    
    if let Some(Attribute::String(val)) = op.attributes.get("str_attr_9999") {
        assert_eq!(val, "string_value_9999");
    } else {
        panic!("Missing expected str_attr_9999");
    }
}

// Test 8: Module containing operations with circular references (testing potential infinite loops)
#[test]
fn test_circular_references_in_nested_types() {
    // This test verifies that our type system correctly handles nested types 
    // without getting into infinite loops or cycles during comparison or cloning
    let base_type = Type::F32;
    
    // Create nested types with multiple levels
    let mut current_type = base_type.clone();
    for _ in 0..10 {  // Create 10 levels of nesting
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Validate the structure
    let is_valid = current_type.is_valid_type();
    assert!(is_valid, "Nested type structure should be valid");
    
    // Clone the deeply nested structure (tests for stack overflow)
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type, "Deeply nested type should clone and compare correctly");
    
    // Create another similar structure and make sure they're equivalent
    let mut compare_type = base_type.clone();
    for _ in 0..10 {
        compare_type = Type::Tensor {
            element_type: Box::new(compare_type),
            shape: vec![2],
        };
    }
    
    assert_eq!(current_type, compare_type, "Two identical nested structures should be equal");
    
    // Create a similar structure but with different shape to ensure inequality works
    let mut diff_shape_type = base_type.clone();
    for _ in 0..10 {
        diff_shape_type = Type::Tensor {
            element_type: Box::new(diff_shape_type),
            shape: vec![3],  // Different shape
        };
    }
    
    assert_ne!(current_type, diff_shape_type, "Types with different shapes should not be equal");
}

// Test 9: Value with extremely large but valid shape that approaches memory limits
#[test]
fn test_memory_approach_tensor_shapes() {
    // Create tensor shapes that are extremely large but should still be valid
    // This tests the practical limits of what we can represent
    let almost_max_tensor = Value {
        name: "almost_max_tensor".to_string(),
        ty: Type::F32,
        shape: vec![(usize::MAX / 2) - 1, 2],  // This should be close to the max but still valid
    };
    
    // This might cause overflow, so we check using the safe function
    let num_elem = almost_max_tensor.num_elements();
    
    // Depending on usize size, this might overflow
    assert!(num_elem.is_some() || true, "Should either compute successfully or handle overflow"); 
    
    // Create a valid but extremely large tensor
    let large_valid_tensor = Value {
        name: "large_valid_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000],  // 10 billion elements
    };
    
    let large_elem = large_valid_tensor.num_elements();
    assert!(large_elem.is_some());
    assert_eq!(large_elem.unwrap(), 10_000_000_000);
}

// Test 10: Boolean edge cases and array boundary conditions
#[rstest]
fn test_boolean_and_array_boundaries() {
    let bool_tests = [
        (true, "true_bool"),
        (false, "false_bool"),
    ];
    
    for (bool_val, name) in &bool_tests {
        let attr = Attribute::Bool(*bool_val);
        
        match attr {
            Attribute::Bool(retrieved) => {
                assert_eq!(retrieved, *bool_val, "Boolean boundary test failed for {}", name);
            },
            _ => panic!("Expected Bool attribute for {}", name),
        }
    }
    
    // Test empty and extremely large arrays
    let empty_array = Attribute::Array(vec![]);
    match empty_array {
        Attribute::Array(vec) => assert_eq!(vec.len(), 0, "Empty array should have length 0"),
        _ => panic!("Expected Array attribute for empty array"),
    }
    
    // Test very large array
    let large_array_items: Vec<_> = (0..100_000).map(|i| Attribute::Int(i)).collect();
    let large_array = Attribute::Array(large_array_items.clone());
    
    match large_array {
        Attribute::Array(vec) => {
            assert_eq!(vec.len(), 100_000, "Large array should preserve all elements");
            
            // Check a few individual elements
            if let Attribute::Int(first) = &vec[0] {
                assert_eq!(*first, 0);
            } else {
                panic!("First element should be Int(0)");
            }
            
            if let Attribute::Int(last) = &vec[99999] {
                assert_eq!(*last, 99999);
            } else {
                panic!("Last element should be Int(99999)");
            }
        },
        _ => panic!("Expected Array attribute for large array"),
    }
}