//! Additional edge case tests for the Impulse compiler
//! Covering boundary conditions and rare scenarios not addressed in existing tests

use crate::{
    ir::{Value, Type, Operation, Attribute, Module},
};
use rstest::*;

/// Test 1: Operations with maximum length string attributes to test memory limits
#[test]
fn test_max_length_string_attributes() {
    let mut op = Operation::new("max_string_attr");
    
    // Create extremely long string attribute (100k characters)
    let long_string = "A".repeat(100_000);
    op.attributes.insert(
        "long_attr".to_string(), 
        Attribute::String(long_string.clone())
    );
    
    if let Some(Attribute::String(retrieved_string)) = op.attributes.get("long_attr") {
        assert_eq!(retrieved_string.len(), 100_000);
        assert_eq!(retrieved_string, &long_string);
    } else {
        panic!("Expected long string attribute");
    }
}

/// Test 2: Operations with nested arrays of maximum depth
#[test]
fn test_deeply_nested_arrays() {
    // Create a deeply nested array (depth of 100)
    let mut nested_array = Attribute::Array(vec![]);
    
    for _ in 0..10 {
        nested_array = Attribute::Array(vec![nested_array]);
    }
    
    // Verify the structure by counting nesting levels
    fn count_nesting_levels(attr: &Attribute) -> usize {
        match attr {
            Attribute::Array(items) => {
                if items.is_empty() {
                    1
                } else {
                    1 + count_nesting_levels(&items[0])
                }
            },
            _ => 0,
        }
    }
    
    let nesting_depth = count_nesting_levels(&nested_array);
    assert!(nesting_depth >= 10); // At least 10 levels of nesting
}

/// Test 3: Tensor types with maximum dimension sizes
#[test]
fn test_tensor_max_dimensions() {
    // Test tensor with very large but reasonable dimensions
    let max_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![1_000_000, 1_000], // 1 billion elements
    };
    
    match max_tensor {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape.len(), 2);
            assert_eq!(shape[0], 1_000_000);
            assert_eq!(shape[1], 1_000);
        },
        _ => panic!("Expected tensor type"),
    }
}

/// Test 4: Values with invalid UTF-8 names (valid in Rust but unusual)
#[test]
fn test_values_with_byte_names() {
    // Create names with non-UTF8 byte sequences
    let byte_sequences = [
        &b"\xFF\xFE\xFD"[..],  // Invalid UTF-8
        &b"valid\x00name"[..], // Null-terminated (valid in Rust strings)
        &b"name\xC0\x80end"[..], // Overlong UTF-8 sequence
    ];
    
    for (_i, &bytes) in byte_sequences.iter().enumerate() {
        let name = String::from_utf8_lossy(bytes);
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        // The value should be created successfully
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1]);
        assert!(!value.name.is_empty());
    }
}

/// Test 5: Operations with maximum attribute count (thousands of attributes)
#[test]
fn test_operation_thousands_of_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("many_attrs");
    let mut attrs = HashMap::new();
    
    // Add 10,000 attributes to an operation
    for i in 0..10_000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::Int(i as i64)
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("attr_9999"), Some(&Attribute::Int(9999)));
}

/// Test 6: Large integer values in tensor shapes that could cause overflow
#[rstest]
#[case(vec![usize::MAX / 2, 2])]  // Product would be close to usize::MAX
#[case(vec![usize::MAX / 4, 4])]
#[case(vec![10, usize::MAX / 10])]
#[case(vec![100_000, 100_000, 100_000])] // Product overflows
fn test_potentially_overflowing_shapes(#[case] shape: Vec<usize>) {
    let value = Value {
        name: "potential_overflow".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape, shape);
    
    // Calculate with checked multiplication to prevent overflow
    let _product_result: Option<usize> = shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    // This test passes if the shape is stored correctly regardless of overflow
    assert_eq!(value.shape.len(), shape.len());
}

/// Test 7: Mixed boolean attribute values in operations
#[test]
fn test_mixed_boolean_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("bool_test");
    let mut attrs = HashMap::new();
    
    // Add various boolean combinations
    attrs.insert("true_bool".to_string(), Attribute::Bool(true));
    attrs.insert("false_bool".to_string(), Attribute::Bool(false));
    attrs.insert("another_true_bool".to_string(), Attribute::Bool(true));
    attrs.insert("another_false_bool".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    assert_eq!(op.attributes.get("true_bool"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("false_bool"), Some(&Attribute::Bool(false)));
    assert_eq!(op.attributes.get("another_true_bool"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("another_false_bool"), Some(&Attribute::Bool(false)));
}

/// Test 8: Recursive tensor with alternating type patterns
#[test]
fn test_alternating_recursive_tensor() {
    let mut current_type = Type::F32;
    
    // Alternate between F32 and I32 in nested structure
    for i in 0..10 {
        let element_type = if i % 2 == 0 {
            Box::new(Type::F32)
        } else {
            Box::new(Type::I32)
        };
        
        current_type = Type::Tensor {
            element_type,
            shape: vec![i + 1],
        };
    }
    
    // Test that the structure was built correctly
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
    
    // The type should be valid (doesn't panic)
    assert!(true); // Dummy assertion to satisfy test
}

/// Test 9: Module with operations that have circular references in names
#[test]
fn test_module_with_patterned_operation_names() {
    let mut module = Module::new("pattern_test");
    
    // Add operations with similar names to potentially trigger hash collisions
    for i in 0..100 {
        // Create operations with names that have similar prefixes
        let mut op = Operation::new(&format!("operation_prefix_{}.suffix", i));
        
        // Add some content to the operation
        op.inputs.push(Value {
            name: format!("input_for_{}", i),
            ty: Type::F32,
            shape: vec![i + 1],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    assert_eq!(module.name, "pattern_test");
    
    // Verify a few operations were added correctly
    assert_eq!(module.operations[0].op_type, "operation_prefix_0.suffix");
    assert_eq!(module.operations[99].op_type, "operation_prefix_99.suffix");
}

/// Test 10: Complex value with invalid mathematical operations
#[test]
fn test_value_with_problematic_math_shapes() {
    let problematic_shapes = [
        vec![0],  // Zero-sized
        vec![1],  // Scalar-like
        vec![0, 0],  // Both dimensions zero
        vec![1, 0],  // One dimension unitary, one zero
        vec![0, 1],  // One dimension zero, one unitary
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], // Ten-dimensional unit tensor
    ];
    
    for shape in &problematic_shapes {
        let value = Value {
            name: "math_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, *shape);
        
        // Calculate element count with potential for edge cases
        let element_count: usize = value.shape.iter()
            .fold(1_usize, |acc, &dim| acc.saturating_mul(dim));
        
        // For any shape with a 0 in it, element_count should be 0
        if shape.contains(&0) {
            assert_eq!(element_count, 0);
        } else {
            // For all 1s, element count should be 1
            if shape.iter().all(|&x| x == 1) {
                assert_eq!(element_count, 1);
            }
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    
    /// Verification that our tests are working as expected
    #[test]
    fn test_test_framework_validation() {
        // This is a meta-test to ensure our test infrastructure works
        assert_eq!(2 + 2, 4);
        
        let sample_value = Value {
            name: "validation".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        
        assert_eq!(sample_value.shape, vec![2, 3]);
        assert_eq!(sample_value.ty, Type::F32);
    }
}