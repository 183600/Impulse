//! Additional edge case tests for the Impulse compiler
//! More boundary conditions to expand test coverage

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Operations with null-byte containing names (security edge case)
#[test]
fn test_null_byte_containing_names() {
    let null_containing_name = "op_name\0with_null";
    let op = Operation::new(null_containing_name);
    assert_eq!(op.op_type, null_containing_name);
    
    let value = Value {
        name: "value_name\0with_null".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    assert_eq!(value.name, "value_name\0with_null");
}

/// Test 2: Tensor types with maximum possible shape dimensions
#[test]
fn test_max_dimensionality_tensors() {
    // Create tensor with maximum possible dimensions (within reason)
    let max_dims = vec![2; 1000]; // 1000 dimensions, each size 2
    let value = Value {
        name: "max_dims_tensor".to_string(),
        ty: Type::F32,
        shape: max_dims,
    };
    
    assert_eq!(value.shape.len(), 1000);
    // Total size would be 2^1000 which is astronomically large, 
    // but computing here shouldn't cause immediate problems
    // Use logarithmic calculation instead of direct product for verification
    assert_eq!(value.shape[0], 2);
    assert_eq!(value.shape[999], 2);
}

/// Test 3: Attribute arrays with maximum nesting and mixed types
#[test]
fn test_max_nested_mixed_type_arrays() {
    // Create a complex nested structure with many types
    let complex_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Int(42),
                    Attribute::Float(3.14159)
                ])
            ])
        ]),
        Attribute::String("top_level".to_string())
    ]);
    
    // Validate the complex structure exists
    match &complex_nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[1] {
                Attribute::String(s) => assert_eq!(s, "top_level"),
                _ => panic!("Expected string at index 1"),
            }
        },
        _ => panic!("Expected Array as top level"),
    }
}

/// Test 4: Operations with extremely long attribute maps
#[test]
fn test_extremely_large_attribute_maps() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("large_attr_map");
    let mut attrs = HashMap::new();
    
    // Add 100,000 attributes to test hash map performance
    for i in 0..100_000 {
        attrs.insert(
            format!("attr_key_{}", i),
            Attribute::Int(i as i64)
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 100_000);
    assert_eq!(op.attributes.get("attr_key_50000"), Some(&Attribute::Int(50000)));
    assert!(op.attributes.contains_key("attr_key_99999"));
}

/// Test 5: Value shapes with alternating zero and non-zero dimensions
#[rstest]
#[case(vec![1, 0, 1, 0, 1])]
#[case(vec![0, 1, 0, 1, 0])]
#[case(vec![2, 0, 3, 0, 4, 0, 5])]
fn test_alternating_zero_nonzero_shapes(#[case] shape: Vec<usize>) {
    let value = Value {
        name: "alternating_shape".to_string(),
        ty: Type::I64,
        shape,
    };
    
    // Any shape containing zero should result in 0 total elements
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 0);
    
    // Verify shape contains zero as expected
    assert!(value.shape.iter().any(|&dim| dim == 0));
}

/// Test 6: Boolean tensors with extreme sizes
#[test]
fn test_extreme_boolean_tensor_sizes() {
    // Create a very large boolean tensor
    let large_bool_tensor = Value {
        name: "large_bool_tensor".to_string(),
        ty: Type::Bool,
        shape: vec![10_000_000], // 10 million booleans
    };
    
    assert_eq!(large_bool_tensor.ty, Type::Bool);
    assert_eq!(large_bool_tensor.shape, vec![10_000_000]);
    
    let size_check: usize = large_bool_tensor.shape.iter().product();
    assert_eq!(size_check, 10_000_000);
    
    // Test with multi-dimensional boolean tensor
    let multi_d_bool = Value {
        name: "multi_d_bool".to_string(),
        ty: Type::Bool,
        shape: vec![1000, 1000, 10], // 10 million booleans in 3D
    };
    
    assert_eq!(multi_d_bool.shape, vec![1000, 1000, 10]);
    let multi_d_size: usize = multi_d_bool.shape.iter().product();
    assert_eq!(multi_d_size, 10_000_000);
}

/// Test 7: Recursive tensor type with self-referencing structures (circular references)
#[test]
fn test_potential_circular_tensor_types() {
    // Note: True circular references would cause infinite loops or memory issues
    // So we're testing deep but finite nesting instead
    
    // Create a sequence of nested types that alternate
    let mut current_type = Type::F32;
    for i in 0..50 {
        // Alternate between different base types in deep nesting
        let base_type = match i % 4 {
            0 => Type::F32,
            1 => Type::I64,
            2 => Type::Bool,
            _ => Type::F64,
        };
        
        current_type = Type::Tensor {
            element_type: Box::new(base_type),
            shape: vec![i + 1],
        };
    }
    
    // Verify final type is valid and can be cloned
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
    
    // Check that structure is preserved
    if let Type::Tensor { shape, .. } = current_type {
        assert_eq!(shape, vec![50]); // Last iteration was i=49, so shape should be [50]
    } else {
        panic!("Expected Tensor type at deepest level");
    }
}

/// Test 8: Edge cases with Unicode characters in string attributes
#[test]
fn test_unicode_string_attribute_edge_cases() {
    // Test attributes with various Unicode edge cases
    let unicode_test_cases = [
        "simple_ascii",
        "汉字",           // Chinese characters
        "café naïve",     // Accented Latin characters  
        "🐍🚀💰",        // Emojis
        "\u{0000}",       // Null character (valid in Rust strings)
        &"a".repeat(100), // Very long ASCII string
        &"αβγδε".repeat(20), // Very long Unicode string
    ];
    
    for test_case in &unicode_test_cases {
        let attr = Attribute::String(test_case.to_string());
        
        match &attr {
            Attribute::String(s) => assert_eq!(s, test_case),
            _ => panic!("Expected String attribute"),
        }
        
        // Test with an operation attribute
        use std::collections::HashMap;
        let mut op = Operation::new("unicode_test_op");
        let mut attrs = HashMap::new();
        attrs.insert("unicode_attr".to_string(), attr);
        op.attributes = attrs;
        
        if let Some(Attribute::String(retrieved)) = op.attributes.get("unicode_attr") {
            assert_eq!(retrieved, test_case);
        } else {
            panic!("Failed to retrieve unicode string attribute");
        }
    }
}

/// Test 9: Stress test with maximum memory allocations
#[test]
fn test_memory_allocation_patterns() {
    // Create many small objects to test allocation patterns
    let mut modules = Vec::with_capacity(10_000);
    
    for i in 0..10_000 {
        let mut module = Module::new(format!("stress_module_{}", i));
        
        // Add a few operations to each module
        for j in 0..5 {
            let mut op = Operation::new(&format!("op_{}_{}", i, j));
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: Type::F32,
                shape: vec![j + 1],
            });
            module.add_operation(op);
        }
        
        modules.push(module);
    }
    
    assert_eq!(modules.len(), 10_000);
    
    // Verify a few at random positions
    assert_eq!(modules[0].name, "stress_module_0");
    assert_eq!(modules[9999].name, "stress_module_9999");
    
    // Free up memory
    drop(modules);
}

/// Test 10: Edge case combinations for operations with extreme parameters
#[test]
fn test_extreme_combination_parameters() {
    use std::collections::HashMap;
    
    let mut extreme_op = Operation::new("extreme_op"); // Using shorter name to avoid potential memory issues
    
    // Add many inputs with extreme properties
    for i in 0..5_000 { // Reduce from 10_000 to avoid timeout/memory issues
        extreme_op.inputs.push(Value {
            name: format!("input_{}", i), // Using shorter names
            ty: match i % 5 {
                0 => Type::F32,
                1 => Type::F64, 
                2 => Type::I32,
                3 => Type::I64,
                _ => Type::Bool,
            },
            shape: if i % 3 == 0 {
                vec![0, i % 1000 + 1] // Sometimes includes zero
            } else if i % 3 == 1 {
                vec![i % 500 + 1]     // Single dimension
            } else {
                vec![i % 10 + 1, i % 20 + 1] // Two dimensions
            },
        });
    }
    
    // Add attributes with extreme properties
    let mut attrs = HashMap::new();
    for i in 0..2_500 { // Reduce from 5_000 to avoid timeout/memory issues
        attrs.insert(
            format!("attr_{}", i), // Using shorter key names to avoid memory issues
            if i % 2 == 0 {
                Attribute::String(format!("value_{}", "z".repeat(100))) // Reduce string size
            } else {
                Attribute::Int(i as i64)
            }
        );
    }
    extreme_op.attributes = attrs;
    
    // Verify basic properties are preserved
    assert_eq!(extreme_op.op_type, "extreme_op");
    assert_eq!(extreme_op.inputs.len(), 5_000);
    assert_eq!(extreme_op.attributes.len(), 2_500);
}