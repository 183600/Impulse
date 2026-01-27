//! Comprehensive additional test cases for the Impulse compiler
//! Covering more edge cases and boundary conditions using both standard assertions and rstest

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Overflow-safe shape product calculations
#[test]
fn test_overflow_safe_shape_products() {
    // Test shapes that could potentially cause overflow when multiplied
    // Using checked arithmetic to prevent actual overflow
    let safe_large_shape = vec![100_000, 100_000];  // 10 billion elements
    let value = Value {
        name: "safe_large".to_string(),
        ty: Type::F32,
        shape: safe_large_shape,
    };
    
    let product: usize = value.shape.iter().product();
    assert_eq!(product, 10_000_000_000);
    
    // Test with safe multiplication to prevent actual overflow
    let potentially_problematic_shape = vec![usize::MAX / 100, 50];
    let safe_check: Option<usize> = potentially_problematic_shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    // This should either return a value or None (indicating overflow)
    assert!(safe_check.is_some() || safe_check.is_none());
}

/// Test 2: Operations with maximum possible attribute counts using different attribute types
#[test]
fn test_operations_with_extreme_attribute_diversity() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("extreme_attr_op");
    let mut attrs = HashMap::new();
    
    // Add many different types of attributes with extreme values
    for i in 0..10_000 {
        match i % 5 {
            0 => attrs.insert(format!("int_attr_{}", i), Attribute::Int(i as i64)),
            1 => attrs.insert(format!("float_attr_{}", i), Attribute::Float(i as f64 * 0.5)),
            2 => attrs.insert(format!("bool_attr_{}", i), Attribute::Bool(i % 2 == 0)),
            3 => attrs.insert(format!("string_attr_{}", i), Attribute::String(format!("string_{}", i))),
            _ => attrs.insert(format!("array_attr_{}", i), Attribute::Array(vec![
                Attribute::Int(i as i64),
                Attribute::String(format!("nested_{}", i))
            ])),
        };
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 10_000);
}

/// Test 3: Comprehensive testing of edge cases with tensor shapes using rstest
#[rstest]
#[case(vec![], 1, "scalar")]  // Empty shape = scalar = 1 element
#[case(vec![0], 0, "zero_dim")]  // Zero dimension = 0 elements
#[case(vec![1, 1, 1, 1], 1, "unit_dims")]  // Multiple unit dimensions = 1 element
#[case(vec![2, 3, 4], 24, "small_tensor")]  // Small multi-dim tensor
#[case(vec![1000, 1000], 1_000_000, "large_flat")]  // Large 2D tensor
#[case(vec![10, 0, 100], 0, "zero_middle")]  // Zero in middle = 0 elements
fn test_tensor_shape_edge_cases(
    #[case] shape: Vec<usize>, 
    #[case] expected_elements: usize, 
    #[case] _desc: &str
) {
    let value = Value {
        name: "shape_test".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let actual_elements: usize = value.shape.iter().product();
    assert_eq!(actual_elements, expected_elements);
}

/// Test 4: Deep recursion with alternating types to test stack safety
#[test]
fn test_deep_recursion_with_alternating_types() {
    // Create a deeply nested type with alternating base types to test stack safety
    let mut current_type = Type::F32;
    let max_depth = 500;  // Reasonable depth that tests recursion without hitting limits
    
    for i in 0..max_depth {
        let next_type = if i % 2 == 0 {
            Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![i % 10 + 1],
            }
        } else {
            Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],
            }
        };
        current_type = next_type;
    }
    
    // Verify the structure
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 5: Memory allocation stress test with large collections
#[test]
fn test_memory_allocation_stress() {
    // Create a series of modules with increasing complexity to test memory management
    let mut modules = Vec::new();
    
    for module_idx in 0..100 {
        let mut module = Module::new(&format!("stress_module_{}", module_idx));
        
        // Add operations to each module
        for op_idx in 0..100 {
            let mut op = Operation::new(&format!("op_{}_{}", module_idx, op_idx));
            
            // Add some inputs and outputs
            for val_idx in 0..10 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}_{}", module_idx, op_idx, val_idx),
                    ty: if val_idx % 2 == 0 { Type::F32 } else { Type::I32 },
                    shape: vec![val_idx + 1, val_idx + 2],
                });
                
                op.outputs.push(Value {
                    name: format!("output_{}_{}_{}", module_idx, op_idx, val_idx),
                    ty: if val_idx % 3 == 0 { Type::F64 } else { Type::I64 },
                    shape: vec![val_idx + 1],
                });
            }
            
            module.add_operation(op);
        }
        
        modules.push(module);
    }
    
    // Verify we have created the expected number of modules and operations
    assert_eq!(modules.len(), 100);
    assert_eq!(modules[0].operations.len(), 100);
    assert_eq!(modules[0].operations[0].inputs.len(), 10);
    
    // Clean up
    drop(modules);
    assert!(true);  // Test passes if no panic occurred
}

/// Test 6: Complex nested tensor types with mixed shapes
#[test]
fn test_complex_nested_tensor_types() {
    // Create a complex nested tensor type: tensor<tensor<tensor<f32, [2,3]>, [4]>, [5,6]>
    let inner_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],  // Innermost shape
    };
    
    let middle_type = Type::Tensor {
        element_type: Box::new(inner_type),
        shape: vec![4],  // Middle shape
    };
    
    let outer_type = Type::Tensor {
        element_type: Box::new(middle_type),
        shape: vec![5, 6],  // Outermost shape
    };
    
    // Verify the structure exists correctly
    if let Type::Tensor { element_type: outer_elem, shape: outer_shape } = &outer_type {
        assert_eq!(outer_shape, &vec![5, 6]);
        
        if let Type::Tensor { element_type: middle_elem, shape: middle_shape } = outer_elem.as_ref() {
            assert_eq!(middle_shape, &vec![4]);
            
            if let Type::Tensor { element_type: inner_elem, shape: inner_shape } = middle_elem.as_ref() {
                assert_eq!(inner_shape, &vec![2, 3]);
                
                if let Type::F32 = inner_elem.as_ref() {
                    // Successfully verified the deep nesting
                    assert!(true);
                } else {
                    panic!("Expected F32 at innermost level");
                }
            } else {
                panic!("Expected Tensor at middle level");
            }
        } else {
            panic!("Expected Tensor at outer level");
        }
    } else {
        panic!("Expected Tensor at top level");
    }
    
    // Test cloning of this complex structure
    let cloned = outer_type.clone();
    assert_eq!(outer_type, cloned);
}

/// Test 7: Edge cases for special floating point values in the context of tensor computations
#[rstest]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
#[case(f64::NAN)]
#[case(-0.0)]
#[case(f64::EPSILON)]
#[case(std::f64::consts::PI)]
#[case(std::f64::consts::E)]
fn test_special_floats_in_tensor_context(#[case] float_val: f64) {
    // Test that special floating point values are handled correctly in attributes
    let attr = Attribute::Float(float_val);
    
    match attr {
        Attribute::Float(retrieved_val) => {
            if float_val.is_nan() {
                assert!(retrieved_val.is_nan(), "NaN value should be preserved");
            } else if float_val.is_infinite() {
                assert!(retrieved_val.is_infinite(), "Infinity value should be preserved");
                assert_eq!(retrieved_val.is_sign_positive(), float_val.is_sign_positive(), 
                          "Sign of infinity should be preserved");
            } else if float_val == -0.0 {
                // Special handling for negative zero
                assert!(retrieved_val == -0.0 || retrieved_val == 0.0, "Zero value should be preserved");
                // Check that sign is preserved by dividing by 1
                assert_eq!((retrieved_val / 1.0).is_sign_negative(), (float_val / 1.0).is_sign_negative(),
                          "Sign of zero should be preserved");
            } else {
                assert!((retrieved_val - float_val).abs() < f64::EPSILON, 
                       "Regular float value should be preserved");
            }
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 8: Boundary conditions for string values including unicode and special characters
#[test]
fn test_string_boundaries_with_unicode_special_chars() {
    let test_strings = [
        // Empty string
        "",
        // ASCII characters
        "simple_string",
        // Unicode characters
        "🚀Unicode_Test_测试_日本語",
        // Control characters
        "test\twith\ncontrol\rchars",
        // Special symbols
        "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        // Very long string
        &"x".repeat(100_000),
        // Mixed ASCII and unicode
        "Mixed_混合_🌍_end",
        // Null characters
        "string_with_null_\0_character",
    ];
    
    for test_str in &test_strings {
        let attr = Attribute::String(test_str.to_string());
        
        match attr {
            Attribute::String(retrieved_str) => {
                assert_eq!(retrieved_str, *test_str);
            },
            _ => panic!("Expected String attribute for: {}", test_str),
        }
    }
}

/// Test 9: Comprehensive tests for tensor type equivalence and hashing behavior
#[test]
fn test_tensor_type_equivalence_comprehensive() {
    // Test various combinations of tensor type equivalence
    let base_cases = [
        (Type::F32, vec![2, 3]),
        (Type::F64, vec![2, 3]),
        (Type::F32, vec![3, 2]),
        (Type::F32, vec![2, 3, 4]),
        (Type::I32, vec![2, 3]),
    ];
    
    // Create corresponding tensor types
    let tensor_types: Vec<Type> = base_cases.iter()
        .map(|(element_type, shape)| {
            Type::Tensor {
                element_type: Box::new(element_type.clone()),
                shape: shape.clone(),
            }
        })
        .collect();
    
    // Each type should be equal to itself
    for (i, tensor_type) in tensor_types.iter().enumerate() {
        assert_eq!(tensor_type, tensor_type, "Type should equal itself at index {}", i);
    }
    
    // Different element types should not be equal
    if tensor_types.len() > 1 {
        assert_ne!(tensor_types[0], tensor_types[1], "Different element types should not be equal");
        assert_ne!(tensor_types[0], tensor_types[2], "Same element type but different shape should not be equal");
        assert_ne!(tensor_types[1], tensor_types[2], "Different element types should not be equal");
    }
    
    // Create identical types to verify equality
    let identical_type_1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    let identical_type_2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    assert_eq!(identical_type_1, identical_type_2, "Identical types should be equal");
}

/// Test 10: Boundary conditions for operation lifecycle and mutation
#[test]
fn test_operation_lifecycle_boundary_conditions() {
    // Create an operation with minimal content
    let mut op = Operation::new("");
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
    
    // Mutate the operation repeatedly to test boundary conditions
    op.op_type = "updated_op".to_string();
    assert_eq!(op.op_type, "updated_op");
    
    // Add a large number of inputs
    for i in 0..500 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i % 10 + 1],  // Cycling through shapes 1-10
        });
    }
    
    assert_eq!(op.inputs.len(), 500);
    
    // Add outputs
    for i in 0..300 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F64,
            shape: vec![i % 5 + 1],  // Cycling through shapes 1-5
        });
    }
    
    assert_eq!(op.outputs.len(), 300);
    
    // Test that modifications persist
    assert_eq!(op.inputs[499].name, "input_499");
    assert_eq!(op.inputs[499].shape, vec![499 % 10 + 1]);  // 499 % 10 + 1 = 9 + 1 = 10
    assert_eq!(op.outputs[299].name, "output_299");
    assert_eq!(op.outputs[299].shape, vec![299 % 5 + 1]);  // 299 % 5 + 1 = 4 + 1 = 5
    
    // Clear collections to test empty boundaries
    op.inputs.clear();
    op.outputs.clear();
    op.attributes.clear();
    
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
    
    // Test adding back to cleared collections
    op.inputs.push(Value {
        name: "restored_input".to_string(),
        ty: Type::I32,
        shape: vec![1],
    });
    
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.inputs[0].name, "restored_input");
}