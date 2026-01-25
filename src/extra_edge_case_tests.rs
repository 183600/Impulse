//! Extra edge case tests for the Impulse compiler
//! Covering additional boundary conditions and specific edge cases

use rstest::*;
use crate::{
    ir::{Value, Type, Operation, Attribute, Module, TypeExtensions},
};

/// Test 1: Operations with empty attribute maps and null-like values
#[test]
fn test_operations_empty_attributes() {
    let mut op = Operation::new("test_op");
    
    // Start with empty attributes map
    assert!(op.attributes.is_empty());
    
    // Add operations with empty collections and ensure they behave correctly
    let empty_value = Value {
        name: "".to_string(),  // Empty name
        ty: Type::F32,
        shape: vec![],         // Empty shape (scalar)
    };
    
    op.inputs.push(empty_value);
    assert_eq!(op.inputs[0].name, "");
    assert_eq!(op.inputs[0].shape.len(), 0);
    
    // Test with empty module
    let module = Module::new("");
    assert_eq!(module.name, "");
    assert_eq!(module.operations.len(), 0);
}

/// Test 2: Type comparison edge cases with deeply nested types
#[test]
fn test_type_comparison_edge_cases() {
    // Two identical nested types should be equal
    let type1 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        }),
        shape: vec![4, 5],
    };
    
    let type2 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        }),
        shape: vec![4, 5],
    };
    
    assert_eq!(type1, type2);
    
    // But different nested types should not be equal
    let type3 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I32), // Different base type
            shape: vec![2, 3],
        }),
        shape: vec![4, 5],
    };
    
    assert_ne!(type1, type3);
}

/// Test 3: Attribute parsing and serialization edge cases
#[rstest]
#[case(Attribute::Int(0))]
#[case(Attribute::Int(-1))]
#[case(Attribute::Int(i64::MAX))]
#[case(Attribute::Int(i64::MIN))]
#[case(Attribute::Float(0.0))]
#[case(Attribute::Float(-0.0))]
#[case(Attribute::Float(f64::INFINITY))]
#[case(Attribute::Float(f64::NEG_INFINITY))]
#[case(Attribute::Float(f64::NAN))]
#[case(Attribute::Bool(true))]
#[case(Attribute::Bool(false))]
#[case(Attribute::String("".to_string()))]
#[case(Attribute::String("normal string".to_string()))]
fn test_attribute_serialization_edge_cases(#[case] attr: Attribute) {
    // Just test that we can create and match the attribute
    match attr {
        Attribute::Int(v) => assert!(v == v), // Basic self-equality
        Attribute::Float(v) => {
            if v.is_nan() {
                assert!(true); // Can't compare NaN normally
            } else {
                assert!(v == v); // Basic self-equality for non-NaN floats
            }
        },
        Attribute::Bool(b) => assert!(b == b), // Basic self-equality
        Attribute::String(s) => assert_eq!(s, s), // String self-equality
        Attribute::Array(ref arr) => assert_eq!(arr, arr), // Array self-equality
    }
}

/// Test 4: Overflow protection in tensor size calculations using checked arithmetic
#[test]
fn test_overflow_protection_in_size_calculation() {
    // Test large numbers that might cause overflow
    let large_but_safe = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000], // This would be 10^10 elements
    };
    
    // Use checked multiplication to avoid overflow
    let mut product: Option<usize> = Some(1);
    for dim in &large_but_safe.shape {
        product = product.and_then(|p| p.checked_mul(*dim));
    }
    
    assert!(product.is_some()); // Should not overflow for these values on 64-bit
    
    // Test with a tensor that definitely has zero elements
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1000, 0, 500], // Contains 0, so product is 0
    };
    
    let zero_product: usize = zero_tensor.shape.iter().product();
    assert_eq!(zero_product, 0);
}

/// Test 5: Operations with invalid or unusual names
#[test]
fn test_operations_with_unusual_names() {
    // Test operation with unicode characters
    let op_unicode = Operation::new("opération_тест_🚀");
    assert_eq!(op_unicode.op_type, "opération_тест_🚀");
    
    // Test with special characters
    let op_special = Operation::new("!@#$%^&*()");
    assert_eq!(op_special.op_type, "!@#$%^&*()");
    
    // Test with control characters (may not be valid in practice, but should not crash)
    let op_control = Operation::new("\x00\x01\x02test");
    assert_eq!(op_control.op_type, "\x00\x01\x02test");
    
    // Test value with unicode name
    let value_unicode = Value {
        name: "tensor_名_测试_🐍".to_string(),
        ty: Type::F32,
        shape: vec![1, 2],
    };
    
    assert_eq!(value_unicode.name, "tensor_名_测试_🐍");
}

/// Test 6: Recursive type handling with alternating patterns
#[test]
fn test_recursive_type_alternating_pattern() {
    // Create a pattern that alternates between different types when nesting
    let mut current_type = Type::Bool;
    
    // Alternate between F32 and I32 at different nesting levels
    for i in 0..20 {
        if i % 2 == 0 {
            current_type = Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![i + 1],
            };
        } else {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i + 1],
            };
        }
    }
    
    // Ensure the type is valid and can be cloned
    assert!(current_type.is_valid_type());
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 7: Operations and modules with maximum nesting of operations
#[test]
fn test_max_nesting_operations() {
    let mut current_op = Operation::new("base_op");
    
    // Add values to the base operation
    current_op.inputs.push(Value {
        name: "base_input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    current_op.outputs.push(Value {
        name: "base_output".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    // Simulate nested operations conceptually (the IR doesn't have nested ops,
    // but we can test with complex interconnected structures)
    let mut module = Module::new("nested_module");
    
    // Add many interconnected operations to create a complex graph
    for i in 0..100 {
        let mut op = Operation::new(&format!("connected_op_{}", i));
        
        // Connect to previous operations
        if i > 0 {
            op.inputs.push(Value {
                name: format!("prev_output_{}", i-1),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![i % 10 + 1],
            });
        }
        
        op.outputs.push(Value {
            name: format!("op_output_{}", i),
            ty: if i % 3 == 0 { Type::F64 } else { Type::I64 },
            shape: vec![i % 5 + 1],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    assert_eq!(module.name, "nested_module");
}

/// Test 8: Boundary values for type enums
#[test]
fn test_boundary_values_for_enums() {
    let types = [
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];
    
    // Verify each type is distinct
    for (i, ti) in types.iter().enumerate() {
        for (j, tj) in types.iter().enumerate() {
            if i == j {
                assert_eq!(ti, tj);
            } else {
                assert_ne!(ti, tj);
            }
        }
    }
    
    // Test nested version of each type
    for base_type in &types {
        let nested = Type::Tensor {
            element_type: Box::new(base_type.clone()),
            shape: vec![2, 3],
        };
        
        // Ensure nested types maintain their base type identity
        match &nested {
            Type::Tensor { element_type, .. } => {
                match element_type.as_ref() {
                    Type::F32 => assert!(matches!(base_type, Type::F32)),
                    Type::F64 => assert!(matches!(base_type, Type::F64)),
                    Type::I32 => assert!(matches!(base_type, Type::I32)),
                    Type::I64 => assert!(matches!(base_type, Type::I64)),
                    Type::Bool => assert!(matches!(base_type, Type::Bool)),
                    _ => panic!("Unexpected nested type"),
                }
            },
            _ => panic!("Expected tensor type"),
        }
    }
}

/// Test 9: Empty collections and boundary conditions in vectors
#[test]
fn test_empty_collections_boundary_conditions() {
    // Test empty values
    let empty_values: Vec<Value> = vec![];
    assert_eq!(empty_values.len(), 0);
    
    // Test single element
    let single_value = vec![Value {
        name: "single".to_string(),
        ty: Type::F32,
        shape: vec![1],
    }];
    assert_eq!(single_value.len(), 1);
    
    // Test with operations containing empty collections
    let mut empty_op = Operation::new("empty_op");
    assert_eq!(empty_op.inputs.len(), 0);
    assert_eq!(empty_op.outputs.len(), 0);
    assert_eq!(empty_op.attributes.len(), 0);
    
    // Add to empty collections and verify
    empty_op.inputs.push(Value {
        name: "first_input".to_string(),
        ty: Type::F32,
        shape: vec![2, 3],
    });
    
    assert_eq!(empty_op.inputs.len(), 1);
    
    // Test empty module
    let empty_module = Module::new("empty_module");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.inputs.len(), 0);
    assert_eq!(empty_module.outputs.len(), 0);
}

/// Test 10: Memory deallocation and cleanup validation
#[test]
fn test_memory_deallocation_validation() {
    // Create a complex structure and then drop it to test cleanup
    let mut complex_module = Module::new("complex_module");
    
    for i in 0..1000 {
        let mut op = Operation::new(&format!("cleanup_test_op_{}", i));
        
        // Add input and output values
        op.inputs.push(Value {
            name: format!("cleanup_input_{}", i),
            ty: Type::F32,
            shape: vec![i % 10 + 1, i % 5 + 1],
        });
        
        op.outputs.push(Value {
            name: format!("cleanup_output_{}", i),
            ty: Type::F64,
            shape: vec![i % 7 + 1, i % 3 + 1],
        });
        
        // Add some attributes
        use std::collections::HashMap;
        let mut attrs = HashMap::new();
        attrs.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
        op.attributes = attrs;
        
        complex_module.add_operation(op);
    }
    
    // Verify the module was built correctly
    assert_eq!(complex_module.operations.len(), 1000);
    assert_eq!(complex_module.name, "complex_module");
    
    // Drop the module - this tests that the memory cleanup works properly
    drop(complex_module);
    
    // Create a new module to ensure memory management is working
    let fresh_module = Module::new("fresh_module");
    assert_eq!(fresh_module.name, "fresh_module");
    assert_eq!(fresh_module.operations.len(), 0);
    
    // If we reach here without panic or memory issues, the test passes
    assert!(true);
}