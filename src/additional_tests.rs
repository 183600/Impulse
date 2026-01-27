//! Additional edge case tests for the Impulse compiler
//! This file contains tests for various edge cases and boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute};

// Test 1: Operations with empty collections
#[test]
fn test_empty_collections() {
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
    assert!(empty_module.operations.is_empty());
    assert!(empty_module.inputs.is_empty());
    assert!(empty_module.outputs.is_empty());

    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    assert!(empty_op.inputs.is_empty());
    assert!(empty_op.outputs.is_empty());
    assert!(empty_op.attributes.is_empty());
}

// Test 2: Extreme numeric values in tensor shapes and attributes
#[test]
fn test_extreme_numeric_values() {
    // Test tensor with maximum possible shape dimensions
    let extreme_value = Value {
        name: "extreme_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };
    
    assert_eq!(extreme_value.shape[0], usize::MAX);

    // Test attribute with maximum integer value
    let int_attr = Attribute::Int(i64::MAX);
    match int_attr {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected Int attribute"),
    }

    // Test attribute with minimum integer value
    let min_int_attr = Attribute::Int(i64::MIN);
    match min_int_attr {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected Int attribute"),
    }

    // Test attribute with extreme float values
    let inf_attr = Attribute::Float(f64::INFINITY);
    match inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float attribute"),
    }

    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    match neg_inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float attribute"),
    }
}

// Test 3: Operations with maximum possible attributes
#[test]
fn test_operations_with_many_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attrs_op");
    let mut attrs = HashMap::new();
    
    // Add many attributes to test hash map performance
    for i in 0..1000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1000);
    assert_eq!(op.op_type, "max_attrs_op");
    
    // Test retrieval of first and last attributes
    assert!(op.attributes.contains_key("attr_0"));
    assert!(op.attributes.contains_key("attr_999"));
    assert_eq!(
        op.attributes.get("attr_0"),
        Some(&Attribute::String("value_0".to_string()))
    );
}

// Test 4: Deeply nested recursive types edge case
#[test]
fn test_very_deeply_nested_types() {
    // Create a very deeply nested type to test recursion limits
    let mut current_type = Type::F32;
    
    // Create 500 levels of nesting
    for i in 0..500 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 10 + 1], // Varying shape to make it interesting
        };
    }
    
    // Verify deep clone still works correctly
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
    
    // Verify it's still a tensor type at the top level
    match current_type {
        Type::Tensor { .. } => (), // Success
        _ => panic!("Top level should still be a Tensor"),
    }
    
    // Verify the cloned version is identical
    assert_eq!(current_type, cloned_type);
}

// Test 5: Complex tensor shape combinations
#[test]
fn test_complex_tensor_shape_combinations() {
    let test_cases = vec![
        (vec![], 1),                          // Scalar: 1 element
        (vec![0], 0),                         // Zero dimension: 0 elements
        (vec![1], 1),                         // Single element: 1 element
        (vec![2, 3, 4], 24),                  // 2×3×4 = 24 elements
        (vec![0, 100, 100], 0),               // Contains zero: 0 elements
        (vec![1, 1, 1, 1, 100], 100),        // Many single dims: 100 elements
        (vec![10, 0, 5, 2], 0),               // Zero in middle: 0 elements
    ];
    
    for (shape, expected_count) in test_cases {
        let value = Value {
            name: "test_shape".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let actual_count: usize = value.shape.iter().product();
        assert_eq!(actual_count, expected_count, "Failed for shape {:?}", shape);
        assert_eq!(value.shape, shape);
    }
}

// Test 6: Unicode and special character handling in names
#[test]
fn test_unicode_and_special_character_names() {
    let unicode_names = vec![
        "αβγ_ΔΣΛ_πθω",           // Greek letters
        "∂∇√∞≡≅∑∏",             // Math symbols
        "🚀🌟💻🎉",              // Emojis
        "café naïve résumé",      // Accented characters
        "日本語_中文_한국어",         // CJK characters
        "🙂😀😎🤖",              // More emojis
        "тест_на_русском",       // Cyrillic
        "اختبار_عربي",           // Arabic
    ];
    
    for name in unicode_names {
        // Test module with unicode name
        let module = Module::new(name);
        assert_eq!(module.name, name);
        
        // Test operation with unicode name
        let op = Operation::new(name);
        assert_eq!(op.op_type, name);
        
        // Test value with unicode name
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        assert_eq!(value.name, name);
    }
}

// Test 7: Zero-sized tensors and empty shapes comprehensive testing
#[test]
fn test_zero_sized_tensors_comprehensive() {
    let zero_cases = vec![
        vec![0],              // Single zero dimension
        vec![0, 5],           // Zero followed by non-zero
        vec![5, 0],           // Non-zero followed by zero
        vec![1, 2, 0, 4],     // Zero in middle
        vec![0, 0, 0],        // All zeros
        vec![2, 0, 3, 0, 5],  // Multiple zeros scattered
    ];
    
    for shape in zero_cases {
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Zero-sized tensors should have 0 elements
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements", shape);
        
        // Verify shape is preserved
        assert_eq!(value.shape, shape);
        assert_eq!(value.ty, Type::F32);
    }
    
    // Test scalar (empty shape) tensors
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::I64,
        shape: vec![],  // Scalar has empty shape
    };
    
    assert!(scalar.shape.is_empty());
    let scalar_elements: usize = scalar.shape.iter().product();
    assert_eq!(scalar_elements, 1); // Scalars have 1 element
}

// Test 8: Nested attribute arrays with complex structure
#[test]
fn test_complex_nested_attributes() {
    let complex_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(1.5),
                Attribute::Array(vec![
                    Attribute::String("deeply_nested".to_string()),
                    Attribute::Bool(true),
                ])
            ]),
        ]),
        Attribute::Array(vec![
            Attribute::Int(2),
            Attribute::Array(vec![
                Attribute::String("another_deep".to_string()),
            ]),
        ]),
    ]);
    
    // Verify the complex nested structure
    match &complex_nested {
        Attribute::Array(outer_array) => {
            assert_eq!(outer_array.len(), 2);
            
            // Check first nested array
            match &outer_array[0] {
                Attribute::Array(first_inner) => {
                    assert_eq!(first_inner.len(), 2);
                    
                    match first_inner[0] {
                        Attribute::Int(1) => (),
                        _ => panic!("First element of first inner array should be Int(1)"),
                    }
                    
                    match &first_inner[1] {
                        Attribute::Array(deeply_nested) => {
                            assert_eq!(deeply_nested.len(), 2);
                            
                            match deeply_nested[0] {
                                Attribute::Float(f) => assert!((f - 1.5).abs() < f64::EPSILON),
                                _ => panic!("First element of deep nest should be Float(1.5)"),
                            }
                            
                            match &deeply_nested[1] {
                                Attribute::Array(deepest) => {
                                    assert_eq!(deepest.len(), 2);
                                    
                                    match &deepest[0] {
                                        Attribute::String(s) => assert_eq!(s, "deeply_nested"),
                                        _ => panic!("First element of deepest nest should be String"),
                                    }
                                    
                                    match deepest[1] {
                                        Attribute::Bool(true) => (),
                                        _ => panic!("Second element of deepest nest should be Bool(true)"),
                                    }
                                },
                                _ => panic!("Deepest level should be an array"),
                            }
                        },
                        _ => panic!("Second element of first inner should be an array"),
                    }
                },
                _ => panic!("First element of outer array should be an array"),
            }
        },
        _ => panic!("Complex nested should be an array attribute"),
    }
    
    // Test cloning of complex nested structure
    let cloned = complex_nested.clone();
    assert_eq!(complex_nested, cloned);
}

// Test 9: Operation with mixed type inputs and outputs
#[test]
fn test_operation_with_mixed_type_entities() {
    let mut op = Operation::new("mixed_types");
    
    // Add various typed inputs
    op.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![2, 3],
    });
    
    op.inputs.push(Value {
        name: "i64_input".to_string(),
        ty: Type::I64,
        shape: vec![5],
    });
    
    op.inputs.push(Value {
        name: "bool_input".to_string(),
        ty: Type::Bool,
        shape: vec![],
    });
    
    // Add various typed outputs
    op.outputs.push(Value {
        name: "f64_output".to_string(),
        ty: Type::F64,
        shape: vec![10, 10, 10],
    });
    
    op.outputs.push(Value {
        name: "i32_output".to_string(),
        ty: Type::I32,
        shape: vec![1, 1, 1, 42],
    });
    
    assert_eq!(op.inputs.len(), 3);
    assert_eq!(op.outputs.len(), 2);
    assert_eq!(op.op_type, "mixed_types");
    
    // Verify the types are correct
    assert_eq!(op.inputs[0].ty, Type::F32);
    assert_eq!(op.inputs[1].ty, Type::I64);
    assert_eq!(op.inputs[2].ty, Type::Bool);
    assert_eq!(op.outputs[0].ty, Type::F64);
    assert_eq!(op.outputs[1].ty, Type::I32);
    
    // Verify shapes are preserved
    assert_eq!(op.inputs[0].shape, vec![2, 3]);
    assert_eq!(op.inputs[1].shape, vec![5]);
    assert!(op.inputs[2].shape.is_empty());
    assert_eq!(op.outputs[0].shape, vec![10, 10, 10]);
    assert_eq!(op.outputs[1].shape, vec![1, 1, 1, 42]);
}

// Test 10: Large memory allocation edge case without overflow
#[test]
fn test_large_but_safe_allocations() {
    // Create a module with many operations to test memory handling
    let mut module = Module::new("stress_test_module");
    
    // Add operations with moderate but significant size
    for i in 0..10_000 {
        let mut op = Operation::new(&format!("stress_op_{}", i));
        
        // Add a few inputs and outputs
        for j in 0..5 {
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: if j % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![j + 1, j + 2],
            });
            
            op.outputs.push(Value {
                name: format!("output_{}_{}", i, j),
                ty: if j % 3 == 0 { Type::F64 } else { Type::I64 },
                shape: vec![j + 2, j + 1],
            });
        }
        
        // Add a few attributes
        use std::collections::HashMap;
        let mut attrs = HashMap::new();
        attrs.insert(
            format!("attr_{}_count", i),
            Attribute::Int(i as i64)
        );
        attrs.insert(
            format!("attr_{}_name", i),
            Attribute::String(format!("operation_{}", i))
        );
        op.attributes = attrs;
        
        module.add_operation(op);
    }
    
    // Verify the module has the expected number of operations
    assert_eq!(module.operations.len(), 10_000);
    assert_eq!(module.name, "stress_test_module");
    
    // Verify that the first and last operations have the expected structure
    assert_eq!(module.operations[0].op_type, "stress_op_0");
    assert_eq!(module.operations[0].inputs.len(), 5);
    assert_eq!(module.operations[0].outputs.len(), 5);
    assert_eq!(module.operations[0].attributes.len(), 2);
    
    assert_eq!(module.operations[9999].op_type, "stress_op_9999");
    assert_eq!(module.operations[9999].inputs.len(), 5);
    assert_eq!(module.operations[9999].outputs.len(), 5);
    assert_eq!(module.operations[9999].attributes.len(), 2);
    
    // Verify module is functioning correctly after large population
    assert_eq!(module.name, "stress_test_module");
}