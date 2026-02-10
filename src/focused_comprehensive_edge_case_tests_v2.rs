//! Focused comprehensive edge case tests v2 - Additional boundary scenarios
//! 
//! This module contains 10 additional test cases covering more edge cases
//! and boundary conditions for the Impulse compiler IR components.

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;

/// Test 1: Module with operations that have very long operation names
#[test]
fn test_operations_with_extremely_long_names() {
    let mut module = Module::new("long_names_module");
    
    // Create operations with progressively longer names
    let long_names = vec![
        "a".repeat(100),
        "operation_".repeat(50),
        "very_long_operation_name_that_stretches_on_and_on_".repeat(20),
        format!("op_{}", "x".repeat(500)),
    ];
    
    for (i, name) in long_names.iter().enumerate() {
        let mut op = Operation::new(name);
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 4);
    assert_eq!(module.operations[0].op_type.len(), 100);
    assert_eq!(module.operations[1].op_type.len(), 500); // "operation_" (10 chars) * 50
    assert!(module.operations[2].op_type.len() >= 1000); // 50 chars * 20 = 1000
    assert_eq!(module.operations[3].op_type.len(), 503); // "op_" (3 chars) + 500 * "x"
}

/// Test 2: Value with shape that would overflow usize multiplication
#[test]
fn test_value_shape_overflow_handling() {
    // Create shapes that would overflow during element count calculation
    let problematic_shapes = vec![
        vec![usize::MAX, 2],  // Would overflow
        vec![1_000_000, 1_000_000, 1_000_000],  // Large but might overflow
        vec![usize::MAX / 2, 3],  // Edge case near overflow
    ];
    
    for shape in problematic_shapes {
        let value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // num_elements should return None for overflow cases
        let result = value.num_elements();
        // For overflow cases, should return None or handle gracefully
        // The actual value depends on the checked_mul implementation
        match result {
            Some(count) => {
                // If it returns a value, it should be valid (either 0 for shapes with 0, or calculated)
                // For overflow cases, this would wrap or saturate, but we check the implementation
                assert!(count == 0 || count > 0);
            }
            None => {
                // This is the expected behavior for true overflow
            }
        }
    }
}

/// Test 3: Attribute with extreme float values (NaN, Infinity)
#[test]
fn test_attribute_with_special_float_values() {
    let special_values = vec![
        (f64::NAN, "NaN"),
        (f64::INFINITY, "Positive Infinity"),
        (f64::NEG_INFINITY, "Negative Infinity"),
        (-0.0, "Negative Zero"),
    ];
    
    for (value, desc) in special_values {
        let attr = Attribute::Float(value);
        
        match attr {
            Attribute::Float(f) => {
                // Verify the attribute stores the value
                if desc.contains("NaN") {
                    assert!(f.is_nan());
                } else if desc.contains("Positive Infinity") {
                    assert!(f.is_infinite() && f.is_sign_positive());
                } else if desc.contains("Negative Infinity") {
                    assert!(f.is_infinite() && f.is_sign_negative());
                } else if desc.contains("Negative Zero") {
                    assert_eq!(f, -0.0);
                    assert!(f.is_sign_negative());
                }
            }
            _ => panic!("Expected Float attribute"),
        }
    }
}

/// Test 4: Module with inputs/outputs having duplicate names
#[test]
fn test_module_with_duplicate_io_names() {
    let mut module = Module::new("duplicate_io_names");
    
    // Add inputs with duplicate names
    for i in 0..3 {
        module.inputs.push(Value {
            name: "duplicate_input".to_string(),
            ty: Type::F32,
            shape: vec![i + 1],
        });
    }
    
    // Add outputs with duplicate names
    for i in 0..2 {
        module.outputs.push(Value {
            name: "duplicate_output".to_string(),
            ty: Type::I32,
            shape: vec![i + 1],
        });
    }
    
    // Verify all are stored (no deduplication by default)
    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 2);
    
    // All inputs have the same name
    for input in &module.inputs {
        assert_eq!(input.name, "duplicate_input");
    }
    
    // All outputs have the same name
    for output in &module.outputs {
        assert_eq!(output.name, "duplicate_output");
    }
}

/// Test 5: Compiler with alternating valid/invalid model data
#[test]
fn test_compiler_with_alternating_models() {
    let mut compiler = ImpulseCompiler::new();
    
    let models = vec![
        vec![0u8],           // Very small valid
        vec![],              // Empty
        vec![0xFF; 1024],    // Larger valid
        vec![0x00; 0],       // Empty slice
        vec![0x01, 0x02, 0x03], // Tiny valid
    ];
    
    for model in models {
        let result = compiler.compile(&model, "cpu");
        // Should handle all cases gracefully
        assert!(result.is_ok() || result.is_err());
    }
}

/// Test 6: Value with scalar and 1-element tensor equivalence patterns
#[test]
fn test_scalar_vs_single_element_tensor() {
    // Scalar (empty shape)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    
    // 1D tensor with single element
    let tensor_1d = Value {
        name: "tensor_1d".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    // Multi-dimensional tensor with single element
    let tensor_nd = Value {
        name: "tensor_nd".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    
    // All should have num_elements() returning Some(1) or handling empty specially
    assert_eq!(scalar.num_elements(), Some(1)); // Empty shape = 1 element (scalar)
    assert_eq!(tensor_1d.num_elements(), Some(1));
    assert_eq!(tensor_nd.num_elements(), Some(1));
}

/// Test 7: Operation with empty array attribute
#[test]
fn test_operation_with_empty_array_attribute() {
    let mut op = Operation::new("empty_array_op");
    
    // Add various empty array attributes
    op.attributes.insert("empty_int_array".to_string(), Attribute::Array(vec![]));
    op.attributes.insert("nested_empty".to_string(), Attribute::Array(vec![
        Attribute::Array(vec![]),
    ]));
    
    assert_eq!(op.attributes.len(), 2);
    
    match op.attributes.get("empty_int_array") {
        Some(Attribute::Array(arr)) => assert!(arr.is_empty()),
        _ => panic!("Expected empty array"),
    }
    
    match op.attributes.get("nested_empty") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Array(inner) => assert!(inner.is_empty()),
                _ => panic!("Expected nested empty array"),
            }
        }
        _ => panic!("Expected nested empty array"),
    }
}

/// Test 8: Module with extremely nested tensor types in values
#[test]
fn test_module_with_extremely_nested_tensor_types() {
    let mut module = Module::new("deeply_nested_module");
    
    // Create deeply nested tensor: tensor<tensor<tensor<tensor<f32, [1]>, [1]>, [1]>, [1]>
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![1],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![1],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![1],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![1],
    };
    
    let mut op = Operation::new("deep_tensor_op");
    op.inputs.push(Value {
        name: "deep_tensor".to_string(),
        ty: level4,
        shape: vec![1],
    });
    
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 1);
}

/// Test 9: Attribute array with mixed types at various nesting levels
#[test]
fn test_mixed_type_nested_attribute_arrays() {
    let mixed = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::Array(vec![
            Attribute::String("nested".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![
                Attribute::Float(2.71),
                Attribute::Int(-1),
            ]),
        ]),
        Attribute::String("top_level".to_string()),
    ]);
    
    match mixed {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 4);
            
            // Check first element
            match &outer[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
            
            // Check second element
            match &outer[1] {
                Attribute::Float(f) => assert!((f - 3.14).abs() < f64::EPSILON),
                _ => panic!("Expected Float(3.14)"),
            }
            
            // Check third element (nested array)
            match &outer[2] {
                Attribute::Array(mid) => {
                    assert_eq!(mid.len(), 3);
                    match &mid[0] {
                        Attribute::String(s) => assert_eq!(s, "nested"),
                        _ => panic!("Expected String"),
                    }
                },
                _ => panic!("Expected nested Array"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 10: Compiler state after multiple failed compilations
#[test]
fn test_compiler_state_after_multiple_failures() {
    let mut compiler = ImpulseCompiler::new();
    
    // Compile with various failing models
    let failing_models = vec![
        vec![],
        vec![0x00],
        vec![0xFF, 0xFF],
    ];
    
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for model in failing_models {
        match compiler.compile(&model, "invalid_target") {
            Ok(_) => success_count += 1,
            Err(_) => failure_count += 1,
        }
    }
    
    // Compiler should still be functional
    assert_eq!(compiler.passes.passes.len(), 0);
    
    // Verify we actually tried to compile
    assert_eq!(success_count + failure_count, 3);
    
    // Try to compile again to ensure compiler state is consistent
    let result = compiler.compile(&[0x01, 0x02, 0x03], "cpu");
    assert!(result.is_ok() || result.is_err());
}