//! Specialized edge case tests covering additional boundary conditions
//! Focus on numerical precision, memory safety, and compiler edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Value with dimension product overflow protection
#[test]
fn test_value_overflow_protection() {
    // Create a shape that would cause overflow if not handled properly
    let overflow_shape = vec![usize::MAX, 2];
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: overflow_shape,
    };
    
    // num_elements should return None when overflow would occur
    let num_elements = value.num_elements();
    assert_eq!(num_elements, None, "Should return None for overflow case");
}

/// Test 2: Module with special characters in operation names
#[test]
fn test_module_special_characters() {
    let mut module = Module::new("special_chars");
    
    // Test operation names with various special characters
    let special_names = [
        "op_with_underscore",
        "op-with-dash",
        "op.with.dot",
        "op:with:colon",
        "op@with@at",
    ];
    
    for name in &special_names {
        let op = Operation::new(name);
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for (i, expected_name) in special_names.iter().enumerate() {
        assert_eq!(module.operations[i].op_type, *expected_name);
    }
}

/// Test 3: Attribute with NaN and Infinity float values
#[test]
fn test_special_float_attributes() {
    let attrs = vec![
        Attribute::Float(f64::NAN),
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::Float(f64::MIN_POSITIVE),
        Attribute::Float(f64::EPSILON),
    ];
    
    // Verify all attributes are created without panic
    for attr in attrs {
        match attr {
            Attribute::Float(val) => {
                // Verify the value is actually a float (even if special)
                assert!(val.is_nan() || val.is_infinite() || val.is_finite());
            }
            _ => panic!("Expected Float attribute"),
        }
    }
}

/// Test 4: Value with asymmetric tensor shapes
#[test]
fn test_asymmetric_tensor_shapes() {
    // Test shapes with highly asymmetric dimensions
    let asymmetric_shapes = vec![
        vec![1, 1000000],
        vec![1000000, 1],
        vec![1, 1, 1000000],
        vec![1, 1000, 1000],
        vec![10, 100, 1000],
    ];
    
    for shape in asymmetric_shapes {
        let value = Value {
            name: "asymmetric".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        let expected: Option<usize> = shape.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
        assert_eq!(value.num_elements(), expected);
    }
}

/// Test 5: Operation with very long attribute values
#[test]
fn test_long_attribute_values() {
    let mut op = Operation::new("long_attrs");
    
    // Create a very long string attribute
    let long_string = "a".repeat(10000);
    op.attributes.insert(
        "long_string".to_string(),
        Attribute::String(long_string.clone()),
    );
    
    // Create a large array attribute
    let large_array: Vec<Attribute> = (0..1000)
        .map(|i| Attribute::Int(i as i64))
        .collect();
    op.attributes.insert(
        "large_array".to_string(),
        Attribute::Array(large_array),
    );
    
    assert_eq!(op.attributes.len(), 2);
    
    match &op.attributes["long_string"] {
        Attribute::String(s) => {
            assert_eq!(s.len(), 10000);
        }
        _ => panic!("Expected String attribute"),
    }
    
    match &op.attributes["large_array"] {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1000);
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: Module with empty operation type
#[test]
fn test_empty_operation_type() {
    let mut module = Module::new("empty_ops");
    let op = Operation::new("");
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].op_type, "");
    assert!(module.operations[0].inputs.is_empty());
    assert!(module.operations[0].outputs.is_empty());
}

/// Test 7: Compiler with zero-length model
#[test]
fn test_compiler_zero_length_model() {
    let mut compiler = ImpulseCompiler::new();
    let empty_model = vec![0u8; 0];
    
    let result = compiler.compile(&empty_model, "cpu");
    // Should handle gracefully without panic
    match result {
        Ok(_) => assert!(true),
        Err(e) => assert!(e.to_string().len() > 0),
    }
}

/// Test 8: Value with single dimension of zero (edge case)
#[test]
fn test_single_zero_dimension() {
    let shapes = vec![
        vec![0],
        vec![0, 1],
        vec![1, 0],
        vec![1, 1, 0],
        vec![0, 1, 1],
    ];
    
    for shape in shapes {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any shape with a zero dimension should have 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 9: Nested array attributes with empty elements
#[test]
fn test_nested_array_with_empty_elements() {
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![]),           // Empty nested array
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![]),        // Empty array in middle
            Attribute::Int(2),
        ]),
        Attribute::Array(vec![Attribute::Bool(true)]),
    ]);
    
    match nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
            
            // Check first element (empty array)
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 0);
                }
                _ => panic!("Expected empty array"),
            }
            
            // Check second element
            match &outer[1] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 3);
                }
                _ => panic!("Expected array with 3 elements"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 10: Module with operations having conflicting input/output types
#[test]
fn test_operation_type_mismatch() {
    let mut module = Module::new("type_mismatch");
    let mut op = Operation::new("mixed_types");
    
    // Add inputs of different types
    op.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op.inputs.push(Value {
        name: "i32_input".to_string(),
        ty: Type::I32,
        shape: vec![10],
    });
    
    // Add outputs of yet another type
    op.outputs.push(Value {
        name: "bool_output".to_string(),
        ty: Type::Bool,
        shape: vec![10],
    });
    
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 2);
    assert_eq!(module.operations[0].outputs.len(), 1);
    assert_eq!(module.operations[0].inputs[0].ty, Type::F32);
    assert_eq!(module.operations[0].inputs[1].ty, Type::I32);
    assert_eq!(module.operations[0].outputs[0].ty, Type::Bool);
}