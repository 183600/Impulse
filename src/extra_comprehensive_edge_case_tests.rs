//! Extra comprehensive edge case tests covering additional boundary conditions
//! Tests focus on numeric boundaries, special values, and extreme scenarios

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Attribute with denormalized floating point values
#[test]
fn test_denormalized_float_attributes() {
    // Test with subnormal (denormalized) float values
    let subnormal_f32 = Attribute::Float(f32::MIN_POSITIVE as f64);
    let subnormal_f64 = Attribute::Float(f64::MIN_POSITIVE);
    
    match subnormal_f32 {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val < 1e-30);
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match subnormal_f64 {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val < 1e-300);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: Value with asymmetric large tensor shapes
#[test]
fn test_asymmetric_tensor_shapes() {
    let test_cases = vec![
        (vec![1, 1_000_000], 1_000_000),       // Very wide 1D
        (vec![1_000_000, 1], 1_000_000),       // Very tall 1D
        (vec![1, 1, 1_000_000], 1_000_000),    // Wide 2D with leading 1s
        (vec![2, 1_000, 1_000], 2_000_000),    // Moderate multi-dim
        (vec![1, 256, 256, 256], 16_777_216),  // 3D image-like
    ];
    
    for (shape, expected_elements) in test_cases {
        let value = Value {
            name: "asymmetric_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let actual_elements = value.num_elements();
        assert_eq!(actual_elements, Some(expected_elements));
    }
}

/// Test 3: Operation with duplicate attribute keys (last one wins)
#[test]
fn test_operation_duplicate_attributes() {
    let mut op = Operation::new("duplicate_attrs");
    let mut attrs = HashMap::new();
    
    // Insert same key multiple times - last insertion wins
    attrs.insert("value".to_string(), Attribute::Int(1));
    attrs.insert("value".to_string(), Attribute::Int(2));
    attrs.insert("value".to_string(), Attribute::Int(3));
    
    op.attributes = attrs;
    
    // Only one entry should exist
    assert_eq!(op.attributes.len(), 1);
    assert_eq!(op.attributes.get("value"), Some(&Attribute::Int(3)));
}

/// Test 4: Module with cyclic-style input/output naming
#[test]
fn test_module_cyclic_naming_pattern() {
    let mut module = Module::new("cyclic_naming");
    
    // Create a pattern where outputs reference inputs conceptually
    let input_a = Value {
        name: "x_0".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    let input_b = Value {
        name: "x_1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    let output_a = Value {
        name: "y_0".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    let output_b = Value {
        name: "y_1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    module.inputs.push(input_a);
    module.inputs.push(input_b);
    module.outputs.push(output_a);
    module.outputs.push(output_b);
    
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    assert!(module.inputs[0].name.starts_with("x_"));
    assert!(module.outputs[0].name.starts_with("y_"));
}

/// Test 5: Type with maximum safe integer boundaries
#[test]
fn test_max_safe_integer_boundaries() {
    // Test with values near i64 boundaries
    let test_values = vec![
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        i64::MIN + 1,
        -1,
        0,
        1,
    ];
    
    for val in test_values {
        let attr = Attribute::Int(val);
        match attr {
            Attribute::Int(v) => assert_eq!(v, val),
            _ => panic!("Expected Int attribute"),
        }
    }
}

/// Test 6: Nested tensor with single-element shapes at each level
#[test]
fn test_single_element_nested_tensors() {
    // Create nested tensors with shape [1] at each level
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
    
    // Verify structure
    match &level3 {
        Type::Tensor { shape, element_type } => {
            assert_eq!(shape, &vec![1]);
            match element_type.as_ref() {
                Type::Tensor { shape: s, .. } => assert_eq!(s, &vec![1]),
                _ => panic!("Expected nested tensor"),
            }
        }
        _ => panic!("Expected tensor type"),
    }
}

/// Test 7: Module with operations having no inputs but with outputs
#[test]
fn test_operations_output_only() {
    let mut module = Module::new("output_only_ops");
    
    // Constant-generating operations (no inputs, only outputs)
    for i in 0..5 {
        let mut op = Operation::new(&format!("const_{}", i));
        op.outputs.push(Value {
            name: format!("constant_{}", i),
            ty: Type::F32,
            shape: vec![],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for (i, op) in module.operations.iter().enumerate() {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 1);
        assert!(op.op_type.starts_with("const_"));
    }
}

/// Test 8: Attribute array with single element repeated many times
#[test]
fn test_repeated_single_element_array() {
    let repeated = Attribute::Array(vec![
        Attribute::Int(1);
        100
    ]);
    
    match repeated {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 100);
            // All elements should be Int(1)
            for (i, elem) in arr.iter().enumerate() {
                match elem {
                    Attribute::Int(1) => (),
                    _ => panic!("Element at index {} should be Int(1)", i),
                }
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Value with all possible primitive types
#[test]
fn test_value_all_primitive_types() {
    let types = vec![
        (Type::F32, "f32_value"),
        (Type::F64, "f64_value"),
        (Type::I32, "i32_value"),
        (Type::I64, "i64_value"),
        (Type::Bool, "bool_value"),
    ];
    
    for (ty, name) in types {
        let value = Value {
            name: name.to_string(),
            ty: ty.clone(),
            shape: vec![1],
        };
        
        assert_eq!(value.ty, ty);
        assert_eq!(value.name, name);
    }
}

/// Test 10: Operation with deeply nested array attribute
#[test]
fn test_deeply_nested_array_attribute() {
    // Create a depth-5 nested array structure
    let depth5 = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Array(vec![Attribute::Int(42)])
                ])
            ])
        ])
    ]);
    
    // Navigate through the nesting to verify structure
    match depth5 {
        Attribute::Array(d5) => {
            assert_eq!(d5.len(), 1);
            match &d5[0] {
                Attribute::Array(d4) => {
                    assert_eq!(d4.len(), 1);
                    match &d4[0] {
                        Attribute::Array(d3) => {
                            assert_eq!(d3.len(), 1);
                            match &d3[0] {
                                Attribute::Array(d2) => {
                                    assert_eq!(d2.len(), 1);
                                    match &d2[0] {
                                        Attribute::Array(d1) => {
                                            assert_eq!(d1.len(), 1);
                                            match &d1[0] {
                                                Attribute::Int(42) => (),
                                                _ => panic!("Expected Int(42) at deepest level"),
                                            }
                                        }
                                        _ => panic!("Expected Array at depth 1"),
                                    }
                                }
                                _ => panic!("Expected Array at depth 2"),
                            }
                        }
                        _ => panic!("Expected Array at depth 3"),
                    }
                }
                _ => panic!("Expected Array at depth 4"),
            }
        }
        _ => panic!("Expected Array at depth 5"),
    }
}