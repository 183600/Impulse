//! Ultimate edge case tests - comprehensive boundary coverage
//! Tests for rare and extreme scenarios in the Impulse compiler

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Value with extreme large dimensions that will cause overflow
#[test]
fn test_value_extreme_large_dimension() {
    // Test with dimensions that will overflow in multiplication
    let value = Value {
        name: "extreme_large".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // This will overflow: MAX * 2
    };
    
    // num_elements should return None due to overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 2: Attribute with special floating point values (inf, nan, negative zero)
#[test]
fn test_special_float_attributes() {
    let attrs = vec![
        ("inf", Attribute::Float(f64::INFINITY)),
        ("neg_inf", Attribute::Float(f64::NEG_INFINITY)),
        ("nan", Attribute::Float(f64::NAN)),
        ("neg_zero", Attribute::Float(-0.0)),
    ];
    
    for (name, attr) in attrs {
        match attr {
            Attribute::Float(f) => {
                // Verify these values are created without panic
                match name {
                    "inf" => assert!(f.is_infinite() && f.is_sign_positive()),
                    "neg_inf" => assert!(f.is_infinite() && f.is_sign_negative()),
                    "nan" => assert!(f.is_nan()),
                    "neg_zero" => assert!(f == 0.0 && f.is_sign_negative()),
                    _ => {}
                }
            }
            _ => panic!("Expected Float attribute"),
        }
    }
}

/// Test 3: Type validation for deeply nested Tensor types
#[test]
fn test_deeply_nested_tensor_validation() {
    // Create a 10-level nested tensor
    let mut nested_type: Type = Type::F32;
    for _i in 1..=10 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![2],
        };
    }
    
    // All nested types should be valid
    assert!(nested_type.is_valid_type());
}

/// Test 4: Module with extremely long operation names
#[test]
fn test_module_very_long_operation_names() {
    let mut module = Module::new("long_names");
    
    // Create an operation with an extremely long name (stress test)
    let long_name = "a".repeat(1000);
    let mut op = Operation::new(&long_name);
    op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    module.add_operation(op);
    
    assert_eq!(module.operations[0].op_type.len(), 1000);
}

/// Test 5: Attribute array with alternating types pattern
#[test]
fn test_alternating_attribute_types() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(1.5),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Int(2),
        Attribute::Float(2.5),
        Attribute::String("test2".to_string()),
        Attribute::Bool(false),
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 8);
            assert!(matches!(arr[0], Attribute::Int(_)));
            assert!(matches!(arr[1], Attribute::Float(_)));
            assert!(matches!(arr[2], Attribute::String(_)));
            assert!(matches!(arr[3], Attribute::Bool(_)));
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 6: Value with all zero dimensions (edge case tensor)
#[test]
fn test_value_all_zero_dimensions() {
    let value = Value {
        name: "all_zero".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0],
    };
    
    assert_eq!(value.num_elements(), Some(0));
    assert_eq!(value.shape.iter().product::<usize>(), 0);
}

/// Test 7: Operation with empty string as attribute values
#[test]
fn test_operation_empty_string_attributes() {
    let mut op = Operation::new("empty_str_test");
    let mut attrs = HashMap::new();
    
    attrs.insert("empty".to_string(), Attribute::String("".to_string()));
    attrs.insert("spaces".to_string(), Attribute::String("   ".to_string()));
    attrs.insert("unicode".to_string(), Attribute::String("🔥".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 3);
    assert_eq!(op.attributes.get("empty"), Some(&Attribute::String("".to_string())));
}

/// Test 8: Compiler with model containing all possible byte values (0x00-0xFF)
#[test]
fn test_compiler_all_byte_values() {
    let mut compiler = ImpulseCompiler::new();
    
    // Create a model with all possible byte values
    let all_bytes: Vec<u8> = (0..=255).collect();
    
    let result = compiler.compile(&all_bytes, "cpu");
    // Should handle gracefully without panic
    assert!(result.is_ok() || result.is_err());
}

/// Test 9: Value with single dimension equal to 1 (broadcast edge case)
#[test]
fn test_value_single_dimension_edge_cases() {
    let test_cases = vec![
        vec![1],           // 1D single element
        vec![1, 1],        // 2D single element
        vec![1, 1, 1],     // 3D single element
        vec![1, 100],      // Can broadcast to [100]
        vec![100, 1],      // Can broadcast to [100]
        vec![1, 100, 1],   // Can broadcast to [100]
    ];
    
    for shape in test_cases {
        let value = Value {
            name: "broadcast_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let elements = value.num_elements().unwrap();
        assert_eq!(elements, shape.iter().product::<usize>());
    }
}

/// Test 10: Module with inputs that have identical shapes but different types
#[test]
fn test_module_identical_shapes_different_types() {
    let mut module = Module::new("mixed_types_same_shape");
    
    // Add inputs with same shape but different types
    module.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    });
    module.inputs.push(Value {
        name: "i32_input".to_string(),
        ty: Type::I32,
        shape: vec![10, 20],
    });
    module.inputs.push(Value {
        name: "f64_input".to_string(),
        ty: Type::F64,
        shape: vec![10, 20],
    });
    
    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.inputs[0].shape, module.inputs[1].shape);
    assert_eq!(module.inputs[1].shape, module.inputs[2].shape);
    assert_ne!(module.inputs[0].ty, module.inputs[1].ty);
    assert_ne!(module.inputs[1].ty, module.inputs[2].ty);
}