//! Ultimate boundary tests new - Additional comprehensive edge case coverage
//! 
//! This module contains 10 additional test cases covering new edge cases
//! and boundary conditions for the Impulse compiler IR components.

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;

/// Test 1: Value with alternating zero and non-zero dimensions pattern
#[test]
fn test_value_alternating_zero_nonzero_dimensions() {
    let patterns = vec![
        vec![1, 0, 1, 0, 1],
        vec![2, 0, 3, 0, 4],
        vec![0, 5, 0, 5, 0],
        vec![10, 0, 10, 0, 10],
    ];
    
    for shape in patterns.iter() {
        let value = Value {
            name: "alternating_pattern".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any pattern containing zero should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 2: Module with operations having cyclic attribute references
#[test]
fn test_module_cyclic_attribute_patterns() {
    let mut module = Module::new("cyclic_attrs_module");
    
    // Create operations with attributes that reference each other's indices
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_{}", i));
        let mut attrs = std::collections::HashMap::new();
        
        // Create cyclic-like pattern
        attrs.insert("prev_idx".to_string(), Attribute::Int(((i as i64) - 1).max(0)));
        attrs.insert("next_idx".to_string(), Attribute::Int(((i + 1) as i64).min(4)));
        attrs.insert("self_idx".to_string(), Attribute::Int(i));
        
        op.attributes = attrs;
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    // Verify the cyclic pattern
    assert_eq!(module.operations[0].attributes.get("prev_idx"), Some(&Attribute::Int(0)));
    assert_eq!(module.operations[0].attributes.get("next_idx"), Some(&Attribute::Int(1)));
    assert_eq!(module.operations[4].attributes.get("next_idx"), Some(&Attribute::Int(4)));
}

/// Test 3: Attribute array with single element of each type
#[test]
fn test_attribute_array_single_element_types() {
    let single_attrs = vec![
        Attribute::Array(vec![Attribute::Int(42)]),
        Attribute::Array(vec![Attribute::Float(3.14)]),
        Attribute::Array(vec![Attribute::String("test".to_string())]),
        Attribute::Array(vec![Attribute::Bool(true)]),
        Attribute::Array(vec![Attribute::Bool(false)]),
    ];
    
    for attr in single_attrs.iter() {
        match attr {
            Attribute::Array(inner) => {
                assert_eq!(inner.len(), 1);
            }
            _ => panic!("Expected Array attribute"),
        }
    }
    
    // Verify the content of each single-element array
    match &single_attrs[0] {
        Attribute::Array(inner) => assert_eq!(inner[0], Attribute::Int(42)),
        _ => panic!("Expected Array with Int"),
    }
}

/// Test 4: Compiler with models containing only repeated bytes
#[test]
fn test_compiler_repeated_byte_models() {
    let mut compiler = ImpulseCompiler::new();
    
    let repeated_patterns = vec![
        vec![0x00; 100],
        vec![0xFF; 100],
        vec![0x55; 50],
        vec![0xAA; 50],
        vec![0x12; 75],
    ];
    
    for pattern in repeated_patterns.iter() {
        let result = compiler.compile(pattern, "cpu");
        // Should handle gracefully without panic
        match result {
            Ok(_) => (),
            Err(e) => {
                assert!(e.to_string().len() > 0);
            }
        }
    }
}

/// Test 5: Value with dimensions that are powers of 2
#[test]
fn test_value_powers_of_two_dimensions() {
    let power_of_2_shapes = vec![
        vec![1, 2, 4, 8, 16],
        vec![2, 4, 8, 16, 32],
        vec![4, 8, 16, 32, 64],
        vec![8, 16, 32, 64, 128],
        vec![16, 32, 64, 128, 256],
    ];
    
    for shape in power_of_2_shapes.iter() {
        let value = Value {
            name: "powers_of_two".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Verify all dimensions are powers of 2
        for dim in shape.iter() {
            assert!(*dim > 0);
            assert!(dim & (dim - 1) == 0); // Power of 2 check
        }
        
        // Verify element count is valid
        assert_eq!(value.num_elements(), Some(shape.iter().product()));
    }
}

/// Test 6: Operation with attributes containing mathematical constants
#[test]
fn test_operation_mathematical_constant_attributes() {
    let mut op = Operation::new("math_constants");
    let mut attrs = std::collections::HashMap::new();
    
    // Add mathematical constants as float attributes
    attrs.insert("pi".to_string(), Attribute::Float(std::f64::consts::PI));
    attrs.insert("e".to_string(), Attribute::Float(std::f64::consts::E));
    attrs.insert("sqrt_2".to_string(), Attribute::Float(std::f64::consts::SQRT_2));
    attrs.insert("ln_10".to_string(), Attribute::Float(std::f64::consts::LN_10));
    attrs.insert("ln_2".to_string(), Attribute::Float(std::f64::consts::LN_2));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    
    // Verify values are stored correctly
    match op.attributes.get("pi") {
        Some(Attribute::Float(val)) => {
            assert!((val - std::f64::consts::PI).abs() < f64::EPSILON);
        }
        _ => panic!("Expected PI float attribute"),
    }
    
    match op.attributes.get("e") {
        Some(Attribute::Float(val)) => {
            assert!((val - std::f64::consts::E).abs() < f64::EPSILON);
        }
        _ => panic!("Expected E float attribute"),
    }
}

/// Test 7: Module with inputs and outputs having identical names
#[test]
fn test_module_identical_input_output_names() {
    let mut module = Module::new("identical_names_module");
    
    let common_name = "shared_name".to_string();
    
    // Add input with common name
    module.inputs.push(Value {
        name: common_name.clone(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    // Add output with same name
    module.outputs.push(Value {
        name: common_name.clone(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, module.outputs[0].name);
    assert_eq!(module.inputs[0].name, "shared_name");
}

/// Test 8: Value with dimensions in decreasing order
#[test]
fn test_value_decreasing_dimension_order() {
    let decreasing_shapes = vec![
        vec![100, 50, 25, 10],
        vec![64, 32, 16, 8, 4, 2, 1],
        vec![1000, 500, 250, 125],
        vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ];
    
    for shape in decreasing_shapes.iter() {
        let value = Value {
            name: "decreasing_dims".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Verify dimensions are in decreasing order
        for i in 0..shape.len().saturating_sub(1) {
            assert!(shape[i] >= shape[i + 1]);
        }
        
        assert_eq!(value.shape, shape.clone());
    }
}

/// Test 9: Attribute with nested arrays containing mixed types
#[test]
fn test_attribute_mixed_type_nested_arrays() {
    let mixed_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("mixed".to_string()),
        ]),
        Attribute::Array(vec![
            Attribute::Bool(true),
            Attribute::Int(100),
        ]),
        Attribute::Array(vec![
            Attribute::String("another".to_string()),
            Attribute::Float(99.9),
        ]),
    ]);
    
    match mixed_nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
            
            // Check first nested array
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 3);
                    assert_eq!(inner[0], Attribute::Int(1));
                    assert_eq!(inner[1], Attribute::Float(2.5));
                    assert_eq!(inner[2], Attribute::String("mixed".to_string()));
                }
                _ => panic!("Expected nested array"),
            }
            
            // Check second nested array
            match &outer[1] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 2);
                    assert_eq!(inner[0], Attribute::Bool(true));
                    assert_eq!(inner[1], Attribute::Int(100));
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected outer Array"),
    }
}

/// Test 10: Compiler with model sizes that are powers of 2
#[test]
fn test_compiler_power_of_two_model_sizes() {
    let mut compiler = ImpulseCompiler::new();
    
    let power_of_2_sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    
    for size in power_of_2_sizes.iter() {
        let model = vec![0xAB; *size];
        
        let result = compiler.compile(&model, "cpu");
        // Should handle gracefully without panic
        match result {
            Ok(_) => (),
            Err(e) => {
                assert!(e.to_string().len() > 0);
            }
        }
    }
}