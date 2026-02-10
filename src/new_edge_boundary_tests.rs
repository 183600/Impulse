//! New edge boundary tests for Impulse compiler
//! Testing edge cases with focus on boundary conditions and unusual inputs

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

#[test]
fn test_value_with_nan_inf_float_attribute() {
    // Test NaN and Infinity float values in attributes
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float attribute"),
    }
    
    match pos_inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float attribute"),
    }
    
    match neg_inf_attr {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float attribute"),
    }
}

#[test]
fn test_module_with_circular_reference_names() {
    // Test module with operation names that suggest circular references
    let mut module = Module::new("circular_module");
    
    // Create operations with names that suggest circular dependency
    let mut op_a = Operation::new("op_a");
    op_a.outputs.push(Value {
        name: "output_a".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op_b = Operation::new("op_b");
    op_b.inputs.push(Value {
        name: "output_a".to_string(),  // References op_a's output
        ty: Type::F32,
        shape: vec![10],
    });
    op_b.outputs.push(Value {
        name: "output_b".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op_a);
    module.add_operation(op_b);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[1].op_type, "op_b");
}

#[test]
fn test_value_with_zero_indices_in_shape() {
    // Test that zero values in shape are handled (they represent empty dimensions)
    // In real IR, zero often means dynamic or empty dimension
    let shapes_with_zeros = vec![
        vec![10, 0],       // Empty dimension
        vec![0, 0],        // All empty
        vec![5, 10, 0],    // Mixed
    ];
    
    for shape in shapes_with_zeros {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        assert_eq!(value.shape, shape);
    }
}

#[test]
fn test_operation_with_empty_string_attributes() {
    // Test operation with various empty string edge cases
    let mut op = Operation::new("empty_strings");
    let mut attrs = HashMap::new();
    
    attrs.insert("empty".to_string(), Attribute::String("".to_string()));
    attrs.insert("spaces".to_string(), Attribute::String("   ".to_string()));
    attrs.insert("whitespace".to_string(), Attribute::String("\t\n\r ".to_string()));
    attrs.insert("unicode_empty".to_string(), Attribute::String("\u{200B}".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    
    match op.attributes.get("empty") {
        Some(Attribute::String(s)) => assert_eq!(s, ""),
        _ => panic!("Expected empty string"),
    }
    
    match op.attributes.get("spaces") {
        Some(Attribute::String(s)) => assert_eq!(s, "   "),
        _ => panic!("Expected spaces string"),
    }
}

#[test]
fn test_tensor_type_with_zero_dimension() {
    // Test tensor types that explicitly have zero dimensions
    let zero_dim_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![0],  // 0-dimensional in size
    };
    
    match zero_dim_tensor {
        Type::Tensor { element_type, shape } => {
            assert_eq!(*element_type, Type::F32);
            assert_eq!(shape, vec![0]);
        }
        _ => panic!("Expected Tensor type"),
    }
}

#[test]
fn test_module_with_single_bit_operations() {
    // Test operations with very small dimensions (1x1)
    let mut module = Module::new("bit_ops");
    
    for i in 0..5 {
        let mut op = Operation::new(&format!("bit_op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::Bool,
            shape: vec![1, 1],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::Bool,
            shape: vec![1, 1],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.inputs[0].ty, Type::Bool);
        assert_eq!(op.inputs[0].shape, vec![1, 1]);
    }
}

#[test]
fn test_attribute_with_max_min_values() {
    // Test attributes with platform-specific max/min values
    let mut attrs = HashMap::new();
    
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    
    // Test with values close to limits
    attrs.insert("near_max_i64".to_string(), Attribute::Int(i64::MAX - 1));
    attrs.insert("near_min_i64".to_string(), Attribute::Int(i64::MIN + 1));
    
    assert_eq!(attrs.len(), 6);
    
    match attrs.get("max_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected max i64"),
    }
    
    match attrs.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected min i64"),
    }
}

#[test]
fn test_value_with_very_large_dimension_count() {
    // Test value with many dimensions (stress test for shape handling)
    let mut shape = Vec::new();
    for i in 0..20 {
        shape.push(i + 1);  // [1, 2, 3, ..., 20]
    }
    
    let value = Value {
        name: "high_dim_tensor".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape.len(), 20);
    assert_eq!(value.shape[0], 1);
    assert_eq!(value.shape[19], 20);
}

#[test]
fn test_module_with_duplicate_value_names() {
    // Test module where multiple operations have values with same names
    let mut module = Module::new("duplicate_names");
    
    let mut op1 = Operation::new("op1");
    op1.inputs.push(Value {
        name: "shared_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op1.outputs.push(Value {
        name: "shared_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(Value {
        name: "shared_input".to_string(),  // Same name as op1's input
        ty: Type::F32,
        shape: vec![10],
    });
    op2.outputs.push(Value {
        name: "shared_output".to_string(),  // Same name as op1's output
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].inputs[0].name, "shared_input");
    assert_eq!(module.operations[1].inputs[0].name, "shared_input");
}

#[test]
fn test_nested_tensor_with_empty_element_shape() {
    // Test nested tensor where inner tensor has empty shape
    let inner_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![],  // Scalar tensor as element
    };
    
    let outer_tensor = Type::Tensor {
        element_type: Box::new(inner_tensor),
        shape: vec![10],  // 10 scalar tensors
    };
    
    match outer_tensor {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![10]);
            match element_type.as_ref() {
                Type::Tensor { shape: inner_shape, .. } => {
                    assert_eq!(inner_shape, &Vec::<usize>::new());
                }
                _ => panic!("Expected nested tensor"),
            }
        }
        _ => panic!("Expected outer tensor"),
    }
}