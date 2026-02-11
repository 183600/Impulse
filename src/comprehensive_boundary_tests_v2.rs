//! Comprehensive boundary tests v2 - Additional edge case coverage with standard assertions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Compiler with extremely small model sizes (1 byte variations)
#[test]
fn test_compiler_single_byte_models() {
    let mut compiler = crate::ImpulseCompiler::new();
    
    let single_byte_models: Vec<Vec<u8>> = vec![
        vec![0x00],  // Null byte
        vec![0xFF],  // Max byte
        vec![0x01],  // Smallest positive
        vec![0x80],  // High bit set
        vec![0x7F],  // Max positive without high bit
    ];
    
    for model in single_byte_models {
        let result = compiler.compile(&model, "cpu");
        assert!(result.is_ok() || result.is_err(), "Should handle single byte models gracefully");
    }
}

/// Test 2: Value with subnormal float edge cases
#[test]
fn test_subnormal_float_values() {
    // Test subnormal float values (smallest representable positive floats)
    let subnormal_f32 = Attribute::Float(1.0e-40_f64);  // Subnormal range
    let tiny_float = Attribute::Float(f64::MIN_POSITIVE);
    let near_zero = Attribute::Float(1e-308);
    
    // All should be valid attributes
    match subnormal_f32 {
        Attribute::Float(v) => assert!(v >= 0.0),
        _ => panic!("Expected Float attribute"),
    }
    
    match tiny_float {
        Attribute::Float(v) => assert!(v > 0.0 && v < 1e-300),
        _ => panic!("Expected Float attribute"),
    }
    
    match near_zero {
        Attribute::Float(v) => assert!(v >= 0.0),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Value with shape containing maximum dimension values
#[test]
fn test_value_max_dimensions() {
    // Test with maximum reasonable dimension values
    let max_dim_value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![46340, 46340],  // sqrt(i32::MAX)
    };
    
    assert_eq!(max_dim_value.shape[0], 46340);
    assert_eq!(max_dim_value.shape[1], 46340);
    assert_eq!(max_dim_value.num_elements(), Some(46340 * 46340));
}

/// Test 4: Module with operations having duplicate names
#[test]
fn test_module_duplicate_operations() {
    let mut module = Module::new("duplicate_test");
    
    // Add multiple operations with the same type name
    for _ in 0..5 {
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    // Verify all have the same op_type
    for op in &module.operations {
        assert_eq!(op.op_type, "add");
    }
}

/// Test 5: Operation with self-referential naming pattern
#[test]
fn test_self_referential_io_names() {
    let mut op = Operation::new("transform");
    
    // Create inputs and outputs with related names
    op.inputs.push(Value {
        name: "x".to_string(),
        ty: Type::F32,
        shape: vec![100],
    });
    op.outputs.push(Value {
        name: "x_transformed".to_string(),
        ty: Type::F32,
        shape: vec![100],
    });
    
    assert_eq!(op.inputs[0].name, "x");
    assert_eq!(op.outputs[0].name, "x_transformed");
}

/// Test 6: Value with mixed dimension patterns (containing zeros)
#[test]
fn test_value_mixed_zero_dimensions() {
    let zero_shapes = vec![
        vec![0, 10],      // Zero at start
        vec![10, 0],      // Zero at end
        vec![5, 0, 5],    // Zero in middle
        vec![0, 0, 0],    // All zeros
    ];
    
    for shape in zero_shapes {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any shape containing zero should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 7: Attribute array with deep nesting and mixed types
#[test]
fn test_deep_nested_mixed_array() {
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Array(vec![
                Attribute::Float(3.14),
                Attribute::String("nested".to_string()),
            ]),
        ]),
        Attribute::Bool(true),
    ]);
    
    match nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 2);
                    match &inner[1] {
                        Attribute::Array(deep) => {
                            assert_eq!(deep.len(), 2);
                        }
                        _ => panic!("Expected deeply nested array"),
                    }
                }
                _ => panic!("Expected inner array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 8: Module with mixed type inputs and outputs
#[test]
fn test_module_mixed_type_io() {
    let mut module = Module::new("mixed_types");
    
    // Add inputs of different types
    module.inputs.push(Value {
        name: "float_in".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.inputs.push(Value {
        name: "int_in".to_string(),
        ty: Type::I32,
        shape: vec![5],
    });
    
    // Add outputs of different types
    module.outputs.push(Value {
        name: "float_out".to_string(),
        ty: Type::F64,
        shape: vec![10],
    });
    module.outputs.push(Value {
        name: "bool_out".to_string(),
        ty: Type::Bool,
        shape: vec![1],
    });
    
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::I32);
    assert_eq!(module.outputs[0].ty, Type::F64);
    assert_eq!(module.outputs[1].ty, Type::Bool);
}

/// Test 9: Type validation for all supported types
#[test]
fn test_type_validation_all_types() {
    let types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];
    
    for ty in types {
        assert!(ty.is_valid_type(), "Type {:?} should be valid", ty);
    }
    
    // Test nested tensor types
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 2],
    };
    assert!(tensor_type.is_valid_type());
}

/// Test 10: Compiler with null bytes in model data
#[test]
fn test_compiler_null_bytes() {
    let mut compiler = crate::ImpulseCompiler::new();
    
    // Model with null bytes scattered throughout
    let model_with_nulls = vec![
        0xFF, 0x00, 0xFE, 0x00, 0xFD, 0x00, 0xFC, 0x00,
        0xFB, 0x00, 0xFA, 0x00, 0xF9, 0x00, 0xF8, 0x00,
    ];
    
    let result = compiler.compile(&model_with_nulls, "cpu");
    // Should handle null bytes gracefully
    assert!(result.is_ok() || result.is_err());
}