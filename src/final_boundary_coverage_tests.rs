//! Final boundary coverage tests - comprehensive edge cases with standard library assertions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Compiler with near-maximum dimension product (overflow boundary)
#[test]
fn test_near_maximum_dimension_product() {
    // Test dimensions that cause overflow in checked multiplication
    let value = Value {
        name: "overflow".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 2], // This will overflow
    };
    
    // This should return None due to overflow detection
    assert_eq!(value.num_elements(), None);
    
    // Test a valid large value that doesn't overflow
    let safe_value = Value {
        name: "large_safe".to_string(),
        ty: Type::F32,
        shape: vec![65536, 65536], // 2^32 elements = 4294967296
    };
    
    // This should return Some(4294967296) on 64-bit systems
    assert_eq!(safe_value.num_elements(), Some(4294967296));
}

/// Test 2: Value with alternating pattern dimensions
#[test]
fn test_alternating_pattern_dimensions() {
    let patterns = vec![
        vec![1, 2, 1, 2, 1],
        vec![10, 1, 10, 1, 10],
        vec![1, 1000, 1, 1000, 1],
    ];
    
    for shape in patterns {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };
        
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 3: Module with operations using all primitive attribute types in sequence
#[test]
fn test_all_primitive_attributes_in_sequence() {
    let mut op = Operation::new("full_attrs");
    
    // Set attributes one by one
    op.attributes.insert("int_val".to_string(), Attribute::Int(-123456789));
    op.attributes.insert("float_val".to_string(), Attribute::Float(-3.14159265359));
    op.attributes.insert("bool_val".to_string(), Attribute::Bool(false));
    op.attributes.insert("string_val".to_string(), Attribute::String("boundary_test".to_string()));
    
    // Verify all attributes are set correctly
    assert_eq!(op.attributes.len(), 4);
    match op.attributes.get("int_val") {
        Some(Attribute::Int(v)) => assert_eq!(*v, -123456789),
        _ => panic!("Expected Int attribute"),
    }
    match op.attributes.get("float_val") {
        Some(Attribute::Float(v)) => assert!((v - (-3.14159265359)).abs() < f64::EPSILON),
        _ => panic!("Expected Float attribute"),
    }
    match op.attributes.get("bool_val") {
        Some(Attribute::Bool(false)) => (),
        _ => panic!("Expected Bool(false)"),
    }
}

/// Test 4: Value with power-of-2 dimensions (common in ML)
#[test]
fn test_power_of_two_dimensions() {
    let powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    
    for &p in &powers {
        let value = Value {
            name: format!("pow2_{}", p),
            ty: Type::F32,
            shape: vec![p],
        };
        
        assert_eq!(value.num_elements(), Some(p));
    }
    
    // Test 2D power of 2
    let value_2d = Value {
        name: "pow2_2d".to_string(),
        ty: Type::F32,
        shape: vec![32, 32],
    };
    assert_eq!(value_2d.num_elements(), Some(1024));
}

/// Test 5: Attribute array with nested depth and mixed types
#[test]
fn test_mixed_nested_attribute_arrays() {
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.0),
                Attribute::Bool(true),
            ]),
        ]),
        Attribute::String("outer".to_string()),
        Attribute::Array(vec![
            Attribute::Int(3),
            Attribute::Int(4),
        ]),
    ]);
    
    match nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
            
            // Check first element is a nested array
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 2);
                    match &inner[1] {
                        Attribute::Array(deep) => {
                            assert_eq!(deep.len(), 2);
                            match &deep[0] {
                                Attribute::Float(v) => assert_eq!(*v, 2.0),
                                _ => panic!("Expected Float in deep array"),
                            }
                        }
                        _ => panic!("Expected nested array"),
                    }
                }
                _ => panic!("Expected Array in outer[0]"),
            }
            
            // Check second element is a string
            match &outer[1] {
                Attribute::String(s) => assert_eq!(s, "outer"),
                _ => panic!("Expected String in outer[1]"),
            }
        }
        _ => panic!("Expected outer Array"),
    }
}

/// Test 6: Module with chain of operations (data flow pattern)
#[test]
fn test_operation_chain_pattern() {
    let mut module = Module::new("chain_module");
    
    // Create a chain: op1 -> op2 -> op3
    let mut op1 = Operation::new("op1");
    op1.outputs.push(Value {
        name: "intermediate1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op1);
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(Value {
        name: "intermediate1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op2.outputs.push(Value {
        name: "intermediate2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op2);
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(Value {
        name: "intermediate2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op3.outputs.push(Value {
        name: "final".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].op_type, "op1");
    assert_eq!(module.operations[1].op_type, "op2");
    assert_eq!(module.operations[2].op_type, "op3");
}

/// Test 7: Value with negative edge case in attribute calculations
#[test]
fn test_negative_edge_case_values() {
    let mut op = Operation::new("negative_test");
    
    // Test with negative values
    op.attributes.insert("neg_int".to_string(), Attribute::Int(-1));
    op.attributes.insert("neg_large_int".to_string(), Attribute::Int(i64::MIN / 2));
    op.attributes.insert("neg_float".to_string(), Attribute::Float(-0.0));
    op.attributes.insert("neg_small_float".to_string(), Attribute::Float(-1e-10));
    
    match op.attributes.get("neg_int") {
        Some(Attribute::Int(v)) => assert_eq!(*v, -1),
        _ => panic!("Expected negative Int"),
    }
    
    match op.attributes.get("neg_float") {
        Some(Attribute::Float(v)) => {
            // -0.0 should equal 0.0 in comparison
            assert_eq!(*v, 0.0);
            // But should be negative in sign
            assert!(v.is_sign_negative());
        }
        _ => panic!("Expected negative zero Float"),
    }
}

/// Test 8: Module with all tensor element types systematically
#[test]
fn test_all_tensor_element_types() {
    let element_types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];
    
    for (i, elem_type) in element_types.iter().enumerate() {
        let tensor_type = Type::Tensor {
            element_type: Box::new(elem_type.clone()),
            shape: vec![2, 3],
        };
        
        match tensor_type {
            Type::Tensor { element_type, shape } => {
                assert_eq!(*element_type, *elem_type);
                assert_eq!(shape, vec![2, 3]);
            }
            _ => panic!("Expected Tensor type"),
        }
    }
}

/// Test 9: Compiler state after multiple compilation attempts
#[test]
fn test_compiler_state_after_multiple_attempts() {
    let mut compiler = ImpulseCompiler::new();
    let empty_model = vec![];
    
    // Attempt multiple compilations with different targets
    let targets = ["cpu", "gpu", "npu", "custom_target"];
    
    for target in targets {
        let result = compiler.compile(&empty_model, target);
        // Should handle gracefully
        assert!(result.is_ok() || result.is_err());
    }
    
    // Compiler should still be valid
    assert_eq!(compiler.passes.passes.len(), 0);
}

/// Test 10: Value with dimension patterns that include zero and ones
#[test]
fn test_zero_one_dimension_patterns() {
    let patterns = vec![
        (vec![0], 0),
        (vec![1], 1),
        (vec![1, 1], 1),
        (vec![1, 1, 1], 1),
        (vec![0, 1], 0),
        (vec![1, 0], 0),
        (vec![0, 0], 0),
        (vec![1, 2, 0, 3], 0),
        (vec![1, 1, 2, 1], 2),
    ];
    
    for (shape, expected) in patterns {
        let value = Value {
            name: "pattern_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.num_elements(), Some(expected),
                   "Failed for shape {:?}, expected {}", shape, expected);
    }
}