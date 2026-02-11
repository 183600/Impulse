//! Advanced comprehensive boundary tests
//! Focus on numerical edge cases, overflow detection, and type safety

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    compiler::Compiler,
};

/// Test 1: Values with extremely large dimensions approaching usize limits
#[test]
fn test_value_with_extreme_large_dimensions() {
    // Test with dimensions that multiply to a very large number (but safe for usize)
    let large_value = Value {
        name: "extreme_large_dim".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 100],  // 100 million elements
    };
    
    // Verify shape is preserved
    assert_eq!(large_value.shape.len(), 2);
    assert_eq!(large_value.shape[0], 1_000_000);
    assert_eq!(large_value.shape[1], 100);
    
    // Verify num_elements handles large values correctly
    assert_eq!(large_value.num_elements(), Some(100_000_000));
}

/// Test 2: Attribute with subnormal float values and extreme precision
#[test]
fn test_subnormal_and_extreme_float_attributes() {
    // Test subnormal floats (very small numbers close to zero)
    let subnormal_pos = Attribute::Float(f64::MIN_POSITIVE);
    let subnormal_neg = Attribute::Float(-f64::MIN_POSITIVE);
    
    // Test extreme precision values
    let very_small = Attribute::Float(1e-307);
    let very_large = Attribute::Float(1e+307);
    
    // Verify attributes are created correctly
    match subnormal_pos {
        Attribute::Float(val) => assert!(val > 0.0 && val < 1e-300),
        _ => panic!("Expected Float attribute"),
    }
    
    match subnormal_neg {
        Attribute::Float(val) => assert!(val < 0.0 && val > -1e-300),
        _ => panic!("Expected Float attribute"),
    }
    
    match very_small {
        Attribute::Float(val) => assert!(val > 0.0 && val < 1e-300),
        _ => panic!("Expected Float attribute"),
    }
    
    match very_large {
        Attribute::Float(val) => assert!(val > 1e+300),
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with operations containing Unicode and special character names
#[test]
fn test_module_with_unicode_operation_names() {
    let mut module = Module::new("unicode_module");
    
    // Test operations with Unicode and special characters
    let unicode_names = vec![
        "操作_中文",  // Chinese characters
        "операция",   // Russian characters
        "αβγδε",      // Greek letters
        "😀🎉",       // Emoji
        "test@#$%^&*()",  // Special characters
    ];
    
    for name in &unicode_names {
        let op = Operation::new(name);
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    // Verify each operation has the correct name
    for (i, expected_name) in unicode_names.iter().enumerate() {
        assert_eq!(module.operations[i].op_type, *expected_name);
    }
}

/// Test 4: Value with alternating dimension pattern to test edge case handling
#[test]
fn test_value_alternating_zero_dimension_pattern() {
    let patterns = vec![
        vec![1, 0, 1, 0, 1],      // Alternating 1 and 0
        vec![0, 1, 0, 1, 0],      // Starts with 0
        vec![2, 0, 2, 0, 2],      // Larger alternating values
        vec![1, 0, 1, 0, 1, 0, 1], // Longer pattern
    ];
    
    for pattern in patterns {
        let value = Value {
            name: "alternating_pattern".to_string(),
            ty: Type::F32,
            shape: pattern.clone(),
        };
        
        // Any pattern containing 0 should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
        assert_eq!(value.shape, pattern);
    }
}

/// Test 5: Attribute array with deeply nested structure and type variety
#[test]
fn test_deeply_nested_mixed_type_array_attribute() {
    // Create a complex nested array structure
    let nested_attr = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::Float(2.5),
                Attribute::String("nested".to_string()),
            ]),
            Attribute::Bool(true),
        ]),
        Attribute::Array(vec![
            Attribute::String("level2".to_string()),
        ]),
        Attribute::Int(42),
    ]);
    
    match nested_attr {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
            
            // Check first element is an array
            match &outer[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 3);
                    // Verify nested structure is preserved
                    match &inner[1] {
                        Attribute::Array(deep) => {
                            assert_eq!(deep.len(), 2);
                        }
                        _ => panic!("Expected deeply nested array"),
                    }
                }
                _ => panic!("Expected Array"),
            }
            
            // Check second element
            match &outer[1] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 1);
                }
                _ => panic!("Expected Array"),
            }
            
            // Check third element is Int
            match outer[2] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 6: Module with many consecutive operations of the same type
#[test]
fn test_module_consecutive_same_operations() {
    let mut module = Module::new("consecutive_ops");
    
    // Add many consecutive operations of the same type
    for _ in 0..1000 {
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: "output".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1000);
    
    // Verify all operations have the same type
    for op in &module.operations {
        assert_eq!(op.op_type, "add");
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
    }
}

/// Test 7: Value with extremely large but valid single dimension
#[test]
fn test_value_single_large_dimension() {
    // Test vectors with very large single dimension
    let large_1d_value = Value {
        name: "large_1d_vector".to_string(),
        ty: Type::F32,
        shape: vec![10_000_000],  // 10 million elements
    };
    
    assert_eq!(large_1d_value.shape.len(), 1);
    assert_eq!(large_1d_value.shape[0], 10_000_000);
    assert_eq!(large_1d_value.num_elements(), Some(10_000_000));
    
    // Test with even larger dimension
    let huge_1d_value = Value {
        name: "huge_1d_vector".to_string(),
        ty: Type::I64,
        shape: vec![100_000_000],  // 100 million elements
    };
    
    assert_eq!(huge_1d_value.shape.len(), 1);
    assert_eq!(huge_1d_value.shape[0], 100_000_000);
    assert_eq!(huge_1d_value.num_elements(), Some(100_000_000));
}

/// Test 8: Operation with maximum and minimum integer attributes
#[test]
fn test_operation_with_extreme_integer_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("extreme_int_op");
    let mut attrs = HashMap::new();
    
    // Add extreme integer values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("max_positive".to_string(), Attribute::Int(9_223_372_036_854_775_807));
    attrs.insert("min_negative".to_string(), Attribute::Int(-9_223_372_036_854_775_808));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    
    // Verify each extreme value is stored correctly
    match op.attributes.get("max_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected max_i64 attribute"),
    }
    
    match op.attributes.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected min_i64 attribute"),
    }
}

/// Test 9: Module with empty strings as names and attributes
#[test]
fn test_module_with_empty_string_names() {
    let mut module = Module::new("");
    
    // Add operation with empty name
    let mut op = Operation::new("");
    op.inputs.push(Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    op.outputs.push(Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    use std::collections::HashMap;
    let mut attrs = HashMap::new();
    attrs.insert("".to_string(), Attribute::String("".to_string()));
    op.attributes = attrs;
    
    module.add_operation(op);
    
    // Verify empty names are handled correctly
    assert_eq!(module.name, "");
    assert_eq!(module.operations[0].op_type, "");
    assert_eq!(module.operations[0].inputs[0].name, "");
    assert_eq!(module.operations[0].outputs[0].name, "");
    assert!(module.operations[0].attributes.contains_key(""));
}

/// Test 10: Compiler with repeated creation to test state management
#[test]
fn test_compiler_repeated_creation_state() {
    // Attempt multiple compiler creations to ensure it works correctly
    let mut compilers = Vec::new();
    
    for _ in 0..10 {
        let compiler = Compiler::new();
        compilers.push(compiler);
    }
    
    // Verify all compilers were created successfully
    assert_eq!(compilers.len(), 10);
}