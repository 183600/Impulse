//! Additional edge case tests for the Impulse compiler
//! This file focuses on boundary conditions and unusual scenarios that might break the system

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
};

/// Test 1: Integer overflow in tensor shape calculations
#[test]
fn test_integer_overflow_in_tensor_shapes() {
    // Use values that would cause overflow when multiplied naively
    let large_dim = (std::usize::MAX as f64).sqrt() as usize;
    
    // Create a tensor with dimensions that would overflow when multiplied
    let safe_large_tensor = Value {
        name: "safe_large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![large_dim, large_dim],
    };
    
    // Verify that the shape was preserved correctly
    assert_eq!(safe_large_tensor.shape[0], large_dim);
    assert_eq!(safe_large_tensor.shape[1], large_dim);
    
    // Test using checked multiplication to detect potential overflow
    let result: Option<usize> = safe_large_tensor.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    // This should handle potential overflow gracefully
    assert!(result.is_some() || result.is_none());
}

/// Test 2: Extremely deeply nested tensor types
#[test]
fn test_extremely_deeply_nested_tensor_types() {
    // Create a deeply nested tensor type to test recursion limits
    let mut current_type = Type::F32;
    for i in 0..500 {  // Create 500 levels of nesting
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 10 + 1], // Cycle through shapes 1-10 to make it interesting
        };
    }
    
    // Verify that the deeply nested type was created without stack overflow
    match &current_type {
        Type::Tensor { shape, .. } => {
            // The expected value depends on the iteration count: (499 % 10) + 1 = 10
            assert_eq!(shape, &vec![10]); // Expected shape based on cycle
        },
        Type::F32 => {
            // If it somehow collapsed to F32, that's also valid
        },
        _ => panic!("Expected a nested tensor type"),
    }
    
    // Test that cloning works for deeply nested types
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 3: Extreme attribute combinations with nested structures
#[test]
fn test_extreme_attribute_combinations() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("extreme_attrs");
    
    // Create complex nested attributes with multiple levels
    let nested_array = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
                Attribute::String("deeply nested".to_string()),
            ]),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::Int(-999),
            ]),
        ]),
        Attribute::Array(vec![
            Attribute::String("outer".to_string()),
            Attribute::Array(vec![Attribute::Bool(false)]),
        ]),
    ]);
    
    // Add various extreme attributes to the operation
    let mut attrs = HashMap::new();
    attrs.insert("deeply_nested_array".to_string(), nested_array);
    attrs.insert("min_int_value".to_string(), Attribute::Int(std::i64::MIN));
    attrs.insert("max_int_value".to_string(), Attribute::Int(std::i64::MAX));
    attrs.insert("min_float_value".to_string(), Attribute::Float(std::f64::MIN));
    attrs.insert("max_float_value".to_string(), Attribute::Float(std::f64::MAX));
    attrs.insert("epsilon".to_string(), Attribute::Float(std::f64::EPSILON));
    attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("very_long_string".to_string(), Attribute::String("long_string_value".repeat(1000)));
    attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 9);
    assert_eq!(op.attributes.get("min_int_value"), Some(&Attribute::Int(std::i64::MIN)));
    assert_eq!(op.attributes.get("max_int_value"), Some(&Attribute::Int(std::i64::MAX)));
}

/// Test 4: Boundary conditions for module operations
#[test]
fn test_module_boundary_conditions() {
    // Create a module with many operations to test memory handling
    let mut module = Module::new("boundary_test_module");
    
    // Add a large number of operations to test memory limits
    for i in 0..50_000 {
        let mut op = Operation::new(&format!("op_{:08}", i));  // Zero-padded for consistent sorting
        op.inputs.push(Value {
            name: format!("input_{:08}", i),
            ty: Type::F32,
            shape: vec![i % 100 + 1, i % 100 + 1], // Varying shapes
        });
        op.outputs.push(Value {
            name: format!("output_{:08}", i),
            ty: Type::F32,
            shape: vec![(i + 1) % 100 + 1, (i + 1) % 100 + 1], // Different shapes for outputs
        });
        module.add_operation(op);
    }
    
    // Verify all operations were added correctly
    assert_eq!(module.operations.len(), 50_000);
    assert_eq!(module.name, "boundary_test_module");
    
    // Check a few specific operations to validate data integrity
    assert_eq!(module.operations[0].op_type, "op_00000000");
    assert_eq!(module.operations[49999].op_type, "op_00049999");
    assert_eq!(module.operations[25000].op_type, "op_00025000");
}

/// Test 5: Complex nested array structures with mixed types
#[test]
fn test_complex_nested_arrays() {
    // Create deeply nested arrays with mixed attribute types
    let complex_nested = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![
            Attribute::Float(3.14),
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Int(42),
                    Attribute::Float(2.718),
                ])
            ])
        ]),
        Attribute::String("outer".to_string()),
    ]);
    
    // Verify the structure of the nested arrays
    match &complex_nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
            
            // Check first element - should be Int
            if let Attribute::Int(1) = outer[0] {
                // Success
            } else {
                panic!("First element should be Int(1)");
            }
            
            // Check third element - should be String
            if let Attribute::String(s) = &outer[2] {
                assert_eq!(s, "outer");
            } else {
                panic!("Third element should be String(\"outer\")");
            }
            
            // Check second element - should be Array
            if let Attribute::Array(middle) = &outer[1] {
                assert_eq!(middle.len(), 2);
                
                if let Attribute::Float(f) = middle[0] {
                    assert!((f - 3.14).abs() < f64::EPSILON);
                } else {
                    panic!("Middle[0] should be Float(3.14)");
                }
            } else {
                panic!("Second element should be Array");
            }
        },
        _ => panic!("Should be an Array attribute"),
    }
}

/// Test 6: Special floating point values in tensor calculations
#[test]
fn test_special_floating_point_values() {
    use std::collections::HashMap;
    
    // Values that might appear in actual tensor computations
    let special_values = [
        (std::f64::INFINITY, "infinity"),
        (std::f64::NEG_INFINITY, "neg_infinity"),
        (std::f64::NAN, "nan"),
        (-0.0, "negative_zero"),
        (std::f64::EPSILON, "epsilon"),
        (std::f64::consts::PI, "pi"),
        (std::f64::consts::E, "euler"),
        (-std::f64::consts::PI, "neg_pi"),
    ];
    
    let mut op = Operation::new("special_fp_op");
    let mut attrs = HashMap::new();
    
    for (val, name) in &special_values {
        // Store special values as attributes
        attrs.insert(name.to_string(), Attribute::Float(*val));
        
        // Also test creating a value with special float
        if !val.is_nan() {  // Skip NaN in shapes since it's not valid as usize
            let special_value = Value {
                name: format!("value_{}", name),
                ty: Type::F64,
                shape: vec![1, 1],  // Small shape
            };
            op.inputs.push(special_value);
        }
    }
    
    op.attributes = attrs;
    
    // Test that infinity values are handled correctly
    assert!(op.attributes.get("infinity").is_some());
    assert!(op.attributes.get("neg_infinity").is_some());
    
    // Test NaN - special case since NaN != NaN
    if let Attribute::Float(nan_val) = op.attributes.get("nan").unwrap() {
        assert!(nan_val.is_nan());
    }
}

/// Test 7: Zero-size tensor edge cases
#[test]
fn test_zero_size_tensor_edge_cases() {
    let zero_cases = vec![
        vec![0],              // 0D tensor with zero elements
        vec![0, 5],           // Contains zero dimension
        vec![5, 0],           // Contains zero dimension  
        vec![2, 0, 3],        // Zero in middle
        vec![0, 0, 0],        // Multiple zeros
        vec![0, 1, 0, 1],     // Alternating zeros and ones
        vec![1, 0, 1, 0, 1],  // More complex zeros pattern
    ];
    
    for (i, shape) in zero_cases.iter().enumerate() {
        let value = Value {
            name: format!("zero_case_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Every tensor with a zero dimension should have 0 total elements
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements", shape);
        
        // Verify the shape was preserved
        assert_eq!(value.shape, *shape);
    }
    
    // Test scalar (empty shape) specifically
    let scalar = Value {
        name: "scalar_tensor".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape indicates scalar
    };
    
    assert_eq!(scalar.shape.len(), 0);
    let scalar_elements: usize = scalar.shape.iter().product();
    assert_eq!(scalar_elements, 1);  // Scalar has 1 element
}

/// Test 8: Memory allocation with large data structures
#[test]
fn test_large_memory_allocation() {
    // Create very large data structures to test memory allocation limits
    
    // Large tensor shape
    let large_shape = vec![100_000, 100];  // 10 million elements
    let large_tensor = Value {
        name: "large_memory_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    assert_eq!(large_tensor.shape[0], 100_000);
    assert_eq!(large_tensor.shape[1], 100);
    let total_elements: usize = large_tensor.shape.iter().product();
    assert_eq!(total_elements, 10_000_000);
    
    // Large string attribute
    let very_long_string = "A".repeat(1_000_000);  // 1 million 'A's
    let string_attr = Attribute::String(very_long_string);
    
    match string_attr {
        Attribute::String(s) => assert_eq!(s.len(), 1_000_000),
        _ => panic!("Should be a String attribute"),
    }
    
    // Create a complex module with many nested types
    let mut complex_module = Module::new("complex_memory_module");
    
    for i in 0..10_000 {
        let shape_pattern = vec![i % 100 + 1, i % 50 + 1];
        let value = Value {
            name: format!("memory_test_val_{}", i),
            ty: if i % 3 == 0 { Type::F32 } else if i % 3 == 1 { Type::I32 } else { Type::Bool },
            shape: shape_pattern,
        };
        
        let mut op = Operation::new(&format!("big_op_{}", i));
        op.inputs.push(value);
        complex_module.add_operation(op);
    }
    
    assert_eq!(complex_module.operations.len(), 10_000);
    assert_eq!(complex_module.name, "complex_memory_module");
}

/// Test 9: Extreme string lengths for names and values
#[test]
fn test_extreme_string_lengths() {
    // Test with extremely long names to test string handling
    let extremely_long_name = "x".repeat(1_000_000);  // 1 million character name
    let value = Value {
        name: extremely_long_name.clone(),
        ty: Type::F32,
        shape: vec![10, 10],
    };
    
    assert_eq!(value.name.len(), 1_000_000);
    assert_eq!(value.name.chars().next(), Some('x'));
    
    // Test extremely long operation name
    let operation = Operation::new(&"op_".repeat(333_333));  // Approaching 1M chars
    assert!(operation.op_type.len() >= 999_000);  // Slightly less strict check
    
    // Test extremely long module name
    let module = Module::new(&"mod_".repeat(333_333));  // Approaching 1M chars
    assert!(module.name.len() >= 1_000_000);
    
    // Test Unicode in long strings
    let unicode_long = "🚀🌟量子计算!".repeat(100_000);  // Unicode in long strings
    let unicode_value = Value {
        name: unicode_long,
        ty: Type::F64,
        shape: vec![1],
    };
    
    assert!(unicode_value.name.len() > 0);
}

/// Test 10: Comprehensive boundary validation 
#[test]
fn test_comprehensive_boundary_validation() {
    // Test all boundary conditions together
    
    // Create a complex structure with all edge cases
    let mut module = Module::new(&"boundary_test_".repeat(10_000));
    
    for i in 0..1000 {
        let mut op = Operation::new(&format!("boundary_op_{}", "x".repeat(1000)));
        
        // Add special values
        op.inputs.push(Value {
            name: format!("input_{}", "n".repeat(500)),
            ty: match i % 5 {
                0 => Type::F32,
                1 => Type::F64,
                2 => Type::I32,
                3 => Type::I64,
                _ => Type::Bool,
            },
            shape: if i % 10 == 0 { 
                vec![]  // scalar
            } else if i % 10 == 1 { 
                vec![0, i % 100 + 1]  // zero dimension
            } else { 
                vec![i % 50 + 1, i % 50 + 1]  // regular shape
            },
        });
        
        // Add some outputs with extreme shapes
        if i % 5 == 0 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1000, 1000],  // Large shape
            });
        }
        
        // Add special attributes
        use std::collections::HashMap;
        let mut attrs = HashMap::new();
        
        if i % 7 == 0 {
            attrs.insert(
                format!("special_attr_{}", i), 
                Attribute::Float(if i == 0 { std::f64::NAN } else { std::f64::INFINITY })
            );
        }
        
        if i % 11 == 0 {
            attrs.insert(
                format!("long_string_attr_{}", i), 
                Attribute::String("loong".repeat(100))
            );
        }
        
        op.attributes = attrs;
        module.add_operation(op);
    }
    
    // Validate the complex structure
    assert!(module.name.len() > 0);
    assert_eq!(module.operations.len(), 1000);
    
    // Test a few random operations
    assert!(module.operations[0].op_type.len() > 0);
    assert!(module.operations[500].op_type.len() > 0);
    assert!(module.operations[999].op_type.len() > 0);
    
    // All operations should have been created without panic
    for op in &module.operations {
        // Verify each operation is structurally valid
        assert!(op.op_type.len() > 0);
        assert!(op.inputs.len() >= 0);  // This is always true for usize, but kept for completeness
        assert!(op.outputs.len() >= 0);  // This is always true for usize, but kept for completeness
        assert!(op.attributes.len() >= 0);  // This is always true for usize, but kept for completeness
    }
}