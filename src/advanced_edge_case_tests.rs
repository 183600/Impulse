//! Advanced edge case tests for the Impulse compiler
//! Testing boundary conditions and extreme scenarios

use impulse::ir::{Module, Operation, Value, Type, Attribute};
use std::collections::HashMap;

/// Test 1: Operations with maximum number of attributes
#[test]
fn test_operation_max_attributes() {
    let mut op = Operation::new("max_attr_test");
    
    // Add maximum possible attributes to test memory handling
    for i in 0..100_000 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    assert_eq!(op.attributes.len(), 100_000);
}

/// Test 2: Extremely deep nested tensor types
#[test]
fn test_extremely_deep_nested_tensors() {
    // Create a deeply nested tensor type to test recursion limits
    let mut current_type = Type::F32;
    for _ in 0..500 {  // Moderate depth to avoid stack overflow
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the structure can be cloned without issues
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 3: Values with maximum length names and shapes
#[rstest::rstest]
fn test_extreme_value_names(
    #[values("x".repeat(1), "x".repeat(1000), "x".repeat(10_000))] name: String
) {
    let value = Value {
        name: name.clone(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value.name, name);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 2, 3]);
}

/// Test 4: Tensor shapes that could cause overflow in size calculations
#[test]
fn test_potential_overflow_shapes() {
    // Test with large but realistic dimensions
    let large_shape = vec![100_000, 100_000];  // Would be 10 billion elements
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    assert_eq!(value.shape, vec![100_000, 100_000]);
    
    // Calculate using checked arithmetic to prevent overflow
    let mut product: Option<usize> = Some(1);
    for &dim in &value.shape {
        product = product.and_then(|p| p.checked_mul(dim));
    }
    
    assert!(product.is_some()); // Should not overflow in this case
}

/// Test 5: Zero-containing tensor shapes (common edge case)
#[rstest::rstest]
#[case(vec![], 1)]  // Scalar has 1 element
#[case(vec![0], 0)]  // Contains 0, so product is 0
#[case(vec![1, 1, 1], 1)]  // All ones
#[case(vec![2, 0, 5], 0)]  // Contains 0, so product is 0
#[case(vec![10, 20], 200)]  // Normal case
fn test_zero_containing_shapes(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let value = Value {
        name: "test_tensor".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let calculated_size: usize = value.shape.iter().product();
    assert_eq!(calculated_size, expected_size);
}

/// Test 6: Unicode and special character handling
#[test]
fn test_unicode_identifiers() {
    let unicode_cases = vec![
        ("tensor_名称_日本語", Type::F32),
        ("tensor_🚀_unicode", Type::I32),
        ("tensor_café_naïve", Type::F64),
        ("tensor_مرحبا_العالم", Type::I64),
        ("tensor_🎉_庆祝", Type::Bool),
    ];
    
    for (name, ty) in unicode_cases {
        let value = Value {
            name: name.to_string(),
            ty: ty.clone(),
            shape: vec![2, 3],
        };
        
        assert_eq!(value.name, name);
        assert_eq!(value.ty, ty);
        assert_eq!(value.shape, vec![2, 3]);
    }
}

/// Test 7: Extreme aspect ratio tensors (very wide or very tall)
#[test]
fn test_extreme_aspect_ratio_tensors() {
    let extreme_cases = vec![
        vec![1, 1_000_000],     // Very wide: 1 row, 1M cols
        vec![1_000_000, 1],     // Very tall: 1M rows, 1 col
        vec![10_000, 10_000],   // Square but large
        vec![2, 2, 2, 2, 250_000], // High dimensional with large last dim
    ];
    
    for shape in extreme_cases {
        let value = Value {
            name: "extreme_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        let total_size: usize = value.shape.iter().product();
        assert!(total_size > 0); // Should be calculable
    }
}

/// Test 8: Operations with extremely large number of inputs/outputs
#[test]
fn test_large_io_operations() {
    let mut op = Operation::new("large_io_op");
    
    // Add many inputs
    for i in 0..10_000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i % 100 + 1], // Varying small shapes
        });
    }
    
    // Add many outputs
    for i in 0..5_000 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![(i + 50) % 100 + 1], // Varying small shapes
        });
    }
    
    assert_eq!(op.inputs.len(), 10_000);
    assert_eq!(op.outputs.len(), 5_000);
    assert_eq!(op.op_type, "large_io_op");
}

/// Test 9: Special floating point attribute values
#[test]
fn test_special_float_attributes() {
    let special_floats = [
        (std::f64::INFINITY, "infinity"),
        (std::f64::NEG_INFINITY, "neg_infinity"),
        (-0.0, "negative_zero"),
        (std::f64::EPSILON, "epsilon"),
        (std::f64::consts::PI, "pi"),
        (std::f64::consts::E, "euler"),
    ];
    
    for (val, desc) in &special_floats {
        let attr = Attribute::Float(*val);
        
        match attr {
            Attribute::Float(retrieved_val) => {
                if val.is_infinite() {
                    assert!(retrieved_val.is_infinite(), "Testing {}", desc);
                    assert_eq!(val.is_sign_positive(), retrieved_val.is_sign_positive(), "Sign test for {}", desc);
                } else if val == &-0.0 {
                    assert!(retrieved_val == 0.0 && retrieved_val.is_sign_negative(), "Negative zero test for {}", desc);
                } else {
                    assert!((retrieved_val - val).abs() < f64::EPSILON, "Value test for {}: {} vs {}", desc, retrieved_val, val);
                }
            },
            _ => panic!("Expected Float attribute for {}", desc),
        }
    }
    
    // Test NaN specially since NaN != NaN
    let nan_attr = Attribute::Float(std::f64::NAN);
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan(), "NaN test failed"),
        _ => panic!("Expected Float attribute for NaN"),
    }
}

/// Test 10: Module with maximum complexity operations
#[test]
fn test_complex_module_structure() {
    let mut module = Module::new("complex_module");
    
    // Add operations with complex structures
    for op_idx in 0..1_000 {
        let mut op = Operation::new(&format!("complex_op_{}", op_idx));
        
        // Add complex inputs
        for inp_idx in 0..50 {
            op.inputs.push(Value {
                name: format!("inp_{}_{}", op_idx, inp_idx),
                ty: if inp_idx % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![op_idx % 10 + 1, inp_idx % 10 + 1],
            });
        }
        
        // Add complex outputs
        for out_idx in 0..25 {
            op.outputs.push(Value {
                name: format!("out_{}_{}", op_idx, out_idx),
                ty: if out_idx % 3 == 0 { Type::F64 } else if out_idx % 3 == 1 { Type::I64 } else { Type::Bool },
                shape: vec![op_idx % 5 + 1],
            });
        }
        
        // Add complex attributes
        let mut attrs = HashMap::new();
        for attr_idx in 0..10 {
            attrs.insert(
                format!("attr_{}_{}", op_idx, attr_idx),
                if attr_idx % 3 == 0 {
                    Attribute::Int((op_idx * 100 + attr_idx) as i64)
                } else if attr_idx % 3 == 1 {
                    Attribute::Float(((op_idx * 100 + attr_idx) as f64) * 0.1)
                } else {
                    Attribute::String(format!("value_{}_{}", op_idx, attr_idx))
                }
            );
        }
        op.attributes = attrs;
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1_000);
    assert_eq!(module.name, "complex_module");
    
    // Verify some specific operations still maintain correct structure
    assert_eq!(module.operations[0].op_type, "complex_op_0");
    assert_eq!(module.operations[999].op_type, "complex_op_999");
    assert_eq!(module.operations[500].op_type, "complex_op_500");
    
    // Verify first operation has correct structure
    assert_eq!(module.operations[0].inputs.len(), 50);
    assert_eq!(module.operations[0].outputs.len(), 25);
    assert_eq!(module.operations[0].attributes.len(), 10);
}