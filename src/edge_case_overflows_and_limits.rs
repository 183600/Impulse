//! Additional edge case tests for the Impulse compiler
//! Focuses on numerical overflows, memory limits, and boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use rstest::rstest;

// Test 1: Overflow detection in shape product calculations using checked arithmetic
#[test]
fn test_shape_product_overflow_detection() {
    // Test the num_elements method which uses checked arithmetic
    // Use values that will definitely cause overflow on 64-bit systems
    let large_shape = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // These numbers when multiplied would definitely overflow
    };
    
    // Direct multiplication would overflow, but our method should handle it gracefully
    // We expect None when overflow occurs
    let maybe_num_elements = large_shape.num_elements();
    
    // The result should be None due to overflow
    assert!(maybe_num_elements.is_none());
    
    // Test with a safe large shape
    let safe_large_shape = Value {
        name: "safe_large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10_000, 10_000],
    };
    
    let safe_result = safe_large_shape.num_elements();
    assert_eq!(safe_result, Some(100_000_000));
}

// Test 2: Empty and zero-size validation edge cases
#[test]
fn test_edge_case_empty_and_zero_shapes() {
    // Test scalar (empty shape) - should have 1 element
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(scalar.num_elements(), Some(1));
    
    // Test tensor with zero in dimensions - should have 0 elements
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100, 0, 50],
    };
    
    assert_eq!(zero_tensor.shape, vec![100, 0, 50]);
    assert_eq!(zero_tensor.num_elements(), Some(0));
    
    // Test single zero dimension
    let single_zero = Value {
        name: "single_zero".to_string(),
        ty: Type::I32,
        shape: vec![0],
    };
    
    assert_eq!(single_zero.shape, vec![0]);
    assert_eq!(single_zero.num_elements(), Some(0));
}

// Test 3: Extreme nesting depth for recursive types using rstest
#[rstest]
#[case(1)]
#[case(10)]
#[case(50)]
fn test_deep_tensor_nesting(#[case] depth: usize) {
    let mut current_type = Type::F32;
    
    // Build a deeply nested type
    for _ in 0..depth {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the type was created properly
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![2]);
        },
        Type::F32 => {
            // Only valid if depth is 0, but we start with F32 so at least 1 level is added
            panic!("Expected nested tensor type");
        },
        _ => panic!("Unexpected type at maximum depth"),
    }
    
    // Ensure the validity check passes
    assert!(current_type.is_valid_type());
    
    // Test that cloning works at this depth
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 4: Memory-intensive operations stress test
#[test]
fn test_operations_with_large_numbers_of_inputs() {
    const NUM_INPUTS: usize = 10_000; // Large but reasonable number
    
    let mut op = Operation::new("stress_test_op");
    
    // Add many inputs to the operation
    for i in 0..NUM_INPUTS {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(op.inputs.len(), NUM_INPUTS);
    assert_eq!(op.op_type, "stress_test_op");
    
    // Verify we can access elements without issue
    assert_eq!(op.inputs[0].name, "input_0");
    assert_eq!(op.inputs[NUM_INPUTS - 1].name, format!("input_{}", NUM_INPUTS - 1));
}

// Test 5: String length limits and special character handling
#[test]
fn test_extremely_long_strings_in_names() {
    
    // Test extremely long operation name
    let long_name = "a".repeat(100_000);
    let op = Operation::new(&long_name);
    assert_eq!(op.op_type, long_name);
    
    // Test extremely long value name
    let long_value_name = "x".repeat(100_000);
    let value = Value {
        name: long_value_name.clone(),
        ty: Type::F32,
        shape: vec![1, 1],
    };
    
    assert_eq!(value.name, long_value_name);
    
    // Test with Unicode characters
    let unicode_name = "🚀".repeat(10_000) + &"🌍".repeat(10_000);
    let unicode_value = Value {
        name: unicode_name.clone(),
        ty: Type::F64,
        shape: vec![2, 2],
    };
    
    assert_eq!(unicode_value.name, unicode_name);
    assert_eq!(unicode_value.ty, Type::F64);
    assert_eq!(unicode_value.shape, vec![2, 2]);
}

// Test 6: Attribute edge cases with extreme values using rstest
#[rstest]
#[case(Attribute::Int(i64::MAX))]
#[case(Attribute::Int(i64::MIN))]
#[case(Attribute::Int(0))]
#[case(Attribute::Float(f64::MAX))]
#[case(Attribute::Float(f64::MIN))]
#[case(Attribute::Float(f64::EPSILON))]
#[case(Attribute::Float(-0.0))]
#[case(Attribute::Bool(true))]
#[case(Attribute::Bool(false))]
#[case(Attribute::String("".to_string()))]
#[case(Attribute::String("a".repeat(100_000)))]
fn test_extreme_attribute_values(#[case] attr: Attribute) {
    // Simply ensure we can create these extreme attributes without issues
    match &attr {
        Attribute::Int(val) => {
            assert!(val == &i64::MAX || val == &i64::MIN || val == &0);
        },
        Attribute::Float(val) => {
            assert!(val.is_finite() || val.is_infinite());
        },
        Attribute::String(s) => {
            assert!(s.is_empty() || s.len() >= 100_000);
        },
        Attribute::Bool(_) => {
            // Bool values are always valid
        },
        Attribute::Array(_) => {
            // Arrays handled separately
        },
    }
    
    // Test cloning of extreme values
    let cloned = attr.clone();
    assert_eq!(attr, cloned);
}

// Test 7: Array attribute with maximum nesting using recursive types
#[test]
fn test_deeply_nested_arrays() {
    // Build a deeply nested array structure
    let mut current_array = Attribute::Array(vec![]);
    
    // Nest 10 levels deep
    for _ in 0..10 {
        current_array = Attribute::Array(vec![current_array]);
    }
    
    // Now add actual values at the deepest level
    let deep_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14159),
        Attribute::String("deep_value".to_string()),
    ]);
    
    // Create a moderate-depth nested array with real values
    let nested_structure = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                deep_array
            ])
        ])
    ]);
    
    // Validate structure integrity
    match &nested_structure {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 1);
            match &outer[0] {
                Attribute::Array(middle) => {
                    assert_eq!(middle.len(), 1);
                    match &middle[0] {
                        Attribute::Array(inner) => {
                            assert_eq!(inner.len(), 1);
                            match &inner[0] {
                                Attribute::Array(deep_vals) => {
                                    assert_eq!(deep_vals.len(), 3);
                                    // Verify the values
                                    if let Attribute::Int(val) = &deep_vals[0] {
                                        assert_eq!(*val, 42);
                                    } else {
                                        panic!("Expected Int(42)");
                                    }
                                    
                                    if let Attribute::Float(val) = &deep_vals[1] {
                                        assert!((val - 3.14159).abs() < f64::EPSILON);
                                    } else {
                                        panic!("Expected Float(3.14159)");
                                    }
                                    
                                    if let Attribute::String(val) = &deep_vals[2] {
                                        assert_eq!(val, "deep_value");
                                    } else {
                                        panic!("Expected String(\"deep_value\")");
                                    }
                                },
                                _ => panic!("Expected Array with values"),
                            }
                        },
                        _ => panic!("Expected Array at inner level"),
                    }
                },
                _ => panic!("Expected Array at middle level"),
            }
        },
        _ => panic!("Expected Array at outer level"),
    }
}

// Test 8: Module creation with extreme number of operations
#[test]
fn test_module_with_extremely_large_number_of_operations() {
    const NUM_OPS: usize = 50_000; // Very large but manageable number
    
    let mut module = Module::new("extremely_large_module");
    
    // Add a very large number of operations
    for i in 0..NUM_OPS {
        let mut op = Operation::new(&format!("operation_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}_", i),
            ty: Type::F32,
            shape: vec![1, 1],
        });
        
        op.outputs.push(Value {
            name: format!("output_{}_", i),
            ty: Type::F32,
            shape: vec![1, 1],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), NUM_OPS);
    assert_eq!(module.name, "extremely_large_module");
    
    // Verify we can access first, middle, and last operations
    assert_eq!(module.operations[0].op_type, "operation_0");
    assert_eq!(module.operations[NUM_OPS/2].op_type, format!("operation_{}", NUM_OPS/2));
    assert_eq!(module.operations[NUM_OPS-1].op_type, format!("operation_{}", NUM_OPS-1));
    
    // Verify input/output names are preserved
    assert_eq!(module.operations[0].inputs[0].name, "input_0_");
    assert_eq!(module.operations[NUM_OPS-1].outputs[0].name, format!("output_{}_", NUM_OPS-1));
}

// Test 9: Special floating point values in computations
#[test]
fn test_special_float_behavior_in_attributes() {
    let special_attrs = vec![
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::Float(f64::NAN),
        Attribute::Float(f64::EPSILON),
        Attribute::Float(f64::MAX),
        Attribute::Float(f64::MIN),
    ];
    
    // Test each special value
    for (i, attr) in special_attrs.iter().enumerate() {
        if let Attribute::Float(val) = attr {
            match i {
                0 => assert!(val.is_infinite() && val.is_sign_positive()),
                1 => assert!(val.is_infinite() && val.is_sign_negative()),
                2 => assert!(val.is_nan()),
                3 => assert_eq!(*val, f64::EPSILON),
                4 => assert_eq!(*val, f64::MAX),
                5 => assert_eq!(*val, f64::MIN),
                _ => panic!("Unexpected index"),
            }
        } else {
            panic!("Expected Float attribute");
        }
    }
    
    // Test that regular float values compare correctly
    let float1 = Attribute::Float(3.14);
    let float2 = Attribute::Float(3.14);
    assert_eq!(float1, float2);
    
    // Test that infinity values compare correctly
    let inf1 = Attribute::Float(f64::INFINITY);
    let inf2 = Attribute::Float(f64::INFINITY);
    assert_eq!(inf1, inf2);
    
    let neg_inf1 = Attribute::Float(f64::NEG_INFINITY);
    let neg_inf2 = Attribute::Float(f64::NEG_INFINITY);
    assert_eq!(neg_inf1, neg_inf2);
    
    // Note: NaN != NaN according to IEEE 754 standard
    // So Attribute::Float(NaN) != Attribute::Float(NaN) which is expected behavior
    let nan1 = Attribute::Float(f64::NAN);
    let nan2 = Attribute::Float(f64::NAN);
    assert_ne!(nan1, nan2);  // This is expected behavior for NaN
    
    // But a value should equal itself when compared to its clone (though this still follows f64 rules)
    let nan_attr = Attribute::Float(f64::NAN);
    let cloned_nan = nan_attr.clone();
    assert_ne!(nan_attr, cloned_nan);  // Even cloned NaNs are not equal
}

// Test 10: Boundary values for all primitive types in tensors
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_all_primitive_types_in_tensor_context(#[case] base_type: Type) {
    // Test each primitive type as the base type for various tensor configurations
    let test_shapes = vec![
        vec![],                // Scalar
        vec![0],               // Zero-dimension
        vec![1],               // Single element
        vec![2, 0],            // Contains zero
        vec![3, 4],            // 2D tensor
        vec![2, 3, 4, 5],      // 4D tensor
    ];
    
    for shape in test_shapes {
        let value = Value {
            name: format!("tensor_{:?}_shape{:?}", base_type, shape).to_string(),
            ty: base_type.clone(),
            shape: shape.clone(),
        };
        
        assert_eq!(value.ty, base_type);
        assert_eq!(value.shape, shape);
        
        // Test that element count calculation works
        let elem_count = value.num_elements();
        if shape.iter().any(|&dim| dim == 0) {
            // If any dimension is 0, total elements should be 0
            assert_eq!(elem_count, Some(0));
        } else if shape.is_empty() {
            // If shape is empty (scalar), total elements should be 1
            assert_eq!(elem_count, Some(1));
        } else {
            // Otherwise, should multiply all dimensions
            let expected: usize = shape.iter().product();
            assert_eq!(elem_count, Some(expected));
        }
    }
}