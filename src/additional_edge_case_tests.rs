//! Additional edge case tests for the Impulse compiler
//! Focuses on boundary conditions and special cases that might break the system

use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
use rstest::rstest;

// Test 1: Various tensor size calculations that could potentially cause overflow
#[test]
fn test_tensor_size_calculations() {
    // Test various combinations that result in the same product but with different shapes
    let cases = vec![
        (vec![1, 1000000], 1_000_000),       // 1 x 1M
        (vec![1000, 1000], 1_000_000),       // 1K x 1K
        (vec![100, 100, 100], 1_000_000),    // 100 x 100 x 100
        (vec![10, 10, 10, 1000], 1_000_000), // 10 x 10 x 10 x 1K
        (vec![0, 1000000], 0),               // Contains 0, so result is 0
        (vec![2, 0, 500000], 0),             // Another zero case
        (vec![], 1),                         // Scalar (empty shape)
    ];

    for (shape, expected_size) in cases {
        let value = Value {
            name: format!("tensor_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let actual_size: usize = value.shape.iter().product();
        assert_eq!(actual_size, expected_size, 
            "Shape {:?} should have {} elements, got {}", 
            shape, expected_size, actual_size);
    }
}

// Test 2: Deeply nested tensor types
#[test]
fn test_deeply_nested_tensor_types() {
    // Create a tensor type nested 10 levels deep
    let mut current_type = Type::F32;
    
    for depth in 0..10 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![depth + 1],  // Increasing shape size with depth
        };
    }
    
    // Verify the final nested structure
    assert!(current_type.is_valid_type());
    
    // Test cloning of deeply nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

// Test 3: Operations with extreme numbers of inputs and outputs
#[test]
fn test_operations_with_extreme_io_counts() {
    let mut op = Operation::new("extreme_io_op");
    
    // Add 5000 inputs to test memory handling
    for i in 0..5000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    // Add 2500 outputs
    for i in 0..2500 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(op.inputs.len(), 5000);
    assert_eq!(op.outputs.len(), 2500);
    assert_eq!(op.op_type, "extreme_io_op");
}

// Test 4: Edge cases for attribute values
#[test]
fn test_attribute_edge_cases() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("attr_edge_case_op");
    let mut attrs = HashMap::new();
    
    // Add various extreme attribute values
    attrs.insert("max_usize".to_string(), Attribute::Int(usize::MAX as i64));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("very_large_number".to_string(), Attribute::Int(999_999_999_999));
    
    // Special float values
    attrs.insert("pos_inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan_val".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    
    // Extremely long strings
    attrs.insert("long_string".to_string(), Attribute::String("a".repeat(50_000)));
    
    // Nested arrays
    attrs.insert("deeply_nested_array".to_string(), Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(42),
                Attribute::Float(3.14),
            ]),
        ]),
    ]));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 12);
    
    // Verify some specific values
    if let Some(attr) = op.attributes.get("pos_inf") {
        match attr {
            Attribute::Float(f) => assert!(f.is_infinite() && f.is_sign_positive()),
            _ => panic!("Expected positive infinity"),
        }
    } else {
        panic!("pos_inf attribute not found");
    }

    if let Some(attr) = op.attributes.get("nan_val") {
        match attr {
            Attribute::Float(f) => assert!(f.is_nan()),
            _ => panic!("Expected NaN value"),
        }
    } else {
        panic!("nan_val attribute not found");
    }
}

// Test 5: Modules with many operations to test memory management
#[test]
fn test_large_module_with_many_operations() {
    let mut module = Module::new("large_test_module");
    
    // Add 1000 operations to test memory handling
    for i in 0..1000 {
        let mut op = Operation::new(&format!("operation_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i % 10 + 1],  // Varying shape sizes
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![(i + 1) % 10 + 1],  // Varying shape sizes
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.name, "large_test_module");
    
    // Check first and last operations to ensure they were stored correctly
    assert_eq!(module.operations[0].op_type, "operation_0");
    assert_eq!(module.operations[999].op_type, "operation_999");
}

// Test 6: Parametrized test using rstest for different tensor types
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_all_base_types_with_extreme_shapes(#[case] base_type: Type) {
    // Test each base type with extreme shapes
    let extreme_shapes = vec![
        vec![],                           // Scalar
        vec![0],                          // Zero-dimension
        vec![0, 1000],                   // Zero followed by large number
        vec![1000, 0],                   // Large number followed by zero
        vec![1, 1, 1, 1, 1],            // Many small dimensions
        vec![10_000, 10_000],            // Large 2D tensor
        vec![100, 100, 100],             // Large 3D tensor
        vec![5, 5, 5, 5, 5],             // Multi-dimensional
    ];
    
    for shape in extreme_shapes {
        let value = Value {
            name: format!("extreme_{}_tensor", format!("{:?}", base_type).to_lowercase()),
            ty: base_type.clone(),
            shape: shape.clone(),
        };
        
        assert_eq!(value.ty, base_type);
        assert_eq!(value.shape, shape);
        
        // Calculate number of elements
        let num_elements: usize = value.shape.iter().product();
        if shape.contains(&0) {
            assert_eq!(num_elements, 0);
        } else if shape.is_empty() {
            assert_eq!(num_elements, 1); // Scalar
        }
    }
}

// Test 7: Very large tensor size calculations that approach usize limits
#[test]
fn test_near_limit_tensor_sizes() {
    // Test tensor shapes that result in very large but valid sizes
    // Using smaller values that won't cause overflow but still test large computations
    
    // A tensor that's 1000x1000 = 1,000,000 elements
    let large_tensor = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000],
    };
    
    let size = large_tensor.shape.iter().product::<usize>();
    assert_eq!(size, 1_000_000);
    
    // Another configuration: 100x100x100 = 1,000,000 elements
    let cube_tensor = Value {
        name: "cube_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100, 100, 100],
    };
    
    let size = cube_tensor.shape.iter().product::<usize>();
    assert_eq!(size, 1_000_000);
    
    // Test tensor approaching the limit where multiplication might overflow
    // Safe values that are large but won't actually overflow on most systems
    let near_limit_tensor = Value {
        name: "near_limit_tensor".to_string(),
        ty: Type::I64,
        shape: vec![46340, 46340],  // 46340^2 ≈ 2.1 billion, near u32::MAX
    };
    
    let size = near_limit_tensor.shape.iter().product::<usize>();
    assert_eq!(size, 46340 * 46340);
}

// Test 8: Extreme string lengths for value and operation names
#[test]
fn test_extremely_long_names() {
    // Test with very long names to check string handling
    let long_name = "a".repeat(50_000);  // 50k character name
    
    let value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value.name, long_name);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 2, 3]);
    
    // Test operation with long name
    let op = Operation::new(&long_name);
    assert_eq!(op.op_type, long_name);
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
    
    // Test module with long name
    let module = Module::new(long_name.clone());
    assert_eq!(module.name, long_name);
    assert_eq!(module.operations.len(), 0);
}

// Test 9: Special tensor shapes that are commonly problematic
#[test]
fn test_problematic_tensor_shapes() {
    let problematic_shapes = vec![
        vec![0],           // Zero dimension
        vec![0, 1],       // Zero followed by 1
        vec![1, 0],       // 1 followed by zero
        vec![0, 0],       // All zeros
        vec![0, 0, 0],    // All zeros (3D)
        vec![1, 0, 1],    // Mixed zero/one pattern
        vec![1, 1, 0, 1], // More complex mix
        vec![1],          // Single dimension
        vec![2],          // Single dimension with value > 1
        vec![1, 1],       // All ones in 2D
    ];
    
    for shape in problematic_shapes {
        let value = Value {
            name: format!("problematic_shape_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Calculate elements - if any dimension is 0, total should be 0
        let num_elements: usize = value.shape.iter().product();
        let contains_zero = shape.iter().any(|&d| d == 0);
        
        if contains_zero {
            assert_eq!(num_elements, 0);
        } else if shape.is_empty() {
            assert_eq!(num_elements, 1); // Scalar case
        } else {
            // Calculate expected value manually
            let expected = shape.iter().product::<usize>();
            assert_eq!(num_elements, expected);
        }
    }
}

// Test 10: Array attributes with extreme nesting and length
#[test]
fn test_array_attribute_extremes() {
    // Create a very long array
    let long_array_attr = Attribute::Array((0..10_000)
        .map(|i| Attribute::Int(i))
        .collect());
    
    match long_array_attr {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 10_000);
            
            // Verify first and last elements
            if let Attribute::Int(first) = arr[0] {
                assert_eq!(first, 0);
            } else {
                panic!("First element should be Int(0)");
            }
            
            if let Attribute::Int(last) = arr[9999] {
                assert_eq!(last, 9999);
            } else {
                panic!("Last element should be Int(9999)");
            }
        },
        _ => panic!("Expected Array attribute"),
    }
    
    // Create deeply nested arrays
    let mut nested_attr = Attribute::Int(42);
    
    // Nest it 10 levels deep
    for _ in 0..10 {
        nested_attr = Attribute::Array(vec![nested_attr]);
    }
    
    // Verify the nesting
    let mut current_attr = &nested_attr;
    for level in 0..10 {
        match current_attr {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
                current_attr = &arr[0];
            },
            _ => panic!("Expected Array at level {}", level),
        }
    }
    
    // At the innermost level, should be the original value
    match current_attr {
        Attribute::Int(42) => (),  // Success
        _ => panic!("Innermost value should be Int(42)"),
    }
}

// Test 11: Error conditions and invalid inputs
#[test]
fn test_error_conditions_and_invalid_inputs() {
    // Test creating values with various invalid or unexpected inputs
    // Note: The current implementation doesn't have explicit error conditions
    // but we can test edge cases that might lead to problems
    
    // Test with very large shape dimensions that might cause issues downstream
    let huge_shape_value = Value {
        name: "huge_shape".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 1000, 1000],  // Attempt to create a near-limit tensor
    };
    
    // The value should still be created successfully
    assert_eq!(huge_shape_value.name, "huge_shape");
    assert_eq!(huge_shape_value.ty, Type::F32);
    
    // Test creating operations with special character names
    let special_chars_name = "op_with_special_chars_!@#$%^&*()_+{}[]|\\:\";'<>?,./";
    let special_op = Operation::new(special_chars_name);
    assert_eq!(special_op.op_type, special_chars_name);
    
    // Test with Unicode characters
    let unicode_name = "operation_测试_тест_テスト";
    let unicode_op = Operation::new(unicode_name);
    assert_eq!(unicode_op.op_type, unicode_name);
    
    // Test creating values with Unicode names
    let unicode_value = Value {
        name: "value_测试_тест_テスト".to_string(),
        ty: Type::I64,
        shape: vec![1, 2, 3],
    };
    assert_eq!(unicode_value.name, "value_测试_тест_テスト");
    assert_eq!(unicode_value.ty, Type::I64);
    assert_eq!(unicode_value.shape, vec![1, 2, 3]);
}

// Test 12: Special floating-point values in attributes
#[test]
fn test_special_float_values() {
    let special_float_attributes = [
        ("pos_infinity", Attribute::Float(f64::INFINITY)),
        ("neg_infinity", Attribute::Float(f64::NEG_INFINITY)),
        ("nan_value", Attribute::Float(f64::NAN)),
        ("zero", Attribute::Float(0.0)),
        ("negative_zero", Attribute::Float(-0.0)),
        ("min_positive", Attribute::Float(f64::MIN_POSITIVE)),
        ("max", Attribute::Float(f64::MAX)),
        ("min", Attribute::Float(f64::MIN)),
        ("epsilon", Attribute::Float(f64::EPSILON)),
    ];
    
    for (name, attr) in special_float_attributes.iter() {
        match attr {
            Attribute::Float(f) => {
                match *name {
                    "pos_infinity" => assert!(f.is_infinite() && f.is_sign_positive(), 
                                   "Value should be positive infinity"),
                    "neg_infinity" => assert!(f.is_infinite() && f.is_sign_negative(),
                                   "Value should be negative infinity"), 
                    "nan_value" => assert!(f.is_nan(), "Value should be NaN"),
                    "zero" => assert!(*f == 0.0 && f.is_sign_positive(), 
                                   "Value should be positive zero"),
                    "negative_zero" => assert!(*f == 0.0 && f.is_sign_negative(),
                                    "Value should be negative zero"),
                    "min_positive" => assert!(*f == f64::MIN_POSITIVE, 
                                      "Value should be f64::MIN_POSITIVE"),
                    "max" => assert!(*f == f64::MAX, "Value should be f64::MAX"),
                    "min" => assert!(*f == f64::MIN, "Value should be f64::MIN"),
                    "epsilon" => assert!(*f == f64::EPSILON, 
                                  "Value should be f64::EPSILON"),
                    _ => panic!("Unknown attribute name: {}", name),
                }
            },
            _ => panic!("Expected Float attribute for {}", name),
        }
    }
}