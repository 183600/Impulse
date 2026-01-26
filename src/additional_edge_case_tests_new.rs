//! Additional edge case tests for the Impulse compiler
//! This file contains more tests covering boundary conditions and edge cases

use rstest::*;
use crate::ir::{Module, Value, Type, Operation, Attribute};

// Test 1: Floating point edge cases including infinity and NaN
#[test]
fn test_floating_point_edge_cases() {
    let special_values = [
        (f64::INFINITY, "positive_infinity"),
        (f64::NEG_INFINITY, "negative_infinity"),
        (f64::NAN, "not_a_number"),
        (-0.0, "negative_zero"),
        (f64::EPSILON, "epsilon"),
        (f64::MAX, "max_f64"),
        (f64::MIN, "min_f64"),
        (std::f64::consts::PI, "pi"),
        (std::f64::consts::E, "euler_number"),
    ];

    for (value, name) in &special_values {
        let attr = Attribute::Float(*value);
        
        match attr {
            Attribute::Float(retrieved_val) => {
                if value.is_nan() {
                    assert!(retrieved_val.is_nan(), "Value {} should be NaN", name);
                } else if value.is_infinite() {
                    assert!(retrieved_val.is_infinite(), "Value {} should be infinite", name);
                    assert_eq!(retrieved_val.is_sign_positive(), value.is_sign_positive(), 
                              "Sign mismatch for {}", name);
                } else {
                    // For finite values, check approximate equality
                    if (value - retrieved_val).abs() > f64::EPSILON {
                        assert!((value - retrieved_val).abs() <= f64::EPSILON, 
                               "Value {} mismatch: expected {}, got {}", name, value, retrieved_val);
                    }
                }
            },
            _ => panic!("Expected Float attribute for {}", name),
        }
    }
}

// Test 2: Extremely large tensor shapes that could cause overflow
#[test]
fn test_extremely_large_tensor_shape_products() {
    // Test cases that might cause overflow when calculating total elements
    let test_cases = vec![
        (vec![100_000, 100_000], 10_000_000_000 as usize),  // 10 billion elements
        (vec![1_000_000, 1_000], 1_000_000_000 as usize),    // 1 billion elements  
        (vec![46340, 46340], 2_147_395_600 as usize),       // Approaching 2^31
        (vec![10, 10, 10, 10, 10, 10], 1_000_000 as usize), // 10^6 elements
    ];
    
    for (shape, expected_product) in test_cases {
        let value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Calculate product safely
        let product_result: Option<usize> = value.shape.iter()
            .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
            
        match product_result {
            Some(product) => assert_eq!(product, expected_product),
            None => panic!("Product calculation overflowed unexpectedly for shape {:?}", shape),
        }
    }
}

// Test 3: Deeply nested type structures to test stack limits
#[test]
fn test_deeply_nested_tensor_types() {
    // Create a very deeply nested tensor type to test stack limits
    let mut current_type = Type::F32;
    const DEPTH: usize = 1000; // Deep nesting level
    
    for i in 0..DEPTH {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 10 + 1], // Varying shapes to make it more complex
        };
    }
    
    // Verify that the deeply nested type is still valid
    let is_valid = match &current_type {
        Type::Tensor { element_type: _, shape } => {
            !shape.is_empty() // At least has some shape
        },
        _ => false,
    };
    
    assert!(is_valid, "Deeply nested type should remain valid");
    
    // Test that we can clone this complex type without stack overflow
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

// Test 4: Memory allocation edge cases
#[test]
fn test_memory_allocation_edge_cases() {
    // Test with maximum possible operations in a module
    let mut module = Module::new("memory_test_module");
    
    // Add many operations to test memory behavior
    const NUM_OPERATIONS: usize = 50_000;
    for i in 0..NUM_OPERATIONS {
        let op = Operation::new(&format!("operation_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), NUM_OPERATIONS);
    assert_eq!(module.name, "memory_test_module");
    
    // Test with operations having many attributes (memory intensive)
    let mut heavy_op = Operation::new("heavy_memory_op");
    for i in 0..10_000 {
        heavy_op.attributes.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    assert_eq!(heavy_op.attributes.len(), 10_000);
    
    // Clean up to test memory deallocation
    drop(module);
    drop(heavy_op);
    assert!(true); // Just ensure no panic occurred during cleanup
}

// Test 5: Invalid UTF-8 and special character handling
#[test]
fn test_special_character_handling() {
    // Test with various string attributes containing special characters
    let test_strings = vec![
        "",                              // Empty string
        "regular_string",               // Regular ASCII
        "unicode_🚀_emoji",             // Unicode emoji
        "chinese_中文_字符",              // Chinese characters
        "arabic_مرحبا",                 // Arabic text
        "special_chars_!@#$%^&*()",     // Special symbols
        "\n\t\r\0\\\"\'",              // Escape sequences
        "mixed_混合_مزيج_🚀_✓",         // Mixed scripts and symbols
    ];
    
    for test_str in test_strings {
        let attr = Attribute::String(test_str.to_string());
        
        match attr {
            Attribute::String(retrieved_str) => {
                assert_eq!(retrieved_str, test_str, "String mismatch for: {}", test_str);
            },
            _ => panic!("Expected String attribute for: {}", test_str),
        }
    }
}

// Test 6: Zero dimension tensors and edge cases with zeros in dimensions
#[test]
fn test_zero_dimension_tensor_cases() {
    let zero_cases = vec![
        vec![],              // Scalar (0-dim tensor)
        vec![0],             // 1-dim tensor with 0 elements
        vec![0, 5],          // 2-dim tensor with 0 elements (0*5=0)
        vec![5, 0],          // 2-dim tensor with 0 elements (5*0=0)
        vec![0, 0],          // 2-dim tensor with 0 elements (0*0=0)
        vec![2, 0, 3],       // 3-dim tensor with 0 elements (2*0*3=0)
        vec![1, 1, 0, 1],    // 4-dim tensor with 0 elements
        vec![0, 1, 0, 1, 0], // 5-dim tensor all containing zeros
    ];
    
    for shape in zero_cases {
        let value = Value {
            name: "zero_test_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Calculate total elements
        let total_elements: usize = value.shape.iter().product();
        
        // All cases with 0 in them should result in 0 total elements
        // Only the empty shape (scalar) should result in 1 element
        if shape.is_empty() {
            assert_eq!(total_elements, 1, "Scalar should have 1 element, got {}", total_elements);
        } else {
            assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements, got {}", shape, total_elements);
        }
    }
}

// Test 7: Recursive operations and complex graph structures
#[test]
fn test_recursive_operation_structures() {
    // Create a complex operation with many nested elements
    let mut complex_op = Operation::new("complex_recursive_op");
    
    // Add many inputs and outputs
    for i in 0..1000 {
        complex_op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: match i % 5 {
                0 => Type::F32,
                1 => Type::F64,
                2 => Type::I32,
                3 => Type::I64,
                _ => Type::Bool,
            },
            shape: vec![i % 10 + 1, i % 5 + 1],
        });
        
        if i % 2 == 0 {  // Add fewer outputs than inputs
            complex_op.outputs.push(Value {
                name: format!("output_{}", i / 2),
                ty: match (i/2) % 5 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    3 => Type::I64,
                    _ => Type::Bool,
                },
                shape: vec![(i/2) % 8 + 1, (i/2) % 3 + 1],
            });
        }
    }
    
    assert_eq!(complex_op.inputs.len(), 1000);
    assert_eq!(complex_op.outputs.len(), 500); // Only added for even indices
    assert_eq!(complex_op.op_type, "complex_recursive_op");
    
    // Add complex attributes with nested structures
    use std::collections::HashMap;
    let mut complex_attrs = HashMap::new();
    
    // Add nested array attributes
    complex_attrs.insert("nested_array".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![
            Attribute::Float(2.5),
            Attribute::Array(vec![
                Attribute::String("deeply_nested".to_string()),
                Attribute::Bool(true),
            ]),
            Attribute::Int(3),
        ]),
        Attribute::Array(vec![
            Attribute::Bool(false),
            Attribute::Int(42),
        ]),
    ]));
    
    // Add multiple simple attributes
    for i in 0..100 {
        complex_attrs.insert(
            format!("simple_attr_{}", i),
            Attribute::Int(i as i64)
        );
    }
    
    complex_op.attributes = complex_attrs;
    
    assert_eq!(complex_op.attributes.len(), 101); // 100 simple + 1 nested array
}

// Test 8: Comprehensive attribute tests with all types and edge combinations
#[test]
fn test_comprehensive_attribute_combinations() {
    use std::collections::HashMap;
    
    // Create operation with all possible attribute types in various combinations
    let mut op = Operation::new("attribute_combo_op");
    let mut attrs = HashMap::new();
    
    // Add all basic attribute types
    attrs.insert("int_min".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("int_max".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("int_zero".to_string(), Attribute::Int(0));
    attrs.insert("int_negative".to_string(), Attribute::Int(-1000));
    attrs.insert("int_positive".to_string(), Attribute::Int(1000));
    
    attrs.insert("float_inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("float_neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("float_nan".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("float_epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("float_pi".to_string(), Attribute::Float(std::f64::consts::PI));
    attrs.insert("float_zero".to_string(), Attribute::Float(0.0));
    attrs.insert("float_negative".to_string(), Attribute::Float(-3.14159));
    
    attrs.insert("string_empty".to_string(), Attribute::String("".to_string()));
    attrs.insert("string_normal".to_string(), Attribute::String("normal_text".to_string()));
    attrs.insert("string_unicode".to_string(), Attribute::String("🚀unicode🌟".to_string()));
    attrs.insert("string_very_long".to_string(), Attribute::String("x".repeat(100_000)));
    
    attrs.insert("bool_true".to_string(), Attribute::Bool(true));
    attrs.insert("bool_false".to_string(), Attribute::Bool(false));
    
    // Add array attributes with mixed types
    attrs.insert("int_array".to_string(), Attribute::Array(vec![
        Attribute::Int(1), Attribute::Int(2), Attribute::Int(3)
    ]));
    
    attrs.insert("float_array".to_string(), Attribute::Array(vec![
        Attribute::Float(1.1), Attribute::Float(2.2), Attribute::Float(3.3)
    ]));
    
    attrs.insert("mixed_array".to_string(), Attribute::Array(vec![
        Attribute::Int(10),
        Attribute::Float(20.5),
        Attribute::String("mixed".to_string()),
        Attribute::Bool(true),
    ]));
    
    attrs.insert("nested_mixed_array".to_string(), Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1), Attribute::Int(2)
        ]),
        Attribute::Array(vec![
            Attribute::Float(3.14), Attribute::Bool(true)
        ])
    ]));
    
    op.attributes = attrs;
    
    // Verify all attributes are present
    assert_eq!(op.attributes.len(), 22); // 6 int + 7 float + 4 string + 2 bool + 3 array types
    
    // Verify specific attribute values
    assert_eq!(op.attributes.get("int_min"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("int_max"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("string_empty"), Some(&Attribute::String("".to_string())));
    assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
    
    // Verify float special values
    if let Some(Attribute::Float(val)) = op.attributes.get("float_inf") {
        assert!(val.is_infinite() && val.is_sign_positive());
    } else {
        panic!("Expected positive infinity");
    }
    
    if let Some(Attribute::Float(val)) = op.attributes.get("float_neg_inf") {
        assert!(val.is_infinite() && !val.is_sign_positive());
    } else {
        panic!("Expected negative infinity");
    }
    
    if let Some(Attribute::Float(val)) = op.attributes.get("float_nan") {
        assert!(val.is_nan());
    } else {
        panic!("Expected NaN");
    }
}

// Test 9: Module operations with extreme values and boundary conditions
#[test]
fn test_module_extreme_operations() {
    // Test module with many operations
    let mut large_module = Module::new("extreme_module");
    
    // Add operations with different characteristics
    for i in 0..25_000 {
        let mut op = Operation::new(&format!("extreme_op_{}", i));
        
        // Add varying numbers of inputs/outputs based on index
        let num_inputs = (i % 10) + 1;  // 1 to 10 inputs
        let num_outputs = (i % 5) + 1;  // 1 to 5 outputs
        
        for j in 0..num_inputs {
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: match (i + j) % 5 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    3 => Type::I64,
                    _ => Type::Bool,
                },
                shape: vec![(j + 1) % 5 + 1, (i + j) % 7 + 1],
            });
        }
        
        for j in 0..num_outputs {
            op.outputs.push(Value {
                name: format!("output_{}_{}", i, j),
                ty: match (i + j + 1) % 5 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    3 => Type::I64,
                    _ => Type::Bool,
                },
                shape: vec![(j + 2) % 5 + 1, (i + j + 1) % 7 + 1],
            });
        }
        
        large_module.add_operation(op);
    }
    
    assert_eq!(large_module.operations.len(), 25_000);
    assert_eq!(large_module.name, "extreme_module");
    
    // Verify some specific operations still have correct data
    let first_op = &large_module.operations[0];
    assert_eq!(first_op.op_type, "extreme_op_0");
    assert_eq!(first_op.inputs.len(), 1); // 0 % 10 + 1 = 1
    assert_eq!(first_op.outputs.len(), 1); // 0 % 5 + 1 = 1
    
    let mid_op = &large_module.operations[12_500];
    assert_eq!(mid_op.op_type, "extreme_op_12500");
    
    let last_op = &large_module.operations[24_999];
    assert_eq!(last_op.op_type, "extreme_op_24999");
    assert_eq!(last_op.inputs.len(), 10); // 24999 % 10 + 1 = 10
    assert_eq!(last_op.outputs.len(), 5); // 24999 % 5 + 1 = 5
}

// Test 10: Using rstest for cross-parameter testing of tensor properties
#[rstest]
#[case(vec![], 1, true)]           // Scalar: empty shape, 1 element, should be valid
#[case(vec![0], 0, true)]          // Contains 0: 0 elements, valid
#[case(vec![1], 1, true)]          // Single element: 1 element, valid
#[case(vec![2, 3], 6, true)]       // 2D: 6 elements, valid
#[case(vec![0, 5], 0, true)]       // Contains 0: 0 elements, valid
#[case(vec![100, 100], 10_000, true)] // Large 2D: 10k elements, valid
fn test_tensor_properties_comprehensive(
    #[case] shape: Vec<usize>, 
    #[case] expected_elements: usize, 
    #[case] should_be_valid: bool
) {
    let value = Value {
        name: "property_test".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape, shape);
    
    let actual_elements: usize = value.shape.iter().product();
    assert_eq!(actual_elements, expected_elements, "For shape {:?}, expected {} elements but got {}", 
               shape, expected_elements, actual_elements);
    
    if should_be_valid {
        // Additional validation for valid tensors
        // Note: usize is always >= 0, so we remove this redundant assertion
        // If no dimension is 0 and shape is not empty, elements should be > 0
        if !shape.is_empty() && !shape.iter().any(|&dim| dim == 0) {
            assert!(actual_elements > 0, "Non-zero dimensional tensor should have elements > 0");
        }
    }
}

// Test using rstest for testing different types with extreme shapes
#[rstest]
fn test_types_with_extreme_shapes(
    #[values(Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool)] data_type: Type,
    #[values(vec![], vec![0], vec![1_000_000], vec![1000, 1000])] shape: Vec<usize>
) {
    let value = Value {
        name: format!("{:?}_extreme_test", data_type).to_string(),
        ty: data_type.clone(),
        shape,
    };
    
    // Verify the value was created correctly
    assert_eq!(value.ty, data_type);
    
    // Calculate total elements
    let total_elements: usize = value.shape.iter().product();
    
    // If shape contains 0, total elements should be 0 (unless it's empty shape)
    if !value.shape.is_empty() && value.shape.iter().any(|&dim| dim == 0) {
        assert_eq!(total_elements, 0);
    } else if value.shape.is_empty() {
        // Empty shape means scalar, which has 1 element
        assert_eq!(total_elements, 1);
    } else {
        // For non-empty shapes without zeros, elements should be > 0
        assert!(total_elements > 0);
    }
}