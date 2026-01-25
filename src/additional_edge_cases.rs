//! Additional edge case tests for the Impulse compiler
//! Covering boundary conditions and edge cases not found in other test modules

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Testing values with extremely large integer values for tensor dimensions
#[test]
fn test_extremely_large_tensor_dimensions() {
    // Test tensor with very large dimensions (but not overflowing)
    let large_value = Value {
        name: "large_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 100_000],  // 100 billion elements if calculated
    };
    
    assert_eq!(large_value.shape, vec![1_000_000, 100_000]);
    let product: usize = large_value.shape.iter().product();
    assert_eq!(product, 100_000_000_000);  // 100 billion
}

/// Test 2: Testing tensor creation with zero elements in different configurations
#[rstest]
#[case(vec![], 1)] // scalar
#[case(vec![0], 0)] // contains 0
#[case(vec![5, 0, 10], 0)] // multiple dimensions with 0
#[case(vec![1, 1, 1, 0], 0)] // trailing zero
#[case(vec![0, 1, 1, 1], 0)] // leading zero
fn test_tensor_zero_element_cases(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let value = Value {
        name: "zero_test_tensor".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let calculated_size: usize = value.shape.iter().product();
    assert_eq!(calculated_size, expected_size);
}

/// Test 3: Testing deeply nested tensor types with maximum possible nesting
#[test]
fn test_extreme_tensor_type_nesting() {
    let mut current_type = Type::I32;
    
    // Create 150 levels of nesting (this should be well within memory limits but still substantial)
    for i in 0..150 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 10 + 1],  // Use variable shapes to make it more complex
        };
    }
    
    // Verify the construction worked
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
    
    // The type should be valid
    use crate::ir::TypeExtensions;
    assert!(current_type.is_valid_type());
}

/// Test 4: Testing operations with extreme numbers of attributes
#[test]
fn test_operation_with_maximum_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attr_op");
    let mut attrs = HashMap::new();
    
    // Add 50,000 attributes to test memory handling
    for i in 0..50_000 {
        attrs.insert(
            format!("attribute_{:05}", i),
            Attribute::Int(i as i64)
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 50_000);
    assert_eq!(op.op_type, "max_attr_op");
    
    // Verify a few specific attributes exist
    assert!(op.attributes.contains_key("attribute_00000"));
    assert!(op.attributes.contains_key("attribute_25000"));
    assert!(op.attributes.contains_key("attribute_49999"));
    
    // Check specific values
    if let Some(Attribute::Int(0)) = op.attributes.get("attribute_00000") {
        assert!(true); // Expected value found
    } else {
        panic!("Expected Int(0) for attribute_00000");
    }
    
    if let Some(Attribute::Int(49999)) = op.attributes.get("attribute_49999") {
        assert!(true); // Expected value found
    } else {
        panic!("Expected Int(49999) for attribute_49999");
    }
}

/// Test 5: Testing values with extremely long names
#[test]
fn test_extremely_long_names() {
    let extremely_long_name = "x".repeat(100_000); // 100k character name
    
    // Test with value
    let value = Value {
        name: extremely_long_name.clone(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    assert_eq!(value.name.len(), 100_000);
    assert_eq!(value.name, extremely_long_name);
    
    // Test with operation
    let op = Operation::new(&extremely_long_name);
    assert_eq!(op.op_type.len(), 100_000);
    assert_eq!(op.op_type, extremely_long_name);
    
    // Test with module
    let module = Module::new(&extremely_long_name);
    assert_eq!(module.name.len(), 100_000);
    assert_eq!(module.name, extremely_long_name);
}

/// Test 6: Testing modules with maximum operation capacity
#[test]
fn test_module_with_maximum_operations() {
    let mut module = Module::new("max_ops_test_module");
    
    // Add 100,000 operations to test memory and performance handling
    for i in 0..100_000 {
        let op_name = format!("operation_{:06}", i);
        let mut op = Operation::new(&op_name);
        
        // Add simple input and output
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100_000);
    assert_eq!(module.name, "max_ops_test_module");
    
    // Verify some operations exist
    assert_eq!(module.operations[0].op_type, "operation_000000");
    assert_eq!(module.operations[50_000].op_type, "operation_050000");
    assert_eq!(module.operations[99_999].op_type, "operation_099999");
}

/// Test 7: Testing floating-point special values in attributes
#[rstest]
#[case(std::f64::INFINITY)]
#[case(std::f64::NEG_INFINITY)]
#[case(std::f64::NAN)]
#[case(-0.0)]  // Negative zero
#[case(std::f64::EPSILON)]
fn test_special_floating_point_values_in_attributes(#[case] value: f64) {
    let attr = Attribute::Float(value);
    
    match attr {
        Attribute::Float(retrieved_value) => {
            if value.is_nan() {
                assert!(retrieved_value.is_nan());
            } else {
                // For infinity, negative zero, epsilon, and normal values
                if value.is_infinite() && retrieved_value.is_infinite() {
                    assert_eq!(value.is_sign_positive(), retrieved_value.is_sign_positive());
                } else if value == -0.0 {
                    assert!(retrieved_value == -0.0 || retrieved_value == 0.0); // Both representations are valid
                    assert!(1.0 / retrieved_value < 0.0); // Ensure it's negative zero
                } else if value == std::f64::EPSILON {
                    assert_eq!(retrieved_value, std::f64::EPSILON);
                } else {
                    assert_eq!(retrieved_value, value);
                }
            }
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 8: Testing complex nested array attributes
#[test]
fn test_complex_nested_array_attributes() {
    // Create a complex nested array structure
    let complex_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Array(vec![
            Attribute::Float(2.5),
            Attribute::Array(vec![
                Attribute::String("deeply_nested".to_string()),
                Attribute::Bool(true),
                Attribute::Int(42),
            ]),
            Attribute::Bool(false),
        ]),
        Attribute::String("top_level".to_string()),
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(100),
                Attribute::Int(200),
            ]),
            Attribute::Float(3.14159),
        ]),
    ]);
    
    // Verify the top-level structure
    if let Attribute::Array(top_level) = &complex_array {
        assert_eq!(top_level.len(), 4);
        
        // Verify first element
        if let Attribute::Int(1) = top_level[0] {
            assert!(true);
        } else {
            panic!("Expected Int(1) at index 0");
        }
        
        // Verify the complex nested structure
        if let Attribute::Array(second_level) = &top_level[1] {
            assert_eq!(second_level.len(), 3);
            
            if let Attribute::Float(val) = second_level[0] {
                assert!((val - 2.5).abs() < f64::EPSILON);
            } else {
                panic!("Expected Float(2.5) at index [1][0]");
            }
            
            // Check the third level
            if let Attribute::Array(third_level) = &second_level[1] {
                assert_eq!(third_level.len(), 3);
                
                if let Attribute::String(ref s) = third_level[0] {
                    assert_eq!(s, "deeply_nested");
                } else {
                    panic!("Expected String at [1][1][0]");
                }
            } else {
                panic!("Expected Array at [1][1]");
            }
        } else {
            panic!("Expected Array at index 1");
        }
    } else {
        panic!("Expected Array at top level");
    }
}

/// Test 9: Testing operations with maximum input/output combinations
#[test]
fn test_operation_with_maximum_io_combinations() {
    let mut op = Operation::new("max_io_combo");
    
    // Add 10,000 inputs
    for i in 0..10_000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i % 100 + 1], // Varying shapes
        });
    }
    
    // Add 5,000 outputs
    for i in 0..5_000 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F64,
            shape: vec![(i + 50) % 200 + 1], // Different varying shapes
        });
    }
    
    assert_eq!(op.inputs.len(), 10_000);
    assert_eq!(op.outputs.len(), 5_000);
    assert_eq!(op.op_type, "max_io_combo");
    
    // Verify some specific inputs and outputs
    assert_eq!(op.inputs[0].name, "input_0");
    assert_eq!(op.inputs[9999].name, "input_9999");
    assert_eq!(op.outputs[0].name, "output_0");
    assert_eq!(op.outputs[4999].name, "output_4999");
    
    // Verify types and shapes
    assert_eq!(op.inputs[0].ty, Type::F32);
    assert_eq!(op.outputs[0].ty, Type::F64);
    assert_eq!(op.inputs[0].shape, vec![1]);  // 0 % 100 + 1 = 1
    assert_eq!(op.outputs[0].shape, vec![51]); // (0 + 50) % 200 + 1 = 51
}

/// Test 10: Testing tensor types with extreme shape variations
#[test]
fn test_tensor_types_with_extreme_shape_variations() {
    // Create various tensor types with different extreme shape characteristics
    let tensor_cases = [
        // Very sparse shape (many 1s with occasional large numbers)
        Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![1, 1, 1, 100_000, 1, 1, 1, 1],
        },
        // Alternating large and small dimensions
        Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![1000, 1, 5000, 1, 100],
        },
        // All large dimensions
        Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![100, 100, 100],
        },
        // Shape with zeros
        Type::Tensor {
            element_type: Box::new(Type::Bool),
            shape: vec![10, 0, 20],
        }
    ];
    
    for (i, tensor_type) in tensor_cases.iter().enumerate() {
        // Test that each tensor can be cloned and compared
        let cloned = tensor_type.clone();
        assert_eq!(tensor_type, &cloned);
        
        // Test validation
        use crate::ir::TypeExtensions;
        assert!(tensor_type.is_valid_type());
        
        // Do specific checks based on index
        match i {
            0 => {
                if let Type::Tensor { ref shape, .. } = tensor_type {
                    assert_eq!(shape[3], 100_000);
                    assert_eq!(shape.len(), 8);
                }
            },
            3 => {
                // For the tensor with a zero in the shape
                if let Type::Tensor { ref shape, .. } = tensor_type {
                    assert!(shape.contains(&0));
                    
                    // Calculate product which should be 0 due to the zero
                    let product: usize = shape.iter().product();
                    assert_eq!(product, 0);
                }
            },
            _ => {} // Other cases just validated above
        }
    }
}