//! Additional boundary condition tests for the Impulse compiler
//! Focuses on mathematical edge cases and safety checks

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;

// Test 1: Comprehensive shape product calculations that test potential overflow boundaries
#[test]
fn test_comprehensive_shape_products() {
    // Test various shape configurations for mathematical correctness
    let test_cases = vec![
        (vec![], 1),                    // Scalar: empty shape, 1 element
        (vec![0], 0),                   // Zero in shape: 0 elements
        (vec![1], 1),                   // Single unit: 1 element
        (vec![1, 1, 1], 1),             // Multiple units: 1 element
        (vec![2, 3, 4], 24),            // Small positive numbers: 24 elements
        (vec![100, 100], 10_000),       // Medium numbers: 10k elements
        (vec![1000, 1000], 1_000_000),  // Large numbers: 1M elements
        (vec![0, 10, 100], 0),          // Contains zero: 0 elements
        (vec![5, 0, 20], 0),            // Zero in middle: 0 elements
    ];

    for (shape, expected_product) in test_cases {
        let value = Value {
            name: "test_value".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        let actual_product: usize = value.shape.iter().product();
        assert_eq!(actual_product, expected_product, "Shape {:?} should have {} elements", shape, expected_product);
        assert_eq!(value.shape, shape);
    }
}

// Test 2: Maximum valid usize tensor dimensions without causing overflow
#[test]
fn test_safe_large_tensor_dimensions() {
    // Use values that are large but won't cause overflow in multiplication
    // This tests that our system can handle large but reasonable tensor sizes
    let max_safe_dimension = ((std::usize::MAX as f64).sqrt()) as usize;
    let large_but_safe_shape = vec![max_safe_dimension, max_safe_dimension];
    
    let value = Value {
        name: "large_but_safe_tensor".to_string(),
        ty: Type::F32,
        shape: large_but_safe_shape,
    };
    
    assert_eq!(value.shape.len(), 2);
    assert_eq!(value.shape[0], max_safe_dimension);
    assert_eq!(value.shape[1], max_safe_dimension);
    
    // The multiplication should not panic
    let product: usize = value.shape.iter().product();
    assert!(product > 0); // Should be a large but valid number
}

// Test 3: Edge cases for tensor types with different primitive types
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_all_base_types_with_various_shapes(#[case] base_type: Type) {
    // Test each base type with different shapes
    let shapes = vec![
        vec![],           // Scalar
        vec![1],          // Single element
        vec![0],          // Zero-sized
        vec![5, 5],      // Square matrix
        vec![2, 3, 4],   // 3D tensor
        vec![100, 100],  // Large 2D
    ];

    for shape in shapes {
        let value = Value {
            name: format!("typed_tensor_{:?}", base_type),
            ty: base_type.clone(),
            shape: shape.clone(),
        };

        assert_eq!(value.ty, base_type);
        assert_eq!(value.shape, shape);
    }
}

// Test 4: Operations with zero, one, and many inputs/outputs
#[test]
fn test_operations_with_varying_io_counts() {
    // Test operation with no inputs or outputs
    let mut op_no_io = Operation::new("no_io_op");
    assert_eq!(op_no_io.op_type, "no_io_op");
    assert_eq!(op_no_io.inputs.len(), 0);
    assert_eq!(op_no_io.outputs.len(), 0);
    assert_eq!(op_no_io.attributes.len(), 0);

    // Test operation with single input/output
    let mut op_single = Operation::new("single_io_op");
    op_single.inputs.push(Value {
        name: "single_input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    op_single.outputs.push(Value {
        name: "single_output".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });

    assert_eq!(op_single.inputs.len(), 1);
    assert_eq!(op_single.outputs.len(), 1);
    assert_eq!(op_single.inputs[0].name, "single_input");
    assert_eq!(op_single.outputs[0].name, "single_output");

    // Test operation with many inputs/outputs
    let mut op_many = Operation::new("many_io_op");
    for i in 0..10 {
        op_many.inputs.push(Value {
            name: format!("input_{}", i),
            ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
            shape: vec![i + 1],
        });
        op_many.outputs.push(Value {
            name: format!("output_{}", i),
            ty: if i % 3 == 0 { Type::F64 } else { Type::I64 },
            shape: vec![i + 2],
        });
    }

    assert_eq!(op_many.inputs.len(), 10);
    assert_eq!(op_many.outputs.len(), 10);
    assert_eq!(op_many.op_type, "many_io_op");
}

// Test 5: Comprehensive attribute testing with all types and edge values
#[test]
fn test_comprehensive_attribute_edge_cases() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("comprehensive_attr_op");
    let mut attrs = HashMap::new();

    // Add attributes of all types with edge values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero_i64".to_string(), Attribute::Int(0));
    attrs.insert("positive_one".to_string(), Attribute::Int(1));
    attrs.insert("negative_one".to_string(), Attribute::Int(-1));

    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("zero_f64".to_string(), Attribute::Float(0.0));
    attrs.insert("epsilon_f64".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("infinity_f64".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_infinity_f64".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan_f64".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("neg_zero_f64".to_string(), Attribute::Float(-0.0));

    attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
    attrs.insert("single_char".to_string(), Attribute::String("x".to_string()));
    attrs.insert("long_string".to_string(), Attribute::String("a".repeat(10000)));

    attrs.insert("true_bool".to_string(), Attribute::Bool(true));
    attrs.insert("false_bool".to_string(), Attribute::Bool(false));

    // Create nested array attributes
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    attrs.insert("single_item_array".to_string(), Attribute::Array(vec![Attribute::Int(42)]));
    attrs.insert("multi_type_array".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.5),
        Attribute::String("three".to_string()),
        Attribute::Bool(true),
    ]));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 19); // Count all the attributes we added
    
    // Verify some specific attribute values
    if let Some(attr) = op.attributes.get("max_i64") {
        match attr {
            Attribute::Int(val) => assert_eq!(*val, i64::MAX),
            _ => panic!("Expected max_i64 to be Int"),
        }
    } else {
        panic!("max_i64 attribute not found");
    }

    if let Some(attr) = op.attributes.get("infinity_f64") {
        match attr {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected infinity_f64 to be Float"),
        }
    } else {
        panic!("infinity_f64 attribute not found");
    }
}

// Test 6: Tensor type comparisons with various nesting levels
#[test]
fn test_tensor_type_comparisons() {
    // Test various tensor type equivalences and differences
    let type1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    let type2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    let type3 = Type::Tensor {
        element_type: Box::new(Type::F64),  // Different element type
        shape: vec![2, 3],
    };
    
    let type4 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![3, 2],  // Different shape
    };

    assert_eq!(type1, type2);     // Same everything
    assert_ne!(type1, type3);     // Different element type
    assert_ne!(type1, type4);     // Different shape
    assert_ne!(type3, type4);     // Both different
}

// Test 7: Modules with different numbers of operations
#[test]
fn test_modules_with_varying_operation_counts() {
    // Test empty module
    let empty_module = Module::new("empty_module");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.name, "empty_module");

    // Test module with single operation
    let mut single_op_module = Module::new("single_op_module");
    single_op_module.add_operation(Operation::new("single_op"));
    assert_eq!(single_op_module.operations.len(), 1);
    assert_eq!(single_op_module.operations[0].op_type, "single_op");

    // Test module with multiple operations
    let mut multi_op_module = Module::new("multi_op_module");
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i + 1],
        });
        multi_op_module.add_operation(op);
    }
    assert_eq!(multi_op_module.operations.len(), 5);
    assert_eq!(multi_op_module.name, "multi_op_module");
    
    // Verify the operations maintain their data correctly
    for (i, op) in multi_op_module.operations.iter().enumerate() {
        assert_eq!(op.op_type, format!("op_{}", i));
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.inputs[0].name, format!("input_{}", i));
        assert_eq!(op.inputs[0].shape, vec![i + 1]);
    }
}

// Test 8: Edge cases for nested tensor types
#[test]
fn test_nested_tensor_edge_cases() {
    // Create a 5-level nested tensor: tensor<tensor<tensor<tensor<tensor<f32, [2]>, [3]>, [4]>, [5]>, [6]>
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![5],
    };
    let level5 = Type::Tensor {
        element_type: Box::new(level4),
        shape: vec![6],
    };

    // Verify the structure is maintained correctly
    if let Type::Tensor { element_type: box_lvl4, shape } = &level5 {
        assert_eq!(shape, &vec![6]);
        
        if let Type::Tensor { element_type: box_lvl3, shape } = box_lvl4.as_ref() {
            assert_eq!(shape, &vec![5]);
            
            if let Type::Tensor { element_type: box_lvl2, shape } = box_lvl3.as_ref() {
                assert_eq!(shape, &vec![4]);
                
                if let Type::Tensor { element_type: box_lvl1, shape } = box_lvl2.as_ref() {
                    assert_eq!(shape, &vec![3]);
                    
                    if let Type::Tensor { element_type: final_type, shape } = box_lvl1.as_ref() {
                        assert_eq!(shape, &vec![2]);
                        
                        if let Type::F32 = final_type.as_ref() {
                            // Success: verified the entire chain of nesting
                        } else {
                            panic!("Innermost type should be F32");
                        }
                    } else {
                        panic!("Expected level 1 to be a tensor");
                    }
                } else {
                    panic!("Expected level 2 to be a tensor");
                }
            } else {
                panic!("Expected level 3 to be a tensor");
            }
        } else {
            panic!("Expected level 4 to be a tensor");
        }
    } else {
        panic!("Expected level 5 to be a tensor");
    }

    // Verify cloning works for deeply nested types
    let cloned = level5.clone();
    assert_eq!(level5, cloned);
}

// Test 9: Special floating-point values in attribute context
#[test]
fn test_special_float_attributes() {
    let special_float_attrs = [
        ("pos_inf", Attribute::Float(f64::INFINITY)),
        ("neg_inf", Attribute::Float(f64::NEG_INFINITY)),
        ("nan_val", Attribute::Float(f64::NAN)),
    ];

    for (name, attr) in &special_float_attrs {
        match attr {
            Attribute::Float(f) => {
                match *name {
                    "pos_inf" => assert!(f.is_infinite() && f.is_sign_positive()),
                    "neg_inf" => assert!(f.is_infinite() && f.is_sign_negative()),
                    "nan_val" => assert!(f.is_nan()),
                    _ => panic!("Unexpected attribute name"),
                }
            },
            _ => panic!("Expected Float attribute"),
        }
    }
}

// Test 10: Zero-dimensional and zero-containing tensor shapes
#[test]
fn test_zero_tensor_shapes() {
    let zero_containing_cases = vec![
        vec![0],              // Scalar-like but zero
        vec![0, 10],         // Leading zero
        vec![10, 0],         // Trailing zero
        vec![5, 0, 7],       // Zero in middle
        vec![0, 0, 0],       // All zeros
        vec![1, 0, 1, 0],    // Scattered zeros
    ];

    for shape in zero_containing_cases {
        let value = Value {
            name: "zero_shape_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, shape);
        
        // Any shape containing 0 should result in 0 total elements
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements", shape);
    }

    // Also test standard scalar (empty shape)
    let scalar = Value {
        name: "scalar_tensor".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape = scalar
    };

    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.shape.len(), 0);
    
    // Scalar has 1 element (empty product is 1)
    let scalar_elements: usize = scalar.shape.iter().product();
    assert_eq!(scalar_elements, 1);
}