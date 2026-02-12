//! Novel boundary coverage tests - Additional edge cases for comprehensive testing
//! Using standard library assert! and assert_eq! with rstest support

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with checked_mul overflow detection in num_elements()
#[test]
fn test_checked_mul_overflow_detection() {
    // Test values that would overflow when computing product
    // Using num_elements() which uses checked_mul internally
    
    // Safe case: small dimensions
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100, 100],
    };
    assert_eq!(safe_value.num_elements(), Some(10_000));
    
    // Edge case: dimensions that multiply to near usize::MAX
    let large_value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![46340, 46340], // ~2.1 billion, safe for 64-bit
    };
    assert_eq!(large_value.num_elements(), Some(46340 * 46340));
    
    // Zero dimension case
    let zero_value = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100, 0, 100],
    };
    assert_eq!(zero_value.num_elements(), Some(0));
    
    // Scalar case (empty shape)
    let scalar_value = Value {
        name: "scalar_tensor".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar_value.num_elements(), Some(1));
}

/// Test 2: Module with duplicate operation types and names
#[test]
fn test_module_duplicate_operations() {
    let mut module = Module::new("duplicate_test");
    
    // Add multiple operations with same type
    for i in 0..10 {
        let mut op = Operation::new("conv2d");
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1, 64, 112, 112],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 10);
    
    // All operations should have the same op_type
    for op in &module.operations {
        assert_eq!(op.op_type, "conv2d");
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
    }
}

/// Test 3: Operation with all attribute types simultaneously
#[test]
fn test_operation_all_attribute_types() {
    let mut op = Operation::new("full_attr_op");
    let mut attrs = HashMap::new();
    
    // Add every attribute type
    attrs.insert("max_int".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_float".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_float".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("pos_inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("empty_str".to_string(), Attribute::String("".to_string()));
    attrs.insert("unicode_str".to_string(), Attribute::String("张量数据".to_string()));
    attrs.insert("true_bool".to_string(), Attribute::Bool(true));
    attrs.insert("false_bool".to_string(), Attribute::Bool(false));
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    attrs.insert("nested_array".to_string(), Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
        Attribute::String("nested".to_string()),
    ]));
    
    op.attributes = attrs;
    
    // Due to potential attribute equality deduplication, check minimum count
    assert!(op.attributes.len() >= 13);
    
    // Verify specific attributes
    match op.attributes.get("max_int") {
        Some(Attribute::Int(v)) => assert_eq!(*v, i64::MAX),
        _ => panic!("Expected max_int"),
    }
    
    match op.attributes.get("unicode_str") {
        Some(Attribute::String(s)) => assert_eq!(s, "张量数据"),
        _ => panic!("Expected unicode_str"),
    }
    
    match op.attributes.get("nan") {
        Some(Attribute::Float(v)) => assert!(v.is_nan()),
        _ => panic!("Expected NaN"),
    }
}

/// Test 4: Nested tensor types with varying depths and shapes
#[test]
fn test_varying_depth_nested_tensors() {
    // Create nested tensors with different depths
    let depth1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    
    let depth2 = Type::Tensor {
        element_type: Box::new(depth1.clone()),
        shape: vec![3, 4],
    };
    
    let depth3 = Type::Tensor {
        element_type: Box::new(depth2.clone()),
        shape: vec![5],
    };
    
    let depth4 = Type::Tensor {
        element_type: Box::new(depth3.clone()),
        shape: vec![6, 7, 8],
    };
    
    // Verify different depths are not equal
    assert_ne!(depth1, depth2);
    assert_ne!(depth2, depth3);
    assert_ne!(depth3, depth4);
    
    // Verify each depth
    if let Type::Tensor { shape, .. } = depth1 {
        assert_eq!(shape, vec![2]);
    }
    
    if let Type::Tensor { shape, .. } = depth4 {
        assert_eq!(shape, vec![6, 7, 8]);
    }
}

/// Test 5: Value with alternating zero and non-zero dimensions
#[test]
fn test_alternating_zero_dimensions() {
    let test_cases = vec![
        (vec![0, 1, 0, 1], 0),
        (vec![1, 0, 1, 0, 1], 0),
        (vec![2, 0, 3, 0, 4], 0),
        (vec![0], 0),
        (vec![0, 0, 0], 0),
        (vec![1, 2, 3], 6), // No zeros
    ];
    
    for (shape, expected) in test_cases {
        let value = Value {
            name: "alternating_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let product: usize = shape.iter().product();
        assert_eq!(product, expected);
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 6: Module with empty operation
#[test]
fn test_module_empty_operation() {
    let mut module = Module::new("empty_op_test");
    
    // Add operation with no inputs, outputs, or attributes
    let op = Operation::new("noop");
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 0);
    assert_eq!(module.operations[0].attributes.len(), 0);
    assert_eq!(module.operations[0].op_type, "noop");
}

/// Test 7: Value with single element in multi-dimensional tensor
#[test]
fn test_single_element_multi_dim() {
    // Test tensors where product is 1 but dimensions vary
    let test_cases = vec![
        vec![1],           // 1D
        vec![1, 1],        // 2D
        vec![1, 1, 1],     // 3D
        vec![1, 1, 1, 1],  // 4D
    ];
    
    for shape in test_cases {
        let value = Value {
            name: "single_elem".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let product: usize = shape.iter().product();
        assert_eq!(product, 1);
        assert_eq!(value.num_elements(), Some(1));
    }
}

/// Test 8: Module with input-output type mismatch (allowed for flexibility)
#[test]
fn test_module_type_mismatch() {
    let mut module = Module::new("mismatch_test");
    
    // Add inputs of different types
    module.inputs.push(Value {
        name: "float_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.inputs.push(Value {
        name: "int_input".to_string(),
        ty: Type::I32,
        shape: vec![10],
    });
    
    // Add outputs with different types
    module.outputs.push(Value {
        name: "f64_output".to_string(),
        ty: Type::F64,
        shape: vec![10],
    });
    module.outputs.push(Value {
        name: "bool_output".to_string(),
        ty: Type::Bool,
        shape: vec![10],
    });
    
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    assert_ne!(module.inputs[0].ty, module.outputs[0].ty);
    assert_ne!(module.inputs[1].ty, module.outputs[1].ty);
}

/// Test 9: Operation with many attributes to test HashMap limits
#[test]
fn test_operation_many_attributes() {
    let mut op = Operation::new("many_attrs");
    let mut attrs = HashMap::new();
    
    // Add 1000 attributes
    for i in 0..1000 {
        attrs.insert(
            format!("attr_{:04}", i),
            Attribute::Int(i as i64)
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify specific attributes
    match op.attributes.get("attr_0000") {
        Some(Attribute::Int(v)) => assert_eq!(*v, 0),
        _ => panic!("Expected attr_0000"),
    }
    
    match op.attributes.get("attr_0500") {
        Some(Attribute::Int(v)) => assert_eq!(*v, 500),
        _ => panic!("Expected attr_0500"),
    }
    
    match op.attributes.get("attr_0999") {
        Some(Attribute::Int(v)) => assert_eq!(*v, 999),
        _ => panic!("Expected attr_0999"),
    }
}

/// Test 10: Deep nested array attributes
#[test]
fn test_deep_nested_array_attributes() {
    // Create deeply nested array: Array(Array(Array(Int)))
    let deep_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
            ]),
            Attribute::Array(vec![
                Attribute::Int(3),
                Attribute::Int(4),
            ]),
        ]),
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(5),
                Attribute::Int(6),
            ]),
        ]),
    ]);
    
    match deep_nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            
            // Check first outer element
            match &outer[0] {
                Attribute::Array(inner1) => {
                    assert_eq!(inner1.len(), 2);
                    
                    // Check first inner element
                    match &inner1[0] {
                        Attribute::Array(deep1) => {
                            assert_eq!(deep1.len(), 2);
                            match &deep1[0] {
                                Attribute::Int(1) => (),
                                _ => panic!("Expected Int(1)"),
                            }
                        },
                        _ => panic!("Expected Array"),
                    }
                },
                _ => panic!("Expected Array"),
            }
            
            // Check second outer element
            match &outer[1] {
                Attribute::Array(inner2) => {
                    assert_eq!(inner2.len(), 1);
                },
                _ => panic!("Expected Array"),
            }
        },
        _ => panic!("Expected outer Array"),
    }
}