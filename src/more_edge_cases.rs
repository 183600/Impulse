//! Additional edge case tests for Impulse compiler
//! Testing more boundary conditions and unusual scenarios

use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};

// Test 1: Value with maximum possible dimensions
#[test]
fn test_value_with_max_dimensions() {
    let max_dims = vec![usize::MAX, 1];  // Large dimension multiplied by 1
    let value = Value {
        name: "max_dims".to_string(),
        ty: Type::F32,
        shape: max_dims,
    };
    
    // Test that the shape is stored correctly
    assert_eq!(value.shape[0], usize::MAX);
    assert_eq!(value.shape[1], 1);
    
    // This would normally overflow, but we test if it's handled properly
    // Note: Actual product may wrap around depending on architecture
    let _product_result: Option<usize> = value.num_elements();
}

// Test 2: Operations with empty string names
#[test]
fn test_operations_with_empty_names() {
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    
    // Test value with empty name
    let value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    assert_eq!(value.name, "");
}

// Test 3: Nested tensors with complex nesting patterns
#[test]
fn test_complex_nested_tensor_patterns() {
    // Create a very complex nested pattern: Tensor<Tensor<Tensor<F32, [2]>, [3]>, [4]>
    let nested_type = Type::Tensor {
        element_type: Box::new(
            Type::Tensor {
                element_type: Box::new(
                    Type::Tensor {
                        element_type: Box::new(Type::F32),
                        shape: vec![2],
                    }
                ),
                shape: vec![3],
            }
        ),
        shape: vec![4],
    };
    
    // Verify the structure is maintained
    if let Type::Tensor { element_type: outer_elem, shape: outer_shape } = &nested_type {
        assert_eq!(outer_shape, &vec![4]);
        
        if let Type::Tensor { element_type: mid_elem, shape: mid_shape } = outer_elem.as_ref() {
            assert_eq!(mid_shape, &vec![3]);
            
            if let Type::Tensor { element_type: inner_elem, shape: inner_shape } = mid_elem.as_ref() {
                assert_eq!(inner_shape, &vec![2]);
                
                if let Type::F32 = inner_elem.as_ref() {
                    // Success - nested structure is intact
                } else {
                    panic!("Innermost type should be F32");
                }
            } else {
                panic!("Mid-level type should be Tensor");
            }
        } else {
            panic!("Outer-level type should be Tensor");
        }
    } else {
        panic!("Nested type should be Tensor");
    }
}

// Test 4: Testing recursive type validation with very deep nesting
#[test]
fn test_deep_recursion_type_validation() {
    let mut current_type = Type::F32;
    const DEPTH: usize = 50; // Not too deep to cause stack overflow but enough to test
    
    // Build nested structure
    for _ in 0..DEPTH {
        current_type = Type::Tensor {
            element_type: Box::new(current_type.clone()),
            shape: vec![2],
        };
    }
    
    // Validate the deeply nested type
    assert!(current_type.is_valid_type());
    
    // Clone it to ensure deep cloning works
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 5: Testing module serialization/deserialization with complex data
#[test]
fn test_module_serialization_complex_data() {
    use serde_json;
    
    let mut module = Module::new("complex_serialization_test");
    
    // Add operations with complex structures
    let mut op = Operation::new("complex_op");
    op.inputs.push(Value {
        name: "complex_input".to_string(),
        ty: Type::F32,
        shape: vec![100, 200, 50],
    });
    op.attributes.insert(
        "complex_attr".to_string(),
        Attribute::Array(vec![
            Attribute::String("nested".to_string()),
            Attribute::Int(42),
            Attribute::Float(3.14159),
        ])
    );
    
    module.add_operation(op);
    
    // Test serialization
    let serialized = serde_json::to_string(&module);
    assert!(serialized.is_ok());
    
    // Test deserialization
    let deserialized: Result<Module, _> = serde_json::from_str(&serialized.unwrap());
    assert!(deserialized.is_ok());
    
    let deserialized_module = deserialized.unwrap();
    assert_eq!(module.name, deserialized_module.name);
    assert_eq!(module.operations.len(), deserialized_module.operations.len());
}

// Test 6: Test value with extremely long shape vectors
#[test]
fn test_value_with_many_dimensions() {
    // Create a value with many dimensions (e.g., for representing complex data structures)
    let many_dims = vec![1; 100]; // 100 dimensions, each of size 1
    let value = Value {
        name: "many_dims".to_string(),
        ty: Type::I64,
        shape: many_dims,
    };
    
    assert_eq!(value.shape.len(), 100);
    assert_eq!(value.shape[0], 1);
    assert_eq!(value.shape[99], 1);
    
    // All dimensions are 1, so total should be 1
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 1);
}

// Test 7: Test operations with duplicate input/output names
#[test]
fn test_operations_with_duplicate_names() {
    let mut op = Operation::new("dup_names_op");
    
    // Add inputs with similar names
    op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op.inputs.push(Value {
        name: "input".to_string(), // Duplicate name
        ty: Type::I32,
        shape: vec![5],
    });
    
    // Add outputs with similar names  
    op.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![3],
    });
    op.outputs.push(Value {
        name: "output".to_string(), // Duplicate name
        ty: Type::F64,
        shape: vec![7],
    });
    
    // Should be allowed - the system shouldn't enforce uniqueness
    assert_eq!(op.inputs.len(), 2);
    assert_eq!(op.outputs.len(), 2);
    assert_eq!(op.inputs[0].name, "input");
    assert_eq!(op.inputs[1].name, "input");
    assert_eq!(op.outputs[0].name, "output");
    assert_eq!(op.outputs[1].name, "output");
}

// Test 8: Test attribute array with maximum recursion
#[test]
fn test_attribute_array_max_recursion() {
    let mut attr: Attribute = Attribute::Int(0);
    
    // Create nested arrays up to a reasonable depth
    for i in 1..20 {
        attr = Attribute::Array(vec![attr, Attribute::Int(i)]);
    }
    
    // Test that the nested structure is preserved
    match &attr {
        Attribute::Array(arr) => {
            assert!(!arr.is_empty());
            // The structure should be maintained regardless of depth
        },
        _ => panic!("Top level should be an Array"),
    }
    
    // Clone to test deep cloning works
    let cloned = attr.clone();
    assert_eq!(attr, cloned);
}

// Test 9: Test floating point precision edge cases in attributes
#[test]
fn test_floating_point_precision_edge_cases() {
    use std::f64;
    
    let float_attrs = [
        Attribute::Float(f64::consts::PI),      // Pi
        Attribute::Float(f64::consts::E),       // Euler's number
        Attribute::Float(f64::MIN_POSITIVE),   // Smallest positive value
        Attribute::Float(f64::MIN),            // Most negative value
        Attribute::Float(f64::MAX),            // Largest finite value
        Attribute::Float(1e-100),              // Very small positive number
        Attribute::Float(1e100),               // Very large positive number
        Attribute::Float(-1e100),              // Very large negative number
    ];
    
    for (i, attr) in float_attrs.iter().enumerate() {
        match attr {
            Attribute::Float(v) => {
                match i {
                    0 => assert!((v - f64::consts::PI).abs() < f64::EPSILON),
                    1 => assert!((v - f64::consts::E).abs() < f64::EPSILON),
                    2 => assert_eq!(*v, f64::MIN_POSITIVE),
                    3 => assert_eq!(*v, f64::MIN),
                    4 => assert_eq!(*v, f64::MAX),
                    5 => assert_eq!(*v, 1e-100),
                    6 => assert_eq!(*v, 1e100),
                    7 => assert_eq!(*v, -1e100),
                    _ => unreachable!(),
                }
            },
            _ => panic!("Expected Float attribute"),
        }
    }
}

// Test 10: Test operations with maximum variety of attribute types
#[test]
fn test_operation_with_full_attribute_variety() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("full_variety_op");
    let mut attrs = HashMap::new();
    
    // Add all types of attributes
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14159));
    attrs.insert("string_attr".to_string(), Attribute::String("hello world".to_string()));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
    
    // Add nested array
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.5),
        Attribute::String("nested".to_string()),
        Attribute::Bool(false),
        Attribute::Array(vec![
            Attribute::Int(10),
            Attribute::Int(20),
        ]),
    ]));
    
    // Add an empty array
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    
    op.attributes = attrs;
    
    // Verify all attributes are present
    assert_eq!(op.attributes.len(), 6);
    assert!(op.attributes.contains_key("int_attr"));
    assert!(op.attributes.contains_key("float_attr"));
    assert!(op.attributes.contains_key("string_attr"));
    assert!(op.attributes.contains_key("bool_attr"));
    assert!(op.attributes.contains_key("array_attr"));
    assert!(op.attributes.contains_key("empty_array"));
    
    // Verify specific values
    if let Some(Attribute::Int(42)) = op.attributes.get("int_attr") {
        // Correct
    } else {
        panic!("int_attr should be Int(42)");
    }
    
    if let Some(Attribute::Array(ref arr)) = op.attributes.get("array_attr") {
        assert_eq!(arr.len(), 5);
        if let Attribute::Array(ref inner_arr) = arr[4] {
            assert_eq!(inner_arr.len(), 2);
            if let Attribute::Int(10) = inner_arr[0] {
                // First element is correct
            } else {
                panic!("Inner array first element should be Int(10)");
            }
        } else {
            panic!("Fifth element should be an array");
        }
    } else {
        panic!("array_attr should be an Array with 5 elements");
    }
    
    // Test empty array
    if let Some(Attribute::Array(ref empty_arr)) = op.attributes.get("empty_array") {
        assert_eq!(empty_arr.len(), 0);
    } else {
        panic!("empty_array should be an empty Array");
    }
}