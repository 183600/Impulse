//! Additional edge case tests for the Impulse compiler
//! Focuses on boundary conditions, numerical edge cases, and edge scenarios

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;

// Test 1: Value with maximum possible shape dimensions (extremely deep tensor)
#[test]
fn test_maximum_shape_dimensions() {
    let max_dims = Value {
        name: "max_dims_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1; 1000], // 1000 dimensions, each of size 1
    };
    
    assert_eq!(max_dims.shape.len(), 1000);
    assert_eq!(max_dims.name, "max_dims_tensor");
    assert_eq!(max_dims.ty, Type::F32);
    
    // The product should be 1 regardless of how many dimensions we have
    let product: usize = max_dims.shape.iter().product();
    assert_eq!(product, 1);
}

// Test 2: Operations with maximum number of attributes
#[test]
fn test_operation_maximum_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attr_op");
    let mut attrs = HashMap::new();
    
    // Add 10000 different attributes to test the limits
    for i in 0..10000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::Int(i as i64)
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10000);
    assert_eq!(op.op_type, "max_attr_op");
    
    // Verify a few random attributes exist
    assert!(op.attributes.contains_key("attr_0"));
    assert!(op.attributes.contains_key("attr_5000"));
    assert!(op.attributes.contains_key("attr_9999"));
}

// Test 3: Nested tensor type with maximum depth (without causing stack overflow)
#[test]
fn test_maximum_nested_tensor_depth() {
    let mut current_type = Type::F32;
    
    // Create 1000 levels of nesting to test deep recursion limits
    for _ in 0..1000 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1],
        };
    }
    
    // Verify the final type is still a tensor with shape [1]
    if let Type::Tensor { shape, .. } = &current_type {
        assert_eq!(shape, &vec![1]);
    } else {
        panic!("Expected tensor type after deep nesting");
    }
    
    // Verify it can be cloned
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 4: Test with all possible combinations of primitive types
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_all_primitive_types_with_complex_shapes(#[case] primitive_type: Type) {
    let shapes_to_test = vec![
        vec![],              // Scalar
        vec![0],             // Zero-size
        vec![1],             // Unit
        vec![1, 1, 1, 1],    // Multi-dimensional unit
        vec![1000, 1000],    // Large 2D
        vec![2, 3, 4, 5, 6], // High-dimension
        vec![0, 100, 200],   // Contains zero
    ];
    
    for shape in shapes_to_test {
        let value = Value {
            name: format!("{:?}_tensor", primitive_type),
            ty: primitive_type.clone(),
            shape: shape.clone(),
        };
        
        assert_eq!(value.ty, primitive_type);
        assert_eq!(value.shape, shape);
        
        if shape.contains(&0) {
            let elements: usize = value.shape.iter().product();
            assert_eq!(elements, 0, "Value with zero in shape should have 0 elements");
        }
    }
}

// Test 5: Edge cases for floating-point values in attributes
#[test]
fn test_floating_point_edge_values_in_attributes() {
    let float_edge_attrs = [
        ("pos_zero", Attribute::Float(0.0)),
        ("neg_zero", Attribute::Float(-0.0)),
        ("pos_inf", Attribute::Float(f64::INFINITY)),
        ("neg_inf", Attribute::Float(f64::NEG_INFINITY)),
        ("nan", Attribute::Float(f64::NAN)),
        ("min_pos", Attribute::Float(f64::MIN_POSITIVE)),
        ("epsilon", Attribute::Float(f64::EPSILON)),
    ];
    
    for (name, attr) in &float_edge_attrs {
        match attr {
            Attribute::Float(value) => {
                match *name {
                    "pos_zero" => assert_eq!(*value, 0.0),
                    "neg_zero" => assert!(value.is_sign_negative() && value.abs() == 0.0),
                    "pos_inf" => assert!(value.is_infinite() && value.is_sign_positive()),
                    "neg_inf" => assert!(value.is_infinite() && value.is_sign_negative()),
                    "nan" => assert!(value.is_nan()),
                    "min_pos" => assert_eq!(*value, f64::MIN_POSITIVE),
                    "epsilon" => assert_eq!(*value, f64::EPSILON),
                    _ => panic!("Unknown attribute name"),
                }
            },
            _ => panic!("Expected Float attribute for {}", name),
        }
    }
}

// Test 6: String attributes with various sizes and content
#[test]
fn test_string_attribute_edge_cases() {
    let string_edge_cases = [
        ("empty", ""),
        ("one_char", "x"),
        ("emoji", "🚀"),
        ("unicode", "café résumé naïve"),
        ("newline_tab", "line1\nline2\ttabbed"),
        ("very_long", &"a".repeat(1_000_000)), // 1MB string
    ];
    
    for (name, content) in &string_edge_cases {
        let attr = Attribute::String(content.to_string());
        
        match attr {
            Attribute::String(s) => {
                assert_eq!(s, *content);
                assert_eq!(s.chars().count(), content.chars().count());
            },
            _ => panic!("Expected String attribute for {}", name),
        }
    }
}

// Test 7: Value with maximum sized shape (testing multiplication overflow)
#[test]
fn test_potential_multiplication_overflow_in_shapes() {
    // Use values that are large but still within usize limits individually
    // But could potentially cause overflow when multiplied together
    let large_but_safe_values = vec![1_000_000, 1_000];
    
    let value = Value {
        name: "large_but_safe_tensor".to_string(),
        ty: Type::F32,
        shape: large_but_safe_values.clone(),
    };
    
    assert_eq!(value.shape, large_but_safe_values);
    
    // Check the multiplication - this should not panic on modern systems
    let product: usize = value.shape.iter().product();
    assert_eq!(product, 1_000_000_000); // 1 billion
    
    // Test the checked multiplication as well
    let checked_product: Option<usize> = value.shape.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
    assert_eq!(checked_product, Some(1_000_000_000));
}

// Test 8: Edge cases for Value.num_elements() method
#[test]
fn test_num_elements_method_edge_cases() {
    // Scalar has 1 element
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
    
    // Zero-containing shape has 0 elements
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![5, 0, 10],
    };
    assert_eq!(zero_tensor.num_elements(), Some(0));
    
    // Regular shape
    let regular = Value {
        name: "regular".to_string(),
        ty: Type::F32,
        shape: vec![2, 3, 4],
    };
    assert_eq!(regular.num_elements(), Some(24));
    
    // Large but safe shape
    let large = Value {
        name: "large".to_string(),
        ty: Type::F32,
        shape: vec![10_000, 10_000],
    };
    assert_eq!(large.num_elements(), Some(100_000_000));
}

// Test 9: Module with complex interdependent operations
#[test]
fn test_complex_interdependent_operations() {
    let mut module = Module::new("complex_module");
    
    // Create a sequence of operations where each depends on previous output
    for i in 0..1000 {
        let mut op = Operation::new(&format!("op_{}", i));
        
        // Previous operation's output becomes next operation's input
        if i > 0 {
            op.inputs.push(Value {
                name: format!("output_{}", i - 1),
                ty: Type::F32,
                shape: vec![i % 100 + 1], // Cycle through shapes 1-100
            });
        }
        
        // Each operation produces an output
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![(i + 1) % 100 + 1], // Cycle through shapes 1-100
        });
        
        // Add a couple of attributes to increase complexity
        op.attributes.insert("iteration".to_string(), Attribute::Int(i as i64));
        op.attributes.insert("modulo".to_string(), Attribute::Int((i % 10) as i64));
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1000);
    assert_eq!(module.name, "complex_module");
    
    // Verify the first and last operations
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[999].op_type, "op_999");
    
    // Verify first operation has no inputs but has outputs and attributes
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 1);
    assert_eq!(module.operations[0].attributes.len(), 2);
    
    // Verify a middle operation has both inputs and outputs
    assert_eq!(module.operations[500].inputs.len(), 1);
    assert_eq!(module.operations[500].outputs.len(), 1);
    assert_eq!(module.operations[500].attributes.len(), 2);
}

// Test 10: Test for deeply nested array attributes
#[test]
fn test_deeply_nested_array_attributes() {
    // Create a deeply nested array structure
    let mut nested_array = Attribute::Array(vec![]);
    
    // Build 10 levels of nested arrays
    for level in 0..10 {
        let new_array = Attribute::Array(vec![
            Attribute::Int(level as i64),
            nested_array,
        ]);
        nested_array = new_array;
    }
    
    // Verify the structure is built correctly
    // The top level should be an array with 2 elements
    if let Attribute::Array(top_level) = &nested_array {
        assert_eq!(top_level.len(), 2);
        
        // The first element should be the level count
        if let Attribute::Int(level_count) = top_level[0] {
            assert_eq!(level_count, 9); // Last level added
        } else {
            panic!("First element should be level count");
        }
    } else {
        panic!("Top level should be an array");
    }
    
    // Test cloning of deeply nested structure
    let cloned_nested = nested_array.clone();
    assert_eq!(nested_array, cloned_nested);
}