//! Additional edge case tests for the Impulse compiler
//! Covering more boundary conditions using standard Rust asserts and rstest

use crate::ir::{Module, Operation, Value, Type, Attribute};
use rstest::rstest;

#[test]
fn test_integer_overflow_in_tensor_shape_calculations() {
    // Test tensor shapes that could potentially cause integer overflow when calculating
    // total number of elements. Using values that might overflow on 32-bit platforms
    // while staying within usize bounds on 64-bit
    
    // Using values that are large but safe on 64-bit systems
    let large_shape = vec![100_000, 100_000]; // This would be 10 billion elements
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    // Calculate using the Value method to check for overflow handling
    let num_elements = value.num_elements();
    assert!(num_elements.is_some());
    assert_eq!(num_elements.unwrap(), 10_000_000_000);
    
    // Test with a shape that would definitely result in 0 elements
    let zero_shape = vec![1000, 0, 5000];
    let zero_value = Value {
        name: "zero_tensor".to_string(),
        ty: Type::I64,
        shape: zero_shape,
    };
    
    let zero_elements = zero_value.num_elements();
    assert!(zero_elements.is_some());
    assert_eq!(zero_elements.unwrap(), 0);
}

#[test]
fn test_deeply_nested_tensor_types() {
    // Test creating a deeply nested tensor type to ensure we don't hit stack limits
    // or have other recursion issues
    let mut current_type = Type::F32;
    
    // Create 1000 levels of nesting (reduce if causing stack issues)
    for _ in 0..500 {  
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the structure is preserved
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![2]);
        },
        _ => panic!("Expected a tensor type after nesting"),
    }
    
    // Test that deeply nested types can be cloned without issue
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(100)]
#[case(10_000)]
fn test_operations_with_extreme_input_output_counts(#[case] count: usize) {
    // Test operations with varying numbers of inputs and outputs to check memory behavior
    let mut op = Operation::new("test_op");
    
    // Add varying numbers of inputs based on test case
    for i in 0..count {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1], // Minimal shape to keep memory usage reasonable
        });
    }
    
    // Add half as many outputs as inputs
    for i in 0..count/2 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(op.inputs.len(), count);
    assert_eq!(op.outputs.len(), count/2);
    assert_eq!(op.op_type, "test_op");
}

#[test]
fn test_special_floating_point_values_in_attributes() {
    // Test special floating point values that can appear in tensor computations
    let special_values = [
        std::f64::INFINITY,
        std::f64::NEG_INFINITY,
        std::f64::NAN,
        -0.0,  // Negative zero
        std::f64::EPSILON,
        std::f64::consts::PI,
        std::f64::consts::E,
    ];
    
    for (i, &val) in special_values.iter().enumerate() {
        let attr = Attribute::Float(val);
        
        match attr {
            Attribute::Float(retrieved_val) => {
                if val.is_nan() {
                    // NaN needs special handling as it's not equal to itself
                    assert!(retrieved_val.is_nan(), 
                           "NaN value {} at index {} was not preserved", val, i);
                } else if val.is_infinite() {
                    assert!(retrieved_val.is_infinite(), 
                           "Infinity value {} at index {} was not preserved", val, i);
                    assert_eq!(val.is_sign_positive(), retrieved_val.is_sign_positive(),
                              "Sign of infinity not preserved for value {} at index {}", val, i);
                } else {
                    // For finite values, check approximate equality for special constants
                    if val == std::f64::EPSILON || val == std::f64::consts::PI || val == std::f64::consts::E {
                        assert!((retrieved_val - val).abs() < f64::EPSILON * 10.0, 
                               "Special constant {} at index {} was not preserved accurately", val, i);
                    } else {
                        assert_eq!(retrieved_val, val, 
                                  "Value {} at index {} was not preserved", val, i);
                    }
                }
            },
            _ => panic!("Expected Float attribute for special value {} at index {}", val, i),
        }
    }
}

#[test]
fn test_large_collection_allocation_in_module() {
    // Test memory allocation with a large number of operations in a module
    let mut module = Module::new("stress_test_module");
    
    // Add 50,000 operations to stress test memory allocation
    for i in 0..50_000 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 50_000);
    assert_eq!(module.name, "stress_test_module");
    
    // Verify some operations at different indices to ensure they were stored correctly
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[49_999].op_type, "op_49999");
    assert_eq!(module.operations[25_000].op_type, "op_25000");
    
    // Test the module can be dropped without memory issues
    drop(module);
    assert!(true); // Placeholder assertion after drop
}

#[test]
fn test_operations_with_extremely_long_names() {
    // Test creating operations with extremely long names to check string handling
    let extremely_long_name = "a".repeat(100_000); // 100k character name
    let op = Operation::new(&extremely_long_name);
    
    assert_eq!(op.op_type.len(), 100_000);
    assert_eq!(op.op_type.chars().next(), Some('a'));
    assert_eq!(op.op_type.chars().last(), Some('a'));
    
    // Also test values with extremely long names
    let value = Value {
        name: "v".repeat(100_000), // 100k character name
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value.name.len(), 100_000);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 2, 3]);
}

#[test]
fn test_zero_sized_tensor_edge_cases() {
    // Test all the different ways zero-sized tensors can occur
    let zero_test_cases = [
        (vec![0], "single zero dimension"),
        (vec![0, 5], "zero followed by positive"),
        (vec![5, 0], "positive followed by zero"),
        (vec![2, 0, 3], "zero in the middle"),
        (vec![0, 0, 0], "multiple zeros"),
        (vec![0, 1, 0, 1], "alternating zeros and ones"),
        (vec![1, 1, 0], "zeros at end"),
    ];

    for (shape, description) in &zero_test_cases {
        let value = Value {
            name: format!("zero_test_{}", description.replace(" ", "_")),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // All tensors with zero in dimensions should have 0 elements
        let elements = value.num_elements();
        assert!(elements.is_some(), "Failed to calculate elements for shape {:?}", shape);
        assert_eq!(elements.unwrap(), 0, "Shape {:?} should have 0 elements ({})", shape, description);
    }
    
    // Test empty shape (scalar) which should have 1 element
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape means scalar
    };
    
    let scalar_elements = scalar.num_elements();
    assert!(scalar_elements.is_some());
    assert_eq!(scalar_elements.unwrap(), 1, "Scalar should have 1 element");
}

#[test]
fn test_unicode_identifiers_in_ir_components() {
    // Test Unicode and special characters in identifiers
    let unicode_cases = [
        ("valid_unicode_🚀", Type::F32),
        ("chinese_chars_中文", Type::I32),
        ("arabic_chars_مرحبا", Type::F64),
        ("accented_chars_café_naïve", Type::I64),
        ("control_chars_\u{0001}_\u{001F}", Type::Bool),
    ];

    for (identifier, data_type) in &unicode_cases {
        // Test values with unicode identifiers
        let value = Value {
            name: identifier.to_string(),
            ty: data_type.clone(),
            shape: vec![1],
        };
        assert_eq!(value.name, *identifier);
        assert_eq!(value.ty, *data_type);

        // Test operations with unicode names
        let op = Operation::new(identifier);
        assert_eq!(op.op_type, *identifier);
        
        // Test modules with unicode names
        let module = Module::new(*identifier);
        assert_eq!(module.name, *identifier);
    }
}

#[test]
fn test_attribute_type_edge_cases() {
    // Test edge cases with attribute types and their equality comparisons
    use std::collections::HashMap;
    
    // Test integer attributes with min/max values
    let int_attrs = [
        Attribute::Int(i64::MAX),
        Attribute::Int(i64::MIN),
        Attribute::Int(0),
        Attribute::Int(-1),
        Attribute::Int(1),
    ];
    
    assert_eq!(int_attrs[0], Attribute::Int(i64::MAX));
    assert_ne!(int_attrs[0], int_attrs[1]);  // MAX != MIN
    assert_ne!(int_attrs[2], int_attrs[3]);  // 0 != -1
    
    // Test string attributes with empty and large strings
    let str_attrs = [
        Attribute::String("".to_string()),
        Attribute::String("short".to_string()),
        Attribute::String("x".repeat(100_000)),  // Large string
    ];
    
    assert_eq!(str_attrs[0], Attribute::String("".to_string()));
    assert_eq!(str_attrs[2], Attribute::String("x".repeat(100_000)));
    assert_ne!(str_attrs[0], str_attrs[1]);  // Empty != short
    
    // Test boolean attributes
    assert_eq!(Attribute::Bool(true), Attribute::Bool(true));
    assert_eq!(Attribute::Bool(false), Attribute::Bool(false));
    assert_ne!(Attribute::Bool(true), Attribute::Bool(false));
    
    // Test complex array attribute with nested structures
    let complex_array = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::String("nested".to_string()),
            ]),
        ]),
        Attribute::Int(42),
    ]);
    
    match &complex_array {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 2);
            
            match &outer[0] {
                Attribute::Array(middle) => {
                    assert_eq!(middle.len(), 3);
                    
                    match &middle[2] {
                        Attribute::Array(inner) => {
                            assert_eq!(inner.len(), 2);
                            assert_eq!(inner[0], Attribute::Bool(true));
                        },
                        _ => panic!("Expected nested array as third element"),
                    }
                },
                _ => panic!("Expected array as first element"),
            }
        },
        _ => panic!("Expected array attribute"),
    }
    
    // Test operation with all attribute types
    let mut op = Operation::new("comprehensive_attr_test");
    let mut attrs = HashMap::new();
    
    attrs.insert("int_max".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("int_min".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("float_inf".to_string(), Attribute::Float(std::f64::INFINITY));
    attrs.insert("float_nan".to_string(), Attribute::Float(std::f64::NAN));
    attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
    attrs.insert("bool_true".to_string(), Attribute::Bool(true));
    attrs.insert("bool_false".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.get("int_max"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("int_min"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
    
    // Special handling for NaN since NaN != NaN
    if let Some(Attribute::Float(val)) = op.attributes.get("float_nan") {
        assert!(val.is_nan());
    } else {
        panic!("Expected NaN float attribute");
    }
}