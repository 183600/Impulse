//! More edge case tests for the Impulse compiler
//! This file contains additional tests for various edge cases and boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::rstest;

// Test 1: Extremely large but valid tensor shapes
#[test]
fn test_extremely_large_shapes() {
    // Test shape that could potentially cause overflow in calculations
    let large_shape = vec![100_000, 100_000];
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    assert_eq!(value.shape, vec![100_000, 100_000]);
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 10_000_000_000); // 10 billion elements
}

// Test 2: Empty operations and values
#[test]
fn test_truly_empty_structures() {
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.inputs.len(), 0);
    assert_eq!(empty_module.outputs.len(), 0);

    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    assert_eq!(empty_op.inputs.len(), 0);
    assert_eq!(empty_op.outputs.len(), 0);
    assert_eq!(empty_op.attributes.len(), 0);
}

// Test 3: Operations with all possible primitive type combinations
#[rstest]
#[case(Type::F32, vec![1, 2])]
#[case(Type::F64, vec![2, 3])]
#[case(Type::I32, vec![3, 4])] 
#[case(Type::I64, vec![4, 5])]
#[case(Type::Bool, vec![5, 6])]
fn test_all_primitive_types(#[case] data_type: Type, #[case] shape: Vec<usize>) {
    let value = Value {
        name: "typed_value".to_string(),
        ty: data_type.clone(),
        shape,
    };
    
    assert_eq!(value.ty, data_type);
}

// Test 4: Operations with maximum possible attributes of different types
#[test]
fn test_operations_with_max_variety_of_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_variety_op");
    let mut attrs = HashMap::new();
    
    // Add many different types of attributes
    attrs.insert("int_attr".to_string(), Attribute::Int(9223372036854775807)); // Max i64
    attrs.insert("float_attr".to_string(), Attribute::Float(1.7976931348623157e308)); // Max f64
    attrs.insert("string_attr".to_string(), Attribute::String("very long string ".repeat(1000)));
    attrs.insert("bool_true_attr".to_string(), Attribute::Bool(true));
    attrs.insert("bool_false_attr".to_string(), Attribute::Bool(false));
    attrs.insert("zero_float_attr".to_string(), Attribute::Float(0.0));
    attrs.insert("negative_int_attr".to_string(), Attribute::Int(-9223372036854775808)); // Min i64
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 7);
    assert_eq!(op.op_type, "max_variety_op");
}

// Test 5: Edge cases with tensor shape calculations that could cause overflow
#[test]
fn test_potential_overflow_shapes() {
    // Create shapes that when multiplied together approach or exceed usize limits
    // For 64-bit systems, we'll use values that are large but not quite overflow
    let near_limit_shape = vec![
        (std::usize::MAX as f64).sqrt() as usize,
        (std::usize::MAX as f64).sqrt() as usize
    ];
    
    let value = Value {
        name: "near_limit_tensor".to_string(),
        ty: Type::F32,
        shape: near_limit_shape,
    };
    
    // Just verify the shape was preserved (the actual multiplication might overflow)
    assert_eq!(value.shape.len(), 2);
}

// Test 6: Special floating point values in attributes
#[test]
fn test_special_float_values_in_attributes() {
    let special_attrs = [
        Attribute::Float(std::f64::INFINITY),
        Attribute::Float(std::f64::NEG_INFINITY),
        Attribute::Float(std::f64::NAN),
        Attribute::Float(-0.0), // Negative zero
        Attribute::Float(std::f64::EPSILON),
    ];
    
    match special_attrs[0] {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected positive infinity"),
    }
    
    match special_attrs[1] {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected negative infinity"),
    }
    
    match special_attrs[2] {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected NaN"),
    }
    
    match special_attrs[3] {
        Attribute::Float(val) => assert!(1.0 / val == std::f64::NEG_INFINITY), // Check for -0.0
        _ => panic!("Expected -0.0"),
    }
    
    match special_attrs[4] {
        Attribute::Float(val) => assert_eq!(val, std::f64::EPSILON),
        _ => panic!("Expected epsilon"),
    }
}

// Test 7: Operations with mixed empty and non-empty inputs/outputs
#[test]
fn test_operations_with_mixed_input_output_patterns() {
    let mut op = Operation::new("mixed_io_op");
    
    // Add a normal input
    op.inputs.push(Value {
        name: "normal_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    // Add an empty input (scalar)
    op.inputs.push(Value {
        name: "scalar_input".to_string(),
        ty: Type::I64,
        shape: vec![],  // scalar
    });
    
    // Add a zero-dimension input
    op.inputs.push(Value {
        name: "zero_input".to_string(),
        ty: Type::Bool,
        shape: vec![0, 10],  // zero dimension
    });
    
    // Add normal outputs
    op.outputs.push(Value {
        name: "normal_output".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    });
    
    // Add scalar output
    op.outputs.push(Value {
        name: "scalar_output".to_string(),
        ty: Type::I32,
        shape: vec![],  // scalar
    });
    
    assert_eq!(op.inputs.len(), 3);
    assert_eq!(op.outputs.len(), 2);
    assert_eq!(op.inputs[0].shape, vec![10, 10]);
    assert!(op.inputs[1].shape.is_empty());
    assert_eq!(op.inputs[2].shape, vec![0, 10]);
    assert_eq!(op.outputs[0].shape, vec![5, 5]);
    assert!(op.outputs[1].shape.is_empty());
}

// Test 8: Deeply nested recursive types with alternating types
#[test]
fn test_alternating_deeply_nested_types() {
    // Create a complex nested type: F32 -> Tensor<F32,[2]> -> Tensor<Tensor<F32,[2]>,[3]> -> ...
    let mut current_type = Type::F32;
    
    for i in 1..=10 { // 10 levels deep
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i], // Varying shape at each level
        };
    }
    
    // Ensure the type is correctly constructed
    match &current_type {
        Type::Tensor { shape, element_type: _ } => {
            assert_eq!(shape, &vec![10]); // Should be level 10
        },
        _ => panic!("Expected a tensor type"),
    }
    
    // Test cloning of complex nested structure
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 9: Boundary values for integer attributes
#[rstest]
#[case(i64::MAX)]
#[case(i64::MIN)]
#[case(0)]
#[case(1)]
#[case(-1)]
fn test_integer_attribute_boundaries(#[case] value: i64) {
    let attr = Attribute::Int(value);
    
    match attr {
        Attribute::Int(v) => assert_eq!(v, value),
        _ => panic!("Expected Int attribute"),
    }
}

// Test 10: Edge cases with Unicode and special characters in names
#[test]
fn test_unicode_and_control_character_names() {
    let test_names = [
        "regular_name",
        "unicode_🚀_name",
        "name_with_\n_newline",
        "name_with_\t_tab",  
        "name_with_\r_carriage",
        "name_with_\0_null_char",
        "chinese_中文_name",
        "arabic_العربية_name",
        "emoji_🔥_name",
    ];
    
    for name in test_names.iter() {
        // Test module creation
        let module = Module::new(*name);
        assert_eq!(&module.name, *name);
        
        // Test operation creation
        let op = Operation::new(*name);
        assert_eq!(&op.op_type, *name);
        
        // Test value creation
        let value = Value {
            name: (*name).to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        assert_eq!(&value.name, *name);
    }
}