//! Additional boundary condition tests for the Impulse compiler
//! Covering more edge cases and boundary conditions beyond existing tests

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Testing behavior with maximum usize values in tensor dimensions (potential overflow)
#[test]
fn test_max_usize_tensor_dimensions() {
    // Test tensor with maximum usize values that would definitely overflow when multiplied
    let value = Value {
        name: "max_usize_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],  // This should cause overflow when calculating product
    };
    
    let product_result: Option<usize> = value.shape.iter().try_fold(1usize, |acc, &x| {
        acc.checked_mul(x)
    });
    
    // Verify that the multiplication would overflow
    assert!(product_result.is_none());
}

/// Test 2: Testing empty and maximum length string attributes
#[rstest]
#[case("")]
#[case("a")]
#[case(&"x".repeat(1000))]
#[case(&"y".repeat(10000))]
fn test_string_attribute_boundaries(#[case] input_str: &str) {
    let attr = Attribute::String(input_str.to_string());
    
    match attr {
        Attribute::String(s) => {
            assert_eq!(s, input_str);
            assert_eq!(s.len(), input_str.len());
        },
        _ => panic!("Expected String attribute"),
    }
}

/// Test 3: Testing operations with empty names and edge case names
#[test]
fn test_operation_name_boundaries() {
    // Test operation with empty name
    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    
    // Test operation with maximum length name
    let max_name = "z".repeat(50_000);
    let max_op = Operation::new(&max_name);
    assert_eq!(max_op.op_type, max_name);
    
    // Test operation with special characters
    let special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~";
    let special_op = Operation::new(special_chars);
    assert_eq!(special_op.op_type, special_chars);
    
    // Test operation with unicode characters
    let unicode_chars = "你好世界🌍🚀✓";
    let unicode_op = Operation::new(unicode_chars);
    assert_eq!(unicode_op.op_type, unicode_chars);
}

/// Test 4: Testing boolean attribute boundary values
#[rstest]
#[case(true)]
#[case(false)]
fn test_boolean_attribute_boundaries(#[case] bool_val: bool) {
    let attr = Attribute::Bool(bool_val);
    
    match attr {
        Attribute::Bool(b) => assert_eq!(b, bool_val),
        _ => panic!("Expected Bool attribute"),
    }
}

/// Test 5: Testing integer attribute boundary values (min/max i64)
#[rstest]
#[case(i64::MIN)]
#[case(i64::MAX)]
#[case(0)]
#[case(-1)]
#[case(1)]
fn test_integer_attribute_boundary_values(#[case] int_val: i64) {
    let attr = Attribute::Int(int_val);
    
    match attr {
        Attribute::Int(i) => assert_eq!(i, int_val),
        _ => panic!("Expected Int attribute"),
    }
}

/// Test 6: Testing modules with empty names and default initialization
#[test]
fn test_module_initialization_boundaries() {
    // Test module with empty name
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
    assert!(empty_module.operations.is_empty());
    assert!(empty_module.inputs.is_empty());
    assert!(empty_module.outputs.is_empty());
    
    // Test module with whitespace name
    let whitespace_module = Module::new("   \t\n  ");
    assert_eq!(whitespace_module.name, "   \t\n  ");
    
    // Test module with special character names
    let special_module = Module::new("!@#$%^&*()");
    assert_eq!(special_module.name, "!@#$%^&*()");
}

/// Test 7: Testing tensor types with maximum nesting and memory pressure
#[test]
fn test_deep_tensor_nesting_memory_pressure() {
    // Create increasingly nested tensors to test memory usage
    let mut current_type = Type::Bool;
    const NESTING_LEVELS: usize = 100; // Reduced from previous test to be more reasonable
    
    for i in 0..NESTING_LEVELS {
        let new_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 5 + 1], // Small shapes to prevent massive expansion
        };
        current_type = new_type;
    }
    
    // Verify the deeply nested type was created successfully
    assert_ne!(std::mem::size_of_val(&current_type), 0);
    
    // Test cloning of the deeply nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 8: Testing complex nested arrays with mixed types at boundary conditions
#[test]
fn test_complex_nested_arrays_boundary_conditions() {
    // Create a nested array with boundary condition values
    let complex_nested = Attribute::Array(vec![
        Attribute::Int(i64::MIN),
        Attribute::Int(i64::MAX),
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::Float(0.0),
        Attribute::Bool(true),
        Attribute::Bool(false),
        Attribute::String("".to_string()),  // Empty string
        Attribute::Array(vec![  // Nested array with boundary values
            Attribute::Int(0),
            Attribute::Float(f64::EPSILON),
            Attribute::String("boundary_test".to_string()),
        ]),
    ]);
    
    // Validate the structure
    if let Attribute::Array(main_arr) = &complex_nested {
        assert_eq!(main_arr.len(), 9);
        
        // Validate nested array
        if let Attribute::Array(nested_arr) = &main_arr[8] {
            assert_eq!(nested_arr.len(), 3);
            
            // Check specific elements
            if let Attribute::String(ref s) = nested_arr[2] {
                assert_eq!(s, "boundary_test");
            } else {
                panic!("Expected string at nested[2]");
            }
        } else {
            panic!("Expected Array at position 8");
        }
    } else {
        panic!("Expected Array at main level");
    }
}

/// Test 9: Testing values with all possible type combinations and extreme shapes
#[rstest]
#[case(Type::F32, vec![])]  // scalar
#[case(Type::F64, vec![1])]
#[case(Type::I32, vec![0])]
#[case(Type::I64, vec![100_000])]
#[case(Type::Bool, vec![1, 1, 1, 1, 1])]
fn test_value_type_shape_combinations(#[case] data_type: Type, #[case] shape: Vec<usize>) {
    let value = Value {
        name: "boundary_test_value".to_string(),
        ty: data_type.clone(),
        shape: shape.clone(),
    };
    
    assert_eq!(value.ty, data_type);
    assert_eq!(value.shape, shape);
    assert_eq!(value.name, "boundary_test_value");
    
    // Calculate the product of shape dimensions
    let product: usize = shape.iter().product();
    
    // If any dimension is 0, total should be 0
    if shape.iter().any(|&dim| dim == 0) {
        assert_eq!(product, 0);
    }
}

/// Test 10: Testing operations with boundary condition attribute counts
#[test]
fn test_operation_attribute_count_boundaries() {
    use std::collections::HashMap;
    
    // Test operation with no attributes
    let mut no_attrs_op = Operation::new("no_attrs");
    assert_eq!(no_attrs_op.attributes.len(), 0);
    
    // Test operation with single attribute
    let mut single_attr_op = Operation::new("single_attr");
    let mut single_attrs = HashMap::new();
    single_attrs.insert("attr1".to_string(), Attribute::Int(42));
    single_attr_op.attributes = single_attrs;
    assert_eq!(single_attr_op.attributes.len(), 1);
    
    // Test operation with many similar attributes (stress test)
    let mut many_attrs_op = Operation::new("many_attrs");
    let mut many_attrs = HashMap::new();
    for i in 0..1000 {
        many_attrs.insert(
            format!("attr_{}", i),
            Attribute::Int(i as i64)
        );
    }
    many_attrs_op.attributes = many_attrs;
    assert_eq!(many_attrs_op.attributes.len(), 1000);
    
    // Verify we can retrieve specific attributes
    assert!(many_attrs_op.attributes.contains_key("attr_0"));
    assert!(many_attrs_op.attributes.contains_key("attr_999"));
    assert_eq!(many_attrs_op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
    assert_eq!(many_attrs_op.attributes.get("attr_999"), Some(&Attribute::Int(999)));
}