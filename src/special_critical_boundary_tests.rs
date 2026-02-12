//! Special critical boundary tests - 10个特别关键边界情况测试，使用标准库 assert! 和 assert_eq!
//! 覆盖编译器核心组件的关键安全边界情况

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Value with overflow prevention using checked arithmetic
#[test]
fn test_checked_arithmetic_overflow_prevention() {
    // Test with dimensions that would overflow usize (max value on 64-bit is about 1.8e19)
    // 10^6 * 10^6 * 10^6 = 10^18, which should be within range
    // 10^7 * 10^7 * 10^7 = 10^21, which overflows 64-bit
    let large_value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![10_000_000, 10_000_000, 10_000_000], // 10^21, overflows 64-bit
    };
    
    // num_elements uses checked_mul and should return None for overflow
    let result = large_value.num_elements();
    assert_eq!(result, None);
}

/// Test 2: Operation with attribute containing negative zero float
#[test]
fn test_negative_zero_float_attribute() {
    let neg_zero = -0.0f64;
    let attr = Attribute::Float(neg_zero);
    
    match attr {
        Attribute::Float(val) => {
            assert!(val == 0.0);
            assert!(val.is_sign_negative()); // Negative zero has negative sign
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with cyclic-like operation chain (by name)
#[test]
fn test_module_with_cyclic_name_pattern() {
    let mut module = Module::new("cyclic_names");
    
    // Create operations with names that form a cycle
    let op1 = Operation::new("op_a");
    let op2 = Operation::new("op_b");
    let op3 = Operation::new("op_c");
    
    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[1].op_type, "op_b");
    assert_eq!(module.operations[2].op_type, "op_c");
}

/// Test 4: Type validation with nested tensor types
#[test]
fn test_nested_type_validation() {
    // Test nested tensor validation
    let tensor_2d = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(tensor_2d.is_valid_type());
    
    let tensor_3d = Type::Tensor {
        element_type: Box::new(tensor_2d.clone()),
        shape: vec![4],
    };
    assert!(tensor_3d.is_valid_type());
}

/// Test 5: Attribute with boolean type edge cases
#[test]
fn test_boolean_attribute_edge_cases() {
    let true_attr = Attribute::Bool(true);
    let false_attr = Attribute::Bool(false);
    
    match true_attr {
        Attribute::Bool(val) => assert!(val),
        _ => panic!("Expected Bool(true)"),
    }
    
    match false_attr {
        Attribute::Bool(val) => assert!(!val),
        _ => panic!("Expected Bool(false)"),
    }
}

/// Test 6: Module with single dimension value (1D tensor)
#[test]
fn test_single_dimension_value() {
    let value_1d = Value {
        name: "vector".to_string(),
        ty: Type::F32,
        shape: vec![1024], // 1D tensor with 1024 elements
    };
    
    assert_eq!(value_1d.shape.len(), 1);
    assert_eq!(value_1d.num_elements(), Some(1024));
}

/// Test 7: Value with extremely small positive integer dimensions
#[test]
fn test_small_positive_dimensions() {
    let cases = vec![
        vec![1, 1, 1],      // 1x1x1 scalar in 3D
        vec![1, 2],         // 1x2 row vector
        vec![2, 1],         // 2x1 column vector
        vec![1],            // Single element
    ];
    
    for shape in cases {
        let value = Value {
            name: "small_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 8: Operation with empty string attributes
#[test]
fn test_empty_string_attribute() {
    let empty_attr = Attribute::String("".to_string());
    
    match empty_attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 0);
            assert_eq!(s, "");
        }
        _ => panic!("Expected empty String attribute"),
    }
}

/// Test 9: Module with operations having special characters in names
#[test]
fn test_special_characters_in_operation_names() {
    let special_names = vec![
        "add-op",
        "multiply_op",
        "conv2d.kernel",
        "layer_norm/eps",
        "transformer.attn",
    ];
    
    let mut module = Module::new("special_ops");
    
    for name in &special_names {
        let op = Operation::new(name);
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for (i, expected_name) in special_names.iter().enumerate() {
        assert_eq!(module.operations[i].op_type, *expected_name);
    }
}

/// Test 10: Value with large dimension count but small product
#[test]
fn test_many_dimensions_small_product() {
    // Create a value with many dimensions but small total size
    let shape = vec![2; 10]; // 10 dimensions, each size 2 = 1024 total elements
    let value = Value {
        name: "high_dim".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape.len(), 10);
    assert_eq!(value.num_elements(), Some(1024));
}