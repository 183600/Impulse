//! Enhanced comprehensive boundary tests - 覆盖更多边界情况
//! 使用标准库 assert! 和 assert_eq! 进行测试

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with extremely large shape dimensions that cause overflow in num_elements()
#[test]
fn test_value_overflow_in_num_elements() {
    // Create a value with dimensions that would overflow when multiplied
    let value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],  // This should cause overflow
    };
    
    // num_elements should return None when overflow would occur
    assert_eq!(value.num_elements(), None);
}

/// Test 2: Attribute with negative zero float
#[test]
fn test_negative_zero_float_attribute() {
    let neg_zero = Attribute::Float(-0.0);
    let pos_zero = Attribute::Float(0.0);
    
    // -0.0 and 0.0 should both be zero but have different sign bits
    match neg_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val == 0.0);
            assert!(val.is_sign_negative());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match pos_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val == 0.0);
            assert!(val.is_sign_positive());
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with cyclic operation dependencies (same input/output names)
#[test]
fn test_module_with_cyclic_dependency_pattern() {
    let mut module = Module::new("cyclic_module");
    
    // Create operations that form a cyclic naming pattern
    let mut op1 = Operation::new("op1");
    op1.outputs.push(Value {
        name: "x".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(Value {
        name: "x".to_string(),  // Uses output from op1
        ty: Type::F32,
        shape: vec![10],
    });
    op2.outputs.push(Value {
        name: "y".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(Value {
        name: "y".to_string(),  // Uses output from op2
        ty: Type::F32,
        shape: vec![10],
    });
    op3.outputs.push(Value {
        name: "x".to_string(),  // Creates cycle
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].outputs[0].name, "x");
    assert_eq!(module.operations[1].inputs[0].name, "x");
    assert_eq!(module.operations[1].outputs[0].name, "y");
    assert_eq!(module.operations[2].inputs[0].name, "y");
    assert_eq!(module.operations[2].outputs[0].name, "x");
}

/// Test 4: Value with shape containing repeated dimensions
#[test]
fn test_value_with_repeated_dimensions() {
    let test_cases = vec![
        (vec![2, 2, 2], 8),
        (vec![1, 1, 1, 1, 1], 1),
        (vec![5, 5], 25),
        (vec![3, 3, 3, 3], 81),
    ];
    
    for (shape, expected_elements) in test_cases {
        let value = Value {
            name: "repeated_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        assert_eq!(value.num_elements(), Some(expected_elements));
    }
}

/// Test 5: Operation with attribute containing special control characters
#[test]
fn test_operation_with_control_character_attributes() {
    let mut op = Operation::new("control_chars");
    let mut attrs = std::collections::HashMap::new();
    
    // Test with control characters
    attrs.insert("null".to_string(), Attribute::String("\0".to_string()));
    attrs.insert("tab".to_string(), Attribute::String("\t".to_string()));
    attrs.insert("newline".to_string(), Attribute::String("\n".to_string()));
    attrs.insert("carriage_return".to_string(), Attribute::String("\r".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    
    // Verify each attribute
    match op.attributes.get("null") {
        Some(Attribute::String(s)) => assert_eq!(s, "\0"),
        _ => panic!("Expected null character string"),
    }
    
    match op.attributes.get("tab") {
        Some(Attribute::String(s)) => assert_eq!(s, "\t"),
        _ => panic!("Expected tab character string"),
    }
}

/// Test 6: Value with very long operation name
#[test]
fn test_operation_with_very_long_name() {
    let long_name = "a".repeat(10_000);
    let op = Operation::new(&long_name);
    
    assert_eq!(op.op_type.len(), 10_000);
    assert!(op.op_type.chars().all(|c| c == 'a'));
}

/// Test 7: Module with operation having identical input and output values
#[test]
fn test_operation_with_identical_input_output() {
    let mut module = Module::new("identical_io");
    
    let same_value = Value {
        name: "same_value".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    };
    
    let mut op = Operation::new("identity");
    op.inputs.push(same_value.clone());
    op.outputs.push(same_value);  // Same value instance
    
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 1);
    assert_eq!(module.operations[0].outputs.len(), 1);
    assert_eq!(module.operations[0].inputs[0].name, "same_value");
    assert_eq!(module.operations[0].outputs[0].name, "same_value");
}

/// Test 8: Attribute array with boolean values covering all combinations
#[test]
fn test_attribute_array_with_boolean_combinations() {
    let bool_combinations = vec![
        Attribute::Bool(true),
        Attribute::Bool(false),
        Attribute::Bool(true),
        Attribute::Bool(false),
    ];
    
    let array = Attribute::Array(bool_combinations);
    
    match array {
        Attribute::Array(vec) => {
            assert_eq!(vec.len(), 4);
            match vec[0] {
                Attribute::Bool(true) => {},
                _ => panic!("Expected true"),
            }
            match vec[1] {
                Attribute::Bool(false) => {},
                _ => panic!("Expected false"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Module with single dimension tensor that is very large
#[test]
fn test_module_with_very_large_single_dimension() {
    let mut module = Module::new("large_1d_module");
    
    let large_1d = Value {
        name: "large_1d".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],  // Maximum single dimension
    };
    
    module.inputs.push(large_1d);
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.inputs[0].shape, vec![usize::MAX]);
    // Should return the max value without overflow
    assert_eq!(module.inputs[0].num_elements(), Some(usize::MAX));
}

/// Test 10: Type with deeply nested tensor structure (5 levels deep)
#[test]
fn test_deeply_nested_tensor_type_five_levels() {
    // Level 1: tensor<f32, [2]>
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    
    // Level 2: tensor<tensor<f32, [2]>, [3]>
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![3],
    };
    
    // Level 3: tensor<tensor<tensor<f32, [2]>, [3]>, [4]>
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };
    
    // Level 4: tensor<tensor<tensor<tensor<f32, [2]>, [3]>, [4]>, [5]>
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![5],
    };
    
    // Level 5: tensor<tensor<tensor<tensor<tensor<f32, [2]>, [3]>, [4]>, [5]>, [6]>
    let level5 = Type::Tensor {
        element_type: Box::new(level4),
        shape: vec![6],
    };
    
    // Verify the nested structure
    match &level5 {
        Type::Tensor { element_type: l5_elem, shape: l5_shape } => {
            assert_eq!(l5_shape, &vec![6]);
            
            match l5_elem.as_ref() {
                Type::Tensor { element_type: l4_elem, shape: l4_shape } => {
                    assert_eq!(l4_shape, &vec![5]);
                    
                    match l4_elem.as_ref() {
                        Type::Tensor { element_type: l3_elem, shape: l3_shape } => {
                            assert_eq!(l3_shape, &vec![4]);
                            
                            match l3_elem.as_ref() {
                                Type::Tensor { element_type: l2_elem, shape: l2_shape } => {
                                    assert_eq!(l2_shape, &vec![3]);
                                    
                                    match l2_elem.as_ref() {
                                        Type::Tensor { element_type: l1_elem, shape: l1_shape } => {
                                            assert_eq!(l1_shape, &vec![2]);
                                            
                                            match l1_elem.as_ref() {
                                                Type::F32 => {}, // Success
                                                _ => panic!("Expected F32 at innermost level"),
                                            }
                                        }
                                        _ => panic!("Expected Tensor at level 1"),
                                    }
                                }
                                _ => panic!("Expected Tensor at level 2"),
                            }
                        }
                        _ => panic!("Expected Tensor at level 3"),
                    }
                }
                _ => panic!("Expected Tensor at level 4"),
            }
        }
        _ => panic!("Expected Tensor at level 5"),
    }
}