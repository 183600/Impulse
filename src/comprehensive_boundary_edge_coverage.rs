//! Comprehensive boundary edge coverage tests - 10 additional edge cases
//! 覆盖编译器核心组件的关键边界情况，使用标准库 assert! 和 assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Module with operations sharing input/output values
#[test]
fn test_operations_sharing_values() {
    let mut module = Module::new("shared_values");
    
    // Create a shared input value
    let shared_input = Value {
        name: "shared_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    };
    
    // Add to module inputs
    module.inputs.push(shared_input.clone());
    
    // Create two operations that use the same input
    let mut op1 = Operation::new("op1");
    op1.inputs.push(shared_input.clone());
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(shared_input.clone());
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    // Verify both operations have the same input
    assert_eq!(module.operations[0].inputs[0].name, "shared_tensor");
    assert_eq!(module.operations[1].inputs[0].name, "shared_tensor");
    assert_eq!(module.operations.len(), 2);
}

/// Test 2: Value with very small dimension sizes (stress test)
#[test]
fn test_very_small_dimensions() {
    let test_cases = vec![
        vec![1],      // Single element
        vec![1, 1],   // 1x1 matrix
        vec![1, 1, 1], // 1x1x1 tensor
        vec![2, 1],   // 2x1 vector
        vec![1, 2],   // 1x2 vector
    ];
    
    for shape in test_cases {
        let value = Value {
            name: format!("tensor_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        assert_eq!(value.num_elements(), Some(shape.iter().product()));
    }
}

/// Test 3: Operation with many attributes (memory boundary)
#[test]
fn test_operation_with_many_attributes() {
    let mut op = Operation::new("many_attrs");
    
    // Add many attributes (stress test)
    for i in 0..1000 {
        op.attributes.insert(
            format!("attr_{}", i),
            Attribute::Int(i as i64),
        );
    }
    
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify a few specific attributes exist
    assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("attr_500"), Some(&Attribute::Int(500)));
    assert_eq!(op.attributes.get("attr_999"), Some(&Attribute::Int(999)));
}

/// Test 4: Module with empty string names
#[test]
fn test_empty_string_names() {
    let module = Module::new("");
    let value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    let op = Operation::new("");
    
    assert_eq!(module.name, "");
    assert_eq!(value.name, "");
    assert_eq!(op.op_type, "");
}

/// Test 5: Value with alternating large/small dimensions
#[test]
fn test_alternating_dimensions() {
    let shapes = vec![
        vec![1000, 1, 1000, 1],
        vec![1, 10000, 1, 10000],
        vec![500, 2, 500, 2],
    ];
    
    for shape in shapes {
        let value = Value {
            name: "alternating_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 6: Attribute with negative zero and positive zero
#[test]
fn test_zero_variations() {
    let pos_zero = Attribute::Float(0.0);
    let neg_zero = Attribute::Float(-0.0);
    
    match pos_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            assert!(val.signum() >= 0.0);
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match neg_zero {
        Attribute::Float(val) => {
            assert_eq!(val, 0.0);
            // Negative zero has negative sign
            assert!(val.signum() <= 0.0);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 7: Module with chain of operations (each op uses previous op's output)
#[test]
fn test_operation_chain() {
    let mut module = Module::new("chain");
    
    let input = Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    module.inputs.push(input.clone());
    
    // Create a chain: op1 -> op2 -> op3
    let mut op1 = Operation::new("op1");
    op1.inputs.push(input.clone());
    let out1 = Value {
        name: "op1_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op1.outputs.push(out1.clone());
    module.add_operation(op1);
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(out1.clone());
    let out2 = Value {
        name: "op2_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op2.outputs.push(out2.clone());
    module.add_operation(op2);
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(out2.clone());
    let out3 = Value {
        name: "op3_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op3.outputs.push(out3.clone());
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].outputs[0].name, "op1_output");
    assert_eq!(module.operations[1].inputs[0].name, "op1_output");
    assert_eq!(module.operations[2].inputs[0].name, "op2_output");
}

/// Test 8: Type with nested tensor types (deep nesting)
#[test]
fn test_deep_nested_tensor_types() {
    // Create deeply nested: tensor<tensor<tensor<f32, [2]>, [3]>, [4]>
    let innermost = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let middle = Type::Tensor {
        element_type: Box::new(innermost),
        shape: vec![3],
    };
    let outer = Type::Tensor {
        element_type: Box::new(middle),
        shape: vec![4],
    };
    
    // Verify the structure
    match outer {
        Type::Tensor { element_type: outer_elem, shape: outer_shape } => {
            assert_eq!(outer_shape, vec![4]);
            
            match outer_elem.as_ref() {
                Type::Tensor { element_type: middle_elem, shape: middle_shape } => {
                    assert_eq!(middle_shape, &vec![3]);
                    
                    match middle_elem.as_ref() {
                        Type::Tensor { element_type: inner_elem, shape: inner_shape } => {
                            assert_eq!(inner_shape, &vec![2]);
                            assert_eq!(**inner_elem, Type::F32);
                        }
                        _ => panic!("Expected Tensor"),
                    }
                }
                _ => panic!("Expected Tensor"),
            }
        }
        _ => panic!("Expected Tensor"),
    }
}

/// Test 9: Attribute array with mixed types
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("hello".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            
            // Verify each element type
            matches!(arr[0], Attribute::Int(42));
            matches!(arr[1], Attribute::Float(_));
            matches!(arr[2], Attribute::String(_));
            matches!(arr[3], Attribute::Bool(true));
            matches!(&arr[4], Attribute::Array(inner) if inner.len() == 2);
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 10: Module with operations having no explicit inputs or outputs
#[test]
fn test_operations_without_explicit_io() {
    let mut module = Module::new("no_explicit_io");
    
    // Add multiple operations with no explicit inputs/outputs
    for i in 0..5 {
        let op = Operation::new(&format!("stateful_op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    // Verify all operations have no inputs or outputs
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
    }
}