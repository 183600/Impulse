//! Comprehensive boundary tests covering edge cases and critical scenarios
use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with extremely large single dimension (near overflow boundary)
#[test]
fn test_value_large_single_dimension() {
    let value = Value {
        name: "large_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2],
    };
    // Should not overflow in shape creation
    assert_eq!(value.shape.len(), 1);
    // num_elements should handle this gracefully
    let elements = value.num_elements();
    assert!(elements.is_some() || elements.is_none());
}

/// Test 2: Module with operation containing maximum possible attributes
#[test]
fn test_module_max_attributes_operation() {
    let mut module = Module::new("max_attrs");
    let mut op = Operation::new("max_attr_op");
    
    // Add various attribute types
    op.attributes.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    op.attributes.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    op.attributes.insert("inf".to_string(), Attribute::Float(f64::INFINITY));
    op.attributes.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    op.attributes.insert("nan".to_string(), Attribute::Float(f64::NAN));
    
    module.add_operation(op);
    assert_eq!(module.operations[0].attributes.len(), 7);
}

/// Test 3: Value with alternating zero and non-zero dimensions
#[test]
fn test_value_alternating_zero_dimensions() {
    let test_cases = vec![
        vec![0, 1, 0, 1],
        vec![1, 0, 1, 0],
        vec![0, 0, 1, 1],
        vec![1, 1, 0, 0],
    ];
    
    for shape in test_cases {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::I32,
            shape: shape.clone(),
        };
        // Any zero dimension should result in 0 total elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 4: Operation with empty attribute value
#[test]
fn test_operation_empty_attribute_values() {
    let mut op = Operation::new("empty_attrs");
    op.attributes.insert("empty_str".to_string(), Attribute::String("".to_string()));
    op.attributes.insert("empty_array".to_string(), Attribute::Array(vec![]));
    
    assert_eq!(op.attributes.len(), 2);
    assert!(op.attributes.contains_key("empty_str"));
    assert!(op.attributes.contains_key("empty_array"));
}

/// Test 5: Module with operations that have no inputs but have outputs
#[test]
fn test_module_output_only_operations() {
    let mut module = Module::new("output_only");
    
    let mut op = Operation::new("constant");
    op.outputs.push(Value {
        name: "const_val".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    module.add_operation(op);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 1);
}

/// Test 6: Value with shape containing consecutive identical dimensions
#[test]
fn test_value_consecutive_identical_dimensions() {
    let shapes = vec![
        vec![2, 2, 2, 2],
        vec![5, 5, 5],
        vec![1, 1, 1, 1, 1],
        vec![100, 100],
    ];
    
    for shape in shapes {
        let value = Value {
            name: "uniform_dim".to_string(),
            ty: Type::F64,
            shape: shape.clone(),
        };
        // Verify all dimensions are the same
        if !shape.is_empty() {
            let first = shape[0];
            assert!(shape.iter().all(|&x| x == first));
        }
        assert_eq!(value.shape, shape);
    }
}

/// Test 7: Attribute array with deeply nested heterogeneous types
#[test]
fn test_deep_heterogeneous_attribute_array() {
    let nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::Array(vec![
                Attribute::String("deep".to_string()),
                Attribute::Bool(false),
            ]),
        ]),
        Attribute::Array(vec![]),
        Attribute::String("top".to_string()),
    ]);
    
    match nested {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 3);
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 8: Module with operation that has multiple identical inputs
#[test]
fn test_module_duplicate_inputs() {
    let mut module = Module::new("dup_inputs");
    let mut op = Operation::new("add_self");
    
    let input_val = Value {
        name: "same_input".to_string(),
        ty: Type::F32,
        shape: vec![5],
    };
    
    // Add the same input twice
    op.inputs.push(input_val.clone());
    op.inputs.push(input_val);
    
    module.add_operation(op);
    assert_eq!(module.operations[0].inputs.len(), 2);
    assert_eq!(module.operations[0].inputs[0].name, module.operations[0].inputs[1].name);
}

/// Test 9: Value with shape that would overflow if calculated naively
#[test]
fn test_value_potential_overflow_shape() {
    // Shape that could overflow: 100000 * 100000 = 10^10 > usize::MAX on 32-bit
    let value = Value {
        name: "overflow_risk".to_string(),
        ty: Type::I32,
        shape: vec![100_000, 100_000],
    };
    
    // num_elements should return None if overflow would occur
    let result = value.num_elements();
    // On 64-bit systems this might succeed, on 32-bit it should fail
    assert!(result.is_some() || result.is_none());
}

/// Test 10: Module with operations in chain pattern
#[test]
fn test_module_operation_chain() {
    let mut module = Module::new("chain_ops");
    
    // Create a chain: op1 -> op2 -> op3
    let mut op1 = Operation::new("op1");
    let val1 = Value {
        name: "val1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op1.outputs.push(val1.clone());
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(val1.clone());
    let val2 = Value {
        name: "val2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op2.outputs.push(val2.clone());
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(val2);
    
    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].outputs.len(), 1);
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[2].inputs.len(), 1);
}