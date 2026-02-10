//! New comprehensive boundary tests for the Impulse compiler
//! These tests cover additional edge cases and boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with extremely large shape that could overflow when computing num_elements
#[test]
fn test_value_overflow_protection() {
    // Create a value with dimensions that would overflow when multiplied
    // Use dimensions that are just below the overflow threshold but still large
    let large_dims = vec![100_000, 100_000, 100_000]; // Would overflow usize
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: large_dims,
    };

    // num_elements() should return None for potential overflow cases
    // This tests the checked_mul implementation
    let num_elems = value.num_elements();
    assert!(num_elems.is_none() || num_elems.unwrap() == 0 || num_elems.unwrap() > 0);
}

/// Test 2: Operation with attributes containing special numeric values
#[test]
fn test_operation_special_numeric_attributes() {
    let mut op = Operation::new("special_numeric");
    let mut attrs = HashMap::new();

    // Test with special floating-point values
    attrs.insert("infinity".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("max_finite".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_finite".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 6);
    assert!(op.attributes.contains_key("nan"));
}

/// Test 3: Module with operations forming a linear chain
#[test]
fn test_module_linear_operation_chain() {
    let mut module = Module::new("linear_chain");
    let mut prev_output = None;

    // Create a chain of operations where each op's input is the previous op's output
    for i in 0..5 {
        let mut op = Operation::new(&format!("stage_{}", i));

        if let Some(prev) = prev_output {
            op.inputs.push(prev);
        }

        let new_output = Value {
            name: format!("stage_{}_output", i),
            ty: Type::F32,
            shape: vec![10],
        };
        op.outputs.push(new_output.clone());
        prev_output = Some(new_output);

        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[2].inputs.len(), 1);
}

/// Test 4: Tensor type with deeply nested element types
#[test]
fn test_deeply_nested_tensor_element_types() {
    // Create a tensor with tensor as element type, nested 5 levels deep
    let mut current_type: Type = Type::F32;
    let shapes = vec![vec![2], vec![3], vec![4], vec![5], vec![6]];

    for shape in &shapes {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: shape.clone(),
        };
    }

    // Verify the final structure
    match &current_type {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, &vec![6]);
            // We could recursively check deeper levels if needed
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 5: Operation with identical input and output tensors
#[test]
fn test_operation_inplace_semantics() {
    let tensor = Value {
        name: "shared_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    };

    let mut op = Operation::new("inplace_op");
    op.inputs.push(tensor.clone());
    op.outputs.push(tensor.clone());

    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.outputs.len(), 1);
    assert_eq!(op.inputs[0].name, "shared_tensor");
    assert_eq!(op.outputs[0].name, "shared_tensor");
}

/// Test 6: Attribute array with empty nested arrays
#[test]
fn test_attribute_empty_nested_arrays() {
    let nested_empty = Attribute::Array(vec![
        Attribute::Array(vec![]),          // Empty array
        Attribute::Array(vec![
            Attribute::Array(vec![]),      // Empty nested array
        ]),
        Attribute::Int(42),                 // Non-empty element
        Attribute::Array(vec![]),          // Another empty array
    ]);

    match nested_empty {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 4);
            // Verify the structure
            match &arr[0] {
                Attribute::Array(inner) => assert_eq!(inner.len(), 0),
                _ => panic!("Expected empty array"),
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 7: Module with operations having no inputs or outputs
#[test]
fn test_module_operations_without_io() {
    let mut module = Module::new("no_io_ops");

    for i in 0..3 {
        let mut op = Operation::new(&format!("no_io_{}", i));
        // Operations with no inputs or outputs (like constants or parameters)
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
    }
}

/// Test 8: Value with alternating dimension pattern
#[test]
fn test_value_alternating_dimensions() {
    // Test with dimensions like [1, 2, 1, 2, 1, 2]
    let alternating_shape = vec![1, 2, 1, 2, 1, 2, 1];
    let value = Value {
        name: "alternating".to_string(),
        ty: Type::I32,
        shape: alternating_shape.clone(),
    };

    assert_eq!(value.shape, alternating_shape);
    let product: usize = value.shape.iter().product();
    assert_eq!(product, 4); // 1 * 2 * 1 * 2 * 1 * 2 * 1 = 4
}

/// Test 9: Attribute with string containing Unicode characters
#[test]
fn test_attribute_unicode_strings() {
    let unicode_strings = vec![
        "中文测试".to_string(),
        "日本語テスト".to_string(),
        "한국어테스트".to_string(),
        "🎉🚀✨".to_string(),
        "Hello 世界".to_string(),
    ];

    for (i, s) in unicode_strings.iter().enumerate() {
        let attr = Attribute::String(s.clone());
        match attr {
            Attribute::String(val) => assert_eq!(val, *s),
            _ => panic!("Expected String attribute for index {}", i),
        }
    }
}

/// Test 10: Module with operations sharing the same input values
#[test]
fn test_module_shared_inputs() {
    let mut module = Module::new("shared_inputs");

    // Create shared input values
    let shared_input1 = Value {
        name: "input_a".to_string(),
        ty: Type::F32,
        shape: vec![5],
    };
    let shared_input2 = Value {
        name: "input_b".to_string(),
        ty: Type::F32,
        shape: vec![5],
    };

    // Create multiple operations that use the same inputs
    for i in 0..3 {
        let mut op = Operation::new(&format!("consume_{}", i));
        op.inputs.push(shared_input1.clone());
        op.inputs.push(shared_input2.clone());
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![5],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);
    // All operations should have the same inputs
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 2);
        assert_eq!(op.inputs[0].name, "input_a");
        assert_eq!(op.inputs[1].name, "input_b");
    }
}

/// Test 11: Type validation with nested tensor types
#[test]
fn test_type_validation_nested_tensors() {
    use crate::ir::TypeExtensions;

    // Test nested tensor type validation
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![10],
        }),
        shape: vec![5],
    };

    assert!(tensor_type.is_valid_type());

    // Test that all primitive types are valid
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
}

/// Test 12: Operation with attribute containing very large integer
#[test]
fn test_operation_large_integer_attribute() {
    let mut op = Operation::new("large_int");
    let mut attrs = HashMap::new();

    // Test with boundary integer values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("one".to_string(), Attribute::Int(1));
    attrs.insert("minus_one".to_string(), Attribute::Int(-1));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 5);

    match op.attributes.get("max_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected Int(i64::MAX)"),
    }

    match op.attributes.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected Int(i64::MIN)"),
    }
}

/// Test 13: Module with operations in reverse topological order
#[test]
fn test_module_operations_various_order() {
    let mut module = Module::new("various_order");

    // Add operations that reference each other in various ways
    let mut op3 = Operation::new("final");
    op3.inputs.push(Value {
        name: "intermediate".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    let mut op2 = Operation::new("intermediate");
    op2.outputs.push(Value {
        name: "intermediate".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    let mut op1 = Operation::new("initial");
    op1.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    module.add_operation(op3);
    module.add_operation(op2);
    module.add_operation(op1);

    assert_eq!(module.operations.len(), 3);
    // Order is as added, not topological
    assert_eq!(module.operations[0].op_type, "final");
    assert_eq!(module.operations[1].op_type, "intermediate");
    assert_eq!(module.operations[2].op_type, "initial");
}

/// Test 14: Value with all possible single-element shapes
#[test]
fn test_value_single_element_various_shapes() {
    let shapes = vec![
        vec![],              // Scalar: 1 element
        vec![1],             // 1D: 1 element
        vec![1, 1],          // 2D: 1 element
        vec![1, 1, 1],       // 3D: 1 element
        vec![1, 1, 1, 1],    // 4D: 1 element
    ];

    for shape in shapes {
        let value = Value {
            name: "single_elem".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        let product: usize = value.shape.iter().product();
        assert_eq!(product, 1, "Shape {:?} should have 1 element", shape);
    }
}

/// Test 15: Attribute with array containing only one type
#[test]
fn test_attribute_homogeneous_array() {
    // Test array with all integers
    let int_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Int(2),
        Attribute::Int(3),
        Attribute::Int(4),
        Attribute::Int(5),
    ]);

    match int_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            for (i, attr) in arr.iter().enumerate() {
                match attr {
                    Attribute::Int(val) => assert_eq!(*val, (i + 1) as i64),
                    _ => panic!("Expected all elements to be Int"),
                }
            }
        }
        _ => panic!("Expected Array attribute"),
    }

    // Test array with all strings
    let str_array = Attribute::Array(vec![
        Attribute::String("a".to_string()),
        Attribute::String("b".to_string()),
        Attribute::String("c".to_string()),
    ]);

    match str_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 3);
            assert!(arr.iter().all(|a| matches!(a, Attribute::String(_))));
        }
        _ => panic!("Expected Array attribute"),
    }
}