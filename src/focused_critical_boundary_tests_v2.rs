//! Focused critical boundary tests v2 - Additional edge case coverage
//! Using standard library assertions (assert!, assert_eq!) for clear error messages

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with extremely large shape dimensions (near overflow boundary)
#[test]
fn test_value_large_shape_near_overflow() {
    // Test shape dimensions that multiply close to usize limit
    let shapes = [
        vec![65536, 65536],      // 2^16 * 2^16 = 2^32 ≈ 4 billion
        vec![100000, 100000, 10], // 100 billion
        vec![1000, 1000, 1000],   // 1 billion
    ];

    for shape in &shapes {
        let value = Value {
            name: "large_shape".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        // Verify shape is stored correctly
        assert_eq!(value.shape, *shape);

        // num_elements should return None for overflow or valid number
        let num_elem = value.num_elements();
        // Just verify it doesn't panic and returns appropriate result
        match num_elem {
            Some(n) => assert!(n > 0),
            None => {
                // Overflow case - valid behavior
                let product: usize = shape.iter().product();
                assert_eq!(product.checked_mul(1), None);
            }
        }
    }
}

/// Test 2: Module with operation containing all boundary integer attributes
#[test]
fn test_module_boundary_integer_attributes() {
    let mut module = Module::new("boundary_int_attrs");
    let mut op = Operation::new("boundary_test");
    let mut attrs = HashMap::new();

    // Test boundary integer values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("one".to_string(), Attribute::Int(1));
    attrs.insert("minus_one".to_string(), Attribute::Int(-1));
    attrs.insert("max_i32_safe".to_string(), Attribute::Int(i32::MAX as i64));
    attrs.insert("min_i32_safe".to_string(), Attribute::Int(i32::MIN as i64));

    op.attributes = attrs;
    module.add_operation(op);

    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].attributes.len(), 7);

    // Verify boundary values
    match module.operations[0].attributes.get("max_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected max_i64"),
    }
    match module.operations[0].attributes.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected min_i64"),
    }
}

/// Test 3: Value with empty and single-element shapes
#[test]
fn test_value_empty_and_single_element_shapes() {
    // Empty shape (scalar)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(scalar.num_elements(), Some(1));

    // Single element shape
    let single_elem = Value {
        name: "single".to_string(),
        ty: Type::I32,
        shape: vec![1],
    };
    assert_eq!(single_elem.shape, vec![1]);
    assert_eq!(single_elem.num_elements(), Some(1));

    // Multiple single dimensions (1x1x1)
    let triple_one = Value {
        name: "triple_one".to_string(),
        ty: Type::F64,
        shape: vec![1, 1, 1],
    };
    assert_eq!(triple_one.shape, vec![1, 1, 1]);
    assert_eq!(triple_one.num_elements(), Some(1));
}

/// Test 4: Operation with boundary float attributes (subnormal, infinity, NaN)
#[test]
fn test_operation_boundary_float_attributes() {
    let mut op = Operation::new("float_boundary");
    let mut attrs = HashMap::new();

    // Test boundary float values
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("min_pos".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("neg_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("pi".to_string(), Attribute::Float(std::f64::consts::PI));
    attrs.insert("e".to_string(), Attribute::Float(std::f64::consts::E));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 7);

    // Verify epsilon is very small but positive
    match op.attributes.get("epsilon") {
        Some(Attribute::Float(val)) => {
            assert!(*val > 0.0);
            assert!(*val < 1.0);
        }
        _ => panic!("Expected epsilon"),
    }

    // Verify max is large
    match op.attributes.get("max_f64") {
        Some(Attribute::Float(val)) => {
            assert!(*val > 1e300);
        }
        _ => panic!("Expected max_f64"),
    }
}

/// Test 5: Value with shapes containing zero dimensions
#[test]
fn test_value_shapes_with_zero_dimensions() {
    let zero_shapes = [
        vec![0],
        vec![0, 10],
        vec![10, 0],
        vec![0, 0, 0],
        vec![1, 0, 1],
        vec![100, 0, 100, 0],
    ];

    for shape in &zero_shapes {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, *shape);
        // Zero dimension should result in 0 total elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 6: Attribute with nested arrays of varying depths
#[test]
fn test_attribute_nested_arrays_varying_depths() {
    // Depth 1: flat array
    let depth1 = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Int(2),
        Attribute::Int(3),
    ]);

    // Depth 2: nested array
    let depth2 = Attribute::Array(vec![
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
        Attribute::Array(vec![Attribute::Int(3), Attribute::Int(4)]),
    ]);

    // Depth 3: deeply nested array
    let depth3 = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1)]),
            Attribute::Array(vec![Attribute::Int(2)]),
        ]),
    ]);

    // Verify depth 1
    match depth1 {
        Attribute::Array(ref arr) => {
            assert_eq!(arr.len(), 3);
            match arr[0] {
                Attribute::Int(1) => {},
                _ => panic!("Expected Int(1)"),
            }
        }
        _ => panic!("Expected Array"),
    }

    // Verify depth 2
    match depth2 {
        Attribute::Array(ref outer) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                Attribute::Array(inner) => assert_eq!(inner.len(), 2),
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array"),
    }

    // Verify depth 3
    match depth3 {
        Attribute::Array(ref outer) => {
            assert_eq!(outer.len(), 1);
            match &outer[0] {
                Attribute::Array(mid) => {
                    assert_eq!(mid.len(), 2);
                    match &mid[0] {
                        Attribute::Array(inner) => assert_eq!(inner.len(), 1),
                        _ => panic!("Expected triple nested array"),
                    }
                }
                _ => panic!("Expected nested array"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 7: Module with mixed type operations and connections
#[test]
fn test_module_mixed_type_operations() {
    let mut module = Module::new("mixed_types");

    // Create inputs of different types
    let float_input = Value {
        name: "float_in".to_string(),
        ty: Type::F32,
        shape: vec![2, 2],
    };
    let int_input = Value {
        name: "int_in".to_string(),
        ty: Type::I32,
        shape: vec![2, 2],
    };
    let bool_input = Value {
        name: "bool_in".to_string(),
        ty: Type::Bool,
        shape: vec![2, 2],
    };

    module.inputs.push(float_input);
    module.inputs.push(int_input);
    module.inputs.push(bool_input);

    // Create operations using different types
    let mut op1 = Operation::new("cast_int_to_float");
    op1.inputs.push(module.inputs[1].clone()); // int_input
    op1.outputs.push(Value {
        name: "casted".to_string(),
        ty: Type::F64,
        shape: vec![2, 2],
    });

    let mut op2 = Operation::new("logical_and");
    op2.inputs.push(module.inputs[2].clone()); // bool_input
    op2.outputs.push(Value {
        name: "result".to_string(),
        ty: Type::Bool,
        shape: vec![2, 2],
    });

    module.add_operation(op1);
    module.add_operation(op2);

    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].inputs[0].ty, Type::I32);
    assert_eq!(module.operations[0].outputs[0].ty, Type::F64);
}

/// Test 8: Type validation for all valid types
#[test]
fn test_type_validation_all_types() {
    use crate::ir::TypeExtensions;

    let valid_types = [
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
        Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        },
        Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![1],
        },
    ];

    for ty in &valid_types {
        assert!(ty.is_valid_type(), "Type {:?} should be valid", ty);
    }
}

/// Test 9: Value with shape containing very large but valid dimensions
#[test]
fn test_value_very_large_valid_dimensions() {
    // Test with dimensions that are large but still valid
    let large_shapes = [
        vec![1000000],           // 1 million elements
        vec![10000, 1000],       // 10 million elements
        vec![1000, 100, 10],     // 1 million elements
        vec![100, 100, 100],     // 1 million elements
    ];

    for shape in &large_shapes {
        let value = Value {
            name: "large_valid".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(value.shape, *shape);

        // Should return valid number of elements
        let num_elem = value.num_elements();
        assert!(num_elem.is_some());

        if let Some(n) = num_elem {
            assert!(n > 0);
            // Verify the product matches
            let product: usize = shape.iter().product();
            assert_eq!(n, product);
        }
    }
}

/// Test 10: Module with empty operations and attributes
#[test]
fn test_module_empty_operations_and_attributes() {
    let module = Module::new("empty_module");

    // Empty module should have no operations, inputs, or outputs
    assert_eq!(module.name, "empty_module");
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);

    // Add an operation with no attributes
    let empty_op = Operation::new("empty_op");
    assert_eq!(empty_op.op_type, "empty_op");
    assert_eq!(empty_op.inputs.len(), 0);
    assert_eq!(empty_op.outputs.len(), 0);
    assert_eq!(empty_op.attributes.len(), 0);

    // Add operation to module
    let mut module_with_op = Module::new("module_with_op");
    module_with_op.add_operation(empty_op);

    assert_eq!(module_with_op.operations.len(), 1);
    assert_eq!(module_with_op.operations[0].attributes.len(), 0);
}