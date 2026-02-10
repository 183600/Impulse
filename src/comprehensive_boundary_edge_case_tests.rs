//! Comprehensive boundary edge case tests for Impulse compiler
//! Testing various boundary conditions and edge cases in IR, module, and operation handling

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Module with extremely large dimension values (boundary check)
#[test]
fn test_module_with_large_dimension_values() {
    let mut module = Module::new("large_dim_module");
    
    // Create value with large dimension values (should not overflow)
    let large_dim_value = Value {
        name: "large_dim".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 10], // 1M elements, valid but large
    };
    
    assert_eq!(large_dim_value.shape, vec![100_000, 10]);
    assert_eq!(large_dim_value.num_elements(), Some(1_000_000));
    
    module.inputs.push(large_dim_value);
    assert_eq!(module.inputs.len(), 1);
}

/// Test 2: Value with zero in various dimension positions
#[test]
fn test_value_with_zero_in_dimensions() {
    let test_cases = vec![
        (vec![0, 10, 10], "zero_first"),
        (vec![10, 0, 10], "zero_middle"),
        (vec![10, 10, 0], "zero_last"),
        (vec![0, 0, 0], "all_zeros"),
    ];
    
    for (shape, name) in test_cases {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any shape containing zero should result in 0 elements
        assert_eq!(value.num_elements(), Some(0), "Failed for shape {:?}", shape);
    }
}

/// Test 3: Scalar tensor (empty shape)
#[test]
fn test_scalar_tensor_empty_shape() {
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![], // Empty shape represents a scalar
    };
    
    assert!(scalar.shape.is_empty());
    // Empty shape product should be 1 (identity element for multiplication)
    assert_eq!(scalar.shape.iter().product::<usize>(), 1);
}

/// Test 4: Attribute with extreme float values (infinity and NaN)
#[test]
fn test_attribute_with_extreme_float_values() {
    let positive_inf = Attribute::Float(f64::INFINITY);
    let negative_inf = Attribute::Float(f64::NEG_INFINITY);
    let nan_value = Attribute::Float(f64::NAN);
    
    // Verify attributes can be created with extreme values
    match positive_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float"),
    }
    
    match negative_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float"),
    }
    
    match nan_value {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float"),
    }
}

/// Test 5: Module with all supported data types
#[test]
fn test_module_with_all_supported_types() {
    let mut module = Module::new("all_types_module");
    
    let types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];
    
    for (i, ty) in types.iter().enumerate() {
        let value = Value {
            name: format!("value_{}", i),
            ty: ty.clone(),
            shape: vec![1, 1],
        };
        module.inputs.push(value);
    }
    
    assert_eq!(module.inputs.len(), 5);
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::F64);
    assert_eq!(module.inputs[2].ty, Type::I32);
    assert_eq!(module.inputs[3].ty, Type::I64);
    assert_eq!(module.inputs[4].ty, Type::Bool);
}

/// Test 6: Operation with empty attribute map
#[test]
fn test_operation_with_empty_attributes() {
    let op = Operation::new("test_op");
    
    assert!(op.attributes.is_empty());
    assert_eq!(op.attributes.len(), 0);
    
    // Verify we can add and remove attributes
    let mut attrs = HashMap::new();
    attrs.insert("key1".to_string(), Attribute::Int(42));
    let mut op_with_attrs = Operation::new("test_op_with_attrs");
    op_with_attrs.attributes = attrs;
    
    assert_eq!(op_with_attrs.attributes.len(), 1);
    assert!(op_with_attrs.attributes.contains_key("key1"));
}

/// Test 7: Value with single dimension (1D tensor)
#[test]
fn test_value_single_dimension() {
    let vec_1d = Value {
        name: "vector".to_string(),
        ty: Type::F32,
        shape: vec![100], // 1D tensor
    };
    
    assert_eq!(vec_1d.shape.len(), 1);
    assert_eq!(vec_1d.num_elements(), Some(100));
}

/// Test 8: Nested tensor types with depth validation
#[test]
fn test_nested_tensor_types() {
    // Create a tensor of tensors
    let inner_type = Type::F32;
    let outer_type = Type::Tensor {
        element_type: Box::new(inner_type),
        shape: vec![2, 3],
    };
    
    match outer_type {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, vec![2, 3]);
            assert_eq!(*element_type, Type::F32);
        }
        _ => panic!("Expected Tensor type"),
    }
}

/// Test 9: Module with multiple operations and dependencies
#[test]
fn test_module_with_operation_chain() {
    let mut module = Module::new("chain_module");
    
    // Create input
    let input = Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    module.inputs.push(input.clone());
    
    // Operation 1: Add
    let mut op1 = Operation::new("add");
    op1.inputs.push(input.clone());
    let mid1 = Value {
        name: "mid1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op1.outputs.push(mid1.clone());
    module.add_operation(op1);
    
    // Operation 2: Multiply
    let mut op2 = Operation::new("multiply");
    op2.inputs.push(mid1);
    let output = Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op2.outputs.push(output.clone());
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].op_type, "add");
    assert_eq!(module.operations[1].op_type, "multiply");
    assert_eq!(module.operations[1].inputs[0].name, "mid1");
}

/// Test 10: Attribute with empty array
#[test]
fn test_attribute_empty_array() {
    let empty_array = Attribute::Array(vec![]);
    
    match empty_array {
        Attribute::Array(vec) => {
            assert!(vec.is_empty());
            assert_eq!(vec.len(), 0);
        }
        _ => panic!("Expected Array attribute"),
    }
}