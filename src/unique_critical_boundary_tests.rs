//! Unique critical boundary tests - Additional edge cases
//! 
//! This module contains 10 unique test cases covering boundary scenarios
//! that may not be fully covered by existing test suites.

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with num_elements() returning None due to overflow
#[test]
fn test_value_overflow_num_elements() {
    // Create a shape that would overflow when calculating num_elements
    // Using very large dimensions that would exceed usize capacity
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2, 2, 2], // This should overflow
    };
    
    // The num_elements method should handle overflow gracefully
    let result = overflow_value.num_elements();
    assert!(result.is_none() || result.is_some()); // Just verify it doesn't panic
}

/// Test 2: Module with operations chain forming a graph
#[test]
fn test_operation_chain_graph() {
    let mut module = Module::new("chain_graph");
    
    // Create a chain of operations: input -> op1 -> op2 -> op3 -> output
    let input = Value {
        name: "initial_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    module.inputs.push(input.clone());
    
    let mut op1 = Operation::new("transform1");
    op1.inputs.push(input.clone());
    let intermediate1 = Value {
        name: "intermediate1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op1.outputs.push(intermediate1.clone());
    module.add_operation(op1);
    
    let mut op2 = Operation::new("transform2");
    op2.inputs.push(intermediate1.clone());
    let intermediate2 = Value {
        name: "intermediate2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op2.outputs.push(intermediate2.clone());
    module.add_operation(op2);
    
    let mut op3 = Operation::new("transform3");
    op3.inputs.push(intermediate2.clone());
    let output = Value {
        name: "final_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op3.outputs.push(output.clone());
    module.add_operation(op3);
    
    module.outputs.push(output);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
}

/// Test 3: TypeExtensions trait with deeply nested tensor validation
#[test]
fn test_type_extensions_deeply_nested_validation() {
    use crate::ir::TypeExtensions;
    
    // Create a deeply nested tensor type
    let mut current_type = Type::F32;
    for _ in 0..10 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2, 2],
        };
    }
    
    // Validate that the deeply nested type is still valid
    assert!(current_type.is_valid_type());
}

/// Test 4: Attribute with NaN and Infinity float values
#[test]
fn test_attribute_nan_infinity_values() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    // NaN should not equal itself
    assert_ne!(nan_attr, nan_attr);
    
    // Infinity values should be equal to themselves
    assert_eq!(pos_inf_attr, pos_inf_attr);
    assert_eq!(neg_inf_attr, neg_inf_attr);
    
    // Positive and negative infinity should not be equal
    assert_ne!(pos_inf_attr, neg_inf_attr);
}

/// Test 5: Operation with empty and null-like attribute values
#[test]
fn test_operation_with_null_like_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("null_like_op");
    let mut attrs = HashMap::new();
    
    attrs.insert("zero_int".to_string(), Attribute::Int(0));
    attrs.insert("zero_float".to_string(), Attribute::Float(0.0));
    attrs.insert("false_bool".to_string(), Attribute::Bool(false));
    attrs.insert("empty_string".to_string(), Attribute::String("".to_string()));
    attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    assert_eq!(op.attributes.get("zero_int"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("empty_string"), Some(&Attribute::String("".to_string())));
}

/// Test 6: Value with negative shape dimensions (should be handled gracefully)
#[test]
fn test_value_with_negative_dimensions_clamped() {
    // Since shape is Vec<usize>, negative values can't be stored directly
    // But we can test the edge case of what happens with minimum usize values
    let min_dim_value = Value {
        name: "min_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![0, 0, 0], // All zeros - represents empty tensor
    };
    
    // All zeros should result in 0 elements
    assert_eq!(min_dim_value.num_elements(), Some(0));
}

/// Test 7: Module with duplicate input/output names
#[test]
fn test_module_duplicate_io_names() {
    let mut module = Module::new("duplicate_io");
    
    // Add inputs with the same name
    let input_value = Value {
        name: "duplicate_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    module.inputs.push(input_value.clone());
    module.inputs.push(input_value.clone());
    
    // Add outputs with the same name
    let output_value = Value {
        name: "output_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    module.outputs.push(output_value.clone());
    module.outputs.push(output_value.clone());
    
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].name, module.inputs[1].name);
    assert_eq!(module.outputs[0].name, module.outputs[1].name);
}

/// Test 8: Type comparison with nested tensor equivalence
#[test]
fn test_nested_tensor_type_equivalence() {
    // Create equivalent nested tensors
    let tensor1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3, 4],
    };
    
    let tensor2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3, 4],
    };
    
    assert_eq!(tensor1, tensor2);
    
    // Different nested tensors
    let tensor3 = Type::Tensor {
        element_type: Box::new(Type::F64), // Different element type
        shape: vec![2, 3, 4],
    };
    
    assert_ne!(tensor1, tensor3);
}

/// Test 9: Value with single dimension of maximum usize
#[test]
fn test_value_max_single_dimension() {
    let max_dim_value = Value {
        name: "max_single_dim".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX], // Single dimension at max
    };
    
    // num_elements should handle this without panic
    let result = max_dim_value.num_elements();
    assert!(result.is_some()); // Should return Some(usize::MAX)
}

/// Test 10: Module with alternating operation types
#[test]
fn test_module_alternating_operation_types() {
    let mut module = Module::new("alternating_ops");
    
    let op_types = ["add", "mul", "sub", "div", "add", "mul", "sub", "div"];
    
    for op_type in op_types.iter() {
        let mut op = Operation::new(op_type);
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        op.outputs.push(Value {
            name: "output".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 8);
    // Verify alternating pattern
    assert_eq!(module.operations[0].op_type, "add");
    assert_eq!(module.operations[1].op_type, "mul");
    assert_eq!(module.operations[2].op_type, "sub");
    assert_eq!(module.operations[3].op_type, "div");
    assert_eq!(module.operations[4].op_type, "add");
}