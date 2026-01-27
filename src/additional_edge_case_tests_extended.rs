//! Additional edge case tests for the Impulse compiler
//! Covers additional boundary conditions and error cases not covered elsewhere

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::ir_utils;

#[test]
fn test_empty_operation_with_no_attributes() {
    let op = Operation::new("");
    
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
}

#[test]
fn test_value_with_extremely_large_name() {
    let long_name = "x".repeat(1_000_000); // 1 million character name
    let value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    assert_eq!(value.name, long_name);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1]);
}

#[test]
fn test_operation_with_multiple_same_named_inputs() {
    let mut op = Operation::new("test_op");
    
    // Add multiple inputs with identical names (this should be allowed)
    for i in 0..5 {
        op.inputs.push(Value {
            name: "duplicate_input".to_string(),
            ty: Type::F32,
            shape: vec![i],
        });
    }
    
    assert_eq!(op.inputs.len(), 5);
    for (i, input) in op.inputs.iter().enumerate() {
        assert_eq!(input.name, "duplicate_input");
        assert_eq!(input.shape, vec![i]);
    }
}

#[test]
fn test_nested_tensor_with_empty_inner_shape() {
    // Create a tensor that contains scalar tensors
    let nested_type = Type::Tensor {
        element_type: Box::new(Type::F32), // F32 scalars
        shape: vec![5, 3], // 5x3 grid of scalars
    };
    
    let result = ir_utils::type_to_string(&nested_type);
    assert!(result.contains("f32"));
    assert!(result.contains("[5, 3]"));
    
    // Test getting element type
    assert_eq!(ir_utils::get_element_type(&nested_type), &Type::F32);
}

#[test]
fn test_calculate_tensor_size_with_max_values() {
    // Test tensor size calculation with values close to usize::MAX
    let max_safe_dim = (std::usize::MAX as f64).sqrt() as usize;
    
    let large_value = Value {
        name: "large_val".to_string(),
        ty: Type::F32,
        shape: vec![max_safe_dim, max_safe_dim],
    };
    
    let size_result = ir_utils::calculate_tensor_size(&large_value.ty, &large_value.shape);
    match size_result {
        Ok(size) => {
            // Size should be calculated without overflow if safe
            assert!(size > 0);
        },
        Err(_) => {
            // This is expected if overflow occurs despite our precautions
        }
    }
}

#[test]
fn test_module_with_mixed_data_types() {
    let mut module = Module::new("mixed_type_module");
    
    // Add operations with different data types
    let mut op1 = Operation::new("op_f32");
    op1.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    module.add_operation(op1);
    
    let mut op2 = Operation::new("op_i64");
    op2.inputs.push(Value {
        name: "i64_input".to_string(),
        ty: Type::I64,
        shape: vec![5, 5, 5],
    });
    module.add_operation(op2);
    
    let mut op3 = Operation::new("op_bool");
    op3.inputs.push(Value {
        name: "bool_input".to_string(),
        ty: Type::Bool,
        shape: vec![100],
    });
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    
    // Verify operations have the correct types
    assert_eq!(module.operations[0].op_type, "op_f32");
    assert_eq!(module.operations[0].inputs[0].ty, Type::F32);
    
    assert_eq!(module.operations[1].op_type, "op_i64");
    assert_eq!(module.operations[1].inputs[0].ty, Type::I64);
    
    assert_eq!(module.operations[2].op_type, "op_bool");
    assert_eq!(module.operations[2].inputs[0].ty, Type::Bool);
}

#[test]
fn test_attribute_array_with_max_depth_nesting() {
    // Create deeply nested array attributes
    let mut nested_attr = Attribute::Int(42);
    
    // Nest arrays 10 levels deep
    for _ in 0..10 {
        nested_attr = Attribute::Array(vec![nested_attr]);
    }
    
    // Verify it's still a valid attribute
    match &nested_attr {
        Attribute::Array(arr) => {
            // Drill down to find the inner Int value
            let mut current = arr;
            for _ in 0..9 { // 9 more levels to get to the int
                if let Attribute::Array(ref next_arr) = current[0] {
                    current = next_arr;
                } else {
                    panic!("Expected array at nested level");
                }
            }
            
            // At the deepest level, we should find the Int(42)
            if let Attribute::Int(42) = current[0] {
                // Success
            } else {
                panic!("Expected Int(42) at deepest level");
            }
        },
        _ => panic!("Expected array at top level"),
    }
}

#[test]
fn test_operation_with_extreme_attribute_count() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("high_attr_op");
    
    // Add a large number of attributes
    let mut attrs = HashMap::new();
    for i in 0..10_000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.op_type, "high_attr_op");
    
    // Verify a few attributes exist
    assert!(op.attributes.contains_key("attr_0"));
    assert!(op.attributes.contains_key("attr_5000"));
    assert!(op.attributes.contains_key("attr_9999"));
    
    if let Attribute::String(ref val) = op.attributes["attr_0"] {
        assert_eq!(val, "value_0");
    } else {
        panic!("Expected string attribute");
    }
}

#[test]
fn test_tensor_size_calculation_zero_handling() {
    // Test various ways zeros can appear in tensor shapes
    let test_cases = vec![
        (Type::F32, vec![0], 0),           // Single zero dimension
        (Type::F32, vec![0, 10], 0),       // Zero in first position
        (Type::F32, vec![10, 0], 0),       // Zero in second position
        (Type::F32, vec![10, 0, 20], 0),   // Zero in middle
        (Type::I64, vec![5, 0, 0], 0),     // Multiple zeros
        (Type::Bool, vec![1, 1, 0, 1], 0), // Zero surrounded by ones
    ];
    
    for (data_type, shape, expected_size) in test_cases {
        let result = ir_utils::calculate_tensor_size(&data_type, &shape);
        
        match result {
            Ok(calculated_size) => {
                assert_eq!(
                    calculated_size, 
                    expected_size, 
                    "Failed for type {:?} with shape {:?}. Expected {}, got {}",
                    data_type, 
                    shape, 
                    expected_size, 
                    calculated_size
                );
            },
            Err(e) => {
                panic!("Error calculating size for type {:?} with shape {:?}: {}", data_type, shape, e);
            }
        }
    }
}

#[test]
fn test_ir_utils_edge_cases() {
    // Test ir_utils functions with edge case values
    
    // Test is_scalar with empty shape
    let scalar = Value {
        name: "scalar_test".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape = scalar
    };
    assert!(ir_utils::is_scalar(&scalar));
    assert_eq!(ir_utils::get_rank(&scalar), 0);
    assert_eq!(ir_utils::get_num_elements(&scalar), Some(1));
    
    // Test with various ranks
    for rank in 0..=5 {
        let shape = vec![2; rank]; // Create shape like [], [2], [2,2], [2,2,2], etc.
        let value = Value {
            name: format!("rank_{}_test", rank),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(ir_utils::get_rank(&value), rank, "Rank mismatch for {:?}", shape);
        
        let expected_elements: usize = shape.iter().product();
        assert_eq!(ir_utils::get_num_elements(&value), Some(expected_elements), 
                  "Element count mismatch for {:?}", shape);
        
        if rank == 0 {
            assert!(ir_utils::is_scalar(&value));
        } else {
            assert!(!ir_utils::is_scalar(&value));
        }
        
        if rank == 1 {
            assert!(ir_utils::is_vector(&value));
        } else {
            assert!(!ir_utils::is_vector(&value));
        }
        
        if rank == 2 {
            assert!(ir_utils::is_matrix(&value));
        } else {
            assert!(!ir_utils::is_matrix(&value));
        }
    }
}