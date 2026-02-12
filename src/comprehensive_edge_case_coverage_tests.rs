//! Comprehensive edge case coverage tests - Additional boundary scenarios with standard library assertions
//! This module adds 10 focused test cases covering previously untested edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

/// Test 1: Value with negative edge case in num_elements calculation
#[test]
fn test_value_checked_multiplication_edge_cases() {
    // Test with dimensions that would overflow unchecked multiplication
    // Using checked_mul to detect overflow safely
    let overflow_risk = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // Would overflow without checked arithmetic
    };
    
    // num_elements should return None for overflow cases
    assert_eq!(overflow_risk.num_elements(), None);
    
    // Test with safe but large dimensions
    let safe_large = Value {
        name: "safe_large".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100], // 10 million elements
    };
    
    assert_eq!(safe_large.num_elements(), Some(10_000_000));
}

/// Test 2: Operation with conflicting attribute names (case sensitivity)
#[test]
fn test_attribute_name_case_sensitivity() {
    let mut op = Operation::new("case_test");
    let mut attrs = HashMap::new();
    
    // Insert same logical name with different cases
    attrs.insert("Value".to_string(), Attribute::Int(1));
    attrs.insert("value".to_string(), Attribute::Int(2));
    attrs.insert("VALUE".to_string(), Attribute::Int(3));
    
    op.attributes = attrs;
    
    // All three should be stored as separate keys
    assert_eq!(op.attributes.len(), 3);
    assert_eq!(op.attributes.get("Value"), Some(&Attribute::Int(1)));
    assert_eq!(op.attributes.get("value"), Some(&Attribute::Int(2)));
    assert_eq!(op.attributes.get("VALUE"), Some(&Attribute::Int(3)));
}

/// Test 3: Module with circular input/output naming pattern
#[test]
fn test_module_circular_naming_pattern() {
    let mut module = Module::new("circular_names");
    
    // Create circular naming: input_1 -> op -> output_1 -> input_2 -> op2 -> output_2
    for i in 0..5 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(module.inputs[i].clone());
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        
        // Clone the output before moving the operation
        let output_value = op.outputs[0].clone();
        module.add_operation(op);
        module.outputs.push(output_value);
    }
    
    assert_eq!(module.inputs.len(), 5);
    assert_eq!(module.outputs.len(), 5);
    assert_eq!(module.operations.len(), 5);
}

/// Test 4: Type with recursive self-reference (edge case for validation)
#[test]
fn test_type_validity_recursive_cases() {
    // Test basic type validity
    assert!(Type::F32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
    
    // Test nested tensor validity
    let nested_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(nested_tensor.is_valid_type());
    
    // Test deeply nested tensor validity
    let mut current = Type::F32;
    for _ in 0..10 {
        current = Type::Tensor {
            element_type: Box::new(current),
            shape: vec![2],
        };
    }
    assert!(current.is_valid_type());
}

/// Test 5: Value with all primitive types including boundary values
#[test]
fn test_value_with_all_primitive_types() {
    let test_cases = vec![
        (Type::F32, vec![1], "f32_scalar"),
        (Type::F64, vec![2, 2], "f64_matrix"),
        (Type::I32, vec![100], "i32_vector"),
        (Type::I64, vec![5, 5, 5], "i64_tensor"),
        (Type::Bool, vec![3, 3, 3], "bool_tensor"),
    ];
    
    for (ty, shape, name) in test_cases {
        let value = Value {
            name: name.to_string(),
            ty: ty.clone(),
            shape: shape.clone(),
        };
        
        assert_eq!(value.ty, ty);
        assert_eq!(value.shape, shape);
        assert_eq!(value.name, name);
    }
}

/// Test 6: Operation with empty operation type string
#[test]
fn test_operation_with_empty_op_type() {
    let mut op = Operation::new(""); // Empty string operation type
    
    assert_eq!(op.op_type, "");
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    
    // Should still be able to add attributes
    op.attributes.insert("test".to_string(), Attribute::Int(42));
    assert_eq!(op.attributes.len(), 1);
}

/// Test 7: Module with operations having same input values (sharing)
#[test]
fn test_module_with_shared_inputs() {
    let mut module = Module::new("shared_inputs");
    
    // Create a single input that's used by multiple operations
    let shared_input = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    module.inputs.push(shared_input.clone());
    
    // Create 5 operations all using the same input
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(shared_input.clone()); // Clone for sharing
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.operations.len(), 5);
    
    // All operations should have the same input
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.inputs[0].name, "shared");
    }
}

/// Test 8: Attribute with very large integer array
#[test]
fn test_large_integer_array_attribute() {
    // Create a large array of integers
    let large_array: Vec<Attribute> = (0..1000)
        .map(|i| Attribute::Int(i as i64))
        .collect();
    
    let attr = Attribute::Array(large_array);
    
    match attr {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1000);
            
            // Verify first and last elements
            match &arr[0] {
                Attribute::Int(0) => (),
                _ => panic!("First element should be Int(0)"),
            }
            
            match &arr[999] {
                Attribute::Int(999) => (),
                _ => panic!("Last element should be Int(999)"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Value with alternating pattern in shape
#[test]
fn test_value_with_alternating_shape_pattern() {
    let patterns = vec![
        vec![1, 2, 1, 2, 1], // Alternating 1 and 2
        vec![10, 1, 10, 1],   // Alternating 10 and 1
        vec![1, 1, 1, 1],     // All ones
        vec![5, 5, 5, 5],     // All same
    ];
    
    for shape in patterns {
        let value = Value {
            name: "pattern_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Verify num_elements calculation
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 10: Module with no operations but with inputs and outputs
#[test]
fn test_module_without_operations() {
    let mut module = Module::new("no_ops");
    
    // Add inputs and outputs without any operations
    for i in 0..3 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        
        module.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
    }
    
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 3);
    assert_eq!(module.name, "no_ops");
}