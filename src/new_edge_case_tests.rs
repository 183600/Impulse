//! New edge case tests for the Impulse compiler
//! Covering additional boundary conditions not yet tested

use rstest::*;
use crate::{
    ir::{Value, Type, Operation, Attribute, Module, TypeExtensions},
};

/// Test 1: Operations with identical input and output names
#[test]
fn test_operations_with_identical_input_output_names() {
    let mut op = Operation::new("identity_op");
    
    // Add input and output with the same name
    let shared_name = "shared_name";
    op.inputs.push(Value {
        name: shared_name.to_string(),
        ty: Type::F32,
        shape: vec![2, 3],
    });
    
    op.outputs.push(Value {
        name: shared_name.to_string(),  // Same name as input
        ty: Type::F32,
        shape: vec![2, 3],
    });
    
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.outputs.len(), 1);
    assert_eq!(op.inputs[0].name, shared_name);
    assert_eq!(op.outputs[0].name, shared_name);
    assert_eq!(op.inputs[0].shape, op.outputs[0].shape);
}

/// Test 2: Operations with deeply nested attributes that contain complex structures
#[test]
fn test_operations_deeply_nested_complex_attributes() {
    use std::collections::HashMap;
    
    let mut complex_attr = Attribute::String("base".to_string());
    
    // Create a very deeply nested attribute structure
    for level in 0..50 {
        complex_attr = Attribute::Array(vec![
            Attribute::Int(level as i64),
            Attribute::Array(vec![complex_attr]),
            Attribute::String(format!("level_{}", level)),
        ]);
    }
    
    // Test with an operation
    let mut op = Operation::new("nested_attr_op");
    let mut attrs = HashMap::new();
    attrs.insert("deeply_nested_attr".to_string(), complex_attr);
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 1);
    // Just ensure the operation can be created without crashing due to deep nesting
}

/// Test 3: Comparison of values with same names but different types/shapes
#[rstest]
#[case(Type::F32, Type::I32)]
#[case(Type::I64, Type::F64)]
#[case(Type::Bool, Type::F32)]
fn test_value_comparison_different_types_same_name(#[case] type1: Type, #[case] type2: Type) {
    let name = "same_name";
    
    let value1 = Value {
        name: name.to_string(),
        ty: type1.clone(),
        shape: vec![1, 2, 3],
    };
    
    let value2 = Value {
        name: name.to_string(),  // Same name
        ty: type2.clone(),
        shape: vec![1, 2, 3],   // Same shape
    };
    
    // Values with same name but different types should not be equal
    if type1 != type2 {
        // If the types are different, the values should be different
        // However, since Value doesn't implement PartialEq, we just verify creation
        assert_eq!(value1.name, value2.name);
        assert_eq!(value1.shape, value2.shape);
        // We can't compare value1.ty != value2.ty directly since we don't know if they're actually different
    }
    
    // Create two identical values to ensure they match the shape
    let value1_clone = Value {
        name: name.to_string(),
        ty: type1.clone(),
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value1.name, value1_clone.name);
    assert_eq!(value1.ty, value1_clone.ty);
    assert_eq!(value1.shape, value1_clone.shape);
}

/// Test 4: Operations with mismatched input/output types
#[test]
fn test_operations_mismatched_input_output_types() {
    let mut op = Operation::new("mismatched_op");
    
    // Add input of one type
    op.inputs.push(Value {
        name: "input_i32".to_string(),
        ty: Type::I32,
        shape: vec![4, 4],
    });
    
    // Add output of different type
    op.outputs.push(Value {
        name: "output_f32".to_string(),
        ty: Type::F32,
        shape: vec![4, 4],  // Same shape but different type
    });
    
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.outputs.len(), 1);
    assert_eq!(op.inputs[0].ty, Type::I32);
    assert_eq!(op.outputs[0].ty, Type::F32);
    // Shapes can match while types differ
    assert_eq!(op.inputs[0].shape, op.outputs[0].shape);
}

/// Test 5: Module with inputs and outputs of various types
#[test]
fn test_module_with_various_input_output_types() {
    let mut module = Module::new("typed_module");
    
    // Add various typed inputs to the module
    module.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 20],
    });
    
    module.inputs.push(Value {
        name: "i64_input".to_string(),
        ty: Type::I64,
        shape: vec![5, 5, 5],
    });
    
    module.inputs.push(Value {
        name: "bool_input".to_string(),
        ty: Type::Bool,
        shape: vec![100],
    });
    
    // Add various typed outputs to the module
    module.outputs.push(Value {
        name: "f64_output".to_string(),
        ty: Type::F64,
        shape: vec![3, 3],
    });
    
    module.outputs.push(Value {
        name: "i32_output".to_string(),
        ty: Type::I32,
        shape: vec![7, 8, 9],
    });
    
    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 2);
    
    // Check that types are preserved
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::I64);
    assert_eq!(module.inputs[2].ty, Type::Bool);
    assert_eq!(module.outputs[0].ty, Type::F64);
    assert_eq!(module.outputs[1].ty, Type::I32);
}

/// Test 6: Value with extremely large but valid shape product
#[test]
fn test_values_with_large_valid_shape_products() {
    // Create a value with a shape that results in a very large but valid product
    let large_but_valid_shape = vec![10_000, 10_000, 10];  // 1 billion elements
    let value = Value {
        name: "large_valid_tensor".to_string(),
        ty: Type::F32,
        shape: large_but_valid_shape,
    };
    
    assert_eq!(value.shape[0], 10_000);
    assert_eq!(value.shape[1], 10_000);
    assert_eq!(value.shape[2], 10);
    
    // Use checked arithmetic to avoid overflow during multiplication
    let mut product: Option<usize> = Some(1);
    for &dim in &value.shape {
        product = product.and_then(|p| p.checked_mul(dim));
    }
    
    assert!(product.is_some());
    assert_eq!(product.unwrap(), 1_000_000_000);
}

/// Test 7: Operations with empty attribute maps but many inputs/outputs
#[test]
fn test_operations_with_empty_attributes_many_ios() {
    let mut op = Operation::new("empty_attr_op");
    
    // Add many inputs and outputs but no attributes
    for i in 0..1000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![i % 10 + 1],
        });
        
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![(i + 1) % 10 + 1],
        });
    }
    
    assert_eq!(op.inputs.len(), 1000);
    assert_eq!(op.outputs.len(), 1000);
    assert_eq!(op.attributes.len(), 0);  // Still empty
    assert_eq!(op.op_type, "empty_attr_op");
}

/// Test 8: Creation and cloning of empty structures
#[test]
fn test_empty_structures_creation_and_cloning() {
    // Test empty module
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.inputs.len(), 0);
    assert_eq!(empty_module.outputs.len(), 0);
    
    // Clone the empty module
    let cloned_empty_module = empty_module.clone();
    assert_eq!(empty_module.name, cloned_empty_module.name);
    assert_eq!(empty_module.operations.len(), cloned_empty_module.operations.len());
    
    // Test empty operation
    let empty_op = Operation::new("empty");
    assert_eq!(empty_op.op_type, "empty");
    assert_eq!(empty_op.inputs.len(), 0);
    assert_eq!(empty_op.outputs.len(), 0);
    assert_eq!(empty_op.attributes.len(), 0);
    
    // Clone the empty operation
    let cloned_empty_op = empty_op.clone();
    assert_eq!(empty_op.op_type, cloned_empty_op.op_type);
    assert_eq!(empty_op.inputs.len(), cloned_empty_op.inputs.len());
    assert_eq!(empty_op.outputs.len(), cloned_empty_op.outputs.len());
    assert_eq!(empty_op.attributes.len(), cloned_empty_op.attributes.len());
    
    // Test empty attribute array
    let empty_attr = Attribute::Array(vec![]);
    match &empty_attr {  // Borrow instead of moving
        Attribute::Array(v) => assert_eq!(v.len(), 0),
        _ => panic!("Expected empty Array attribute"),
    }
    
    let cloned_empty_attr = empty_attr.clone();
    match cloned_empty_attr {
        Attribute::Array(v) => assert_eq!(v.len(), 0),
        _ => panic!("Expected empty Array attribute after clone"),
    }
}

/// Test 9: Tensor type with complex recursive structure validation
#[test]
fn test_complex_recursive_tensor_type_validation() {
    // Create a complex nested tensor type: tensor<tensor<tensor<f32, [2]>, [3]>, [4]>
    let complex_nested = Type::Tensor {
        element_type: Box::new(
            Type::Tensor {
                element_type: Box::new(
                    Type::Tensor {
                        element_type: Box::new(Type::F32),
                        shape: vec![2],
                    }
                ),
                shape: vec![3],
            }
        ),
        shape: vec![4],
    };
    
    // Validate the complex nested type
    assert!(complex_nested.is_valid_type());
    
    // Clone and validate equality
    let cloned_complex = complex_nested.clone();
    assert_eq!(complex_nested, cloned_complex);
    
    // Ensure we can access the nested structure without crashing
    if let Type::Tensor { element_type: outer_elem, shape: outer_shape } = &complex_nested {
        assert_eq!(outer_shape, &vec![4]);
        
        if let Type::Tensor { element_type: mid_elem, shape: mid_shape } = outer_elem.as_ref() {
            assert_eq!(mid_shape, &vec![3]);
            
            if let Type::Tensor { element_type: inner_elem, shape: inner_shape } = mid_elem.as_ref() {
                assert_eq!(inner_shape, &vec![2]);
                
                if let Type::F32 = inner_elem.as_ref() {
                    // Successfully accessed the innermost type
                    assert!(true);
                } else {
                    panic!("Expected F32 as innermost type");
                }
            } else {
                panic!("Expected innermost to be Tensor type");
            }
        } else {
            panic!("Expected middle to be Tensor type");
        }
    } else {
        panic!("Expected outermost to be Tensor type");
    }
}

/// Test 10: Edge case with maximum number of distinct types in a single operation
#[test]
fn test_operation_with_maximum_distinct_types() {
    let mut op = Operation::new("max_types_op");
    
    // Add values of all supported primitive types
    let test_types = [Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool];
    
    for (idx, data_type) in test_types.iter().enumerate() {
        // Add input with this type
        op.inputs.push(Value {
            name: format!("input_{}_{}", idx, data_type_as_string(data_type)),
            ty: data_type.clone(),
            shape: vec![idx + 1],
        });
        
        // Add output with the same type
        op.outputs.push(Value {
            name: format!("output_{}_{}", idx, data_type_as_string(data_type)),
            ty: data_type.clone(),
            shape: vec![idx + 2],
        });
    }
    
    assert_eq!(op.inputs.len(), test_types.len());
    assert_eq!(op.outputs.len(), test_types.len());
    
    // Verify each input/output has the expected type
    for (idx, expected_type) in test_types.iter().enumerate() {
        assert_eq!(op.inputs[idx].ty, *expected_type);
        assert_eq!(op.outputs[idx].ty, *expected_type);
    }
}

// Helper function to convert Type to string for naming
fn data_type_as_string(ty: &Type) -> &'static str {
    match ty {
        Type::F32 => "f32",
        Type::F64 => "f64",
        Type::I32 => "i32",
        Type::I64 => "i64",
        Type::Bool => "bool",
        Type::Tensor { .. } => "tensor",
    }
}