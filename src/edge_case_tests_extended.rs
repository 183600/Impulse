//! Extended edge case tests for the Impulse compiler IR
//! This file contains additional tests covering more boundary conditions

use rstest::*;
use crate::ir::{Module, Value, Type, Operation};
use crate::utils::ir_utils;

// Test edge cases for tensor shapes with zeros and empty shapes
#[rstest]
#[case(vec![], 1)]  // scalar (empty shape)
#[case(vec![0], 0)]  // dimension of 0
#[case(vec![0, 10], 0)]  // contains 0
#[case(vec![1, 0, 100], 0)]  // contains 0
#[case(vec![1], 1)]  // single element
#[case(vec![1, 1, 1], 1)]  // all ones
fn test_tensor_shape_edge_cases(#[case] shape: Vec<usize>, #[case] expected_elements: usize) {
    let value = Value {
        name: "test".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let num_elements = ir_utils::get_num_elements(&value).unwrap();
    assert_eq!(num_elements, expected_elements);
}

// Test different data types with extreme shapes
#[rstest]
#[case(Type::F32, vec![], 4)]           // scalar F32
#[case(Type::F64, vec![], 8)]           // scalar F64
#[case(Type::Bool, vec![], 1)]          // scalar Bool
#[case(Type::I32, vec![0], 0)]          // zero-element I32 tensor
#[case(Type::F32, vec![1000, 1000], 4 * 1000 * 1000)]  // large F32 tensor
fn test_tensor_size_calculation(#[case] data_type: Type, #[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let actual_size = ir_utils::calculate_tensor_size(&data_type, &shape).unwrap();
    assert_eq!(actual_size, expected_size);
}

// Test deeply nested tensor types
#[test]
fn test_deeply_nested_tensor_types() {
    // Create multiple levels of nested tensors
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![5],
    };

    // Verify the nested structure
    if let Type::Tensor { element_type: box3, shape: outer_shape } = &level4 {
        assert_eq!(outer_shape, &vec![5]);
        
        if let Type::Tensor { element_type: box2, shape: mid_shape } = box3.as_ref() {
            assert_eq!(mid_shape, &vec![4]);
            
            if let Type::Tensor { element_type: box1, shape: inner_shape } = box2.as_ref() {
                assert_eq!(inner_shape, &vec![3]);
                
                if let Type::Tensor { element_type: base_type, shape: base_shape } = box1.as_ref() {
                    assert_eq!(base_shape, &vec![2]);
                    
                    if let Type::F32 = base_type.as_ref() {
                        // Successful nested structure verification
                        assert!(true);
                    } else {
                        panic!("Expected F32 at the base level");
                    }
                } else {
                    panic!("Expected Tensor at the third level");
                }
            } else {
                panic!("Expected Tensor at the second level");
            }
        } else {
            panic!("Expected Tensor at the first level");
        }
    } else {
        panic!("Expected Tensor as outer type");
    }
}

// Test operations with extreme numbers of inputs and outputs
#[test]
fn test_operation_extreme_io_counts() {
    let mut op = Operation::new("extreme_io_op");
    
    // Add 10,000 inputs
    for i in 0..10000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    // Add 5,000 outputs
    for i in 0..5000 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
    }
    
    assert_eq!(op.inputs.len(), 10000);
    assert_eq!(op.outputs.len(), 5000);
    assert_eq!(op.op_type, "extreme_io_op");
}

// Test attribute edge cases including nested arrays
#[test]
fn test_attribute_edge_cases() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("attr_test_op");
    let mut attrs = HashMap::new();
    
    // Test deeply nested array attributes
    let nested_array = crate::ir::Attribute::Array(vec![
        crate::ir::Attribute::Array(vec![
            crate::ir::Attribute::Int(1),
            crate::ir::Attribute::Int(2),
        ]),
        crate::ir::Attribute::Array(vec![
            crate::ir::Attribute::Float(3.14),
            crate::ir::Attribute::Float(2.71),
        ]),
        crate::ir::Attribute::Array(vec![
            crate::ir::Attribute::String("nested".to_string()),
            crate::ir::Attribute::Bool(true),
        ]),
    ]);
    
    attrs.insert("deeply_nested".to_string(), nested_array);
    
    // Large string attribute
    let long_string = "a".repeat(100_000); // 100k character string
    attrs.insert("long_string".to_string(), crate::ir::Attribute::String(long_string));
    
    // Large numeric values
    attrs.insert("large_int".to_string(), crate::ir::Attribute::Int(i64::MAX));
    attrs.insert("small_int".to_string(), crate::ir::Attribute::Int(i64::MIN));
    attrs.insert("zero_int".to_string(), crate::ir::Attribute::Int(0));  // This makes it 5 total
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    assert!(op.attributes.contains_key("deeply_nested"));
    assert!(op.attributes.contains_key("long_string"));
    assert!(op.attributes.contains_key("large_int"));
    assert!(op.attributes.contains_key("small_int"));
    assert!(op.attributes.contains_key("zero_int"));
}

// Test for potential overflow in tensor size calculations
#[test]
fn test_potential_overflow_scenarios() {
    // Very large but potentially safe dimensions that could cause overflow
    let large_shape = vec![100_000, 100_000];  // 10 billion elements
    
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    // This should successfully compute the number of elements
    let num_elements = ir_utils::get_num_elements(&value);
    
    // Check that it doesn't overflow
    if let Some(elements) = num_elements {
        assert_eq!(elements, 10_000_000_000);
    } else {
        // If it returns None, that's also acceptable as it indicates overflow protection
        assert!(true);
    }
    
    // Test with zero dimensions (should return Some(0))
    let zero_shape = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1000, 0, 500],
    };
    
    let zero_elements = ir_utils::get_num_elements(&zero_shape).unwrap();
    assert_eq!(zero_elements, 0);
}

// Test module with many operations and values
#[test]
fn test_large_module_operations() {
    let mut module = Module::new("large_module");
    
    // Add 50,000 operations to the module
    for i in 0..50_000 {
        let mut op = Operation::new(&format!("operation_{}", i));
        
        // Add a few inputs and outputs to each operation
        op.inputs.push(Value {
            name: format!("input_{}_0", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        op.outputs.push(Value {
            name: format!("output_{}_0", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 50_000);
    assert_eq!(module.name, "large_module");
    
    // Verify a few operations still maintain valid data
    let first_op = &module.operations[0];
    let last_op = &module.operations[49_999];
    let middle_op = &module.operations[25_000];
    
    assert_eq!(first_op.op_type, "operation_0");
    assert_eq!(last_op.op_type, "operation_49999");
    assert_eq!(middle_op.op_type, "operation_25000");
}

// Test utility functions with edge cases
#[rstest]
#[case(vec![], true, 0, 1)]  // scalar
#[case(vec![5], false, 1, 5)]  // vector
#[case(vec![3, 4], false, 2, 12)]  // matrix
#[case(vec![2, 3, 4], false, 3, 24)]  // 3D tensor
#[case(vec![0], false, 1, 0)]  // zero-size tensor
#[case(vec![1, 1, 1, 1], false, 4, 1)]  // multi-dimensional unit tensor
fn test_ir_utility_functions(
    #[case] shape: Vec<usize>,
    #[case] expected_scalar: bool,
    #[case] expected_rank: usize,
    #[case] expected_elements: usize
) {
    let value = Value {
        name: "test_value".to_string(),
        ty: Type::F32,
        shape,
    };
    
    assert_eq!(ir_utils::is_scalar(&value), expected_scalar);
    assert_eq!(ir_utils::get_rank(&value), expected_rank);
    
    let num_elements = ir_utils::get_num_elements(&value).unwrap();
    assert_eq!(num_elements, expected_elements);
    
    if expected_rank == 1 {
        assert!(ir_utils::is_vector(&value));
    } else {
        assert!(!ir_utils::is_vector(&value));
    }
    
    if expected_rank == 2 {
        assert!(ir_utils::is_matrix(&value));
    } else {
        assert!(!ir_utils::is_matrix(&value));
    }
}

// Test nested tensor size calculations
#[test]
fn test_nested_tensor_size_calculation() {
    // Create a nested tensor: tensor<tensor<f32, [2, 3]>, [4]>
    let inner_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],  // 6 F32 elements = 24 bytes
    };
    
    // Calculate size of inner tensor: 2 * 3 * 4 bytes per F32 = 24 bytes
    let inner_size = ir_utils::calculate_tensor_size(&inner_tensor, &[]).unwrap();
    assert_eq!(inner_size, 24); // 2 * 3 * 4 bytes per F32
    
    // Now create the outer tensor as a regular tensor with nested element type
    // The size for a regular tensor with nested type should be calculated as:
    // number_of_outer_elements * size_of_inner_tensor
    let outer_shape = vec![4]; // 4 outer elements
    // This represents 4 copies of the inner tensor
    let calculated_size = ir_utils::calculate_tensor_size(&inner_tensor, &outer_shape).unwrap();
    assert_eq!(calculated_size, 96); // 4 * 24 bytes
}

// Test module validation with edge cases
#[test]
fn test_module_operations_counting() {
    let mut module = Module::new("count_test_module");
    
    // Add operations of different types
    for _i in 0..100 {
        let op = Operation::new("add");
        module.add_operation(op);
    }
    
    for _i in 0..50 {
        let op = Operation::new("multiply");
        module.add_operation(op);
    }
    
    for _i in 0..25 {
        let op = Operation::new("conv2d");
        module.add_operation(op);
    }
    
    // Test counting operations by type
    let counts = ir_utils::count_operations_by_type(&module);
    assert_eq!(counts.get("add"), Some(&100));
    assert_eq!(counts.get("multiply"), Some(&50));
    assert_eq!(counts.get("conv2d"), Some(&25));
    
    // Test finding operations by type
    let add_ops = ir_utils::find_operations_by_type(&module, "add");
    assert_eq!(add_ops.len(), 100);
    
    let multiply_ops = ir_utils::find_operations_by_type(&module, "multiply");
    assert_eq!(multiply_ops.len(), 50);
    
    let nonexistent_ops = ir_utils::find_operations_by_type(&module, "nonexistent");
    assert_eq!(nonexistent_ops.len(), 0);
}