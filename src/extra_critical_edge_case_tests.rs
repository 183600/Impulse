//! Extra critical edge case tests for the Impulse compiler
//! Using standard library assert! and assert_eq! for boundary condition coverage

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::{calculate_tensor_size, calculate_tensor_size_safe, get_rank};

#[test]
fn test_value_with_num_elements_overflow_protection() {
    // Test that num_elements() returns None when overflow would occur
    let huge_dims: Vec<usize> = vec![usize::MAX / 2, 3]; // This would overflow
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: huge_dims,
    };
    
    let result = value.num_elements();
    assert!(result.is_none(), "Expected None for overflow case");
}

#[test]
fn test_calculate_tensor_size_with_zero_dimensions() {
    // Test various zero dimension patterns
    let shapes = vec![
        vec![0],
        vec![10, 0],
        vec![0, 10],
        vec![5, 0, 3],
        vec![0, 0, 0],
    ];
    
    for shape in shapes {
        let size = calculate_tensor_size(&Type::F32, &shape);
        assert_eq!(size.unwrap(), 0, "Zero dimensions should result in size 0");
    }
}

#[test]
fn test_nested_tensor_size_edge_case() {
    // Test nested tensor with zero inner dimensions
    let inner_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![0], // Zero-sized inner tensor
    };
    let outer_tensor = Type::Tensor {
        element_type: Box::new(inner_tensor),
        shape: vec![10],
    };
    
    let size = calculate_tensor_size(&outer_tensor, &[]);
    assert_eq!(size.unwrap(), 0, "Nested tensor with zero inner size should be 0");
}

#[test]
fn test_value_rank_edge_cases() {
    // Test rank for various tensor dimensions
    let test_cases = vec![
        (vec![], 0),           // scalar
        (vec![1], 1),          // 1D
        (vec![2, 3], 2),       // 2D
        (vec![4, 5, 6], 3),    // 3D
        (vec![1, 1, 1, 1], 4), // 4D with all ones
        (vec![100], 1),        // 1D large
        (vec![10, 0], 2),      // 2D with zero dim
    ];
    
    for (shape, expected_rank) in test_cases {
        let value = Value {
            name: format!("rank_test_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        assert_eq!(get_rank(&value), expected_rank);
    }
}

#[test]
fn test_calculate_tensor_size_safe_max_usize() {
    // Test that calculate_tensor_size_safe handles usize::MAX properly
    let overflow_shape = vec![usize::MAX, 2];
    let result = calculate_tensor_size_safe(&overflow_shape);
    assert!(result.is_none(), "Should return None for overflow");
}

#[test]
fn test_scalar_tensor_size_for_all_types() {
    // Test scalar (empty shape) size for all data types
    let types_and_sizes = vec![
        (Type::F32, 4),
        (Type::F64, 8),
        (Type::I32, 4),
        (Type::I64, 8),
        (Type::Bool, 1),
    ];
    
    for (dtype, expected_size) in types_and_sizes {
        let size = calculate_tensor_size(&dtype, &[]);
        assert_eq!(size.unwrap(), expected_size, "Scalar size mismatch for {:?}", dtype);
    }
}

#[test]
fn test_module_with_empty_operations_list() {
    // Test module with explicitly empty operations
    let mut module = Module::new("empty_ops_module");
    assert_eq!(module.operations.len(), 0);
    
    // Try to add and immediately remove operations
    let op = Operation::new("temp_op");
    module.add_operation(op);
    assert_eq!(module.operations.len(), 1);
    
    module.operations.clear();
    assert_eq!(module.operations.len(), 0);
}

#[test]
fn test_attribute_with_special_float_values() {
    // Test attribute creation with special float values
    let special_floats = vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        0.0,
        -0.0,
        f64::MIN,
        f64::MAX,
    ];
    
    for float_val in special_floats {
        let attr = Attribute::Float(float_val);
        match attr {
            Attribute::Float(v) => {
                // Verify the value is stored (even if NaN doesn't equal itself)
                if float_val.is_nan() {
                    assert!(v.is_nan(), "NaN should remain NaN");
                } else if float_val.is_infinite() {
                    assert!(v.is_infinite(), "Infinity should remain infinite");
                } else {
                    assert_eq!(v, float_val, "Float value mismatch");
                }
            }
            _ => panic!("Expected Float attribute"),
        }
    }
}

#[test]
fn test_value_with_single_dimension_tensors() {
    // Test 1D tensors of various sizes
    let test_sizes = vec![0, 1, 2, 10, 100, 1000, 10000];
    
    for size in test_sizes {
        let value = Value {
            name: format!("vec_{}", size),
            ty: Type::I32,
            shape: vec![size],
        };
        
        assert_eq!(value.num_elements(), Some(size));
        assert_eq!(get_rank(&value), 1);
        
        let expected_bytes = size * 4; // I32 = 4 bytes
        let tensor_size = calculate_tensor_size(&Type::I32, &vec![size]);
        assert_eq!(tensor_size.unwrap(), expected_bytes);
    }
}

#[test]
fn test_module_with_multiple_operations_same_type() {
    // Test module with many operations of the same type
    let mut module = Module::new("multi_ops_same_type");
    
    // Add 100 operations all of type "add"
    for i in 0..100 {
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: format!("in_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: format!("out_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    
    // Verify all operations are "add"
    for op in &module.operations {
        assert_eq!(op.op_type, "add");
    }
}