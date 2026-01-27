//! Additional edge case tests for the Impulse compiler
//! Covering boundary conditions and extreme inputs that may not be tested elsewhere

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils,
};
use rstest::*;

// Test for empty shapes and extreme edge cases in tensor calculations
#[rstest]
#[case(vec![], 1)]  // scalar
#[case(vec![0], 0)]  // zero-dimension tensor
#[case(vec![1], 1)]  // single element
#[case(vec![0, 1], 0)]  // mixed zeros
#[case(vec![1, 0], 0)]  // mixed zeros
#[case(vec![0, 0, 0], 0)]  // all zeros
#[case(vec![1, 1, 1], 1)]  // all ones
fn test_edge_case_shape_products(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
    let actual_product: usize = shape.iter().product();
    assert_eq!(actual_product, expected_product);
}

// Test with very large numbers that could potentially cause overflow in calculations
#[test]
fn test_large_shape_values_no_overflow() {
    // Test large but reasonable shapes
    let large_shape = vec![10_000, 10_000];
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    // Calculate with checked multiplication to prevent overflow
    let product_result: Option<usize> = value.shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
    assert!(product_result.is_some());
    
    if let Some(product) = product_result {
        assert_eq!(product, 100_000_000); // 10k * 10k
    }
}

// Test potential overflow in tensor size calculations
#[test]
fn test_tensor_size_calculation_overflow() {
    // Test with values that could cause overflow
    let huge_dims = vec![usize::MAX, 2];
    let value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: huge_dims,
    };
    
    // Use the safe calculation method
    let result = value.num_elements();
    assert!(result.is_none());  // Should return None due to overflow
}

// Test extreme tensor rank (dimension count)
#[test]
fn test_extremely_high_rank_tensor() {
    // Create a tensor with many dimensions (rank 20)
    let high_rank_shape = vec![1; 20];  // 20 dimensions, each size 1
    let value = Value {
        name: "high_rank_tensor".to_string(),
        ty: Type::F32,
        shape: high_rank_shape,
    };
    
    assert_eq!(value.shape.len(), 20);
    assert_eq!(value.shape.iter().product::<usize>(), 1); // Product should be 1
    
    // Test with different ranks
    assert_eq!(ir_utils::get_rank(&value), 20);
    assert_eq!(ir_utils::is_scalar(&value), false);
    assert_eq!(ir_utils::is_vector(&value), false);
    assert_eq!(ir_utils::is_matrix(&value), false);
}

// Test operations with empty input/output lists
#[test]
fn test_operation_empty_io() {
    let mut op = Operation::new("empty_io_op");
    
    // Operation with no inputs or outputs
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
    
    // Still has a valid type
    assert_eq!(op.op_type, "empty_io_op");
    
    // Add some attributes but no I/O
    op.attributes.insert("param".to_string(), Attribute::Int(123));
    assert_eq!(op.attributes.len(), 1);
    assert_eq!(op.attributes.get("param"), Some(&Attribute::Int(123)));
}

// Test extreme string lengths for names
#[test]
fn test_extremely_long_names() {
    // Test with very long names (10k characters each)
    let long_name = "x".repeat(10_000);
    let module = Module::new(long_name.clone());
    assert_eq!(module.name, long_name);
    assert_eq!(module.operations.len(), 0);
    
    // Test with long value names
    let long_value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(long_value.name, long_name);
}

// Test deeply nested tensor types
#[test]
fn test_extremely_deeply_nested_tensors() {
    // Create a very deep nested tensor type
    let mut nested_type = Type::F32;
    for _ in 0..50 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![2],
        };
    }
    
    // Verify that the type is valid and can be compared
    assert!(matches!(
        nested_type,
        Type::Tensor { .. }
    ));
    
    // Check that the innermost type is F32
    let element_type = ir_utils::get_element_type(&nested_type);
    assert_eq!(element_type, &Type::F32);
    
    // Test cloning of deeply nested type
    let cloned = nested_type.clone();
    assert_eq!(nested_type, cloned);
}

// Parameterized test for different scalar vs tensor detection
#[rstest]
#[case(vec![], true, false, false)]  // scalar
#[case(vec![1], false, true, false)]  // vector of size 1
#[case(vec![5], false, true, false)]  // vector
#[case(vec![1, 1], false, false, true)]  // 1x1 matrix
#[case(vec![2, 3], false, false, true)]  // 2x3 matrix
#[case(vec![1, 1, 1], false, false, false)]  // 3D tensor
fn test_rank_classification(
    #[case] shape: Vec<usize>,
    #[case] expected_is_scalar: bool,
    #[case] expected_is_vector: bool,
    #[case] expected_is_matrix: bool
) {
    let value = Value {
        name: "test".to_string(),
        ty: Type::F32,
        shape: shape,
    };
    
    assert_eq!(ir_utils::is_scalar(&value), expected_is_scalar);
    assert_eq!(ir_utils::is_vector(&value), expected_is_vector);
    assert_eq!(ir_utils::is_matrix(&value), expected_is_matrix);
    assert_eq!(ir_utils::get_rank(&value), value.shape.len());
}

// Test tensor size calculation with all primitive types
#[rstest]
#[case(Type::F32, 4)]
#[case(Type::F64, 8)]
#[case(Type::I32, 4)]
#[case(Type::I64, 8)]
#[case(Type::Bool, 1)]
fn test_primitive_type_sizes(#[case] dtype: Type, #[case] expected_bytes_per_element: usize) {
    // Test scalar
    let scalar_size = ir_utils::calculate_tensor_size(&dtype, &[]).unwrap();
    assert_eq!(scalar_size, expected_bytes_per_element);
    
    // Test 1D tensor with 10 elements
    let vector_size = ir_utils::calculate_tensor_size(&dtype, &[10]).unwrap();
    assert_eq!(vector_size, 10 * expected_bytes_per_element);
    
    // Test 2D tensor with 6 elements
    let matrix_size = ir_utils::calculate_tensor_size(&dtype, &[2, 3]).unwrap();
    assert_eq!(matrix_size, 6 * expected_bytes_per_element);
    
    // Test tensor with zero dimension
    let zero_size = ir_utils::calculate_tensor_size(&dtype, &[5, 0, 10]).unwrap();
    assert_eq!(zero_size, 0);
}

// Test empty module behavior
#[test]
fn test_empty_module_operations() {
    let module = Module::new("");
    assert_eq!(module.name, "");
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
    
    // Test finding operations in empty module
    let counts = ir_utils::count_operations_by_type(&module);
    assert_eq!(counts.len(), 0);
    
    let ops = ir_utils::find_operations_by_type(&module, "any_op");
    assert_eq!(ops.len(), 0);
}

// Test with maximum possible values for shape dimensions
#[test]
fn test_max_size_tensor_dimensions() {
    let max_shape = vec![std::cmp::min(1_000_000, usize::MAX / 1000); 2]; // Limit size to prevent real overflow
    let value = Value {
        name: "max_tensor".to_string(),
        ty: Type::F32,
        shape: max_shape.clone(),
    };
    
    assert_eq!(value.shape, max_shape);
    assert_eq!(value.shape.len(), 2);
    
    // Calculate safely to avoid actual overflow
    let num_elements = value.num_elements();
    assert!(num_elements.is_some());
}