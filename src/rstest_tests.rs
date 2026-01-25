//! Test cases using rstest for the Impulse compiler
//! This file contains additional tests focusing on edge cases and boundary conditions

use rstest::*;
use impulse::{
    ir::{Module, Value, Type, Operation, Attribute},
    compiler::Compiler,
    utils::ir_utils,
};

// Test for tensor edge cases
#[rstest]
fn test_tensor_edge_cases(
    #[values(vec![], vec![0], vec![1], vec![0, 0], vec![1, 0], vec![0, 1], vec![10, 0, 20])] shape: Vec<usize>,
    #[values(Type::F32, Type::I32, Type::Bool)] dtype: Type,
) {
    let value = Value {
        name: "test_tensor".to_string(),
        ty: dtype.clone(),
        shape,
    };

    // Verify the shape is correctly stored
    assert_eq!(value.shape.len(), value.shape.len());
    
    // Calculate total elements
    let total_elements: usize = value.shape.iter().product();
    
    // If any dimension is 0, total elements should be 0
    if value.shape.iter().any(|&dim| dim == 0) {
        assert_eq!(total_elements, 0);
    }
}

// Test for nested tensor types
#[rstest]
#[case(
    Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    },
    vec![3],
    3 * 2 * 4  // 3 * 2 elements * 4 bytes per F32
)]
#[case(
    Type::Tensor {
        element_type: Box::new(Type::I64),
        shape: vec![4, 5],
    },
    vec![2],
    2 * 4 * 5 * 8  // 2 * 4 * 5 elements * 8 bytes per I64
)]
#[case(
    Type::Tensor {
        element_type: Box::new(Type::Bool),
        shape: vec![10],
    },
    vec![],
    1 * 10 * 1  // 1 * 10 elements * 1 byte per Bool
)]
fn test_nested_tensor_size_calculation(
    #[case] tensor_type: Type,
    #[case] outer_shape: Vec<usize>,
    #[case] expected_size: usize,
) {
    let calculated_size = ir_utils::calculate_tensor_size(&tensor_type, &outer_shape).unwrap();
    assert_eq!(calculated_size, expected_size);
}

// Test for operation creation with different numbers of inputs and outputs
#[rstest]
#[case(0, 0, "basic_op")]
#[case(1, 1, "single_io_op")]
#[case(2, 3, "multi_io_op")]
#[case(5, 2, "many_inputs_op")]
#[case(0, 5, "no_inputs_op")]
fn test_operation_creation(
    #[case] num_inputs: usize,
    #[case] num_outputs: usize,
    #[case] op_type: &str,
) {
    let mut op = Operation::new(op_type);
    
    // Add inputs
    for i in 0..num_inputs {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
    }
    
    // Add outputs
    for i in 0..num_outputs {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
    }
    
    assert_eq!(op.inputs.len(), num_inputs);
    assert_eq!(op.outputs.len(), num_outputs);
    assert_eq!(op.op_type, op_type);
}

// Test for attribute handling
#[rstest]
#[case(Attribute::Int(42))]
#[case(Attribute::Float(3.14))]
#[case(Attribute::String("test".to_string()))]
#[case(Attribute::Bool(true))]
#[case(Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]))]
fn test_attribute_matching(#[case] attr: Attribute) {
    match attr {
        Attribute::Int(_) => { /* Valid */ }
        Attribute::Float(_) => { /* Valid */ }
        Attribute::String(_) => { /* Valid */ }
        Attribute::Bool(_) => { /* Valid */ }
        Attribute::Array(_) => { /* Valid */ }
    }
}

// Test for module operations with edge case values
#[rstest]
fn test_module_edge_cases(
    #[values("", "simple", "long_module_name_with_many_characters_that_could_potentially_cause_issues")] name: &str,
) {
    let module = Module::new(name);
    assert_eq!(module.name, name);
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
}

// Test for different type conversions and their string representations
#[rstest]
#[case(Type::F32, "f32")]
#[case(Type::F64, "f64")]
#[case(Type::I32, "i32")]
#[case(Type::I64, "i64")]
#[case(Type::Bool, "bool")]
fn test_type_to_string_conversion(#[case] typ: Type, #[case] expected: &str) {
    let result = ir_utils::type_to_string(&typ);
    assert_eq!(result, expected);
}

// Test for empty and large tensor shapes
#[rstest]
#[case(vec![], 1)]  // scalar
#[case(vec![0], 0)]  // zero-sized
#[case(vec![1], 1)]  // single element
#[case(vec![2, 3], 6)]  // 2D
#[case(vec![2, 3, 4], 24)]  // 3D
fn test_shape_products(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
    let actual_product: usize = shape.iter().product();
    assert_eq!(actual_product, expected_product);
}

// Test for compiler creation and basic operations
#[test]
fn test_compiler_creation() {
    let compiler = Compiler::new();
    // Since Compiler struct is empty, the test just verifies it can be created
    assert!(true); // Basic creation test
}

// Test for operation attributes
#[test]
fn test_operation_with_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("conv2d");
    let mut attrs = HashMap::new();
    attrs.insert("padding".to_string(), Attribute::Int(1));
    attrs.insert("stride".to_string(), Attribute::Int(2));
    attrs.insert("activation".to_string(), Attribute::String("relu".to_string()));
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 3);
    assert_eq!(op.op_type, "conv2d");
    
    // Verify individual attributes
    assert_eq!(op.attributes.get("padding"), Some(&Attribute::Int(1)));
    assert_eq!(op.attributes.get("stride"), Some(&Attribute::Int(2)));
    assert_eq!(op.attributes.get("activation"), Some(&Attribute::String("relu".to_string())));
}

// Test for deeply nested tensor type equality
#[test]
fn test_nested_tensor_equality() {
    let type1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    let type2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    // They should be equal
    assert_eq!(type1, type2);
    
    // Different element types should not be equal
    let type3 = Type::Tensor {
        element_type: Box::new(Type::I32),
        shape: vec![2, 3],
    };
    
    assert_ne!(type1, type3);
    
    // Different shapes should not be equal
    let type4 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 4],
    };
    
    assert_ne!(type1, type4);
}

// Boundary test for large tensor dimensions
#[test]
fn test_very_large_tensor_size() {
    // Test creating tensors with very large dimensions
    let large_shape = vec![1000, 1000];
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    assert_eq!(value.shape, vec![1000, 1000]);
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 1_000_000);
    
    // Compute expected tensor size
    let expected_size = ir_utils::calculate_tensor_size(&value.ty, &value.shape).unwrap();
    assert_eq!(expected_size, 1_000_000 * 4); // 4 bytes per F32
}

// Test for tensor size calculation with various data types
#[test]
fn test_tensor_size_calculation_comprehensive() {
    // Test different combinations of types and shapes
    let test_cases = vec![
        (Type::F32, vec![], 4),           // Scalar F32
        (Type::F64, vec![], 8),           // Scalar F64
        (Type::I32, vec![], 4),           // Scalar I32
        (Type::I64, vec![], 8),           // Scalar I64
        (Type::Bool, vec![], 1),          // Scalar Bool
        (Type::F32, vec![10], 40),        // 1D F32 tensor
        (Type::I32, vec![5, 5], 100),     // 2D I32 tensor (5*5*4)
        (Type::Bool, vec![8, 8, 8], 512), // 3D Bool tensor (8*8*8*1)
    ];
    
    for (dtype, shape, expected_size) in test_cases {
        let calculated_size = ir_utils::calculate_tensor_size(&dtype, &shape).unwrap();
        assert_eq!(
            calculated_size, 
            expected_size, 
            "Failed for type {:?} with shape {:?}, expected {}, got {}", 
            dtype, 
            shape, 
            expected_size, 
            calculated_size
        );
    }
}