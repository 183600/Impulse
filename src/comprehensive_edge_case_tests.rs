//! Comprehensive edge case tests for the Impulse compiler
//! Focuses on boundary conditions, overflow scenarios, and unusual inputs

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use rstest::rstest;

// Test 1: Integer overflow prevention in shape calculations using checked arithmetic
#[test]
fn test_shape_product_overflow_prevention() {
    // Test that our system properly handles potentially overflowing calculations
    // Using checked_mul to prevent overflow in num_elements implementation
    
    // Large values that could potentially overflow in multiplication
    // On 64-bit systems, usize is typically u64, so we'll test values near sqrt(u64::MAX)
    let near_limit = (std::u64::MAX as usize) >> 32; // Rough approximation
    
    let value = Value {
        name: "potential_overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![near_limit, near_limit],
    };
    
    // Test the safe num_elements method that handles potential overflow
    let num_elements = value.num_elements();
    // The result should be None indicating overflow would occur, or a valid value
    println!("Potential overflow test: {:?}", num_elements);
    
    // Test with a shape that definitely contains 0
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![near_limit, 0, near_limit],
    };
    
    assert_eq!(zero_tensor.num_elements(), Some(0));
}

// Test 2: Deeply nested tensor types at maximum reasonable depth
#[rstest]
fn test_maximum_depth_tensor_nesting() {
    // Test creating very deeply nested tensor types to check recursion limits
    let mut current_type = Type::F32;
    
    // Nest up to a reasonable depth to avoid stack overflow
    for _ in 0..10 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the nested structure is valid
    assert!(current_type.is_valid_type());
    
    // Create another identical nested type and compare
    let mut comparison_type = Type::F32;
    for _ in 0..10 {
        comparison_type = Type::Tensor {
            element_type: Box::new(comparison_type),
            shape: vec![2],
        };
    }
    
    assert_eq!(current_type, comparison_type);
}

// Test 3: Special floating point values (NaN, infinity) in operations
#[test]
fn test_special_float_handling() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("special_float_op");
    let mut attrs = HashMap::new();
    
    attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
    attrs.insert("pos_inf_value".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_inf_value".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("normal_value".to_string(), Attribute::Float(3.14159));
    
    op.attributes = attrs;
    
    // Verify NaN handling - note that NaN != NaN, so special handling is needed
    if let Some(Attribute::Float(f)) = op.attributes.get("nan_value") {
        assert!(f.is_nan());
    } else {
        panic!("NaN value not stored properly");
    }
    
    // Verify infinity handling
    if let Some(Attribute::Float(f)) = op.attributes.get("pos_inf_value") {
        assert!(f.is_infinite() && f.is_sign_positive());
    }
    
    if let Some(Attribute::Float(f)) = op.attributes.get("neg_inf_value") {
        assert!(f.is_infinite() && f.is_sign_negative());
    }
    
    if let Some(Attribute::Float(f)) = op.attributes.get("normal_value") {
        assert!((f - 3.14159).abs() < f64::EPSILON);
    }
}

// Test 4: Type validation with edge cases including invalid combinations
#[test]
fn test_type_validation_edge_cases() {
    // Valid types should return true
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
    
    // Valid nested types
    let nested_valid = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![10, 20],
    };
    assert!(nested_valid.is_valid_type());
    
    // Very deeply nested valid types
    let mut deep_type = Type::F32;
    for _ in 0..50 {
        deep_type = Type::Tensor {
            element_type: Box::new(deep_type),
            shape: vec![2],
        };
    }
    assert!(deep_type.is_valid_type());
}

// Test 5: Boundary conditions for tensor dimensions (very large dims)
#[test]
fn test_extreme_tensor_dimensions() {
    // Test tensors with very large dimensions but still computationally manageable
    let extremely_large = Value {
        name: "extreme_tensor".to_string(),
        ty: Type::F32,
        shape: vec![std::u32::MAX as usize / 100000, 10],  // Very large first dim, small second
    };
    
    assert_eq!(extremely_large.shape.len(), 2);
    assert_eq!(extremely_large.shape[1], 10);
    assert!(extremely_large.shape[0] > 0);
    
    // Test shape with maximum possible dimensions count
    let many_dims = Value {
        name: "high_dimensional_tensor".to_string(),
        ty: Type::I64,
        shape: vec![2; 100],  // 100 dimensions, each size 2
    };
    
    assert_eq!(many_dims.shape.len(), 100);
    assert!(many_dims.shape.iter().all(|&x| x == 2));
    
    // Calculate total elements (should be 2^100 if computed, but that's huge)
    // Using a smaller example to verify our understanding
    let product: usize = many_dims.shape.iter().take(10).product(); // Only first 10 dims
    assert_eq!(product, 1024); // 2^10
}

// Test 6: Extreme attribute values with large strings and nested arrays
#[test]
fn test_extreme_attribute_values() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("extreme_attr_op");
    let mut attrs = HashMap::new();
    
    // Very large string attribute
    let large_string = "A".repeat(1_000_000); // 1MB string
    attrs.insert("large_string".to_string(), Attribute::String(large_string));
    
    // Deeply nested array
    let mut nested = Attribute::Array(vec![]);
    for _ in 0..10 {
        nested = Attribute::Array(vec![nested]);
    }
    attrs.insert("deeply_nested_array".to_string(), nested);
    
    // Large numeric values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    
    op.attributes = attrs;
    
    // Verify the attributes were stored correctly
    assert_eq!(op.attributes.len(), 4);
    assert!(op.attributes.contains_key("large_string"));
    assert!(op.attributes.contains_key("deeply_nested_array"));
    
    if let Some(Attribute::Int(val)) = op.attributes.get("max_i64") {
        assert_eq!(*val, i64::MAX);
    } else {
        panic!("max_i64 not stored properly");
    }
}

// Test 7: Recursive operations with complex interdependencies
#[test]
fn test_recursive_operation_structure() {
    // Create a complex module structure with interconnected operations
    let mut module = Module::new("recursive_test_module");
    
    // Create operations where outputs feed into other operations
    let mut op1 = Operation::new("initial_op");
    op1.outputs.push(Value {
        name: "intermediate_result".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    module.add_operation(op1);
    
    // Second operation that consumes the output of the first
    let mut op2 = Operation::new("second_op");
    op2.inputs.push(Value {
        name: "intermediate_result".to_string(),  // Matches output of op1
        ty: Type::F32,
        shape: vec![10, 10],
    });
    op2.outputs.push(Value {
        name: "final_result".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    });
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].op_type, "initial_op");
    assert_eq!(module.operations[1].op_type, "second_op");
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[1].inputs[0].name, "intermediate_result");
}

// Test 8: Memory-intensive operations testing allocation/deallocation
#[test]
fn test_large_memory_allocation_operations() {
    // Create operations with large numbers of inputs/outputs to test memory management
    let mut op = Operation::new("memory_intensive_op");
    
    // Add 10,000 inputs with minimal data
    for i in 0..10_000 {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],  // Minimal shape
        });
    }
    
    // Add 5,000 outputs
    for i in 0..5_000 {
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1],  // Minimal shape
        });
    }
    
    assert_eq!(op.inputs.len(), 10_000);
    assert_eq!(op.outputs.len(), 5_000);
    assert_eq!(op.op_type, "memory_intensive_op");
    
    // Verify all inputs have proper names
    assert_eq!(op.inputs[0].name, "input_0");
    assert_eq!(op.inputs[9999].name, "input_9999");
    
    // Verify all outputs have proper names
    assert_eq!(op.outputs[0].name, "output_0");
    assert_eq!(op.outputs[4999].name, "output_4999");
}

// Test 9: Checked arithmetic operations in value calculations
#[test]
fn test_checked_arithmetic_operations() {
    // Test the num_elements method that uses checked arithmetic
    let normal_tensor = Value {
        name: "normal_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10, 20, 5],  // 1000 elements
    };
    
    assert_eq!(normal_tensor.num_elements(), Some(1000));
    
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::I64,
        shape: vec![100, 0, 50],  // 0 elements due to zero
    };
    
    assert_eq!(zero_tensor.num_elements(), Some(0));
    
    let scalar_tensor = Value {
        name: "scalar_tensor".to_string(),
        ty: Type::Bool,
        shape: vec![],  // Scalar, 1 element
    };
    
    assert_eq!(scalar_tensor.num_elements(), Some(1));
    
    // Test very large but finite shape
    let large_finite = Value {
        name: "large_finite".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000, 1_000],  // May overflow in multiplication
    };
    
    // The result depends on whether the multiplication would actually overflow
    let result = large_finite.num_elements();
    println!("Large finite tensor result: {:?}", result);
}

// Test 10: Invalid tensor types and error conditions
#[test]
fn test_invalid_type_scenarios() {
    // Test various scenarios that could lead to invalid types
    // Though the current Type definition doesn't easily allow truly invalid types,
    // we can test the validation methods
    
    // A valid tensor type
    let valid_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![10, 20],
    };
    
    assert!(valid_tensor.is_valid_type());
    
    // Test cloning of complex types
    let cloned_tensor = valid_tensor.clone();
    assert_eq!(valid_tensor, cloned_tensor);
    
    // Test deeply nested valid types still pass validation
    let mut nested_type = Type::F32;
    for _ in 0..10 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![2, 2],
        };
    }
    
    assert!(nested_type.is_valid_type());
    
    // Create a module with complex valid types
    let mut module = Module::new("validation_test_module");
    
    let value_with_complex_type = Value {
        name: "complex_typed_value".to_string(),
        ty: nested_type,
        shape: vec![3, 3],
    };
    
    // Add an operation using this complex value
    let mut op = Operation::new("complex_type_op");
    op.inputs.push(value_with_complex_type);
    module.add_operation(op);
    
    assert_eq!(module.name, "validation_test_module");
    assert_eq!(module.operations.len(), 1);
}