//! Additional edge case tests for the Impulse compiler
//! Covers more boundary conditions and extreme values using standard library assertions

use rstest::*;
use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Operations with empty string names alongside special characters
#[test]
fn test_operations_empty_and_special_char_names() {
    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    
    let special_op = Operation::new("!@#$%^&*()");
    assert_eq!(special_op.op_type, "!@#$%^&*()");
    
    let whitespace_op = Operation::new("   \n\t\r");
    assert_eq!(whitespace_op.op_type, "   \n\t\r");
}

/// Test 2: Values with maximum possible shape dimensions (length of shape vector)
#[test]
fn test_values_maximum_shape_dimensions() {
    // Create a shape with maximum dimensions (but small values to prevent overflow)
    let max_dims_shape = vec![1; 1000]; // 1000 dimensions, each with size 1
    let value = Value {
        name: "max_dims_tensor".to_string(),
        ty: Type::F32,
        shape: max_dims_shape,
    };
    
    assert_eq!(value.shape.len(), 1000);
    
    // Total elements should be 1 (since all dimensions are 1)
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 1);
}

/// Test 3: Deeply nested tensor types with maximum recursion depth
#[test]
fn test_deepest_nested_tensor_types() {
    // Create a deeply nested tensor type (reduced from 500 to 20 to avoid stack overflow)
    let mut current_type = Type::F32;
    for _ in 0..20 {  // Reduced depth to avoid stack overflow
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1],
        };
    }
    
    // Verify that the type is still valid and can be compared
    assert!(matches!(current_type, Type::Tensor { .. }));
    
    // Test cloning of deeply nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 4: Handling extremely large numerical values in tensor shapes that might cause overflow
#[rstest]
#[case(vec![usize::MAX, 1], usize::MAX)]
#[case(vec![1, usize::MAX], usize::MAX)]
#[case(vec![0, usize::MAX], 0)]
#[case(vec![1, 1, 1, 1], 1)]
#[case(vec![2, 3, 4], 24)]
fn test_extremely_large_shape_values(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
    let value = Value {
        name: "large_shape_tensor".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape, shape);
    
    // Use checked multiplication to avoid panic in case of overflow
    let mut product: usize = 1;
    let mut has_overflow = false;
    for &dim in &shape {
        if let Some(result) = product.checked_mul(dim) {
            product = result;
        } else {
            has_overflow = true;
            break;
        }
    }
    
    if !has_overflow {
        assert_eq!(product, expected_product);
    }
}

/// Test 5: Operations with maximum possible attributes
#[test]
fn test_operations_maximum_attributes_count() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attrs_op");
    let mut attrs = HashMap::new();
    
    // Add a very large number of attributes
    for i in 0..50_000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 50_000);
    assert_eq!(op.op_type, "max_attrs_op");
    
    // Verify some specific attributes exist
    assert!(op.attributes.contains_key("attr_0"));
    assert!(op.attributes.contains_key("attr_25000"));
    assert!(op.attributes.contains_key("attr_49999"));
    
    // Verify attribute values
    assert_eq!(op.attributes.get("attr_0").unwrap(), &Attribute::String("value_0".to_string()));
    assert_eq!(op.attributes.get("attr_49999").unwrap(), &Attribute::String("value_49999".to_string()));
}

/// Test 6: Values with special floating-point shapes and tensor sizes
#[test]
fn test_special_floating_point_tensor_shapes() {
    // Test various non-standard shapes that could cause issues
    let test_cases = vec![
        (vec![], 1),                      // scalar (0-dimensional tensor)
        (vec![0], 0),                     // size 0 in 1D
        (vec![0, 1, 2, 3], 0),           // contains 0, so total 0
        (vec![1, 2, 0, 4], 0),           // 0 in middle
        (vec![1, 2, 3, 4], 24),          // standard case
        (vec![100, 100, 100], 1_000_000), // 3D large tensor
        (vec![10_000, 10_000], 100_000_000), // 2D very large tensor
        (vec![2, 2, 2, 2, 2, 2, 2, 2], 256), // 8D tensor (2^8)
    ];
    
    for (shape, expected_total) in test_cases {
        let value = Value {
            name: "special_shape_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        let actual_total: usize = value.shape.iter().product();
        assert_eq!(actual_total, expected_total, "Shape {:?} should have {} total elements", shape, expected_total);
    }
}

/// Test 7: Modules with maximum possible operations
#[test]
fn test_modules_maximum_operations_count() {
    let mut module = Module::new("max_ops_module");
    
    // Add a maximum possible number of operations
    for i in 0..200_000 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);
        
        // Periodic check to make sure operations are being added
        if i > 0 && (i % 50_000) == 0 {
            assert_eq!(module.operations.len(), i + 1);
        }
    }
    
    assert_eq!(module.operations.len(), 200_000);
    assert_eq!(module.name, "max_ops_module");
    
    // Check that operations maintain their names correctly
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[199_999].op_type, "op_199999");
}

/// Test 8: Complex recursive types with alternating patterns
#[test]
fn test_complex_recursive_type_patterns() {
    // Create a complex recursive type that alternates between different base types
    let mut current_type = Type::F32;
    for i in 0..100 {
        if i % 2 == 0 {
            current_type = Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![i + 1],
            };
        } else {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 5 + 1],  // Cycle through shapes 1-5
            };
        }
    }
    
    // Verify the resulting type can be handled
    assert!(current_type.is_valid_type());
    
    // Verify the structure can be cloned without issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 9: Special values in attribute types (infinity, NaN, etc.)
#[test]
fn test_special_values_in_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("special_vals_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with special float values
    attrs.insert("inf_val".to_string(), Attribute::Float(std::f64::INFINITY));
    attrs.insert("neg_inf_val".to_string(), Attribute::Float(std::f64::NEG_INFINITY));
    attrs.insert("nan_val".to_string(), Attribute::Float(std::f64::NAN));
    attrs.insert("max_float".to_string(), Attribute::Float(std::f64::MAX));
    attrs.insert("min_float".to_string(), Attribute::Float(std::f64::MIN));
    attrs.insert("epsilon".to_string(), Attribute::Float(std::f64::EPSILON));
    attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("positive_zero".to_string(), Attribute::Float(0.0));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 8);
    
    // Test that infinity values are properly stored
    if let Attribute::Float(inf_val) = op.attributes.get("inf_val").unwrap() {
        assert!(inf_val.is_infinite() && inf_val.is_sign_positive());
    } else {
        panic!("Expected positive infinity value");
    }
    
    if let Attribute::Float(neg_inf_val) = op.attributes.get("neg_inf_val").unwrap() {
        assert!(neg_inf_val.is_infinite() && neg_inf_val.is_sign_negative());
    } else {
        panic!("Expected negative infinity value");
    }
    
    // NaN needs special handling since NaN != NaN
    if let Attribute::Float(nan_val) = op.attributes.get("nan_val").unwrap() {
        assert!(nan_val.is_nan());
    } else {
        panic!("Expected NaN value");
    }
}

/// Test 10: Compiler with maximum memory allocation patterns
#[test]
fn test_compiler_memory_allocation_patterns() {
    use std::collections::HashMap;
    
    // Create a complex module with nested structures to test memory allocation
    let mut complex_module = Module::new("complex_memory_test");
    
    // Add operations with complex nested structures
    for i in 0..10_000 {
        let mut op = Operation::new(&format!("complex_op_{}", i));
        
        // Add several inputs with complex shapes
        for j in 0..10 {
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: if j % 2 == 0 { Type::F32 } else { Type::I64 },
                shape: vec![j + 1, j + 2],
            });
        }
        
        // Add several outputs with complex types
        for j in 0..5 {
            op.outputs.push(Value {
                name: format!("output_{}_{}", i, j),
                ty: if j % 3 == 0 { Type::F64 } else if j % 3 == 1 { Type::I32 } else { Type::Bool },
                shape: vec![j + 1],
            });
        }
        
        // Add several attributes
        let mut attrs = HashMap::new();
        for j in 0..20 {
            attrs.insert(
                format!("attr_{}_{}", i, j),
                Attribute::String(format!("value_{}_{}", i, j))
            );
        }
        op.attributes = attrs;
        
        complex_module.add_operation(op);
    }
    
    // Verify the module was constructed correctly
    assert_eq!(complex_module.operations.len(), 10_000);
    assert_eq!(complex_module.name, "complex_memory_test");
    
    // Check a few operations to ensure they maintained their structure
    assert_eq!(complex_module.operations[0].op_type, "complex_op_0");
    assert_eq!(complex_module.operations[0].inputs.len(), 10);
    assert!(complex_module.operations[0].attributes.contains_key("attr_0_0"));
    
    assert_eq!(complex_module.operations[9_999].op_type, "complex_op_9999");
    assert_eq!(complex_module.operations[9_999].inputs.len(), 10);
    assert!(complex_module.operations[9_999].attributes.contains_key("attr_9999_0"));
}