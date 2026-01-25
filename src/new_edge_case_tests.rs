//! New edge case tests for the Impulse compiler
//! Covering additional boundary conditions and error scenarios

use rstest::*;
use crate::{
    ir::{Value, Type, Operation, Attribute, Module},
};

/// Test 1: Operations with malformed UTF-8 byte sequences
#[test]
fn test_operations_with_invalid_utf8_sequences() {
    // Test creating operations with byte sequences that are invalid UTF-8
    // This ensures the system handles invalid input gracefully
    
    // Construct a string with invalid UTF-8 sequence
    let invalid_utf8_bytes = vec![0xC0, 0xAF]; // Invalid UTF-8 sequence
    let invalid_string = String::from_utf8_lossy(&invalid_utf8_bytes);
    
    let op = Operation::new(&invalid_string);
    assert_eq!(op.op_type, invalid_string);
    
    // Test with value names containing invalid UTF-8
    let value = Value {
        name: invalid_string.to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value.name, invalid_string);
}

/// Test 2: Zero-sized tensor operations and memory allocation
#[test]
fn test_zero_sized_tensor_allocation() {
    // Test tensors with zero-sized dimensions in different configurations
    let zero_configs = [
        vec![0],           // 0-dimensional tensor with 0 elements
        vec![0, 1],        // 2D tensor with 0 width
        vec![1, 0],        // 2D tensor with 0 height  
        vec![0, 0],        // 2D tensor with 0 width and height
        vec![2, 0, 4],     // 3D tensor with 0 in middle
        vec![0, 10, 0],    // 3D tensor with zeros at start and end
    ];
    
    for shape in zero_configs.iter() {
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 0, "Shape {:?} should have 0 total elements", shape);
    }
}

/// Test 3: Extreme type conversions and type checking
#[rstest]
#[case(Type::F32, "F32")]
#[case(Type::F64, "F64")]
#[case(Type::I32, "I32")]
#[case(Type::I64, "I64")]
#[case(Type::Bool, "Bool")]
fn test_basic_type_properties(#[case] type_enum: Type, #[case] _type_name: &str) {
    // Verify that basic types maintain consistent properties
    let cloned_type = type_enum.clone();
    assert_eq!(type_enum, cloned_type);
    
    // Test type validation
    use crate::ir::TypeExtensions;
    assert!(type_enum.is_valid_type());
}

/// Test 4: Complex attribute operations with mixed types
#[test]
fn test_complex_attribute_operations() {
    use std::collections::HashMap;
    
    // Create an operation with multiple complex attribute combinations
    let mut op = Operation::new("complex_attr_op");
    let mut attrs = HashMap::new();
    
    // Add deeply nested array attributes
    let deepest_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::String("deep".to_string()),
    ]);
    let nested_array = Attribute::Array(vec![
        Attribute::Float(3.14),
        deepest_array,
        Attribute::Bool(true),
    ]);
    let outer_array = Attribute::Array(vec![
        Attribute::String("outer".to_string()),
        nested_array,
        Attribute::Int(42),
    ]);
    
    attrs.insert("complex_array".to_string(), outer_array);
    attrs.insert("simple_int".to_string(), Attribute::Int(123));
    attrs.insert("simple_float".to_string(), Attribute::Float(2.718));
    attrs.insert("simple_string".to_string(), Attribute::String("hello".to_string()));
    attrs.insert("simple_bool".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    assert!(op.attributes.contains_key("complex_array"));
    assert!(op.attributes.contains_key("simple_int"));
}

/// Test 5: Tensor shape permutations and equivalent shapes
#[test]
fn test_tensor_shape_permutations() {
    // Test that different arrangements of the same dimensions yield same total size
    let shapes_with_same_size = vec![
        vec![2, 3, 4],    // 24 elements
        vec![3, 4, 2],    // 24 elements (reordered)
        vec![4, 3, 2],    // 24 elements (reordered)
        vec![6, 4],       // 24 elements (different arrangement)
        vec![24],         // 24 elements (1D)
    ];
    
    let expected_size = 24;
    
    for shape in shapes_with_same_size {
        let value = Value {
            name: "permuted_tensor".to_string(),
            ty: Type::F32,
            shape,
        };
        
        let calculated_size: usize = value.shape.iter().product();
        assert_eq!(calculated_size, expected_size, "Shape {:?} should have {} elements", value.shape, expected_size);
    }
}

/// Test 6: Boundary value testing for numeric types
#[test]
fn test_numeric_boundary_values() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("boundary_test_op");
    let mut attrs = HashMap::new();
    
    // Test boundary values for integer attributes
    attrs.insert("i64_min".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("i64_max".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("i64_zero".to_string(), Attribute::Int(0));
    attrs.insert("i64_neg_one".to_string(), Attribute::Int(-1));
    
    // Test boundary values for float attributes
    attrs.insert("f64_min".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("f64_max".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("f64_min_positive".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("f64_epsilon".to_string(), Attribute::Float(f64::EPSILON));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 8);
    assert_eq!(op.attributes.get("i64_min"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("i64_max"), Some(&Attribute::Int(i64::MAX)));
}

/// Test 7: Empty operations and modules handling
#[test]
fn test_empty_structure_handling() {
    // Test creating and manipulating truly empty structures
    let empty_op = Operation::new("");
    assert_eq!(empty_op.op_type, "");
    assert!(empty_op.inputs.is_empty());
    assert!(empty_op.outputs.is_empty());
    assert!(empty_op.attributes.is_empty());
    
    let empty_module = Module::new("");
    assert_eq!(empty_module.name, "");
    assert!(empty_module.operations.is_empty());
    assert!(empty_module.inputs.is_empty());
    assert!(empty_module.outputs.is_empty());
    
    // Test adding an empty operation to an empty module
    let mut module_with_empty_op = Module::new("module_with_empty_op");
    module_with_empty_op.add_operation(empty_op.clone());
    
    assert_eq!(module_with_empty_op.operations.len(), 1);
    assert_eq!(module_with_empty_op.operations[0].op_type, "");
}

/// Test 8: Recursive type validation with maximum recursion protection
#[test]
fn test_recursive_type_validation_limits() {
    // Test to make sure deeply nested types don't cause stack overflow
    // but are still handled correctly
    
    let mut current_type = Type::F32;
    let max_depth = 500; // Reasonable depth to avoid stack overflow while testing recursion
    
    // Build deeply nested type
    for _ in 0..max_depth {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Validate the deeply nested type
    use crate::ir::TypeExtensions;
    assert!(current_type.is_valid_type());
    
    // Test equality of two similar deeply nested types
    let mut other_type = Type::F32;
    for _ in 0..max_depth {
        other_type = Type::Tensor {
            element_type: Box::new(other_type),
            shape: vec![2],
        };
    }
    
    assert_eq!(current_type, other_type);
    
    // Test equality with a slightly different version (different shape)
    let mut different_type = Type::F32;
    for _ in 0..max_depth {
        different_type = Type::Tensor {
            element_type: Box::new(different_type),
            shape: vec![3], // Different shape
        };
    }
    
    assert_ne!(current_type, different_type);
}

/// Test 9: Shape validation with unusual but valid configurations
#[test]
fn test_unusual_valid_shape_configurations() {
    // Test tensors with unusual but valid shape configurations
    
    // Singleton dimensions (many single-element dimensions)
    let singleton_tensor = Value {
        name: "singleton_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1, 10, 1, 1, 1],
    };
    
    let total_elements: usize = singleton_tensor.shape.iter().product();
    assert_eq!(total_elements, 10); // Only the non-singleton dimension matters
    
    // Extremely rectangular tensors (very high ratio of one dimension to another)
    let rectangular_tensor = Value {
        name: "rectangular_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1, 1_000_000], // 1 x 1M - very wide
    };
    
    let rectangular_elements: usize = rectangular_tensor.shape.iter().product();
    assert_eq!(rectangular_elements, 1_000_000);
    
    // Many small dimensions
    let fragmented_tensor = Value {
        name: "fragmented_tensor".to_string(),
        ty: Type::I32,
        shape: vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], // 2^10 = 1024
    };
    
    let fragmented_elements: usize = fragmented_tensor.shape.iter().product();
    assert_eq!(fragmented_elements, 1024);
}

/// Test 10: Error scenario handling with malformed data
#[test]
fn test_error_scenario_malformed_data_handling() {
    use std::collections::HashMap;
    
    // Test creating operations with potentially problematic data to ensure graceful handling
    
    // Test with duplicate input/output names (valid scenario)
    let mut op = Operation::new("duplicate_names_op");
    op.inputs.push(Value {
        name: "duplicate_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op.inputs.push(Value {
        name: "duplicate_name".to_string(), // Same name as previous
        ty: Type::I32,
        shape: vec![5],
    });
    
    op.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![8],
    });
    op.outputs.push(Value {
        name: "output".to_string(), // Same name as previous
        ty: Type::I64,
        shape: vec![4],
    });
    
    // Operations should allow duplicate names - this might be valid in context
    assert_eq!(op.inputs.len(), 2);
    assert_eq!(op.outputs.len(), 2);
    
    // Test with complex nested types that might be problematic
    let mut complex_module = Module::new("complex_module");
    
    // Add operation with a mix of all attribute types
    let mut complex_op = Operation::new("complex_op");
    let mut attrs = HashMap::new();
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14159));
    attrs.insert("string_attr".to_string(), Attribute::String("test_string".to_string()));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::String("nested".to_string()),
        Attribute::Float(2.718),
        Attribute::Bool(false),
        Attribute::Array(vec![Attribute::Int(99), Attribute::Int(100)])
    ]));
    complex_op.attributes = attrs;
    
    complex_module.add_operation(complex_op);
    
    assert_eq!(complex_module.operations.len(), 1);
    assert_eq!(complex_module.operations[0].attributes.len(), 5);
    assert!(complex_module.operations[0].attributes.contains_key("array_attr"));
}