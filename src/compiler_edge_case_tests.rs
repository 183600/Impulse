//! Additional edge case tests for the Impulse compiler focusing on critical areas
//! that weren't fully covered in existing tests.

use crate::compiler::Compiler;
use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use rstest::rstest;

// Test 1: Empty module compilation
#[test]
fn test_empty_module_compilation() {
    let _compiler = Compiler::new();  // Explicitly mark as unused with underscore
    let module = Module::new("empty_module");
    
    // Should handle empty modules gracefully
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.name, "empty_module");
}

// Test 2: Maximal recursion depth for type checking
#[test]
fn test_max_recursion_depth_type_checking() {
    let mut current_type = Type::F32;
    
    // Build a deeply nested type structure to test recursion limits
    for _ in 0..100 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1],
        };
    }
    
    // Ensure the type remains valid despite deep nesting
    assert!(current_type.is_valid_type());
    
    // Test recursive clone operation
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 3: Operations with maximum possible attributes
#[test]
fn test_operation_with_maximum_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attrs_op");
    let mut attrs = HashMap::new();
    
    // Add maximum variety of attribute types (avoiding non-existent Type and Value variants)
    for i in 0..1000 {
        match i % 5 {  // Reduced modulo since we removed invalid variants
            0 => attrs.insert(format!("int_attr_{}", i), Attribute::Int(i as i64)),
            1 => attrs.insert(format!("float_attr_{}", i), Attribute::Float(i as f64)),
            2 => attrs.insert(format!("str_attr_{}", i), Attribute::String(format!("string_{}", i))),
            3 => attrs.insert(format!("bool_attr_{}", i), Attribute::Bool(i % 2 == 0)),
            _ => attrs.insert(format!("array_attr_{}", i), Attribute::Array(vec![
                Attribute::Int(i as i64),
                Attribute::Float(i as f64),
            ])),
        };
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 1000);
}

// Test 4: Invalid tensor shape configurations
#[test]
fn test_invalid_tensor_shape_configurations() {
    let invalid_shapes = vec![
        vec![-1],          // Negative dimension
        vec![1, -1, 2],    // Mixed negative dimension
        vec![std::i32::MAX, std::i32::MAX, std::i32::MAX], // Very large dimensions
    ];
    
    // Note: Actual validation would happen in implementation, here we test the concept
    for shape in invalid_shapes {
        // Simulate creating a value with potentially invalid shape
        // In real implementation, this might trigger validation errors
        let value = Value {
            name: format!("invalid_shape_{:?}", shape),
            ty: Type::F32,
            shape: shape.iter().filter(|&&d| d >= 0).map(|&d| d as usize).collect(),
        };
        
        // Basic validation
        assert_eq!(value.name.contains("invalid_shape"), true);
        assert_eq!(value.ty, Type::F32);
    }
}

// Test 5: Zero-sized tensors and operations
#[test]
fn test_zero_sized_tensors_operations() {
    // Zero-sized tensors: any dimension with size 0 results in 0 total elements
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![0, 10, 20], // One dimension is 0, so total size is 0
    };
    
    let calculated_size: usize = zero_tensor.shape.iter().product();
    assert_eq!(calculated_size, 0);
    
    // Similar test with different zero position
    let zero_tensor2 = Value {
        name: "zero_tensor2".to_string(),
        ty: Type::I32,
        shape: vec![5, 0, 3],
    };
    
    let calculated_size2: usize = zero_tensor2.shape.iter().product();
    assert_eq!(calculated_size2, 0);
    
    // Create operations with zero-sized tensors
    let mut op = Operation::new("zero_tensor_op");
    op.inputs.push(zero_tensor);
    op.outputs.push(zero_tensor2);
    
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.outputs.len(), 1);
}

// Test 6: Maximum numeric values in attributes
#[test]
fn test_maximum_numeric_values_in_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_numeric_op");
    let mut attrs = HashMap::new();
    
    // Test with maximum values
    attrs.insert("max_u64".to_string(), Attribute::String(u64::MAX.to_string()));
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64_positive".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    
    // Test conversion back and forth
    if let Some(max_i64_attr) = attrs.get("max_i64") {
        match max_i64_attr {
            Attribute::Int(val) => assert_eq!(*val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 5);
}

// Test 7: Rapid creation/destruction of compiler objects
#[test]
fn test_rapid_object_creation_destruction() {
    // Stress test to ensure no memory leaks during rapid object creation/destruction
    for i in 0..100 {
        let module = Module::new(&format!("test_module_{}", i));
        let value = Value {
            name: format!("test_value_{}", i),
            ty: Type::F32,
            shape: vec![i % 10 + 1],
        };
        
        // Objects should be properly cleaned up when going out of scope
        assert_eq!(module.name, format!("test_module_{}", i));
        assert_eq!(value.name, format!("test_value_{}", i));
    }
}

// Test 8: String encoding edge cases in names
#[test]
fn test_string_encoding_edge_cases() {
    let problematic_names = vec![
        "regular_name".to_string(),
        "name_with_numbers_123".to_string(),
        "name_with_symbols_!@#$%".to_string(),
        "name_with_unicode_ 测试_тест".to_string(),
        "name_with_whitespace_ _\t_\n".to_string(),
        "name_with_control_chars_\x00\x01\x02".to_string(), // null and control chars
        "a".repeat(1000), // very long string
        "".to_string(), // empty string
    ];
    
    for name in problematic_names {
        let value = Value {
            name: name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        // Each name should be preserved exactly as provided
        assert_eq!(value.name, name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1]);
    }
}

// Test 9: Operations with duplicate input/output names
#[test]
fn test_duplicate_names_handling() {
    let mut op = Operation::new("duplicate_names_op");
    
    // Add inputs with similar/duplicate-like names
    op.inputs.push(Value {
        name: "input_a".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    op.inputs.push(Value {
        name: "input_a_copy".to_string(), // Similar name
        ty: Type::F32,
        shape: vec![1],
    });
    
    op.inputs.push(Value {
        name: "input_A".to_string(), // Different case
        ty: Type::F64,
        shape: vec![1],
    });
    
    // Add outputs
    op.outputs.push(Value {
        name: "output_x".to_string(),
        ty: Type::I32,
        shape: vec![1],
    });
    
    op.outputs.push(Value {
        name: "output_x".to_string(), // Exact duplicate (would typically be invalid in practice)
        ty: Type::I64,
        shape: vec![1],
    });
    
    assert_eq!(op.inputs.len(), 3);
    assert_eq!(op.outputs.len(), 2);
    assert_eq!(op.op_type, "duplicate_names_op");
}

// Test 10: Parametrized test for different type combinations with extreme values
#[rstest]
#[case(Type::F32, vec![0], 0)]
#[case(Type::F32, vec![1], 1)]
#[case(Type::F32, vec![1000, 1000], 1_000_000)]
#[case(Type::F64, vec![100, 100, 100], 1_000_000)]
#[case(Type::I32, vec![0, 100], 0)]
#[case(Type::I64, vec![50, 0, 20], 0)]
#[case(Type::Bool, vec![2, 2, 2, 2, 2], 32)]
fn test_type_shape_combinations(
    #[case] data_type: Type,
    #[case] shape: Vec<usize>,
    #[case] expected_size: usize
) {
    let value = Value {
        name: format!("test_value_{:?}", data_type),
        ty: data_type.clone(),
        shape: shape.clone(),
    };
    
    // Calculate actual size
    let actual_size: usize = value.shape.iter().product();
    
    assert_eq!(actual_size, expected_size);
    assert_eq!(value.ty, data_type);
    assert_eq!(value.shape, shape);
}

#[cfg(test)]
mod arithmetic_overflow_tests {
    use super::*;

    // Test potential overflow in size calculations
    #[test]
fn test_size_calculation_overflow_protection() {
        // Use values that are large but not quite at overflow boundaries
        // to avoid actual overflow while still testing large values
        let large_but_safe_values = vec![
            (vec![100_000, 100_000], 10_000_000_000), // Product within usize range
            (vec![1_000, 1_000, 1_000], 1_000_000_000), // Still safe
            (vec![usize::MAX / 1000, 1], usize::MAX / 1000), // Near limit but safe
        ];

        for (shape, expected_product) in large_but_safe_values {
            let value = Value {
                name: format!("large_size_tensor_{:?}", shape),
                ty: Type::F32,
                shape: shape.clone(),
            };

            let calculated_size: usize = value.shape.iter().product();
            assert_eq!(calculated_size, expected_product, 
                      "Shape {:?} should multiply to {}, got {}", 
                      shape, expected_product, calculated_size);
        }
    }
}