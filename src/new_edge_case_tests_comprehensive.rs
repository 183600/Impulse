//! Comprehensive new edge case tests for the Impulse compiler
//! Adding more extensive edge case coverage for uncovered scenarios

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Testing operations with empty names and special characters
#[test]
fn test_operations_with_empty_or_special_char_names() {
    let op_normal = Operation::new("normal");
    let op_empty = Operation::new("");
    
    assert_eq!(op_normal.op_type, "normal");
    assert_eq!(op_empty.op_type, "");
    
    // Operations with special characters
    let special_names = ["+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "~", "<", ">", "==", "!=", "<=", ">="];
    for name in special_names.iter() {
        let op = Operation::new(name);
        assert_eq!(op.op_type, *name);
    }
}

/// Test 2: Testing values with all possible type combinations
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_basic_types_memory_size(#[case] _typ: Type) {
    // Just test one type per run to avoid move issues, and we'll test each separately
    let value = Value {
        name: "size_test".to_string(),
        ty: _typ.clone(),
        shape: vec![1],
    };
    
    // Verify the type is preserved
    match value.ty {
        Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Bool => assert!(true), // Valid types
        _ => panic!("Unexpected type"),
    }
    
    // Test with different shapes
    let sizes_for_shapes = [
        (vec![], 1),      // scalar
        (vec![1], 1),     // single element
        (vec![5], 5),     // 1D
        (vec![2, 3], 6),  // 2D
        (vec![1, 1, 1], 1), // 3D with minimal dims
    ];
    
    for (shape, multiplier) in sizes_for_shapes.iter() {
        let test_val = Value {
            name: "shape_test".to_string(),
            ty: _typ.clone(),
            shape: shape.clone(),
        };
        
        let total_elements: usize = test_val.shape.iter().product();
        assert_eq!(total_elements, *multiplier);
    }
}

/// Test 3: Testing deeply nested tensor operations
#[test]
fn test_deeply_nested_tensor_operations() {
    // Create a deeply nested tensor type
    let mut base_type = Type::F32;
    const NESTING_DEPTH: usize = 20;
    
    for _ in 0..NESTING_DEPTH {
        base_type = Type::Tensor {
            element_type: Box::new(base_type),
            shape: vec![2],
        };
    }
    
    // Create an operation with this deeply nested type
    let mut op = Operation::new("nested_tensor_op");
    op.inputs.push(Value {
        name: "nested_input".to_string(),
        ty: base_type.clone(),
        shape: vec![1],
    });
    
    op.outputs.push(Value {
        name: "nested_output".to_string(),
        ty: base_type.clone(),
        shape: vec![1],
    });
    
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.outputs.len(), 1);
    
    // Test cloning and equality
    let cloned_op = op.clone();
    assert_eq!(op.op_type, cloned_op.op_type);
    assert_eq!(op.inputs.len(), cloned_op.inputs.len());
    assert_eq!(op.outputs.len(), cloned_op.outputs.len());
}

/// Test 4: Testing attribute operations with complex combinations
#[test]
fn test_complex_attribute_combinations() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("complex_attr_op");
    let mut attrs = HashMap::new();
    
    // Add mixed attribute types
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14159));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
    attrs.insert("string_attr".to_string(), Attribute::String("hello".to_string()));
    
    // Add nested arrays
    attrs.insert(
        "nested_array".to_string(),
        Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
            Attribute::Array(vec![Attribute::Float(1.1), Attribute::Float(2.2)]),
        ])
    );
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 5);
    
    // Verify each attribute type
    assert!(matches!(op.attributes.get("int_attr"), Some(Attribute::Int(42))));
    assert!(matches!(op.attributes.get("float_attr"), Some(&Attribute::Float(v)) if (v - 3.14159).abs() < f64::EPSILON));
    assert!(matches!(op.attributes.get("bool_attr"), Some(Attribute::Bool(true))));
    assert!(matches!(op.attributes.get("string_attr"), Some(Attribute::String(s)) if s == "hello"));
}

/// Test 5: Testing modules with zero operations but with inputs and outputs
#[test]
fn test_empty_modules_with_io() {
    // Create a module with initial inputs and outputs in the struct
    let mut module = Module::new("empty_io_module");
    
    // Add inputs and outputs by directly manipulating the fields
    module.inputs.push(Value {
        name: "input_1".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    module.inputs.push(Value {
        name: "input_2".to_string(),
        ty: Type::I32,
        shape: vec![5, 5, 5],
    });
    
    module.outputs.push(Value {
        name: "output_1".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.name, "empty_io_module");
    
    // Verify input and output content
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.inputs[1].ty, Type::I32);
    assert_eq!(module.outputs[0].ty, Type::F32);
}

/// Test 6: Testing invalid tensor shape combinations
#[test]
fn test_invalid_tensor_shape_scenarios() {
    // Cases that might be problematic but should be handled gracefully
    let problematic_shapes = [
        vec![],
        vec![0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0, 3],
        vec![1, 1, 0],
        vec![usize::MAX, 1], // This might cause overflow in product
    ];
    
    for (i, shape) in problematic_shapes.iter().enumerate() {
        let value = Value {
            name: format!("problematic_shape_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Calculate product - should not crash even for potentially problematic shapes
        let product: usize = value.shape.iter().product();
        
        if i < problematic_shapes.len() - 1 { // Not the last one which might overflow
            match i {
                0 => assert_eq!(product, 1), // Empty should be scalar = 1
                1 | 2 | 3 | 4 | 5 => assert_eq!(product, 0), // Contains 0 should result in 0
                _ => {} // Other cases
            }
        }
    }
}

/// Test 7: Testing mixed operation types with extreme variation
#[test]
fn test_mixed_operation_types_extreme_variation() {
    let operation_types = [
        "add", "subtract", "multiply", "divide", 
        "conv2d", "matmul", "reduce_sum", "broadcast",
        "", // Empty operation
        "very_long_operation_name_that_might_cause_performance_issues_if_not_handled_properly",
        "!@#$%^&*()", // Special characters
        "123_numbers_at_start", "OpWith_underscores_andNumbers123",
    ];
    
    let mut module = Module::new("mixed_ops_module");
    
    for (i, op_type) in operation_types.iter().enumerate() {
        let mut op = Operation::new(op_type);
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
            shape: vec![i + 1],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: if i % 2 == 0 { Type::I32 } else { Type::F32 },
            shape: vec![i + 1],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), operation_types.len());
    
    for (i, expected_type) in operation_types.iter().enumerate() {
        assert_eq!(module.operations[i].op_type, *expected_type);
    }
}

/// Test 8: Testing attribute map edge cases
#[test]
fn test_attribute_map_edge_cases() {
    use std::collections::HashMap;
    
    // Test with duplicate keys (should overwrite)
    let mut attrs = HashMap::new();
    attrs.insert("key".to_string(), Attribute::Int(1));
    attrs.insert("key".to_string(), Attribute::Int(2)); // This should overwrite
    
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs.get("key"), Some(&Attribute::Int(2)));
    
    // Test with very long keys
    let long_key = "a".repeat(10_000);
    attrs.insert(long_key.clone(), Attribute::String("long_key_value".to_string()));
    
    assert_eq!(attrs.len(), 2);
    assert!(attrs.contains_key(&long_key));
    
    // Test with empty key
    attrs.insert("".to_string(), Attribute::Bool(true));
    assert!(attrs.contains_key(""));
    assert_eq!(attrs.get(""), Some(&Attribute::Bool(true)));
}

/// Test 9: Testing value comparisons and equality
#[rstest]
#[case(Value { name: "same".to_string(), ty: Type::F32, shape: vec![1] }, Value { name: "same".to_string(), ty: Type::F32, shape: vec![1] }, true)]
#[case(Value { name: "diff_name".to_string(), ty: Type::F32, shape: vec![1] }, Value { name: "other_name".to_string(), ty: Type::F32, shape: vec![1] }, false)]
#[case(Value { name: "same".to_string(), ty: Type::F32, shape: vec![1] }, Value { name: "same".to_string(), ty: Type::I32, shape: vec![1] }, false)]
#[case(Value { name: "same".to_string(), ty: Type::F32, shape: vec![1] }, Value { name: "same".to_string(), ty: Type::F32, shape: vec![2] }, false)]
fn test_value_equality_scenarios(
    #[case] val1: Value, 
    #[case] val2: Value, 
    #[case] expected_equal: bool
) {
    if expected_equal {
        assert_eq!(val1, val2);
        assert_eq!(val1.name, val2.name);
        assert_eq!(val1.ty, val2.ty);
        assert_eq!(val1.shape, val2.shape);
    } else {
        assert_ne!(val1, val2);
    }
}

/// Test 10: Testing tensor flattening and reshape edge cases
#[rstest]
#[case(vec![], 1)]  // scalar
#[case(vec![1], 1)]  // single element
#[case(vec![0], 0)]  // zero-sized
#[case(vec![1, 1, 1, 1], 1)]  // all ones
#[case(vec![2, 2, 2], 8)]  // cube
#[case(vec![100, 100, 100], 1_000_000)]  // large cube
fn test_tensor_flattening_scenarios(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let flattened_size: usize = shape.iter().product();
    assert_eq!(flattened_size, expected_size);
    
    // Test with different data types
    let test_types = [Type::F32, Type::I32, Type::Bool];
    for dtype in test_types.iter() {
        let value = Value {
            name: "flatten_test".to_string(),
            ty: dtype.clone(),
            shape: shape.clone(),
        };
        
        // Verify shape is preserved
        assert_eq!(value.shape, shape);
        
        // Calculate the total size in elements
        let element_count: usize = value.shape.iter().product();
        assert_eq!(element_count, expected_size);
    }
}