//! Additional comprehensive edge case tests for the Impulse compiler
//! Focuses on new edge cases not covered in existing tests

use crate::ir::{Module, Operation, Value, Type, Attribute};
use rstest::rstest;

/// Test 1: Potential overflow in tensor size calculations
#[test]
fn test_tensor_size_overflow_scenarios() {
    // Test cases that could potentially cause overflow in tensor size calculations
    let test_cases = vec![
        // Cases that result in 0 elements due to zero dimension
        (vec![0, 1000000], 0),
        (vec![1000000, 0, 500], 0),
        (vec![1, 2, 0, 4, 5], 0),
        
        // Large but safe tensor sizes
        (vec![46340, 46340], 2_147_395_600), // Nearly u32::MAX
        
        // Scalar (empty shape)
        (vec![], 1),
        
        // Single dimension
        (vec![1], 1),
        (vec![1000000], 1000000),
    ];
    
    for (shape, expected_elements) in test_cases {
        let value = Value {
            name: format!("tensor_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Manual calculation
        let calculated_elements: usize = value.shape.iter().product();
        assert_eq!(calculated_elements, expected_elements);
        
        // Test the num_elements method 
        let method_result = value.num_elements();
        assert_eq!(method_result, Some(expected_elements));
    }
}

/// Test 2: Operations with empty inputs/outputs but valid attributes
#[test]
fn test_operation_with_empty_io_but_attributes() {
    let mut op = Operation::new("empty_io_op");
    op.attributes.insert("param".to_string(), Attribute::Int(42));
    op.attributes.insert("name".to_string(), Attribute::String("test_op".to_string()));
    
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 2);
    
    // Verify attribute access works correctly
    assert_eq!(op.attributes.get("param"), Some(&Attribute::Int(42)));
    assert_eq!(op.attributes.get("name"), Some(&Attribute::String("test_op".to_string())));
}

/// Test 3: Module with duplicate operation names
#[test]
fn test_module_with_duplicate_operation_names() {
    let mut module = Module::new("duplicate_names_module");
    
    // Add two operations with the same name but different functionality
    let mut op1 = Operation::new("same_name_op");
    op1.inputs.push(Value {
        name: "input1".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    
    let mut op2 = Operation::new("same_name_op"); // Same name
    op2.inputs.push(Value {
        name: "input2".to_string(),
        ty: Type::I32,
        shape: vec![2, 2],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    // Both operations have the same type name
    assert_eq!(module.operations[0].op_type, "same_name_op");
    assert_eq!(module.operations[1].op_type, "same_name_op");
    // But different characteristics
    assert_eq!(module.operations[0].inputs[0].ty, Type::F32);
    assert_eq!(module.operations[1].inputs[0].ty, Type::I32);
}

/// Test 4: Value with various string names to test UTF-8 handling
#[rstest]
#[case("")]
#[case("valid_ascii")]
#[case("混合_multilingual")]
#[case("Многоязычный_text")]
#[case("🎉 unicode_emoji 🚀")]
#[case("!@#$%^&*()")]  // Special characters
#[case("a".repeat(1000))]  // Long ASCII string
fn test_value_with_various_names(#[case] name: String) {
    let value = Value {
        name: name.clone(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    
    assert_eq!(value.name, name);
    assert_eq!(value.ty, Type::F32);
    assert_eq!(value.shape, vec![1, 2, 3]);
}

/// Test 5: Type comparison edge cases
#[test]
fn test_type_comparison_edge_cases() {
    // Test tensor types with same element type but different shapes
    let tensor1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 2],
    };
    
    let tensor2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![4],  // Same number of elements but different shape
    };
    
    assert_ne!(tensor1, tensor2);  // Different shapes make them unequal
    
    // Test nested tensor comparison
    let nested1 = Type::Tensor {
        element_type: Box::new(
            Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![2],
            }
        ),
        shape: vec![3],
    };
    
    let nested2 = Type::Tensor {
        element_type: Box::new(
            Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![2],  // Same as nested1's inner shape
            }
        ),
        shape: vec![3],  // Same outer shape
    };
    
    assert_eq!(nested1, nested2);  // Should be equal
}

/// Test 6: Attribute handling with max/min integer bounds
#[test]
fn test_attribute_integer_bounds() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("bounds_test_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with boundary integer values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("negative_one".to_string(), Attribute::Int(-1));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    
    // Verify each value is stored correctly
    match op.attributes.get("max_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected max_i64 to be stored as Int"),
    }
    
    match op.attributes.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected min_i64 to be stored as Int"),
    }
    
    match op.attributes.get("zero") {
        Some(Attribute::Int(val)) => assert_eq!(*val, 0),
        _ => panic!("Expected zero to be stored as Int"),
    }
    
    match op.attributes.get("negative_one") {
        Some(Attribute::Int(val)) => assert_eq!(*val, -1),
        _ => panic!("Expected negative_one to be stored as Int"),
    }
}

/// Test 7: Deeply nested operations in module
#[test]
fn test_operations_with_circular_dependency_simulation() {
    // Although true circular dependencies might not be allowed in the final IR,
    // we test that the data structures can handle references that conceptually represent cycles
    
    let mut module = Module::new("dependency_test_module");
    
    // Create operations that reference each other conceptually
    let mut op1 = Operation::new("producer");
    op1.outputs.push(Value {
        name: "result".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("consumer");
    op2.inputs.push(Value {
        name: "result".to_string(),  // Same name as output from op1
        ty: Type::F32,
        shape: vec![10],  // Shape matches
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].op_type, "producer");
    assert_eq!(module.operations[1].op_type, "consumer");
}

/// Test 8: Values with extremely sparse shapes (many single dimensions)
#[test]
fn test_sparse_dimension_shapes() {
    // Test shapes with many dimensions of size 1 (sparse shapes)
    let sparse_shapes = vec![
        vec![1],                          // 1D single element
        vec![1, 1],                      // 2D single element
        vec![1, 1, 1, 1, 1],            // 5D single element  
        vec![1, 100, 1, 1, 50],         // Mostly single dimensions
        vec![2, 1, 1, 1, 1000],         // Scattered dimensions
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], // 10D single element
    ];
    
    for shape in sparse_shapes {
        let value = Value {
            name: format!("sparse_tensor_{:?}", shape),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Calculate expected number of elements
        let expected_elements: usize = shape.iter().product();
        let calculated_elements: usize = value.shape.iter().product();
        
        assert_eq!(calculated_elements, expected_elements);
        
        // Test the num_elements method too
        assert_eq!(value.num_elements(), Some(expected_elements));
    }
}

/// Test 9: Attribute array with heterogeneous values but in a controlled way
#[test]
fn test_attribute_array_edge_cases() {
    use std::collections::HashMap;
    
    // Test empty array
    let empty_array = Attribute::Array(vec![]);
    match empty_array {
        Attribute::Array(arr) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected empty array attribute"),
    }
    
    // Test array with single element
    let single_array = Attribute::Array(vec![Attribute::Int(42)]);
    match single_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected single int in array"),
            }
        },
        _ => panic!("Expected single-element array attribute"),
    }
    
    // Test array with mixed but specific pattern
    let pattern_array = Attribute::Array(vec![
        Attribute::Int(1),
        Attribute::Float(2.0),
        Attribute::Int(3),
        Attribute::Float(4.0),
    ]);
    
    match pattern_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 4);
            
            // Check alternating pattern
            if let Attribute::Int(1) = arr[0] {} else { panic!("Expected Int(1) at index 0"); }
            if let Attribute::Float(2.0) = arr[1] {} else { panic!("Expected Float(2.0) at index 1"); } 
            if let Attribute::Int(3) = arr[2] {} else { panic!("Expected Int(3) at index 2"); }
            if let Attribute::Float(4.0) = arr[3] {} else { panic!("Expected Float(4.0) at index 3"); }
        },
        _ => panic!("Expected pattern array attribute"),
    }
}

/// Test 10: Error-prone tensor shape patterns that could confuse calculation logic
#[test]
fn test_confusing_tensor_shape_patterns() {
    // Patterns that might be confused with each other during shape manipulation
    
    let value1 = Value {
        name: "single_large".to_string(),
        ty: Type::F32,
        shape: vec![1_000_000],  // 1M elements in 1D
    };
    
    let value2 = Value {
        name: "multi_small".to_string(),
        ty: Type::F32,
        shape: vec![100, 100, 100],  // 1M elements in 3D
    };
    
    let value3 = Value {
        name: "flat_rect".to_string(),
        ty: Type::F32,
        shape: vec![10_000, 100],  // 1M elements in 2D
    };
    
    // All should have the same number of elements despite different shapes
    assert_eq!(value1.shape.iter().product::<usize>(), 1_000_000);
    assert_eq!(value2.shape.iter().product::<usize>(), 1_000_000);
    assert_eq!(value3.shape.iter().product::<usize>(), 1_000_000);
    
    // Verify num_elements gives same result
    assert_eq!(value1.num_elements(), Some(1_000_000));
    assert_eq!(value2.num_elements(), Some(1_000_000));
    assert_eq!(value3.num_elements(), Some(1_000_000));
    
    // Verify shapes are indeed different
    assert_ne!(value1.shape, value2.shape);
    assert_ne!(value1.shape, value3.shape);
    assert_ne!(value2.shape, value3.shape);
}
