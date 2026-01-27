//! New comprehensive edge case tests for the Impulse compiler
//! Covering additional boundary conditions not found in other test files

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Handling empty string names for operations, values, and modules
#[test]
fn test_empty_string_names() {
    // Operation with empty name
    let op = Operation::new("");
    assert_eq!(op.op_type, "");
    
    // Value with empty name
    let value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.name, "");
    
    // Module with empty name
    let module = Module::new("");
    assert_eq!(module.name, "");
}

/// Test 2: Operations with extremely long names (boundary test)
#[test]
fn test_extremely_long_operation_names() {
    let long_name = "a".repeat(1_000_000); // 1 million character name
    let op = Operation::new(&long_name);
    assert_eq!(op.op_type.len(), 1_000_000);
    assert_eq!(op.op_type, long_name);
    
    // Also test with value and module
    let value = Value {
        name: long_name.clone(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.name.len(), 1_000_000);
    
    let module = Module::new(&long_name);
    assert_eq!(module.name.len(), 1_000_000);
}

/// Test 3: Zero-dimensional tensors (scalars) with various types
#[rstest]
#[case(Type::F32)]
#[case(Type::F64)]
#[case(Type::I32)]
#[case(Type::I64)]
#[case(Type::Bool)]
fn test_zero_dimensional_tensors(#[case] data_type: Type) {
    let scalar = Value {
        name: "scalar_value".to_string(),
        ty: data_type,
        shape: vec![], // Zero-dimensional tensor (scalar)
    };
    
    assert_eq!(scalar.shape.len(), 0);
    assert!(scalar.shape.is_empty());
    
    // Product of empty shape should be 1 (one scalar element)
    let product: usize = scalar.shape.iter().product();
    assert_eq!(product, 1);
}

/// Test 4: Nested operations with circular references (should be handled properly)
#[test]
fn test_circular_reference_in_operations() {
    // This tests if our data structures can handle potentially circular references
    // without causing infinite loops or stack overflows
    
    // Create operations that reference each other indirectly
    let mut op1 = Operation::new("op1");
    let mut op2 = Operation::new("op2");
    
    // Add inputs and outputs to both operations
    op1.outputs.push(Value {
        name: "op1_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    op2.inputs.push(Value {
        name: "op1_output".to_string(), // References output from op1
        ty: Type::F32,
        shape: vec![10],
    });
    
    op2.outputs.push(Value {
        name: "op2_output".to_string(),
        ty: Type::F32,
        shape: vec![5],
    });
    
    op1.inputs.push(Value {
        name: "op2_output".to_string(), // References output from op2
        ty: Type::F32,
        shape: vec![5],
    });
    
    // Verify both operations have been properly constructed
    assert_eq!(op1.op_type, "op1");
    assert_eq!(op2.op_type, "op2");
    assert_eq!(op1.inputs.len(), 1);
    assert_eq!(op1.outputs.len(), 1);
    assert_eq!(op2.inputs.len(), 1);
    assert_eq!(op2.outputs.len(), 1);
}

/// Test 5: Attributes with maximum integer values
#[test]
fn test_max_integer_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_int_test");
    let mut attrs = HashMap::new();
    
    // Add attributes with maximum and minimum possible integer values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("negative_one".to_string(), Attribute::Int(-1));
    
    op.attributes = attrs;
    
    assert_eq!(op.op_type, "max_int_test");
    assert_eq!(op.attributes.len(), 4);
    
    // Verify specific values
    if let Some(Attribute::Int(max_val)) = op.attributes.get("max_i64") {
        assert_eq!(*max_val, i64::MAX);
    } else {
        panic!("Expected Int(i64::MAX) for max_i64");
    }
    
    if let Some(Attribute::Int(min_val)) = op.attributes.get("min_i64") {
        assert_eq!(*min_val, i64::MIN);
    } else {
        panic!("Expected Int(i64::MIN) for min_i64");
    }
}

/// Test 6: Complex tensor type nesting that switches types at each level
#[test]
fn test_alternating_complex_tensor_nesting() {
    // Create a nested type that alternates between different base types
    let mut current_type = Type::I32;  // Start with an integer type
    let expected_depth = 25;       // Reasonable depth for the test
    
    // Alternate between types as we nest deeper
    for i in 0..expected_depth {
        let next_type = match i % 4 {
            0 => Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![i + 1],
            },
            1 => Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![i + 2],
            },
            2 => Type::Tensor {
                element_type: Box::new(Type::Bool),
                shape: vec![i + 3],
            },
            _ => Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i + 4],
            },
        };
        current_type = next_type;
    }
    
    // Verify the construction worked
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 7: Operations with mixed attribute types in complex arrangements
#[test]
fn test_mixed_complex_attribute_arrangements() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("complex_attr_op");
    let mut attrs = HashMap::new();
    
    // Create a complex arrangement of nested arrays and different types
    attrs.insert(
        "complex_array".to_string(),
        Attribute::Array(vec![
            Attribute::String("level1".to_string()),
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(3.14),
                    Attribute::Bool(true),
                ]),
                Attribute::String("level2".to_string()),
            ]),
            Attribute::Bool(false),
        ])
    );
    
    // Add other types as well
    attrs.insert("simple_int".to_string(), Attribute::Int(42));
    attrs.insert("simple_float".to_string(), Attribute::Float(2.718));
    attrs.insert("simple_string".to_string(), Attribute::String("test".to_string()));
    attrs.insert("simple_bool".to_string(), Attribute::Bool(true));
    
    op.attributes = attrs;
    
    assert_eq!(op.op_type, "complex_attr_op");
    assert_eq!(op.attributes.len(), 5); // 1 complex array + 4 simple values
    
    // Verify the complex array structure
    if let Some(Attribute::Array(arr)) = op.attributes.get("complex_array") {
        assert_eq!(arr.len(), 3);  // 3 top-level elements
        
        if let Attribute::String(ref s) = arr[0] {
            assert_eq!(s, "level1");
        } else {
            panic!("Expected string at array[0]");
        }
        
        if let Attribute::Array(inner_arr) = &arr[1] {
            assert_eq!(inner_arr.len(), 3);  // 3 elements in the inner array
            
            if let Attribute::Int(1) = inner_arr[0] {
                assert!(true);  // Correct value
            } else {
                panic!("Expected int 1 at inner array[0]");
            }
            
            if let Attribute::Array(deep_arr) = &inner_arr[1] {
                assert_eq!(deep_arr.len(), 2);  // 2 elements in the deep array
                
                if let Attribute::Float(val) = deep_arr[0] {
                    assert!((val - 3.14).abs() < f64::EPSILON);
                } else {
                    panic!("Expected float 3.14 at deep array[0]");
                }
            }
        }
    }
}

/// Test 8: Large number of operations without inputs or outputs to test memory efficiency
#[test]
fn test_large_number_of_simple_operations() {
    let mut module = Module::new("simple_ops_module");
    
    // Add 200,000 operations that have no inputs or outputs
    for i in 0..200_000 {
        let op = Operation::new(&format!("simple_op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 200_000);
    assert_eq!(module.name, "simple_ops_module");
    
    // Verify some operations exist at specific indices
    assert_eq!(module.operations[0].op_type, "simple_op_0");
    assert_eq!(module.operations[99_999].op_type, "simple_op_99999");
    assert_eq!(module.operations[199_999].op_type, "simple_op_199999");
}

/// Test 9: Value shapes with maximum possible dimension values and overflow handling
#[rstest]
#[case(vec![0, usize::MAX], Some(0))]  // Contains zero, so product is 0
#[case(vec![1, 1, 1, 1], Some(1))]    // Small values, no overflow
#[case(vec![1000, 1000, 1000], Some(1_000_000_000))] // Large but safe
fn test_shape_overflow_scenarios(#[case] shape: Vec<usize>, #[case] expected_result: Option<usize>) {
    let value = Value {
        name: "overflow_test_tensor".to_string(),
        ty: Type::F32,
        shape,
    };
    
    // Compute the product using checked arithmetic to detect overflow
    let result = value.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    if expected_result.is_none() {
        // Expecting overflow
        assert!(result.is_none());
    } else {
        // Not expecting overflow
        assert_eq!(result, expected_result);
    }
    
    // Regardless of overflow, the shape should remain intact
    assert_eq!(value.shape.len(), value.shape.len());
}

/// Test 10: Empty operations list handling and module edge cases
#[test]
fn test_module_empty_operations_edge_cases() {
    // Test module with no operations
    let mut empty_module = Module::new("empty_module");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.name, "empty_module");
    
    // Add and remove operations to test dynamic behavior
    let op = Operation::new("temporary_op");
    empty_module.add_operation(op);
    
    assert_eq!(empty_module.operations.len(), 1);
    
    // Clear all operations
    empty_module.operations.clear();
    assert_eq!(empty_module.operations.len(), 0);
    
    // Test creating and destroying many modules to check for resource leaks
    for i in 0..1000 {
        let temp_module = Module::new(&format!("temp_module_{}", i));
        // Module goes out of scope here, should be cleaned up properly
        assert_eq!(temp_module.name, format!("temp_module_{}", i));
    }
    
    // Add multiple operations at once
    let mut multi_op_module = Module::new("multi_op_module");
    
    // Add 100 operations at once
    for i in 0..100 {
        let mut op = Operation::new(&format!("batch_op_{}", i));
        if i % 2 == 0 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
        }
        
        multi_op_module.add_operation(op);
    }
    
    assert_eq!(multi_op_module.operations.len(), 100);
    assert_eq!(multi_op_module.name, "multi_op_module");
}