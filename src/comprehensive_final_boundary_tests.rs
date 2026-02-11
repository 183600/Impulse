//! Comprehensive final boundary tests - advanced edge cases and boundary conditions
//! Using standard library assert! and assert_eq! macros

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ir::TypeExtensions;
use std::collections::HashMap;

/// Test 1: Operation with extremely small and large float attributes
#[test]
fn test_operation_extreme_float_attributes() {
    let mut op = Operation::new("extreme_float_op");
    let mut attrs = HashMap::new();
    
    // Add extreme float values
    attrs.insert("min_positive".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("large_val".to_string(), Attribute::Float(1e300));
    attrs.insert("tiny_val".to_string(), Attribute::Float(-1e-300));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    
    // Verify min positive float
    if let Attribute::Float(val) = &op.attributes["min_positive"] {
        assert_eq!(*val, f64::MIN_POSITIVE);
        assert!(*val > 0.0);
        assert!(*val < 1e-300);
    } else {
        panic!("Expected Float attribute");
    }
    
    // Verify epsilon
    if let Attribute::Float(val) = &op.attributes["epsilon"] {
        assert_eq!(*val, f64::EPSILON);
    } else {
        panic!("Expected Float attribute");
    }
}

/// Test 2: Value with dimension overflow protection using checked arithmetic
#[test]
fn test_value_dimension_overflow_protection() {
    // Test shapes that would cause overflow with naive multiplication
    // Use checked_mul pattern through num_elements() method
    
    // Shape with product that overflows usize
    let overflow_shape = vec![usize::MAX, 2];
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: overflow_shape,
    };
    
    // num_elements() should return None for overflow case
    assert_eq!(overflow_value.num_elements(), None);
    
    // Valid large shape should still work
    let valid_large_shape = vec![1000, 1000, 100];
    let valid_value = Value {
        name: "valid_large".to_string(),
        ty: Type::F32,
        shape: valid_large_shape,
    };
    assert_eq!(valid_value.num_elements(), Some(100_000_000));
}

/// Test 3: Module with chained operations forming a DAG structure
#[test]
fn test_module_dag_structure() {
    let mut module = Module::new("dag_module");
    
    // Create a DAG: input -> op1 -> op2 -> output
    //                       -> op3
    let input_val = Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    module.inputs.push(input_val.clone());
    
    // op1 takes input
    let mut op1 = Operation::new("op1");
    op1.inputs.push(input_val.clone());
    let mid_val1 = Value {
        name: "mid1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op1.outputs.push(mid_val1.clone());
    module.add_operation(op1);
    
    // op2 takes mid1
    let mut op2 = Operation::new("op2");
    op2.inputs.push(mid_val1.clone());
    let mid_val2 = Value {
        name: "mid2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op2.outputs.push(mid_val2.clone());
    module.add_operation(op2);
    
    // op3 also takes mid1 (branch)
    let mut op3 = Operation::new("op3");
    op3.inputs.push(mid_val1.clone());
    let mid_val3 = Value {
        name: "mid3".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    op3.outputs.push(mid_val3.clone());
    module.add_operation(op3);
    
    // Verify DAG structure
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.operations[0].outputs[0].name, "mid1");
    assert_eq!(module.operations[1].inputs[0].name, "mid1");
    assert_eq!(module.operations[2].inputs[0].name, "mid1");
}

/// Test 4: Type validation with deeply nested invalid structures
#[test]
fn test_deeply_nested_type_validation() {
    // Create deeply nested tensor types
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![3],
    };
    
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![4],
    };
    
    // All nested types should be valid
    assert!(level3.is_valid_type());
    
    // Create another nested structure with different base type
    let bool_level1 = Type::Tensor {
        element_type: Box::new(Type::Bool),
        shape: vec![5],
    };
    
    assert!(bool_level1.is_valid_type());
}

/// Test 5: Module with malformed attribute values
#[test]
fn test_module_malformed_attributes() {
    let mut module = Module::new("malformed_attr_module");
    let mut op = Operation::new("malformed_attr_op");
    
    // Test attributes with edge case values
    let mut attrs = HashMap::new();
    attrs.insert("max_int".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("large_float".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("small_float".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("empty_string".to_string(), Attribute::String(String::new()));
    attrs.insert("long_string".to_string(), Attribute::String("x".repeat(1000)));
    
    op.attributes = attrs;
    module.add_operation(op);
    
    // Verify all attributes are stored correctly
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].attributes.len(), 6);
    
    // Verify specific attributes
    if let Attribute::Int(val) = &module.operations[0].attributes["max_int"] {
        assert_eq!(*val, i64::MAX);
    } else {
        panic!("Expected Int attribute");
    }
    
    if let Attribute::String(s) = &module.operations[0].attributes["empty_string"] {
        assert!(s.is_empty());
    } else {
        panic!("Expected String attribute");
    }
    
    if let Attribute::String(s) = &module.operations[0].attributes["long_string"] {
        assert_eq!(s.len(), 1000);
    } else {
        panic!("Expected String attribute");
    }
}

/// Test 6: Value with alternating dimension patterns
#[test]
fn test_value_alternating_dimension_patterns() {
    // Test various alternating dimension patterns
    let patterns = [
        vec![1, 2, 1, 2],           // Alternating 1,2
        vec![2, 1, 2, 1, 2, 1],     // Alternating 2,1
        vec![1, 1, 1, 1],           // All ones
        vec![1, 0, 1, 0, 1],        // Alternating 1,0 (zero in middle)
    ];
    
    for shape in patterns.iter() {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::F32,
            shape: shape.to_vec(),
        };
        
        assert_eq!(value.shape, *shape);
        
        // Calculate expected elements
        let expected: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
    
    // Special case: shape with zeros should return 0
    let zero_shape = vec![1, 0, 1, 0, 1];
    let zero_value = Value {
        name: "zero_pattern".to_string(),
        ty: Type::F32,
        shape: zero_shape,
    };
    assert_eq!(zero_value.num_elements(), Some(0));
}

/// Test 7: Attribute with empty and single-element arrays
#[test]
fn test_attribute_empty_and_single_element_arrays() {
    // Empty array
    let empty_arr = Attribute::Array(vec![]);
    if let Attribute::Array(arr) = empty_arr {
        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
    } else {
        panic!("Expected Array attribute");
    }
    
    // Single element arrays of different types
    let single_int = Attribute::Array(vec![Attribute::Int(42)]);
    if let Attribute::Array(arr) = single_int {
        assert_eq!(arr.len(), 1);
        if let Attribute::Int(42) = arr[0] {
            assert!(true);
        } else {
            panic!("Expected Int(42)");
        }
    }
    
    let single_bool = Attribute::Array(vec![Attribute::Bool(false)]);
    if let Attribute::Array(arr) = single_bool {
        assert_eq!(arr.len(), 1);
        if let Attribute::Bool(false) = arr[0] {
            assert!(true);
        } else {
            panic!("Expected Bool(false)");
        }
    }
}

/// Test 8: Module with operations having no inputs or no outputs
#[test]
fn test_module_operations_with_missing_io() {
    let mut module = Module::new("missing_io_module");
    
    // Operation with no inputs
    let mut no_input_op = Operation::new("no_input");
    no_input_op.outputs.push(Value {
        name: "generated".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    module.add_operation(no_input_op);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 1);
    
    // Operation with no outputs
    let mut no_output_op = Operation::new("no_output");
    no_output_op.inputs.push(Value {
        name: "consumed".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    module.add_operation(no_output_op);
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[1].outputs.len(), 0);
    
    // Operation with neither inputs nor outputs
    let empty_op = Operation::new("empty");
    module.add_operation(empty_op);
    assert_eq!(module.operations[2].inputs.len(), 0);
    assert_eq!(module.operations[2].outputs.len(), 0);
    
    assert_eq!(module.operations.len(), 3);
}

/// Test 9: Value with Unicode characters in names
#[test]
fn test_value_unicode_names() {
    let unicode_names = [
        "张量",
        "テンソル",
        "텐서",
        "Tensor_αβγ",
        "tënsör_中文",
        "𝕋𝕖𝕟𝕤𝕠𝕣",
        "Ténsör-Δ",
    ];
    
    for (i, name) in unicode_names.iter().enumerate() {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![i + 1],
        };
        
        assert_eq!(value.name, *name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape.len(), 1);
    }
}

/// Test 10: Integer boundary conditions for attributes
#[test]
fn test_integer_boundary_attributes() {
    let mut op = Operation::new("boundary_int_op");
    let mut attrs = HashMap::new();
    
    // Test integer boundaries
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero".to_string(), Attribute::Int(0));
    attrs.insert("positive_one".to_string(), Attribute::Int(1));
    attrs.insert("negative_one".to_string(), Attribute::Int(-1));
    attrs.insert("max_safe_int".to_string(), Attribute::Int(9_007_199_254_740_991)); // Number.MAX_SAFE_INTEGER equivalent
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 6);
    
    // Verify max value
    if let Attribute::Int(val) = op.attributes["max_i64"] {
        assert_eq!(val, i64::MAX);
        assert!(val > 0);
    } else {
        panic!("Expected Int attribute");
    }
    
    // Verify min value
    if let Attribute::Int(val) = op.attributes["min_i64"] {
        assert_eq!(val, i64::MIN);
        assert!(val < 0);
    } else {
        panic!("Expected Int attribute");
    }
    
    // Verify zero
    if let Attribute::Int(val) = op.attributes["zero"] {
        assert_eq!(val, 0);
    } else {
        panic!("Expected Int attribute");
    }
    
    // Verify positive and negative one
    if let Attribute::Int(val) = op.attributes["positive_one"] {
        assert_eq!(val, 1);
    }
    if let Attribute::Int(val) = op.attributes["negative_one"] {
        assert_eq!(val, -1);
    }
}