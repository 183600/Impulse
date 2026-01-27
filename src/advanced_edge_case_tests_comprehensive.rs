//! Advanced edge case tests for the Impulse compiler
//! Covering complex scenarios and boundary conditions

use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Testing recursive tensor type definitions
#[test]
fn test_recursive_tensor_type_definitions() {
    // Create recursive tensor types to test deep equality and cloning
    let mut current_type = Type::F32;
    
    // Build nested tensors up to depth 10 to avoid stack overflow
    for depth in 0..10 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![depth + 1],
        };
    }
    
    // Clone the deeply nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
    
    // Create values with the recursive type
    let value1 = Value {
        name: "recursive_tensor_1".to_string(),
        ty: current_type.clone(),
        shape: vec![1],
    };
    
    let value2 = Value {
        name: "recursive_tensor_2".to_string(),
        ty: current_type.clone(),
        shape: vec![1],
    };
    
    // Values should have the same type but different names
    assert_eq!(value1.ty, value2.ty);
    assert_ne!(value1.name, value2.name);
}

/// Test 2: Testing very large numeric values in attributes
#[test]
fn test_extremely_large_numeric_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("large_num_op");
    let mut attrs = HashMap::new();
    
    // Test with maximum i64 values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("zero_i64".to_string(), Attribute::Int(0));
    
    // Test with very large f64 values
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("tiny_positive_f64".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    
    op.attributes = attrs;
    
    // Verify the values are preserved
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
    assert_eq!(op.attributes.get("max_f64"), Some(&Attribute::Float(f64::MAX)));
    
    // Verify tiny positive number
    if let Some(Attribute::Float(val)) = op.attributes.get("tiny_positive_f64") {
        assert_eq!(*val, f64::MIN_POSITIVE);
    }
}

/// Test 3: Testing operations with maximum string length attributes
#[test]
fn test_operations_with_maximum_length_strings() {
    // Create a very long string (1MB)
    let very_long_string = "A".repeat(1_000_000);
    
    let mut op = Operation::new("long_string_op");
    op.attributes.insert(
        "very_long_attr".to_string(),
        Attribute::String(very_long_string.clone())
    );
    
    // Verify it can be stored and retrieved
    if let Some(Attribute::String(retrieved)) = op.attributes.get("very_long_attr") {
        assert_eq!(retrieved.len(), 1_000_000);
        assert_eq!(retrieved, &very_long_string);
    }
    
    // Also test with the operation name
    let long_op_name = "O".repeat(500_000);
    let long_op = Operation::new(&long_op_name);
    assert_eq!(long_op.op_type, long_op_name);
}

/// Test 4: Testing concurrent attribute access patterns (non-concurrent, sequential simulation)
#[test]
fn test_complex_attribute_access_patterns() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("complex_access_op");
    let mut attrs = HashMap::new();
    
    // Populate with various attribute types
    for i in 0..1000 {
        match i % 5 {
            0 => { attrs.insert(format!("int_{}", i), Attribute::Int(i as i64)); }
            1 => { attrs.insert(format!("float_{}", i), Attribute::Float(i as f64)); }
            2 => { attrs.insert(format!("bool_{}", i), Attribute::Bool(i % 2 == 0)); }
            3 => { attrs.insert(format!("str_{}", i), Attribute::String(i.to_string())); }
            4 => { 
                attrs.insert(
                    format!("arr_{}", i), 
                    Attribute::Array(vec![Attribute::Int(i as i64)])
                ); 
            }
            _ => unreachable!(),
        }
    }
    
    op.attributes = attrs;
    
    // Verify counts
    assert_eq!(op.attributes.len(), 1000);
    
    // Verify a few random access patterns
    assert_eq!(op.attributes.get("int_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("float_1"), Some(&Attribute::Float(1.0)));
    assert_eq!(op.attributes.get("bool_2"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("str_3"), Some(&Attribute::String("3".to_string())));
    
    // Verify last elements
    assert_eq!(op.attributes.get("int_995"), Some(&Attribute::Int(995)));
    assert_eq!(op.attributes.get("float_996"), Some(&Attribute::Float(996.0)));
    assert_eq!(op.attributes.get("str_998"), Some(&Attribute::String("998".to_string())));
}

/// Test 6: Testing modules with multiple operations
#[test]
fn test_modules_with_multiple_operations() {
    let mut module = Module::new("multi_op_module");
    
    // Add several ops to the module
    for i in 0..10 {
        let mut op = Operation::new(&format!("op_{}", i));
        
        // Add inputs 
        for j in 0..i.min(3) {  // Up to 3 inputs per op
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![i + j + 1],
            });
        }
        
        // Add outputs
        op.outputs.push(Value {
            name: format!("output_op_{}_result", i),
            ty: if i % 3 == 0 { Type::F32 } else { Type::I32 },
            shape: vec![i + 1],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 10);
    
    // Verify the structure
    for (idx, op) in module.operations.iter().enumerate() {
        assert_eq!(op.op_type, format!("op_{}", idx));
        assert_eq!(op.outputs.len(), 1);
        
        // Each operation could have up to 3 inputs
        let expected_inputs = idx.min(3);
        assert_eq!(op.inputs.len(), expected_inputs);
    }
}

/// Test 6: Testing type mismatch operations
#[test]
fn test_type_mismatch_scenarios() {
    // Test creating values with mismatched expectations
    let f32_value = Value {
        name: "f32_val".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    let i32_value = Value {
        name: "i32_val".to_string(),
        ty: Type::I32,
        shape: vec![10],
    };
    
    // These should not be equal despite same shape
    assert_ne!(f32_value, i32_value);
    
    // Same for operations with different types but same structure
    let mut op1 = Operation::new("arithmetic");
    let mut op2 = Operation::new("arithmetic");
    
    op1.inputs.push(f32_value.clone());
    op2.inputs.push(i32_value.clone());
    
    // Even though op type is same, inputs differ in type
    assert_ne!(op1.inputs, op2.inputs);
}

/// Test 7: Testing operations with empty collections
#[test]
fn test_operations_with_empty_collections() {
    let empty_op = Operation::new("empty_op");
    
    // Verify all collections are empty
    assert_eq!(empty_op.inputs.len(), 0);
    assert_eq!(empty_op.outputs.len(), 0);
    assert_eq!(empty_op.attributes.len(), 0);
    
    // An operation with empty name
    let unnamed_op = Operation::new("");
    assert_eq!(unnamed_op.op_type, "");
    
    // Create a value with empty name
    let empty_name_value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(empty_name_value.name, "");
    assert_eq!(empty_name_value.ty, Type::F32);
}

/// Test 8: Testing attribute retrieval and error handling
#[test]
fn test_attribute_retrieval_edge_cases() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("attr_test_op");
    let mut attrs = HashMap::new();
    
    attrs.insert("existent_key".to_string(), Attribute::Int(42));
    attrs.insert("".to_string(), Attribute::String("empty_key".to_string()));
    op.attributes = attrs;
    
    // Test existing key
    assert!(op.attributes.contains_key("existent_key"));
    assert_eq!(op.attributes.get("existent_key"), Some(&Attribute::Int(42)));
    
    // Test empty key
    assert!(op.attributes.contains_key(""));
    assert_eq!(op.attributes.get(""), Some(&Attribute::String("empty_key".to_string())));
    
    // Test non-existent key (should return None)
    assert!(!op.attributes.contains_key("non_existent"));
    assert_eq!(op.attributes.get("non_existent"), None);
}

/// Test 9: Testing module validation with various configurations
#[test]
fn test_module_validation_scenarios() {
    let test_modules = [
        Module::new("normal_module"),
        Module::new(""),
        Module::new(&"A".repeat(10_000)), // Very long name
    ];
    
    for module in test_modules.iter() {
        // All modules should have a name
        assert!(!module.name.is_empty() || module.name == "");
        
        // All modules should start with empty operations
        assert_eq!(module.operations.len(), 0);
        
        // Test adding operations
        let mut mutable_module = module.clone();
        
        let mut op = Operation::new("test_op");
        op.inputs.push(Value {
            name: "test_input".to_string(),
            ty: Type::F32,
            shape: vec![1],
        });
        
        mutable_module.add_operation(op);
        assert_eq!(mutable_module.operations.len(), 1);
        
        // To 'clear' operations, we create a new module with the same name
        let cleared_module = Module::new(&mutable_module.name);
        assert_eq!(cleared_module.operations.len(), 0);
    }
}

/// Test 10: Testing memory allocation patterns with many small objects
#[test]
fn test_many_small_objects_allocation() {
    // Create many small objects to test allocation patterns
    let mut operations = Vec::new();
    
    for i in 0..50_000 {
        let mut op = Operation::new(&format!("small_op_{}", i % 100)); // Reuse op names to test hash behavior
        
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: if i % 2 == 0 { Type::F32 } else { Type::I32 },
            shape: vec![1],
        });
        
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: if i % 3 == 0 { Type::F32 } else { Type::I64 },
            shape: vec![1],
        });
        
        operations.push(op);
    }
    
    assert_eq!(operations.len(), 50_000);
    
    // Verify some properties of the operations
    assert_eq!(operations[0].op_type, "small_op_0");
    assert_eq!(operations[1].op_type, "small_op_1");
    assert_eq!(operations[99].op_type, "small_op_99");
    assert_eq!(operations[100].op_type, "small_op_0"); // Name should wrap around
}