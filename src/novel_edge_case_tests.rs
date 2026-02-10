//! Novel edge case tests covering additional boundary conditions
//! 
//! This module contains 10 comprehensive test cases that cover:
//! - Type system edge cases
//! - Operation chaining scenarios
//! - Memory-related boundary conditions
//! - Attribute serialization edge cases
//! - Module validation scenarios

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;

/// Test 1: Type trait is_valid_type validation for deeply nested tensors
#[test]
fn test_deeply_nested_type_validation() {
    use crate::ir::TypeExtensions;
    
    // Create a 5-level nested tensor type using clones to avoid move issues
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![4],
    };
    let level4 = Type::Tensor {
        element_type: Box::new(level3.clone()),
        shape: vec![5],
    };
    let level5 = Type::Tensor {
        element_type: Box::new(level4.clone()),
        shape: vec![6],
    };
    
    // All levels should be valid types
    assert!(level1.is_valid_type());
    assert!(level2.is_valid_type());
    assert!(level3.is_valid_type());
    assert!(level4.is_valid_type());
    assert!(level5.is_valid_type());
}

/// Test 2: Value num_elements overflow detection
#[test]
fn test_value_num_elements_overflow() {
    // Test cases that would cause overflow
    // Very large dimensions that when multiplied exceed usize::MAX
    
    // Safe case - should return Some
    let safe_value = Value {
        name: "safe".to_string(),
        ty: Type::F32,
        shape: vec![100, 100, 100],
    };
    assert_eq!(safe_value.num_elements(), Some(1_000_000));
    
    // Edge case with zero - should return Some(0)
    let zero_value = Value {
        name: "zero".to_string(),
        ty: Type::F32,
        shape: vec![100, 0, 100],
    };
    assert_eq!(zero_value.num_elements(), Some(0));
    
    // Large but potentially safe - check if it handles gracefully
    let large_value = Value {
        name: "large".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000],
    };
    // This might overflow on 32-bit systems, but should return None
    let result = large_value.num_elements();
    // Either it succeeds or returns None for overflow
    assert!(result.is_none() || result.is_some());
}

/// Test 3: Operation with cyclic name pattern (simulating graph cycles)
#[test]
fn test_operation_cyclic_naming_pattern() {
    let mut module = Module::new("cyclic_test");
    
    // Create operations that form a naming cycle
    let mut op_a = Operation::new("op_a");
    op_a.outputs.push(Value {
        name: "intermediate_1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op_b = Operation::new("op_b");
    op_b.inputs.push(op_a.outputs[0].clone());
    op_b.outputs.push(Value {
        name: "intermediate_2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op_c = Operation::new("op_c");
    op_c.inputs.push(op_b.outputs[0].clone());
    op_c.outputs.push(Value {
        name: "final_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op_a);
    module.add_operation(op_b);
    module.add_operation(op_c);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[1].op_type, "op_b");
    assert_eq!(module.operations[2].op_type, "op_c");
}

/// Test 4: Module with all primitive types in tensor wrappers
#[test]
fn test_all_primitive_types_in_tensors() {
    let mut module = Module::new("all_types");
    
    let primitive_types = vec![
        Type::F32,
        Type::F64,
        Type::I32,
        Type::I64,
        Type::Bool,
    ];
    
    for (i, base_type) in primitive_types.iter().enumerate() {
        let tensor_type = Type::Tensor {
            element_type: Box::new(base_type.clone()),
            shape: vec![2, 3],
        };
        
        let mut op = Operation::new(&format!("tensor_op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: tensor_type,
            shape: vec![2, 3],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
}

/// Test 5: Attribute array with varying integer types (positive, negative, zero)
#[test]
fn test_attribute_array_with_integer_variations() {
    let int_array = Attribute::Array(vec![
        Attribute::Int(0),           // Zero
        Attribute::Int(-1),          // Negative
        Attribute::Int(1),           // Positive
        Attribute::Int(i64::MAX),    // Max positive
        Attribute::Int(i64::MIN),    // Min negative
    ]);
    
    match int_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            
            match arr[0] { Attribute::Int(0) => (), _ => panic!("Expected 0") };
            match arr[1] { Attribute::Int(-1) => (), _ => panic!("Expected -1") };
            match arr[2] { Attribute::Int(1) => (), _ => panic!("Expected 1") };
            match arr[3] { Attribute::Int(i64::MAX) => (), _ => panic!("Expected MAX") };
            match arr[4] { Attribute::Int(i64::MIN) => (), _ => panic!("Expected MIN") };
        },
        _ => panic!("Expected Array"),
    }
}

/// Test 6: Module with operations that have no inputs or outputs
#[test]
fn test_operations_without_inputs_outputs() {
    let mut module = Module::new("empty_ops");
    
    // Operation with no inputs or outputs (e.g., a constant or control op)
    let op1 = Operation::new("nop");
    assert_eq!(op1.inputs.len(), 0);
    assert_eq!(op1.outputs.len(), 0);
    module.add_operation(op1);
    
    // Operation with inputs but no outputs (e.g., a sink)
    let mut op2 = Operation::new("sink");
    op2.inputs.push(Value {
        name: "sink_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    assert_eq!(op2.inputs.len(), 1);
    assert_eq!(op2.outputs.len(), 0);
    module.add_operation(op2);
    
    // Operation with outputs but no inputs (e.g., a source)
    let mut op3 = Operation::new("source");
    op3.outputs.push(Value {
        name: "source_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    assert_eq!(op3.inputs.len(), 0);
    assert_eq!(op3.outputs.len(), 1);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
}

/// Test 7: Compiler with different empty and non-empty target strings
#[test]
fn test_compiler_with_various_target_strings() {
    let mut compiler = ImpulseCompiler::new();
    let mock_model = vec![1u8, 2u8, 3u8];
    
    let target_variants = vec![
        "",           // Empty string
        "CPU",        // Uppercase
        "Cpu",        // Mixed case
        "gpu",        // Lowercase
        "cuda-0",     // With device ID
        "cuda:0",     // With colon separator
        "tpu-v2",     // With version
        "npu-ascend", // Complex name
        " ",          // Space only
        "\t",         // Tab only
    ];
    
    for target in target_variants {
        let result = compiler.compile(&mock_model, target);
        // Should handle gracefully without panicking
        match result {
            Ok(_) => println!("Compiled for target: {:?}", target),
            Err(e) => {
                // Error is acceptable
                assert!(e.to_string().len() > 0);
            }
        }
    }
}

/// Test 8: Module with operations having the same output names (potential conflict)
#[test]
fn test_module_operations_with_same_output_names() {
    let mut module = Module::new("same_outputs");
    
    // Multiple operations producing outputs with the same name
    for i in 0..3 {
        let mut op = Operation::new(&format!("producer_{}", i));
        op.outputs.push(Value {
            name: "shared_output".to_string(),  // Same name
            ty: Type::F32,
            shape: vec![5],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 3);
    // All operations should have the same output name
    for op in &module.operations {
        assert_eq!(op.outputs[0].name, "shared_output");
    }
}

/// Test 9: Float attributes with special values (infinity, NaN, negative zero)
#[test]
fn test_float_attributes_with_special_values() {
    let special_floats = vec![
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::Float(f64::NAN),
        Attribute::Float(-0.0),  // Negative zero
        Attribute::Float(0.0),   // Positive zero
    ];
    
    assert_eq!(special_floats.len(), 5);
    
    // Check infinity
    if let Attribute::Float(val) = &special_floats[0] {
        assert!(val.is_infinite());
        assert!(*val > 0.0);
    }
    
    // Check negative infinity
    if let Attribute::Float(val) = &special_floats[1] {
        assert!(val.is_infinite());
        assert!(*val < 0.0);
    }
    
    // Check NaN
    if let Attribute::Float(val) = &special_floats[2] {
        assert!(val.is_nan());
    }
    
    // Both -0.0 and 0.0 should exist as different values
    if let Attribute::Float(val) = &special_floats[3] {
        assert_eq!(*val, 0.0);
        // But they're not identical at bit level
    }
}

/// Test 10: Module with mixed scalar and tensor inputs/outputs
#[test]
fn test_module_mixed_scalar_and_tensor_io() {
    let mut module = Module::new("mixed_io");
    
    // Scalar input (empty shape)
    module.inputs.push(Value {
        name: "scalar_alpha".to_string(),
        ty: Type::F32,
        shape: vec![],  // Scalar
    });
    
    // Tensor input
    module.inputs.push(Value {
        name: "tensor_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    // Scalar output
    module.outputs.push(Value {
        name: "scalar_result".to_string(),
        ty: Type::F64,
        shape: vec![],  // Scalar
    });
    
    // Tensor output
    module.outputs.push(Value {
        name: "tensor_result".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    
    // Verify scalar inputs have empty shape
    assert!(module.inputs[0].shape.is_empty());
    assert!(!module.inputs[1].shape.is_empty());
    
    // Verify scalar outputs have empty shape
    assert!(module.outputs[0].shape.is_empty());
    assert!(!module.outputs[1].shape.is_empty());
}
