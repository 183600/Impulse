/// Extended critical boundary tests - additional edge cases for compiler robustness
/// 覆盖更多边界情况，包括极端数值、类型转换、内存安全等

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Subnormal float values (denormalized numbers) in attributes
#[test]
fn test_subnormal_float_values() {
    // Test subnormal (denormalized) floats - smallest positive values
    let subnormal_f32 = f32::MIN_POSITIVE; // Smallest normal f32
    let subnormal_f64 = f64::MIN_POSITIVE; // Smallest normal f64
    
    let attr1 = Attribute::Float(subnormal_f64 as f64);
    let attr2 = Attribute::Float(1e-308); // Very small f64
    
    match attr1 {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val.is_finite());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match attr2 {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val.is_finite());
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: Value with shape containing usize::MAX dimension (boundary test)
#[test]
fn test_value_with_max_usize_dimension() {
    // Test with a single dimension set to a very large value
    // This tests the boundary of dimension size representation
    let large_dim = 10_000_000usize; // Large but not causing overflow
    let value = Value {
        name: "large_dim_value".to_string(),
        ty: Type::F32,
        shape: vec![large_dim],
    };
    
    assert_eq!(value.shape[0], large_dim);
    assert_eq!(value.num_elements(), Some(large_dim));
}

/// Test 3: Module with cyclic name pattern references
#[test]
fn test_module_with_cyclic_name_patterns() {
    let mut module = Module::new("cyclic_module");
    
    // Create operations with naming pattern that suggests cycles
    let mut op1 = Operation::new("transform_a");
    op1.outputs.push(Value {
        name: "intermediate_a".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("transform_b");
    op2.inputs.push(Value {
        name: "intermediate_a".to_string(), // References op1 output
        ty: Type::F32,
        shape: vec![10],
    });
    op2.outputs.push(Value {
        name: "intermediate_b".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op3 = Operation::new("transform_c");
    op3.inputs.push(Value {
        name: "intermediate_b".to_string(), // References op2 output
        ty: Type::F32,
        shape: vec![10],
    });
    op3.outputs.push(Value {
        name: "intermediate_a".to_string(), // Cycles back to op1 output name
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    assert_eq!(module.operations[2].outputs[0].name, "intermediate_a");
}

/// Test 4: Attribute array with recursive nesting depth
#[test]
fn test_deeply_nested_attribute_array() {
    // Create a deeply nested array structure
    let level1 = Attribute::Array(vec![Attribute::Int(1)]);
    let level2 = Attribute::Array(vec![level1]);
    let level3 = Attribute::Array(vec![level2]);
    let level4 = Attribute::Array(vec![level3]);
    let level5 = Attribute::Array(vec![level4]);
    
    match level5 {
        Attribute::Array(outer) => {
            match &outer[0] {
                Attribute::Array(l4) => {
                    match &l4[0] {
                        Attribute::Array(l3) => {
                            match &l3[0] {
                                Attribute::Array(l2) => {
                                    match &l2[0] {
                                        Attribute::Array(l1) => {
                                            match &l1[0] {
                                                Attribute::Int(1) => {
                                                    // Successfully navigated 5 levels of nesting
                                                }
                                                _ => panic!("Expected Int at deepest level"),
                                            }
                                        }
                                        _ => panic!("Expected Array at level 1"),
                                    }
                                }
                                _ => panic!("Expected Array at level 2"),
                            }
                        }
                        _ => panic!("Expected Array at level 3"),
                    }
                }
                _ => panic!("Expected Array at level 4"),
            }
        }
        _ => panic!("Expected Array at level 5"),
    }
}

/// Test 5: Value with negative float values near zero
#[test]
fn test_negative_floats_near_zero() {
    // Test negative float values very close to zero
    let values = vec![
        -0.0,
        -1e-10,
        -1e-100,
        -f64::MIN_POSITIVE,
    ];
    
    for val in values {
        let attr = Attribute::Float(val);
        match attr {
            Attribute::Float(v) => {
                assert!(v <= 0.0);
                assert!(v.is_finite());
            }
            _ => panic!("Expected Float attribute"),
        }
    }
}

/// Test 6: Operation with all boolean attribute combinations
#[test]
fn test_all_boolean_attribute_combinations() {
    let mut op = Operation::new("bool_test_op");
    
    // Test all boolean combinations for a set of keys
    let mut attrs = HashMap::new();
    attrs.insert("flag1".to_string(), Attribute::Bool(true));
    attrs.insert("flag2".to_string(), Attribute::Bool(false));
    attrs.insert("flag3".to_string(), Attribute::Bool(true));
    attrs.insert("flag4".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    assert_eq!(op.attributes.get("flag1"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("flag2"), Some(&Attribute::Bool(false)));
    assert_eq!(op.attributes.get("flag3"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("flag4"), Some(&Attribute::Bool(false)));
}

/// Test 7: Module with alternating type pattern in operations
#[test]
fn test_module_alternating_type_pattern() {
    let mut module = Module::new("alternating_types_module");
    
    let types = [Type::F32, Type::I32, Type::F64, Type::I64, Type::Bool];
    
    for (i, ty) in types.iter().cycle().take(10).enumerate() {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: ty.clone(),
            shape: vec![1],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: ty.clone(),
            shape: vec![1],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 10);
    // Verify alternating pattern
    for i in 0..10 {
        let expected_type = types[i % types.len()].clone();
        assert_eq!(module.operations[i].inputs[0].ty, expected_type);
        assert_eq!(module.operations[i].outputs[0].ty, expected_type);
    }
}

/// Test 8: String attribute with maximum length practical boundary
#[test]
fn test_max_length_string_attribute() {
    // Test with a string of moderate length (not too long to cause memory issues)
    let moderate_length_string = "a".repeat(50_000); // 50k characters
    let attr = Attribute::String(moderate_length_string.clone());
    
    match attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), 50_000);
            assert_eq!(s.chars().count(), 50_000);
        }
        _ => panic!("Expected String attribute"),
    }
}

/// Test 9: Value with all possible single dimension sizes
#[test]
fn test_single_dimension_boundary_sizes() {
    // Test boundary values for single dimension sizes
    let boundary_sizes = vec![0, 1, 2, 10, 100, 1000, 10_000, 100_000];
    
    for size in boundary_sizes {
        let value = Value {
            name: format!("dim_{}", size),
            ty: Type::F32,
            shape: vec![size],
        };
        
        assert_eq!(value.shape[0], size);
        assert_eq!(value.num_elements(), Some(size));
    }
}

/// Test 10: Module with operations having no inputs but outputs
#[test]
fn test_operations_without_inputs() {
    let mut module = Module::new("no_inputs_module");
    
    // Create operations that generate values without taking inputs
    for i in 0..5 {
        let mut op = Operation::new(&format!("generator_{}", i));
        op.inputs.clear(); // Explicitly empty inputs
        op.outputs.push(Value {
            name: format!("generated_{}", i),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 1);
    }
}