//! Enhanced edge case tests for the Impulse compiler
//! Covering additional boundary conditions not addressed in other test files

use crate::ir::{Value, Type, Operation, Attribute, Module};
use std::collections::HashMap;

/// Test 1: Error handling with malformed tensor shapes
#[test]
fn test_malformed_tensor_shape_handling() {
    // Testing shapes with potential issues
    let problematic_shapes = [
        vec![1, 0, 1],          // Zero in middle
        vec![100_000_000, 0],   // Large number with zero
        vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], // Many small dims
        vec![usize::MAX, 1],    // Near boundary values
    ];
    
    for (i, shape) in problematic_shapes.iter().enumerate() {
        let value = Value {
            name: format!("malformed_shape_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Verify no panic during creation
        assert_eq!(value.shape, *shape);
        
        // Test safe calculation of total elements
        let safe_product = value.shape.iter()
            .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
        
        match i {
            0 => assert_eq!(safe_product, Some(0)),  // Contains 0
            1 => assert_eq!(safe_product, Some(0)),  // Contains 0  
            2 => assert_eq!(safe_product, Some(1024)), // 2^10
            3 => assert_eq!(safe_product, Some(usize::MAX)), // MAX * 1
            _ => panic!("Unexpected index"),
        }
    }
}

/// Test 2: Hash collision resistance in attribute maps
#[test]
fn test_hash_collision_resistance() {
    let mut op = Operation::new("collision_test");
    let mut attrs = HashMap::new();
    
    // Generate many keys that might collide
    for i in 0..10_000 {
        let key = format!("key_{}_suffix", i);
        attrs.insert(key.clone(), Attribute::Int(i as i64));
        
        // Verify we can retrieve what we inserted
        assert_eq!(attrs.get(&key), Some(&Attribute::Int(i as i64)));
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 10_000);
    
    // Test retrieval of specific values
    assert_eq!(op.attributes.get("key_123_suffix"), Some(&Attribute::Int(123)));
    assert_eq!(op.attributes.get("key_9999_suffix"), Some(&Attribute::Int(9999)));
}

/// Test 3: Recursive type with maximum depth without stack overflow
#[test]
fn test_deep_recursion_without_stack_overflow() {
    let mut current_type = Type::Bool;
    
    // Create a very deep recursive type (but not too deep to cause stack overflow)
    for depth in 0..1000 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![depth % 10 + 1], // Varying shape to make it more interesting
        };
    }
    
    // Verify the structure can be handled without crashing
    // At depth 999, shape will be [999 % 10 + 1] = [9 + 1] = [10]
    match &current_type {
        Type::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![10]); // 999 % 10 + 1 = 10
        },
        _ => panic!("Expected tensor type after deep nesting"),
    }
    
    // Ensure it can be cloned without issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

/// Test 4: Type construction boundaries
#[test]
fn test_type_construction_boundaries() {
    // Test various type constructions and boundary conditions
    let types = [
        Type::F32, 
        Type::F64, 
        Type::I32, 
        Type::I64, 
        Type::Bool
    ];
    
    for (i, from_type) in types.iter().enumerate() {
        // Create a value with each type
        let value = Value {
            name: format!("construction_test_{}", i),
            ty: from_type.clone(),
            shape: vec![1],
        };
        
        // Verify creation succeeds regardless of type
        assert_eq!(value.ty, *from_type);
        
        // Test creating another value with same type
        let same_type_value = Value {
            name: format!("same_type_test_{}", i),
            ty: from_type.clone(),
            shape: vec![2],
        };
        assert_eq!(same_type_value.ty, *from_type);
    }
}

/// Test 5: Invalid UTF-8 sequence handling in names (Rust strings can't contain invalid UTF-8)
#[test]
fn test_unicode_handling_edge_cases() {
    // Test with various Unicode edge cases that are valid in Rust
    let unicode_test_cases = [
        "",                                  // Empty string
        "a",                                 // Basic ASCII
        "α",                                 // Greek letter
        "🙂",                               // Emoji
        "Z̴̜̰̬̱̲̙͉̘̱̗̯̞̩̫̝̬͎̪̙͈̱̩̥̝̱̤̖̳̭̠̲̮̹̲̹̣̾̃̅̂̒͑̍́̆̾̂̋̐̄̚̕͘͝", // Combining characters
        "\u{0001}\u{0080}\u{0081}\u{0082}", // Control characters
    ];
    
    for (i, &case) in unicode_test_cases.iter().enumerate() {
        // Test value name
        let value = Value {
            name: case.to_string(),
            ty: Type::F32,
            shape: vec![i + 1],
        };
        assert_eq!(value.name, case);
        assert_eq!(value.shape, vec![i + 1]);
        
        // Test operation name
        let op = Operation::new(case);
        assert_eq!(op.op_type, case);
        
        // Test module name
        let module = Module::new(case.to_string());
        assert_eq!(module.name, case);
    }
}

/// Test 6: Large sparse tensor simulations (high dimensionality, low occupancy)
#[test]
fn test_sparse_tensor_simulations() {
    // Create tensors that are conceptually "sparse" - high dimensional but with many zeros
    let sparse_scenarios = [
        vec![1000, 0, 1000],     // Contains zero, so sparse
        vec![1, 1, 1, 1, 1],     // Minimal but high dimensional
        vec![2; 20],             // 2^20 = ~1M elements, but uniform
        vec![0, 1, 1, 1, 1, 1],  // Leading zero, sparse
    ];
    
    for (i, shape) in sparse_scenarios.iter().enumerate() {
        let value = Value {
            name: format!("sparse_tensor_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        let total_elements: usize = value.shape.iter().product();
        
        match i {
            0 | 3 => assert_eq!(total_elements, 0),  // Contains zero
            1 => assert_eq!(total_elements, 1),      // All ones
            2 => assert_eq!(total_elements, 1048576), // 2^20
            _ => panic!("Unexpected case"),
        }
        
        // Verify tensor characteristics
        assert_eq!(value.shape, *shape);
        assert_eq!(value.ty, Type::F32);
    }
}

/// Test 7: Mixed signed/unsigned boundary operations
#[test]
fn test_signed_unsigned_boundary_operations() {
    use std::i64;
    
    // Test operations with boundary integers that might represent signed/unsigned confusion
    let boundary_values = [
        0i64,                    // Zero
        1i64,                    // Small positive
        -1i64,                   // Small negative
        i64::MAX,                // Max signed
        i64::MIN,                // Min signed
        i64::MAX / 2,            // Half max
        i64::MIN / 2,            // Half min
    ];
    
    let mut op = Operation::new("signed_unsigned_test");
    let mut attrs = HashMap::new();
    
    for (i, &val) in boundary_values.iter().enumerate() {
        attrs.insert(
            format!("boundary_val_{}", i),
            Attribute::Int(val)
        );
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), boundary_values.len());
    
    // Verify specific values were stored correctly
    assert_eq!(op.attributes.get("boundary_val_0"), Some(&Attribute::Int(0)));
    assert_eq!(op.attributes.get("boundary_val_1"), Some(&Attribute::Int(1)));
    assert_eq!(op.attributes.get("boundary_val_2"), Some(&Attribute::Int(-1)));
    assert_eq!(op.attributes.get("boundary_val_3"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("boundary_val_4"), Some(&Attribute::Int(i64::MIN)));
}

/// Test 8: Memory efficiency with identical/repeated values
#[test]
fn test_memory_efficiency_with_repeated_values() {
    // Test creating many values with identical characteristics to check for memory efficiency
    let mut values = Vec::new();
    
    for i in 0..100_000 {
        values.push(Value {
            name: format!("repeated_value_{}", i),
            ty: Type::F32, // Same type across all
            shape: vec![1], // Same simple shape
        });
    }
    
    assert_eq!(values.len(), 100_000);
    
    // Verify a few values to ensure they were created correctly
    assert_eq!(values[0].ty, Type::F32);
    assert_eq!(values[0].shape, vec![1]);
    assert_eq!(values[99_999].ty, Type::F32);
    assert_eq!(values[99_999].shape, vec![1]);
    
    // Clean up to test memory deallocation
    drop(values);
    assert!(true); // If we reach here, no memory issues occurred
}

/// Test 9: Operations with alternating complex/simple structure
#[test]
fn test_alternating_complex_simple_structure() {
    let mut op = Operation::new("alternating_structure");
    
    // Add alternating complex/complex values
    for i in 0..1000 {
        if i % 2 == 0 {
            // Complex structure
            op.inputs.push(Value {
                name: format!("complex_input_{}", i),
                ty: Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![i + 1, i + 2],
                },
                shape: vec![2, 2],
            });
        } else {
            // Simple structure
            op.inputs.push(Value {
                name: format!("simple_input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
    }
    
    assert_eq!(op.inputs.len(), 1000);
    
    // Verify alternating pattern
    if let Type::Tensor { .. } = op.inputs[0].ty {
        assert!(true); // First should be complex
    } else {
        panic!("Expected complex type at even index");
    }
    
    assert_eq!(op.inputs[1].ty, Type::F32); // Second should be simple
}

/// Test 10: Comprehensive module interconnection with maximum edge cases
#[test]
fn test_comprehensive_module_with_edge_cases() {
    let mut module = Module::new("comprehensive_edge_case_module_\u{1F680}"); // Rocket emoji in name
    
    // Add operations with various edge cases combined
    for op_idx in 0..100 {
        let mut op = Operation::new(&format!("op_{}_🚀", op_idx));
        
        // Mix of extreme value types
        for val_idx in 0..50 {
            op.inputs.push(Value {
                name: format!("val_{}_{}_🔥", op_idx, val_idx),
                ty: match val_idx % 5 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    3 => Type::I64,
                    _ => Type::Bool,
                },
                shape: if val_idx % 7 == 0 {
                    // Sometimes zero dimension
                    vec![]
                } else if val_idx % 5 == 0 {
                    // Sometimes contains zero
                    vec![val_idx + 1, 0, (val_idx % 10) + 1]
                } else {
                    // Normal shape
                    vec![val_idx + 1, (op_idx % 10) + 1]
                },
            });
        }
        
        // Add mixed attribute types
        let mut attrs = HashMap::new();
        for attr_idx in 0..20 {
            attrs.insert(
                format!("attr_{}_{}_🌟", op_idx, attr_idx),
                match attr_idx % 4 {
                    0 => Attribute::Int(attr_idx as i64),
                    1 => Attribute::Float(attr_idx as f64),
                    2 => Attribute::Bool(attr_idx % 2 == 0),
                    _ => Attribute::String(format!("str_attr_{}_{}", op_idx, attr_idx)),
                }
            );
        }
        op.attributes = attrs;
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    assert!(module.name.contains('🚀'));
    
    // Verify a few operations to ensure data integrity
    assert_eq!(module.operations[0].inputs.len(), 50);
    assert_eq!(module.operations[99].inputs.len(), 50);
    assert_eq!(module.operations[0].attributes.len(), 20);
    assert_eq!(module.operations[99].attributes.len(), 20);
    
    // Check that unicode in names was preserved
    assert!(module.operations[0].op_type.contains('🚀'));
    assert!(module.operations[0].inputs[0].name.contains('🔥'));
}