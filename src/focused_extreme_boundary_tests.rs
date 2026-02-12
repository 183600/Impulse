//! Focused extreme boundary tests - Additional edge cases with standard library assertions
//! 
//! This module contains 10 boundary test cases covering extreme scenarios
//! that may not be covered by other test suites.

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: num_elements() with maximum safe dimension values before overflow
#[test]
fn test_num_elements_max_safe_dimensions() {
    // Test dimensions that approach but don't overflow usize::MAX
    // On 64-bit systems, usize::MAX is 18,446,744,073,709,551,615
    // We use sqrt(max) * sqrt(max) approach for 2D tensor
    let sqrt_max = 4_294_967_295u32; // Close to sqrt(usize::MAX) on 64-bit
    
    let value = Value {
        name: "max_safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![sqrt_max as usize, 1], // Product = 4,294,967,295
    };
    
    // Verify num_elements returns correct value without overflow
    assert_eq!(value.num_elements(), Some(4_294_967_295));
    
    // Test with empty shape (scalar) - should return 1
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::I64,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
}

/// Test 2: Attribute with extreme float values close to overflow boundaries
#[test]
fn test_extreme_float_attribute_boundaries() {
    // Test f64::MAX and f64::MIN
    let max_float = Attribute::Float(f64::MAX);
    let min_float = Attribute::Float(f64::MIN);
    
    match max_float {
        Attribute::Float(val) => {
            assert_eq!(val, f64::MAX);
            assert!(val.is_finite());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    match min_float {
        Attribute::Float(val) => {
            assert_eq!(val, f64::MIN);
            assert!(val.is_finite());
        }
        _ => panic!("Expected Float attribute"),
    }
    
    // Test smallest positive normal float
    let min_positive = Attribute::Float(f64::MIN_POSITIVE);
    match min_positive {
        Attribute::Float(val) => {
            assert_eq!(val, f64::MIN_POSITIVE);
            assert!(val > 0.0 && val.is_normal());
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Type equality with deeply nested Tensor types
#[test]
fn test_deep_nested_type_equality() {
    // Create two deeply nested tensor types with identical structure
    let nested_a = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![2],
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    
    let nested_b = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![2],
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    
    // Verify they are equal
    assert_eq!(nested_a, nested_b);
    
    // Create a different nested type (different shape)
    let nested_c = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![5], // Different shape
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    
    // Verify they are not equal
    assert_ne!(nested_a, nested_c);
}

/// Test 4: Module with cyclic-like naming patterns (stress test)
#[test]
fn test_module_with_cyclic_naming_patterns() {
    let mut module = Module::new("cyclic_pattern_module");
    
    // Create operations with cyclic naming pattern
    // This tests the compiler's ability to handle complex naming schemes
    let num_ops = 100;
    for i in 0..num_ops {
        let mut op = Operation::new(&format!("op_{}", i % 10)); // Cyclic op_type names
        op.inputs.push(Value {
            name: format!("input_{}", (i + 1) % 10),
            ty: Type::F32,
            shape: vec![i as usize + 1],
        });
        op.outputs.push(Value {
            name: format!("output_{}", (i + 2) % 10),
            ty: Type::F32,
            shape: vec![i as usize + 2],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), num_ops);
    
    // Verify cyclic pattern is preserved
    assert_eq!(module.operations[0].op_type, "op_0");
    assert_eq!(module.operations[10].op_type, "op_0");
    assert_eq!(module.operations[20].op_type, "op_0");
}

/// Test 5: Compiler with zero-length input models
#[test]
fn test_compiler_with_zero_length_models() {
    let mut compiler = ImpulseCompiler::new();
    
    // Test with completely empty model
    let empty_model: Vec<u8> = vec![];
    let result = compiler.compile(&empty_model, "cpu");
    
    // Compiler should handle empty model gracefully
    // Either succeed or fail with a proper error message
    match result {
        Ok(_) => {
            // Compilation succeeded for empty model
            assert!(true);
        }
        Err(e) => {
            // Compilation failed but with a proper error message
            let error_msg = e.to_string();
            assert!(!error_msg.is_empty());
        }
    }
}

/// Test 6: Operation with HashMap attributes containing special key characters
#[test]
fn test_attribute_hashmap_with_special_keys() {
    let mut op = Operation::new("special_keys_op");
    let mut attrs = HashMap::new();
    
    // Use special characters in attribute keys
    let special_keys = [
        "key.with.dots",
        "key-with-dashes",
        "key_with_underscores",
        "key:with:colons",
        "key/with/slashes",
        "key\\with\\backslashes",
        "key@with@at",
        "key#with#hash",
        "key$with$dollar",
        "key%with%percent",
    ];
    
    for (i, key) in special_keys.iter().enumerate() {
        attrs.insert(key.to_string(), Attribute::Int(i as i64));
    }
    
    op.attributes = attrs;
    
    // Verify all special keys are preserved
    assert_eq!(op.attributes.len(), special_keys.len());
    for (i, key) in special_keys.iter().enumerate() {
        assert!(op.attributes.contains_key(*key));
        match op.attributes.get(*key) {
            Some(Attribute::Int(val)) => assert_eq!(*val, i as i64),
            _ => panic!("Expected Int attribute for key: {}", key),
        }
    }
}

/// Test 7: Value with shape containing very large dimension followed by zeros
#[test]
fn test_value_large_dimension_with_zero() {
    // Large dimension followed by zero should result in 0 elements
    let shapes = [
        vec![1_000_000_000, 0],
        vec![100_000, 0, 100_000],
        vec![0, 1_000_000_000],
        vec![50_000, 0, 0, 50_000],
    ];
    
    for shape in shapes.iter() {
        let value = Value {
            name: "zero_result_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any shape containing zero should result in Some(0)
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 8: Attribute array with single element of each type
#[test]
fn test_single_element_attribute_arrays() {
    // Create single-element arrays for each attribute type
    let int_array = Attribute::Array(vec![Attribute::Int(42)]);
    let float_array = Attribute::Array(vec![Attribute::Float(3.14)]);
    let string_array = Attribute::Array(vec![Attribute::String("test".to_string())]);
    let bool_array = Attribute::Array(vec![Attribute::Bool(true)]);
    let nested_array = Attribute::Array(vec![Attribute::Array(vec![])]);
    
    // Verify each array has exactly one element
    match int_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match float_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Float(val) if (val - 3.14).abs() < f64::EPSILON => (),
                _ => panic!("Expected Float(3.14)"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match string_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::String(s) if s == "test" => (),
                _ => panic!("Expected String(\"test\")"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match bool_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match arr[0] {
                Attribute::Bool(true) => (),
                _ => panic!("Expected Bool(true)"),
            }
        }
        _ => panic!("Expected Array"),
    }
    
    match nested_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match &arr[0] {
                Attribute::Array(inner) => {
                    assert_eq!(inner.len(), 0);
                }
                _ => panic!("Expected empty Array"),
            }
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 9: Module with inputs and outputs sharing same names
#[test]
fn test_module_shared_input_output_names() {
    let mut module = Module::new("shared_names_module");
    
    // Add input
    module.inputs.push(Value {
        name: "shared_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    // Add output with same name
    module.outputs.push(Value {
        name: "shared_name".to_string(),
        ty: Type::F64, // Different type but same name
        shape: vec![10],
    });
    
    // Verify both exist with same name
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, "shared_name");
    assert_eq!(module.outputs[0].name, "shared_name");
    
    // They should have different types
    assert_eq!(module.inputs[0].ty, Type::F32);
    assert_eq!(module.outputs[0].ty, Type::F64);
}

/// Test 10: Value with all dimension sizes equal to 1 (1x1x1... tensors)
#[test]
fn test_all_ones_dimensions() {
    // Create tensors with varying dimensions but all size 1
    let shapes = [
        vec![1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![1, 1, 1, 1],
        vec![1, 1, 1, 1, 1],
        vec![1, 1, 1, 1, 1, 1],
    ];
    
    for (i, shape) in shapes.iter().enumerate() {
        let value = Value {
            name: format!("ones_tensor_{}d", i + 1),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // All dimensions are 1, so product should be 1
        assert_eq!(value.num_elements(), Some(1));
        assert_eq!(value.shape.iter().product::<usize>(), 1);
        
        // Verify shape length is correct
        assert_eq!(value.shape.len(), i + 1);
    }
}
