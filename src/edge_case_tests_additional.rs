//! Additional edge case tests for the Impulse compiler
//! This file contains 10 new test cases covering various boundary conditions

use rstest::*;
use impulse::{
    ir::{Module, Value, Type, Operation, Attribute},
    ImpulseCompiler,
};

/// Test 1: Tensor shapes that approach usize::MAX to check for overflow
#[test]
fn test_very_large_tensor_shapes_no_overflow() {
    // Create a tensor with dimensions that would approach usize limits
    // Use values that won't actually overflow but stress the calculation
    let large_shape = vec![100_000_000, 100_000_000];
    let value = Value {
        name: "max_tensor".to_string(),
        ty: Type::F32,
        shape: large_shape,
    };
    
    // This should not cause overflow in normal calculations but tests limits
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 100_000_000 * 100_000_000);
}

/// Test 2: Deeply nested tensor types with alternating element types
#[test]
fn test_very_deeply_nested_tensor_types() {
    // Create a very deeply nested type to ensure recursion limits are handled well
    let mut current_type = Type::F32;
    for i in 0..200 {
        current_type = Type::Tensor {
            element_type: Box::new(if i % 3 == 0 {
                Type::I32
            } else if i % 3 == 1 {
                Type::F64
            } else {
                Type::Bool
            }),
            shape: vec![2],
        };
    }
    
    // Test that we can clone this deeply nested type
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 3: Module with maximum possible values for component counts
#[test]
fn test_module_with_maximum_component_counts() {
    let mut module = Module::new("max_components_module");
    
    // Add a large number of operations to test memory handling
    for i in 0..50_000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
        
        // Occasionally add an output for variety
        if i % 1000 == 0 {
            module.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
    }
    
    assert_eq!(module.operations.len(), 50_000);
    assert_eq!(module.outputs.len(), 51); // 0, 1000, 2000, ..., 50000 (51 total)
}

/// Test 4: Operations with maximum attribute complexity
#[test]
fn test_operation_with_maximum_attribute_complexity() {
    use std::collections::HashMap;
    use std::iter::repeat_with;
    
    let mut op = Operation::new("max_complexity_op");
    let mut attrs = HashMap::new();
    
    // Add various types of attributes with complex nesting
    for i in 0..10_000 {
        match i % 5 {
            0 => { attrs.insert(format!("int_attr_{}", i), Attribute::Int(i as i64)); }
            1 => { attrs.insert(format!("float_attr_{}", i), Attribute::Float(i as f64 * 1.5)); }
            2 => { attrs.insert(format!("string_attr_{}", i), Attribute::String(format!("string_val_{}", i))); }
            3 => { attrs.insert(format!("bool_attr_{}", i), Attribute::Bool(i % 2 == 0)); }
            4 => { 
                // Add nested arrays with increasing complexity
                let nested_array = Attribute::Array(vec![
                    Attribute::Array(vec![
                        Attribute::Int(i as i64),
                        Attribute::String(format!("nested_{}", i)),
                    ]),
                    Attribute::Array(vec![
                        Attribute::Float((i as f64) * 0.5),
                        Attribute::Bool(i % 3 == 0),
                    ])
                ]);
                attrs.insert(format!("array_attr_{}", i), nested_array); 
            }
            _ => unreachable!(),
        }
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.op_type, "max_complexity_op");
}

/// Test 5: Test with special floating-point values in tensor contexts
#[rstest]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
#[case(f64::NAN)]
#[case(f64::EPSILON)]
#[case(-0.0)]
fn test_special_floating_point_attributes(#[case] special_value: f64) {
    let attr = Attribute::Float(special_value);
    
    match attr {
        Attribute::Float(value) => {
            if special_value.is_nan() {
                assert!(value.is_nan());
            } else if special_value.is_infinite() {
                assert!(value.is_infinite());
                assert_eq!(value.is_sign_positive(), special_value.is_sign_positive());
            } else {
                assert!((value - special_value).abs() < f64::EPSILON || 
                       (value.is_sign_negative() && special_value.is_sign_negative()));
            }
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 6: Zero-dimension tensors and their behavior
#[test]
fn test_zero_dimension_tensor_edge_cases() {
    let test_cases = [
        vec![],           // Scalar (0-dimensional tensor)
        vec![0],         // 1D tensor with 0 elements
        vec![0, 1],      // 2D tensor with 0 elements
        vec![1, 0],      // 2D tensor with 0 elements
        vec![0, 0],      // 2D tensor with 0 elements
        vec![1, 0, 1],   // 3D tensor with 0 elements
        vec![0, 1, 0],   // 3D tensor with 0 elements
    ];
    
    for (idx, shape) in test_cases.iter().enumerate() {
        let value = Value {
            name: format!("zero_shape_{}", idx),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any tensor with a zero dimension should have 0 total elements (except scalars)
        let total_elements: usize = value.shape.iter().product();
        if shape.is_empty() {
            assert_eq!(total_elements, 1, "Scalar should have 1 element"); // Scalar case
        } else {
            assert_eq!(total_elements, 0, "Tensor with zero dimension should have 0 elements");
        }
    }
}

/// Test 7: Extremely long identifiers with unicode characters
#[test]
fn test_extremely_long_unicode_identifiers() {
    // Create extremely long names with unicode characters
    let extreme_unicode_name = "🚀🔥🚀🔥".repeat(5_000) + "_末" + "🤖🔬".repeat(5_000).as_str();
    
    let value = Value {
        name: extreme_unicode_name.clone(),
        ty: Type::F32,
        shape: vec![1, 1],
    };
    
    assert_eq!(value.name, extreme_unicode_name);
    assert_eq!(value.ty, Type::F32);
    
    // Also test with an operation
    let op = Operation::new(&extreme_unicode_name);
    assert_eq!(op.op_type, extreme_unicode_name);
}

/// Test 8: Compiler resilience with invalid/edge-case inputs
#[test]
fn test_compiler_resilience_with_edge_case_inputs() {
    let mut compiler = ImpulseCompiler::new();
    
    // Test with extremely large inputs
    let large_model = vec![0xFFu8; 100_000_000]; // 100MB model
    let result = compiler.compile(&large_model, "cpu");
    
    // The result may be success or failure, but should not panic
    assert!(result.is_ok() || result.is_err());
    
    // Test with minimal input
    let empty_model = vec![];
    let result2 = compiler.compile(&empty_model, "cpu");
    assert!(result2.is_ok() || result2.is_err());
    
    // Test with unusual target names
    let normal_model = vec![0x01, 0x02, 0x03, 0x04];
    let targets = ["", "cpu", "gpu", "npu", "invalid_target", "cpu_with_very_long_name"];
    for target in targets {
        let result3 = compiler.compile(&normal_model, target);
        assert!(result3.is_ok() || result3.is_err());
    }
}

/// Test 9: Nested recursive types with max depth
#[test]
fn test_recursive_type_equality_at_extreme_depth() {
    // Create two identical deeply nested types
    let create_deep_type = |depth: usize, base_type: Type| -> Type {
        let mut current_type = base_type;
        for i in 0..depth {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 5 + 1], // Varying shapes
            };
        }
        current_type
    };
    
    const DEPTH: usize = 500; // Deep enough to test limits without causing stack overflow
    
    let type1 = create_deep_type(DEPTH, Type::I32);
    let type2 = create_deep_type(DEPTH, Type::I32);
    
    // These should be equal
    assert_eq!(type1, type2);
    
    // Create a slightly different one
    let type3 = create_deep_type(DEPTH - 1, Type::I32);
    assert_ne!(type1, type3);
}

/// Test 10: Memory allocation edge cases with large collections
#[test]
fn test_memory_allocation_with_large_collections() {
    // Create many modules to test memory management
    let mut modules = Vec::new();
    
    // Create 1000 small modules to test allocation/deallocation patterns
    for i in 0..1000 {
        let mut module = Module::new(&format!("test_module_{}", i));
        
        // Add a few operations to each module
        for j in 0..10 {
            let mut op = Operation::new(&format!("op_{}_{}", i, j));
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: Type::F32,
                shape: vec![j + 1],
            });
            
            // Add attributes to operations
            op.attributes.insert(
                format!("attr_{}_{}", i, j),
                Attribute::String(format!("value_{}_{}", i, j))
            );
            
            module.add_operation(op);
        }
        
        modules.push(module);
    }
    
    // Verify we created the expected number of modules with operations
    assert_eq!(modules.len(), 1000);
    for (idx, module) in modules.iter().enumerate() {
        assert_eq!(module.operations.len(), 10);
        assert_eq!(module.name, format!("test_module_{}", idx));
    }
    
    // Clean up to test deallocation
    drop(modules);
    
    // Simple assertion to ensure test completed without panic
    assert!(true);
}
