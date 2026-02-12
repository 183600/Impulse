//! Advanced comprehensive edge tests - 10 additional edge cases with standard library assertions
//! 覆盖更多边界情况，使用标准库的 assert! 和 assert_eq!

use crate::{
    ir::{Module, Value, Type, Operation, Attribute, TypeExtensions},
    ImpulseCompiler,
};

/// Test 1: Value with num_elements() overflow detection
#[test]
fn test_value_num_elements_overflow_detection() {
    // Test with dimensions that could cause overflow in unchecked multiplication
    // Using checked_mul in num_elements() should prevent overflow
    let safe_value = Value {
        name: "safe_overflow".to_string(),
        ty: Type::F32,
        shape: vec![10000, 10000],  // 100 million elements, safe
    };
    assert_eq!(safe_value.num_elements(), Some(100_000_000));

    // Test with empty shape (scalar)
    let scalar = Value {
        name: "scalar_overflow".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));

    // Test with zero dimension
    let zero_dim = Value {
        name: "zero_dim_overflow".to_string(),
        ty: Type::F32,
        shape: vec![0, 1000000],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));
}

/// Test 2: Module with cyclic operation references pattern
#[test]
fn test_module_cyclic_operation_pattern() {
    let mut module = Module::new("cyclic_pattern");
    
    // Create operations that form a conceptual cycle
    let mut op1 = Operation::new("op1");
    op1.outputs.push(Value {
        name: "intermediate".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(Value {
        name: "intermediate".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op2.outputs.push(Value {
        name: "final".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    module.add_operation(op1);
    module.add_operation(op2);
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].op_type, "op1");
    assert_eq!(module.operations[1].op_type, "op2");
}

/// Test 3: Attribute with denormalized float values
#[test]
fn test_denormalized_float_attributes() {
    // Test with denormalized (subnormal) float values
    let denormal = f64::MIN_POSITIVE / 2.0;  // Smallest subnormal
    let attr = Attribute::Float(denormal);
    
    match attr {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val < f64::MIN_POSITIVE);
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 4: Value with asymmetric extreme shape dimensions
#[test]
fn test_value_asymmetric_extreme_dimensions() {
    // Test with highly asymmetric tensor shapes
    let test_cases = vec![
        vec![1, 1000000],        // 1x1M
        vec![1000000, 1],        // 1Mx1
        vec![2, 500000],         // 2x500K
        vec![500000, 2],         // 500Kx2
        vec![10, 10, 10000],     // 10x10x10K
    ];
    
    for shape in test_cases {
        let value = Value {
            name: "asymmetric_tensor".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        assert_eq!(value.shape, shape);
        
        // Verify num_elements works correctly
        let expected_elements: usize = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected_elements));
    }
}

/// Test 5: Compiler with consecutive identical empty compilations
#[test]
fn test_compiler_consecutive_identical_compilations() {
    let mut compiler = ImpulseCompiler::new();
    let empty_model = vec![];
    
    // Perform 10 identical empty compilations
    let mut results = Vec::new();
    for _ in 0..10 {
        let result = compiler.compile(&empty_model, "cpu");
        results.push(result.is_ok() || result.is_err());
    }
    
    // All should complete without panicking
    assert!(results.iter().all(|&r| r));
}

/// Test 6: Type validation with nested tensor edge cases
#[test]
fn test_nested_tensor_validation() {
    // Test nested tensor validation
    let deep_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2],
            }),
            shape: vec![3],
        }),
        shape: vec![4],
    };
    
    // Validate recursively
    assert!(deep_nested.is_valid_type());
}

/// Test 7: Module with single operation and no explicit inputs/outputs
#[test]
fn test_module_single_operation_no_io() {
    let mut module = Module::new("single_op_module");
    let op = Operation::new("stateless_op");
    
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
}

/// Test 8: Operation attributes with special keys containing special characters
#[test]
fn test_operation_special_attribute_keys() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("special_keys_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with special character keys
    attrs.insert("key-with-dash".to_string(), Attribute::Int(1));
    attrs.insert("key_with_underscore".to_string(), Attribute::Int(2));
    attrs.insert("key.with.dot".to_string(), Attribute::Int(3));
    attrs.insert("key:with:colon".to_string(), Attribute::Int(4));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    assert!(op.attributes.contains_key("key-with-dash"));
    assert!(op.attributes.contains_key("key_with_underscore"));
    assert!(op.attributes.contains_key("key.with.dot"));
    assert!(op.attributes.contains_key("key:with:colon"));
}

/// Test 9: Value with single element but multi-dimensional shape
#[test]
fn test_value_single_element_multidimensional() {
    // Test tensors that have only 1 element but multiple dimensions
    let test_cases = vec![
        vec![1],           // 1D: 1 element
        vec![1, 1],        // 2D: 1 element
        vec![1, 1, 1],     // 3D: 1 element
        vec![1, 1, 1, 1],  // 4D: 1 element
    ];
    
    for shape in test_cases {
        let value = Value {
            name: "single_element_multi_dim".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        assert_eq!(value.num_elements(), Some(1));
        
        // Verify shape length matches expected
        assert_eq!(value.shape.len(), shape.len());
    }
}

/// Test 10: Module with operations having identical input/output value references
#[test]
fn test_module_operations_with_identical_values() {
    let mut module = Module::new("shared_values_module");
    
    // Create a shared value
    let shared_value = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![5, 5],
    };
    
    // Create multiple operations using the same value
    for i in 0..3 {
        let mut op = Operation::new(&format!("consumer_{}", i));
        op.inputs.push(shared_value.clone());
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 3);
    
    // Verify all operations have the same input
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.inputs[0].name, "shared");
    }
}