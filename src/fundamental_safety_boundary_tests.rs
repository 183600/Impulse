//! Fundamental safety boundary tests - 基础安全边界测试
//! 覆盖编译器核心组件的关键安全边界情况，使用标准库 assert! 和 assert_eq!

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Value with checked_mul overflow detection in num_elements
#[test]
fn test_num_elements_overflow_detection() {
    // Test with dimensions that would cause overflow in unchecked multiplication
    // Use values that when multiplied together exceed usize::MAX
    // The num_elements() method should return None for overflow cases
    
    // Safe case: product fits in usize
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000], // 10 billion
    };
    assert_eq!(safe_value.num_elements(), Some(10_000_000_000));
    
    // Edge case: single dimension at maximum usize
    let max_dim_value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };
    assert_eq!(max_dim_value.num_elements(), Some(usize::MAX));
    
    // Zero dimension should return Some(0)
    let zero_value = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100, 0, 100],
    };
    assert_eq!(zero_value.num_elements(), Some(0));
    
    // Empty shape (scalar) should return Some(1)
    let scalar_value = Value {
        name: "scalar_tensor".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar_value.num_elements(), Some(1));
}

/// Test 2: Operation attribute with negative zero float
#[test]
fn test_negative_zero_float_attribute() {
    // Test -0.0 vs 0.0 behavior in float attributes
    let pos_zero = Attribute::Float(0.0);
    let neg_zero = Attribute::Float(-0.0);
    
    // -0.0 and 0.0 should be equal according to IEEE 754
    assert_eq!(pos_zero, neg_zero);
    
    // But they have different bit representations
    match (pos_zero, neg_zero) {
        (Attribute::Float(p), Attribute::Float(n)) => {
            assert_eq!(p, 0.0);
            assert_eq!(n, -0.0);
            // -0.0 has a different sign bit than 0.0
            assert_ne!(p.to_bits(), n.to_bits());
            // The sign bit is the only difference
            assert_eq!(p.to_bits() ^ n.to_bits(), 0x8000_0000_0000_0000);
        }
        _ => panic!("Expected Float attributes"),
    }
}

/// Test 3: Module with deeply nested tensor types
#[test]
fn test_deeply_nested_tensor_types() {
    // Create a deeply nested tensor type to test recursion limits
    // tensor<tensor<tensor<tensor<f32, [2]>, [2]>, [2]>, [2]>
    
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![2],
    };
    
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![2],
    };
    
    let level4 = Type::Tensor {
        element_type: Box::new(level3),
        shape: vec![2],
    };
    
    // Verify the deepest type is F32
    if let Type::Tensor { element_type, shape } = &level4 {
        assert_eq!(shape, &vec![2]);
        if let Type::Tensor { element_type: inner, shape: inner_shape } = element_type.as_ref() {
            assert_eq!(inner_shape, &vec![2]);
            if let Type::Tensor { element_type: inner2, shape: inner2_shape } = inner.as_ref() {
                assert_eq!(inner2_shape, &vec![2]);
                if let Type::Tensor { element_type: inner3, shape: inner3_shape } = inner2.as_ref() {
                    assert_eq!(inner3_shape, &vec![2]);
                    assert_eq!(*inner3.as_ref(), Type::F32);
                } else {
                    panic!("Expected Tensor at level 3");
                }
            } else {
                panic!("Expected Tensor at level 2");
            }
        } else {
            panic!("Expected Tensor at level 1");
        }
    } else {
        panic!("Expected Tensor at outer level");
    }
}

/// Test 4: Value with shape containing usize::MAX
#[test]
fn test_value_with_max_usize_dimension() {
    // Test creating a value with a dimension at usize::MAX
    let max_dim_value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX],
    };
    
    assert_eq!(max_dim_value.shape[0], usize::MAX);
    // Single dimension with usize::MAX should return Some(usize::MAX)
    assert_eq!(max_dim_value.num_elements(), Some(usize::MAX));
    
    // Test with multiple dimensions that are safe (no overflow)
    let safe_multi_dims = Value {
        name: "safe_multi".to_string(),
        ty: Type::F32,
        shape: vec![10_000, 10_000], // 100 million elements
    };
    
    assert_eq!(safe_multi_dims.num_elements(), Some(100_000_000));
    
    // Test with dimensions that will overflow when multiplied
    // On 64-bit, usize::MAX / 2 + 1 * 2 will overflow
    let half_max = usize::MAX / 2 + 1; // This is > MAX/2, so *2 overflows
    let overflow_dims = Value {
        name: "overflow_dims".to_string(),
        ty: Type::F32,
        shape: vec![2, half_max],
    };
    
    // This should cause overflow and return None
    assert_eq!(overflow_dims.num_elements(), None);
    
    // Verify: half_max * 2 would overflow
    assert_eq!(half_max.checked_mul(2), None);
}

/// Test 5: Attribute array with circular-like structure (via cloning)
#[test]
fn test_attribute_array_cloning_preserves_structure() {
    // Test that cloning nested arrays preserves structure correctly
    let original = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
        ]),
        Attribute::Float(3.14),
    ]);
    
    let cloned = original.clone();
    
    assert_eq!(original, cloned);
    
    // Verify nested structure is preserved
    match (original, cloned) {
        (Attribute::Array(orig_arr), Attribute::Array(cloned_arr)) => {
            assert_eq!(orig_arr.len(), cloned_arr.len());
            
            match (&orig_arr[0], &cloned_arr[0]) {
                (Attribute::Array(orig_inner), Attribute::Array(cloned_inner)) => {
                    assert_eq!(orig_inner, cloned_inner);
                }
                _ => panic!("Expected nested arrays"),
            }
        }
        _ => panic!("Expected Array attributes"),
    }
}

/// Test 6: Module with operation using all numeric types
#[test]
fn test_module_with_all_numeric_types() {
    let mut module = Module::new("numeric_types_module");
    
    // Create operations with different numeric types
    let mut f32_op = Operation::new("f32_op");
    f32_op.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(f32_op);
    
    let mut f64_op = Operation::new("f64_op");
    f64_op.inputs.push(Value {
        name: "f64_input".to_string(),
        ty: Type::F64,
        shape: vec![10],
    });
    module.add_operation(f64_op);
    
    let mut i32_op = Operation::new("i32_op");
    i32_op.inputs.push(Value {
        name: "i32_input".to_string(),
        ty: Type::I32,
        shape: vec![10],
    });
    module.add_operation(i32_op);
    
    let mut i64_op = Operation::new("i64_op");
    i64_op.inputs.push(Value {
        name: "i64_input".to_string(),
        ty: Type::I64,
        shape: vec![10],
    });
    module.add_operation(i64_op);
    
    let mut bool_op = Operation::new("bool_op");
    bool_op.inputs.push(Value {
        name: "bool_input".to_string(),
        ty: Type::Bool,
        shape: vec![10],
    });
    module.add_operation(bool_op);
    
    assert_eq!(module.operations.len(), 5);
    assert_eq!(module.operations[0].inputs[0].ty, Type::F32);
    assert_eq!(module.operations[1].inputs[0].ty, Type::F64);
    assert_eq!(module.operations[2].inputs[0].ty, Type::I32);
    assert_eq!(module.operations[3].inputs[0].ty, Type::I64);
    assert_eq!(module.operations[4].inputs[0].ty, Type::Bool);
}

/// Test 7: Operation with attribute containing very large integers
#[test]
fn test_operation_with_large_integer_attributes() {
    let mut op = Operation::new("large_int_op");
    let mut attrs = HashMap::new();
    
    // Test with large integer values
    attrs.insert("large_positive".to_string(), Attribute::Int(i64::MAX / 2));
    attrs.insert("large_negative".to_string(), Attribute::Int(i64::MIN / 2));
    attrs.insert("power_of_two".to_string(), Attribute::Int(2_i64.pow(62)));
    attrs.insert("negative_power".to_string(), Attribute::Int(-2_i64.pow(62)));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    
    match op.attributes.get("large_positive") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX / 2),
        _ => panic!("Expected Int attribute"),
    }
    
    match op.attributes.get("large_negative") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN / 2),
        _ => panic!("Expected Int attribute"),
    }
}

/// Test 8: Value with shape containing consecutive ones
#[test]
fn test_value_with_consecutive_ones_shape() {
    // Test shapes with consecutive ones (common in broadcasting scenarios)
    let test_cases = vec![
        vec![1, 1, 1],
        vec![1, 10, 1],
        vec![1, 1, 100],
        vec![10, 1, 1],
        vec![1, 1, 1, 1],
    ];
    
    for shape in test_cases {
        let value = Value {
            name: "ones_shape".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        // Verify num_elements calculation
        let expected = shape.iter().product();
        assert_eq!(value.num_elements(), Some(expected));
    }
}

/// Test 9: Module with operations sharing input values
#[test]
fn test_module_operations_sharing_inputs() {
    let mut module = Module::new("shared_inputs_module");
    
    // Create a shared input value
    let shared_input = Value {
        name: "shared_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    };
    
    // Create multiple operations that use the same input
    let mut op1 = Operation::new("op1");
    op1.inputs.push(shared_input.clone());
    module.add_operation(op1);
    
    let mut op2 = Operation::new("op2");
    op2.inputs.push(shared_input.clone());
    module.add_operation(op2);
    
    let mut op3 = Operation::new("op3");
    op3.inputs.push(shared_input);
    module.add_operation(op3);
    
    assert_eq!(module.operations.len(), 3);
    
    // All operations should have the same input name
    assert_eq!(module.operations[0].inputs[0].name, "shared_input");
    assert_eq!(module.operations[1].inputs[0].name, "shared_input");
    assert_eq!(module.operations[2].inputs[0].name, "shared_input");
}

/// Test 10: Type validation for all valid types
#[test]
fn test_type_validation_for_all_types() {
    use crate::ir::TypeExtensions;
    
    // Test all primitive types are valid
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
    
    // Test nested tensor types
    let tensor_f32 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(tensor_f32.is_valid_type());
    
    let tensor_i64 = Type::Tensor {
        element_type: Box::new(Type::I64),
        shape: vec![10],
    };
    assert!(tensor_i64.is_valid_type());
    
    // Test deeply nested tensor types
    let nested_tensor = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        }),
        shape: vec![3],
    };
    assert!(nested_tensor.is_valid_type());
}