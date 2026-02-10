//! New critical boundary tests for the Impulse compiler
//! Covers additional edge cases and boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::{gcd, lcm, round_up_to_multiple, next_power_of_2};

/// Test 1: Check for integer overflow in num_elements calculation
#[test]
fn test_value_num_elements_overflow_protection() {
    // Test with shapes that might cause overflow in multiplication
    // Using checked_mul through try_fold in num_elements
    
    // Shape with product that fits in usize
    let safe_value = Value {
        name: "safe_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000, 100], // 100M elements
    };
    assert_eq!(safe_value.num_elements(), Some(100_000_000));
    
    // Shape with zero (should return 0, not overflow)
    let zero_dim_value = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 0, 100],
    };
    assert_eq!(zero_dim_value.num_elements(), Some(0));
    
    // Empty shape (scalar) should return 1
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: Type::I32,
        shape: vec![],
    };
    assert_eq!(scalar_value.num_elements(), Some(1));
}

/// Test 2: Test math utility functions with extreme boundary values
#[test]
fn test_math_utils_extreme_values() {
    // Test GCD with 0
    assert_eq!(gcd(0, 0), 0);
    assert_eq!(gcd(0, 100), 100);
    assert_eq!(gcd(100, 0), 100);
    
    // Test GCD with same values
    assert_eq!(gcd(42, 42), 42);
    assert_eq!(gcd(1, 1), 1);
    
    // Test LCM boundary conditions
    assert_eq!(lcm(0, 0), 0);
    assert_eq!(lcm(0, 100), 0);
    assert_eq!(lcm(100, 0), 0);
    
    // Test round_up_to_multiple edge cases
    assert_eq!(round_up_to_multiple(0, 10), 0);
    assert_eq!(round_up_to_multiple(0, 0), 0);
    assert_eq!(round_up_to_multiple(100, 1), 100);
    assert_eq!(round_up_to_multiple(1, 1), 1);
    
    // Test next_power_of_2 with 0 and 1
    assert_eq!(next_power_of_2(0), 1);
    assert_eq!(next_power_of_2(1), 1);
}

/// Test 3: Test Type validation with all possible types
#[test]
fn test_type_validation_for_all_types() {
    use crate::ir::TypeExtensions;
    
    // Test all basic types are valid
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
    
    // Test nested tensor types
    let nested1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(nested1.is_valid_type());
    
    // Test deeply nested tensor
    let nested2 = Type::Tensor {
        element_type: Box::new(nested1.clone()),
        shape: vec![4, 5],
    };
    assert!(nested2.is_valid_type());
}

/// Test 4: Test Module with maximum number of operations (stress test)
#[test]
fn test_module_maximum_operations() {
    let mut module = Module::new("max_ops_module");
    
    // Add operations with increasing complexity
    for i in 0..100 {
        let mut op = Operation::new(&format!("stress_op_{}", i));
        
        // Add varying number of inputs
        for j in 0..(i % 10 + 1) {
            op.inputs.push(Value {
                name: format!("input_{}_{}", i, j),
                ty: Type::F32,
                shape: vec![10, 10],
            });
        }
        
        // Add outputs
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    
    // Verify first and last operations
    assert_eq!(module.operations[0].op_type, "stress_op_0");
    assert_eq!(module.operations[99].op_type, "stress_op_99");
}

/// Test 5: Test Attribute with special float values
#[test]
fn test_attribute_special_float_values() {
    // Test infinity
    let inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    
    match inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float(INFINITY)"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float(NEG_INFINITY)"),
    }
    
    // Test NaN
    let nan = Attribute::Float(f64::NAN);
    match nan {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float(NAN)"),
    }
    
    // Test negative zero
    let neg_zero = Attribute::Float(-0.0);
    match neg_zero {
        Attribute::Float(val) => assert_eq!(val, 0.0),
        _ => panic!("Expected Float(-0.0)"),
    }
}

/// Test 6: Test empty and single element containers
#[test]
fn test_empty_and_single_element_containers() {
    // Empty module
    let empty_module = Module::new("empty");
    assert_eq!(empty_module.operations.len(), 0);
    assert_eq!(empty_module.inputs.len(), 0);
    assert_eq!(empty_module.outputs.len(), 0);
    
    // Empty operation
    let empty_op = Operation::new("empty_op");
    assert_eq!(empty_op.inputs.len(), 0);
    assert_eq!(empty_op.outputs.len(), 0);
    assert_eq!(empty_op.attributes.len(), 0);
    
    // Empty attribute array
    let empty_array = Attribute::Array(vec![]);
    match empty_array {
        Attribute::Array(arr) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected empty Array"),
    }
    
    // Single element shapes
    let single_element = Value {
        name: "single".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1],
    };
    assert_eq!(single_element.num_elements(), Some(1));
}

/// Test 7: Test Operation with all possible input/output type combinations
#[test]
fn test_operation_mixed_type_combinations() {
    let mut op = Operation::new("mixed_types_op");
    
    // Add inputs of all types
    op.inputs.push(Value {
        name: "f32_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    op.inputs.push(Value {
        name: "i64_input".to_string(),
        ty: Type::I64,
        shape: vec![5],
    });
    op.inputs.push(Value {
        name: "bool_input".to_string(),
        ty: Type::Bool,
        shape: vec![3],
    });
    
    // Add outputs of all types
    op.outputs.push(Value {
        name: "f64_output".to_string(),
        ty: Type::F64,
        shape: vec![10],
    });
    op.outputs.push(Value {
        name: "i32_output".to_string(),
        ty: Type::I32,
        shape: vec![5],
    });
    
    assert_eq!(op.inputs.len(), 3);
    assert_eq!(op.outputs.len(), 2);
    assert_eq!(op.inputs[0].ty, Type::F32);
    assert_eq!(op.inputs[1].ty, Type::I64);
    assert_eq!(op.inputs[2].ty, Type::Bool);
    assert_eq!(op.outputs[0].ty, Type::F64);
    assert_eq!(op.outputs[1].ty, Type::I32);
}

/// Test 8: Test tensor with very small but non-zero dimensions
#[test]
fn test_tensor_very_small_dimensions() {
    // Test with dimension size of 1
    let ones_tensor = Value {
        name: "ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1],
    };
    assert_eq!(ones_tensor.num_elements(), Some(1));
    
    // Test with alternating 1s and small numbers
    let mixed_small = Value {
        name: "mixed_small".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 1, 3, 1],
    };
    assert_eq!(mixed_small.num_elements(), Some(6));
    
    // Test with 1D tensor of size 1
    let single_1d = Value {
        name: "single_1d".to_string(),
        ty: Type::I32,
        shape: vec![1],
    };
    assert_eq!(single_1d.num_elements(), Some(1));
}

/// Test 9: Test Attribute with maximum/minimum integer values
#[test]
fn test_attribute_extreme_integer_values() {
    // Test with maximum values
    let max_i64 = Attribute::Int(i64::MAX);
    let min_i64 = Attribute::Int(i64::MIN);
    let zero = Attribute::Int(0);
    let max_positive = Attribute::Int(2147483647); // i32::MAX
    let min_negative = Attribute::Int(-2147483648); // i32::MIN
    
    match max_i64 {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected Int(i64::MAX)"),
    }
    
    match min_i64 {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected Int(i64::MIN)"),
    }
    
    match zero {
        Attribute::Int(val) => assert_eq!(val, 0),
        _ => panic!("Expected Int(0)"),
    }
    
    match max_positive {
        Attribute::Int(val) => assert_eq!(val, 2147483647),
        _ => panic!("Expected Int(2147483647)"),
    }
    
    match min_negative {
        Attribute::Int(val) => assert_eq!(val, -2147483648),
        _ => panic!("Expected Int(-2147483648)"),
    }
}

/// Test 10: Test Module with chained operation dependencies
#[test]
fn test_module_chained_operation_dependencies() {
    let mut module = Module::new("chained_module");
    
    // Create a chain: op1 -> op2 -> op3 -> op4
    let mut op1 = Operation::new("source");
    op1.outputs.push(Value {
        name: "intermediate_1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op1);
    
    let mut op2 = Operation::new("process_1");
    op2.inputs.push(module.operations[0].outputs[0].clone());
    op2.outputs.push(Value {
        name: "intermediate_2".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op2);
    
    let mut op3 = Operation::new("process_2");
    op3.inputs.push(module.operations[1].outputs[0].clone());
    op3.outputs.push(Value {
        name: "intermediate_3".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op3);
    
    let mut op4 = Operation::new("sink");
    op4.inputs.push(module.operations[2].outputs[0].clone());
    op4.outputs.push(Value {
        name: "final_output".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.add_operation(op4);
    
    // Verify the chain
    assert_eq!(module.operations.len(), 4);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[1].inputs.len(), 1);
    assert_eq!(module.operations[2].inputs.len(), 1);
    assert_eq!(module.operations[3].inputs.len(), 1);
    
    // Verify input names match previous output names
    assert_eq!(
        module.operations[1].inputs[0].name,
        module.operations[0].outputs[0].name
    );
    assert_eq!(
        module.operations[2].inputs[0].name,
        module.operations[1].outputs[0].name
    );
    assert_eq!(
        module.operations[3].inputs[0].name,
        module.operations[2].outputs[0].name
    );
}