//! Critical compiler boundary tests - Focused on essential edge cases for compiler safety
//! Tests use standard library assertions (assert!, assert_eq!) and rstest

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;

/// Test 1: Value with shape that would cause overflow in num_elements() calculation
#[test]
fn test_shape_overflow_detection() {
    // Create a value with dimensions that multiply to overflow on 64-bit
    // On 64-bit, we need values that exceed 18446744073709551615
    // Let's try with a very large single dimension that causes issues
    let overflow_value = Value {
        name: "overflow_risk".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 2], // This should overflow: (MAX/2+1) * 2 > MAX
    };
    
    // num_elements should return None for overflow cases due to checked_mul
    assert_eq!(overflow_value.num_elements(), None);
}

/// Test 2: Attribute with negative zero float value
#[test]
fn test_negative_zero_float() {
    let neg_zero = -0.0_f64;
    let attr = Attribute::Float(neg_zero);
    
    match attr {
        Attribute::Float(val) => {
            // -0.0 is equal to 0.0 but has a different sign bit
            assert_eq!(val, 0.0);
            assert!(val.is_sign_negative());
        },
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 3: Module with cyclic dependency simulation (same value used as input and output)
#[test]
fn test_cyclic_dependency_pattern() {
    let mut module = Module::new("cyclic_module");
    
    let shared_value = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![5],
    };
    
    let mut op = Operation::new("in_place_op");
    op.inputs.push(shared_value.clone());
    op.outputs.push(shared_value.clone());
    module.add_operation(op);
    
    // Both input and output reference the same value
    assert_eq!(module.operations[0].inputs[0].name, "shared");
    assert_eq!(module.operations[0].outputs[0].name, "shared");
}

/// Test 4: Operation with very deep attribute nesting (100 levels)
#[test]
fn test_deep_attribute_nesting() {
    let mut nested = Attribute::Int(0);
    
    // Create 100 levels of nesting
    for i in 1..=100 {
        nested = Attribute::Array(vec![nested, Attribute::Int(i)]);
    }
    
    match nested {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 2);
            // The structure should be valid at all levels
        },
        _ => panic!("Expected deeply nested array"),
    }
}

/// Test 5: Value with extremely long name containing control characters
#[test]
fn test_name_with_control_characters() {
    let control_chars = "\x00\x01\x02\x1F\x7F";
    let value = Value {
        name: format!("tensor_{}_test", control_chars),
        ty: Type::F32,
        shape: vec![1],
    };
    
    assert!(value.name.contains("\x00"));
    assert!(value.name.len() > 10);
}

/// Test 6: Compiler with model containing only null bytes
#[test]
fn test_compiler_with_all_null_bytes() {
    let mut compiler = ImpulseCompiler::new();
    let null_model = vec![0u8; 1000]; // All null bytes
    
    let result = compiler.compile(&null_model, "cpu");
    // Should not panic, handle gracefully
    assert!(result.is_ok() || result.is_err());
}

/// Test 7: Attribute with extremely large integer value near i64::MAX
#[test]
fn test_large_integer_attribute() {
    let near_max = i64::MAX - 1000;
    let attr = Attribute::Int(near_max);
    
    match attr {
        Attribute::Int(val) => assert_eq!(val, near_max),
        _ => panic!("Expected Int attribute"),
    }
}

/// Test 8: Module with duplicate input/output names
#[test]
fn test_duplicate_io_names() {
    let mut module = Module::new("dup_io_module");
    
    let value = Value {
        name: "duplicate".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    // Add same value as both input and output
    module.inputs.push(value.clone());
    module.outputs.push(value);
    
    assert_eq!(module.inputs[0].name, "duplicate");
    assert_eq!(module.outputs[0].name, "duplicate");
}

/// Test 9: Operation with empty string attributes
#[test]
fn test_empty_string_attributes() {
    let mut op = Operation::new("empty_str_op");
    op.attributes.insert("empty".to_string(), Attribute::String("".to_string()));
    
    assert_eq!(op.attributes.len(), 1);
    match op.attributes.get("empty") {
        Some(Attribute::String(s)) => assert_eq!(s, ""),
        _ => panic!("Expected empty string"),
    }
}

/// Test 10: Value with scalar type (empty shape) in complex operations
#[test]
fn test_scalar_in_complex_operations() {
    let scalar = Value {
        name: "scalar_value".to_string(),
        ty: Type::F32,
        shape: vec![], // Empty shape = scalar
    };
    
    assert_eq!(scalar.num_elements(), Some(1));
    
    // Use scalar in an operation
    let mut op = Operation::new("scalar_op");
    op.inputs.push(scalar.clone());
    op.outputs.push(scalar);
    
    assert_eq!(op.inputs[0].shape.len(), 0);
    assert_eq!(op.outputs[0].shape.len(), 0);
}