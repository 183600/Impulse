/// Critical boundary safety tests - 10 essential edge cases for compiler robustness
use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: num_elements overflow detection with checked_mul
#[test]
fn test_num_elements_overflow_protection() {
    // Test with values that would overflow on naive multiplication
    let value = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // Would overflow if multiplied directly
    };
    
    // num_elements uses checked_mul and should return None on overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 2: NaN and Infinity float attribute handling
#[test]
fn test_special_float_values_in_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    
    // Verify NaN attribute is stored correctly
    if let Attribute::Float(val) = nan_attr {
        assert!(val.is_nan());
    } else {
        panic!("Expected Float attribute with NaN");
    }
    
    // Verify positive infinity
    if let Attribute::Float(val) = pos_inf_attr {
        assert!(val.is_infinite() && val.is_sign_positive());
    } else {
        panic!("Expected Float attribute with +inf");
    }
    
    // Verify negative infinity
    if let Attribute::Float(val) = neg_inf_attr {
        assert!(val.is_infinite() && val.is_sign_negative());
    } else {
        panic!("Expected Float attribute with -inf");
    }
}

/// Test 3: Zero and negative integer edge cases
#[test]
fn test_extreme_integer_attributes() {
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let zero_int = Attribute::Int(0);
    
    assert_eq!(max_int, Attribute::Int(i64::MAX));
    assert_eq!(min_int, Attribute::Int(i64::MIN));
    assert_eq!(zero_int, Attribute::Int(0));
    
    // Verify they are distinct
    assert_ne!(max_int, min_int);
    assert_ne!(max_int, zero_int);
    assert_ne!(min_int, zero_int);
}

/// Test 4: Empty shape (scalar) vs shape with single 1
#[test]
fn test_scalar_vs_single_element_tensor() {
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    
    let single_1d = Value {
        name: "single_1d".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    
    let single_2d = Value {
        name: "single_2d".to_string(),
        ty: Type::F32,
        shape: vec![1, 1],
    };
    
    // All should have 1 element
    assert_eq!(scalar.num_elements(), Some(1));
    assert_eq!(single_1d.num_elements(), Some(1));
    assert_eq!(single_2d.num_elements(), Some(1));
    
    // But shapes are different
    assert_eq!(scalar.shape.len(), 0);
    assert_eq!(single_1d.shape.len(), 1);
    assert_eq!(single_2d.shape.len(), 2);
}

/// Test 5: Shape containing zero dimensions
#[test]
fn test_zero_dimension_handling() {
    let zero_dim_cases = vec![
        vec![0],          // Single zero dimension
        vec![10, 0, 5],   // Zero in middle
        vec![0, 10, 10],  // Zero at start
        vec![10, 10, 0],  // Zero at end
        vec![0, 0, 0],    // All zeros
    ];
    
    for shape in zero_dim_cases {
        let value = Value {
            name: "zero_dim_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any shape containing zero should result in 0 elements
        assert_eq!(value.num_elements(), Some(0));
    }
}

/// Test 6: Subnormal float values (very small numbers)
#[test]
fn test_subnormal_float_attributes() {
    let min_positive = Attribute::Float(f64::MIN_POSITIVE);
    let subnormal = Attribute::Float(f64::MIN_POSITIVE / 2.0);
    
    if let Attribute::Float(val) = min_positive {
        assert!(val > 0.0);
        assert!(val < 1e-300);
    }
    
    if let Attribute::Float(val) = subnormal {
        // Subnormals are still positive but smaller than MIN_POSITIVE
        assert!(val >= 0.0);
    }
}

/// Test 7: Empty module, operation, and attribute structures
#[test]
fn test_empty_structures() {
    let module = Module::new("empty");
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);
    
    let op = Operation::new("empty_op");
    assert_eq!(op.inputs.len(), 0);
    assert_eq!(op.outputs.len(), 0);
    assert_eq!(op.attributes.len(), 0);
    
    let empty_array = Attribute::Array(vec![]);
    if let Attribute::Array(arr) = empty_array {
        assert_eq!(arr.len(), 0);
    }
}

/// Test 8: Type validation for nested tensor types
#[test]
fn test_nested_tensor_type_validation() {
    use crate::ir::TypeExtensions;
    
    let simple_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    
    // Clone to avoid moving simple_tensor
    let nested_tensor = Type::Tensor {
        element_type: Box::new(simple_tensor.clone()),
        shape: vec![4],
    };
    
    // Both should be valid types
    assert!(simple_tensor.is_valid_type());
    assert!(nested_tensor.is_valid_type());
    
    // Verify the nested structure
    if let Type::Tensor { element_type, shape } = nested_tensor {
        assert_eq!(shape, vec![4]);
        if let Type::Tensor { shape: inner_shape, .. } = element_type.as_ref() {
            assert_eq!(inner_shape, &vec![2, 3]);
        }
    }
}

/// Test 9: Attribute with very long string
#[test]
fn test_long_string_attribute() {
    let long_string = "x".repeat(100_000);
    let attr = Attribute::String(long_string.clone());
    
    if let Attribute::String(s) = attr {
        assert_eq!(s.len(), 100_000);
        assert_eq!(&s[..10], "xxxxxxxxxx");
        assert_eq!(&s[s.len()-10..], "xxxxxxxxxx");
    }
}

/// Test 10: Module with operations having no inputs or outputs
#[test]
fn test_operations_without_io() {
    let mut module = Module::new("stateless_module");
    
    // Add multiple operations with no explicit inputs/outputs
    for i in 0..5 {
        let op = Operation::new(&format!("noop_op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
    }
}