//! Essential boundary tests - core edge cases for IR components
//! Tests focus on critical boundaries: overflow detection, NaN handling, extreme values

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value num_elements() overflow detection with checked_mul
#[test]
fn test_value_num_elements_overflow_detection() {
    // Create a value that would cause overflow in shape product calculation
    // Use dimensions that would overflow when multiplied
    let overflow_value = Value {
        name: "overflow_tensor".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2], // Would overflow on multiplication
    };
    
    // num_elements() should return None when overflow occurs
    let result = overflow_value.num_elements();
    assert_eq!(result, None, "Expected None for overflow case");
}

/// Test 2: NaN and Infinity float attribute handling
#[test]
fn test_nan_infinity_float_attributes() {
    // Test NaN values (different representations)
    let nan_attr = Attribute::Float(f64::NAN);
    let neg_nan_attr = Attribute::Float(-f64::NAN);
    
    // NaN is not equal to itself
    if let Attribute::Float(val) = nan_attr {
        assert!(val.is_nan());
    }
    
    if let Attribute::Float(val) = neg_nan_attr {
        assert!(val.is_nan());
    }
    
    // Test positive infinity
    let pos_inf_attr = Attribute::Float(f64::INFINITY);
    if let Attribute::Float(val) = pos_inf_attr {
        assert!(val.is_infinite() && val.is_sign_positive());
    }
    
    // Test negative infinity
    let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
    if let Attribute::Float(val) = neg_inf_attr {
        assert!(val.is_infinite() && val.is_sign_negative());
    }
    
    // Test very large finite float
    let max_attr = Attribute::Float(f64::MAX);
    if let Attribute::Float(val) = max_attr {
        assert!(val.is_finite() && val > 1e300);
    }
}

/// Test 3: Module with empty string name validation
#[test]
fn test_module_empty_string_name() {
    let module = Module::new("");
    assert_eq!(module.name, "");
    assert!(module.name.is_empty());
    assert!(module.operations.is_empty());
    assert!(module.inputs.is_empty());
    assert!(module.outputs.is_empty());
}

/// Test 4: Value with single usize::MAX dimension (edge of valid range)
#[test]
fn test_value_with_max_single_dimension() {
    let value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::I32,
        shape: vec![usize::MAX], // Single dimension at max
    };
    
    assert_eq!(value.shape.len(), 1);
    assert_eq!(value.shape[0], usize::MAX);
    
    // num_elements should return the value itself for single dimension
    let result = value.num_elements();
    assert_eq!(result, Some(usize::MAX));
}

/// Test 5: Operation with attribute keys being empty strings
#[test]
fn test_operation_empty_attribute_keys() {
    let mut op = Operation::new("test_op");
    op.attributes.insert("".to_string(), Attribute::Int(42));
    op.attributes.insert("normal_key".to_string(), Attribute::String("value".to_string()));
    
    assert_eq!(op.attributes.len(), 2);
    assert!(op.attributes.contains_key(""));
    assert_eq!(op.attributes.get(""), Some(&Attribute::Int(42)));
}

/// Test 6: Module inputs/outputs with identical names (should allow)
#[test]
fn test_module_duplicate_io_names() {
    let mut module = Module::new("duplicate_io_test");
    
    // Add input and output with same name
    let shared_value = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    module.inputs.push(shared_value.clone());
    module.outputs.push(shared_value);
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, "shared");
    assert_eq!(module.outputs[0].name, "shared");
}

/// Test 7: Attribute array with all zeros
#[test]
fn test_attribute_array_all_zeros() {
    let zero_array = Attribute::Array(vec![
        Attribute::Int(0),
        Attribute::Float(0.0),
        Attribute::String("".to_string()),
        Attribute::Bool(false),
    ]);
    
    match zero_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Attribute::Int(0));
            assert_eq!(arr[1], Attribute::Float(0.0));
            assert_eq!(arr[2], Attribute::String("".to_string()));
            assert_eq!(arr[3], Attribute::Bool(false));
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 8: Value with all dimensions equal to 1
#[test]
fn test_value_all_ones_dimensions() {
    let value = Value {
        name: "all_ones_tensor".to_string(),
        ty: Type::F64,
        shape: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1], // 10 dimensions, all 1
    };
    
    assert_eq!(value.shape.len(), 10);
    assert!(value.shape.iter().all(|&dim| dim == 1));
    
    // Product of all ones should be 1
    let result = value.num_elements();
    assert_eq!(result, Some(1));
}

/// Test 9: Integer attribute with extreme boundary values
#[test]
fn test_integer_extreme_boundary_values() {
    // Test i64 boundary values
    let max_int = Attribute::Int(i64::MAX);
    let min_int = Attribute::Int(i64::MIN);
    let zero_int = Attribute::Int(0);
    let neg_one = Attribute::Int(-1);
    let pos_one = Attribute::Int(1);
    
    // Verify all can be created and matched
    match max_int {
        Attribute::Int(val) => assert_eq!(val, i64::MAX),
        _ => panic!("Expected Int"),
    }
    
    match min_int {
        Attribute::Int(val) => assert_eq!(val, i64::MIN),
        _ => panic!("Expected Int"),
    }
    
    match zero_int {
        Attribute::Int(val) => assert_eq!(val, 0),
        _ => panic!("Expected Int"),
    }
    
    match neg_one {
        Attribute::Int(val) => assert_eq!(val, -1),
        _ => panic!("Expected Int"),
    }
    
    match pos_one {
        Attribute::Int(val) => assert_eq!(val, 1),
        _ => panic!("Expected Int"),
    }
}

/// Test 10: Nested module with operations having no inputs and no outputs
#[test]
fn test_module_operations_with_no_io() {
    let mut module = Module::new("no_io_module");
    
    // Add operations with no inputs or outputs
    for i in 0..5 {
        let op = Operation::new(&format!("no_io_op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    // Verify all operations have no inputs or outputs
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
    }
}