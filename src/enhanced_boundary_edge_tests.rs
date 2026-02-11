//! Enhanced boundary and edge case tests for Impulse compiler
//! Tests covering numerical precision, overflow scenarios, and edge conditions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::{gcd, lcm, round_up_to_multiple, next_power_of_2};

/// Test 1: Module with extremely large dimension values that are just below overflow threshold
#[test]
fn test_module_with_near_overflow_dimensions() {
    let mut module = Module::new("near_overflow");
    
    // Use dimensions that are large but safe (product won't overflow)
    let large_dim = 46340; // sqrt(2^31) - safe for i32
    let value = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![large_dim, large_dim],
    };
    
    let mut op = Operation::new("test_op");
    op.inputs.push(value);
    module.add_operation(op);
    
    assert_eq!(module.operations[0].inputs[0].shape, vec![46340, 46340]);
    assert_eq!(module.operations[0].inputs[0].num_elements(), Some(46340 * 46340));
}

/// Test 2: GCD function edge cases with extreme values
#[test]
fn test_gcd_extreme_values() {
    // Test with prime numbers
    assert_eq!(gcd(2_147_483_647, 2_147_483_647), 2_147_483_647); // i32::MAX (as usize on 64-bit)
    
    // Test with consecutive numbers (always coprime)
    assert_eq!(gcd(999_999, 1_000_000), 1);
    
    // Test with power of 2
    assert_eq!(gcd(1_048_576, 2_097_152), 1_048_576); // 2^20 and 2^21
    
    // Test with zero
    assert_eq!(gcd(0, usize::MAX), usize::MAX);
    assert_eq!(gcd(usize::MAX, 0), usize::MAX);
}

/// Test 3: LCM function with potential overflow scenarios
#[test]
fn test_lcm_edge_cases() {
    // Test with same number
    assert_eq!(lcm(42, 42), 42);
    
    // Test with coprime numbers
    assert_eq!(lcm(17, 19), 323);
    
    // Test where one is multiple of other
    assert_eq!(lcm(7, 21), 21);
    assert_eq!(lcm(21, 7), 21);
    
    // Test with 1
    assert_eq!(lcm(1, 999_999), 999_999);
    assert_eq!(lcm(999_999, 1), 999_999);
}

/// Test 4: next_power_of_2 with power-of-2 boundaries
#[test]
fn test_next_power_of_2_boundaries() {
    // Test exactly at powers of 2
    assert_eq!(next_power_of_2(1024), 1024);
    assert_eq!(next_power_of_2(2048), 2048);
    assert_eq!(next_power_of_2(4096), 4096);
    
    // Test one less than powers of 2
    assert_eq!(next_power_of_2(1023), 1024);
    assert_eq!(next_power_of_2(2047), 2048);
    assert_eq!(next_power_of_2(4095), 4096);
    
    // Test one more than powers of 2
    assert_eq!(next_power_of_2(1025), 2048);
    assert_eq!(next_power_of_2(2049), 4096);
    assert_eq!(next_power_of_2(4097), 8192);
}

/// Test 5: round_up_to_multiple with edge cases
#[test]
fn test_round_up_to_multiple_edge_cases() {
    // Test with multiple equals to value
    assert_eq!(round_up_to_multiple(100, 100), 100);
    
    // Test with value is exact multiple
    assert_eq!(round_up_to_multiple(200, 50), 200);
    
    // Test with value slightly less than next multiple
    assert_eq!(round_up_to_multiple(99, 100), 100);
    assert_eq!(round_up_to_multiple(149, 50), 150);
    
    // Test with multiple = 1 (should return value)
    assert_eq!(round_up_to_multiple(12345, 1), 12345);
}

/// Test 6: Value shape with alternating dimension patterns
#[test]
fn test_value_alternating_dimensions() {
    // Test shapes with alternating small/large dimensions
    let alternating = Value {
        name: "alternating".to_string(),
        ty: Type::F32,
        shape: vec![1, 1000, 1, 1000, 1], // Product = 1,000,000
    };
    
    assert_eq!(alternating.shape.len(), 5);
    assert_eq!(alternating.num_elements(), Some(1_000_000));
    
    // Verify specific dimension values
    assert_eq!(alternating.shape[0], 1);
    assert_eq!(alternating.shape[1], 1000);
    assert_eq!(alternating.shape[2], 1);
    assert_eq!(alternating.shape[3], 1000);
    assert_eq!(alternating.shape[4], 1);
}

/// Test 7: Module with all integer type operations
#[test]
fn test_module_all_integer_types() {
    let mut module = Module::new("int_types");
    
    // Test all integer types with various bit widths
    let int_types = vec![
        Type::I32,
        Type::I64,
    ];
    
    for (i, ty) in int_types.into_iter().enumerate() {
        let mut op = Operation::new(&format!("int_op_{}", i));
        op.inputs.push(Value {
            name: format!("int_input_{}", i),
            ty: ty.clone(),
            shape: vec![5, 5],
        });
        op.outputs.push(Value {
            name: format!("int_output_{}", i),
            ty: ty,
            shape: vec![5, 5],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].inputs[0].ty, Type::I32);
    assert_eq!(module.operations[1].inputs[0].ty, Type::I64);
}

/// Test 8: Attribute with special float values (subnormal, denormalized)
#[test]
fn test_attribute_special_float_values() {
    // Test subnormal float (very small positive)
    let subnormal = Attribute::Float(f64::MIN_POSITIVE);
    
    // Test infinity values
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    
    // Test NaN
    let nan_attr = Attribute::Float(f64::NAN);
    
    // Verify attributes are created (comparing with self should work for infinity)
    assert_eq!(pos_inf, pos_inf);
    assert_eq!(neg_inf, neg_inf);
    
    // NaN is never equal to itself
    assert_ne!(nan_attr, nan_attr);
    
    // Subnormal is just a very small positive number
    match subnormal {
        Attribute::Float(val) => {
            assert!(val > 0.0);
            assert!(val < f64::MIN_POSITIVE * 2.0);
        }
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 9: Module with operations that have empty attributes map
#[test]
fn test_module_operations_empty_attributes() {
    let mut module = Module::new("empty_attrs");
    
    // Add operations with empty attributes
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_no_attrs_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![2, 2],
        });
        // attributes remain empty (HashMap default)
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    
    // Verify all operations have empty attributes
    for op in &module.operations {
        assert!(op.attributes.is_empty());
        assert_eq!(op.attributes.len(), 0);
    }
}

/// Test 10: Value with shape containing consecutive repeated dimensions
#[test]
fn test_value_consecutive_repeated_dimensions() {
    // Test shapes with repeated dimension values
    let repeated = Value {
        name: "repeated_dims".to_string(),
        ty: Type::F64,
        shape: vec![3, 3, 3, 3, 3], // Five 3s: 3^5 = 243
    };
    
    assert_eq!(repeated.shape.len(), 5);
    assert_eq!(repeated.num_elements(), Some(243));
    
    // All dimensions should be equal
    for dim in &repeated.shape {
        assert_eq!(*dim, 3);
    }
    
    // Test another repeated pattern
    let repeated2 = Value {
        name: "repeated_dims_2".to_string(),
        ty: Type::I32,
        shape: vec![10, 10, 10], // Three 10s: 10^3 = 1000
    };
    
    assert_eq!(repeated2.shape, vec![10, 10, 10]);
    assert_eq!(repeated2.num_elements(), Some(1000));
}