//! Advanced edge and boundary tests for the Impulse compiler
//! Covers numerical precision, overflow prevention, and edge conditions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::utils::{math_utils, ir_utils};

/// Test 1: Math utils with extreme values - GCD with zero and identity
#[test]
fn test_gcd_edge_cases() {
    // GCD with zero
    assert_eq!(math_utils::gcd(0, 0), 0);
    assert_eq!(math_utils::gcd(0, 5), 5);
    assert_eq!(math_utils::gcd(5, 0), 5);

    // GCD with large primes
    assert_eq!(math_utils::gcd(2_147_483_647, 2_147_483_647), 2_147_483_647);
    assert_eq!(math_utils::gcd(2_147_483_647, 1), 1);

    // GCD of consecutive Fibonacci numbers (should be 1)
    assert_eq!(math_utils::gcd(832040, 514229), 1);
}

/// Test 2: Math utils with extreme values - LCM with zero and overflow handling
#[test]
fn test_lcm_edge_cases() {
    // LCM with zero
    assert_eq!(math_utils::lcm(0, 0), 0);
    assert_eq!(math_utils::lcm(0, 5), 0);
    assert_eq!(math_utils::lcm(5, 0), 0);

    // LCM of co-prime numbers
    assert_eq!(math_utils::lcm(17, 19), 323);
    assert_eq!(math_utils::lcm(7, 13), 91);

    // LCM where a divides b
    assert_eq!(math_utils::lcm(8, 16), 16);
    assert_eq!(math_utils::lcm(1, 1000), 1000);
}

/// Test 3: Next power of 2 with large values
#[test]
fn test_next_power_of_2_large_values() {
    // Test with values near powers of 2
    assert_eq!(math_utils::next_power_of_2(1 << 15), 1 << 15);
    assert_eq!(math_utils::next_power_of_2((1 << 15) + 1), 1 << 16);

    // Test with very large values
    assert_eq!(math_utils::next_power_of_2(1_000_000_000), 1_073_741_824);
    assert_eq!(math_utils::next_power_of_2(2_147_483_647), 2_147_483_648);
}

/// Test 4: Round up to multiple with boundary conditions
#[test]
fn test_round_up_to_multiple_edge_cases() {
    // Round up when already aligned
    assert_eq!(math_utils::round_up_to_multiple(0, 16), 0);
    assert_eq!(math_utils::round_up_to_multiple(32, 16), 32);

    // Round up with small multiples
    assert_eq!(math_utils::round_up_to_multiple(7, 8), 8);
    assert_eq!(math_utils::round_up_to_multiple(15, 16), 16);

    // Round up with very large multiples
    assert_eq!(math_utils::round_up_to_multiple(1_000_000, 64), 1_000_000);
    // 1_000_001 % 64 = 33, so round up to 1_000_001 + (64 - 33) = 1_000_032
    // But actual result is 1_000_064 because 1_000_001 / 64 = 15625 remainder 1
    // So round_up_to_multiple(1_000_001, 64) = 1_000_001 + (64 - 1) = 1_000_064
    assert_eq!(math_utils::round_up_to_multiple(1_000_001, 64), 1_000_064);
}

/// Test 5: Value with extreme shape combinations
#[test]
fn test_value_extreme_shape_combinations() {
    // Scalar (empty shape)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));

    // Very long shape with ones
    let long_shape = vec![1; 100];
    let long_one = Value {
        name: "long_one".to_string(),
        ty: Type::F32,
        shape: long_shape,
    };
    assert_eq!(long_one.num_elements(), Some(1));

    // Shape with alternating ones and twos
    let alternating = vec![1, 2, 1, 2, 1, 2];
    let alternating_value = Value {
        name: "alternating".to_string(),
        ty: Type::F32,
        shape: alternating,
    };
    assert_eq!(alternating_value.num_elements(), Some(8));
}

/// Test 6: IR utils - calculate tensor size with edge cases
#[test]
fn test_calculate_tensor_size_edge_cases() {
    // Scalar F32
    let scalar_size = ir_utils::calculate_tensor_size(&Type::F32, &[]).unwrap();
    assert_eq!(scalar_size, 4);

    // Empty tensor (contains zero dimension)
    let empty_size = ir_utils::calculate_tensor_size(&Type::F32, &[0, 10]).unwrap();
    assert_eq!(empty_size, 0);

    // Very large tensor that might overflow
    let overflow_test = ir_utils::calculate_tensor_size(&Type::F32, &[usize::MAX / 4, 5]);
    assert!(overflow_test.is_err());

    // Nested tensor with empty inner shape
    let nested_empty = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![0],
    };
    let nested_size = ir_utils::calculate_tensor_size(&nested_empty, &[2]).unwrap();
    assert_eq!(nested_size, 0);
}

/// Test 7: Type validation with nested structures
#[test]
fn test_nested_type_validation() {
    // Deeply nested tensor types
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let level2 = Type::Tensor {
        element_type: Box::new(level1.clone()),
        shape: vec![3],
    };
    let level3 = Type::Tensor {
        element_type: Box::new(level2.clone()),
        shape: vec![4],
    };

    assert!(level1.is_valid_type());
    assert!(level2.is_valid_type());
    assert!(level3.is_valid_type());

    // Extract element type from deep nesting
    assert_eq!(ir_utils::get_element_type(&level3), &Type::F32);
}

/// Test 8: Operation with attribute array containing mixed types
#[test]
fn test_operation_mixed_attribute_array() {
    let mut op = Operation::new("mixed_attrs");
    op.attributes.insert("mixed_array".to_string(), Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]));

    assert_eq!(op.attributes.len(), 1);
    match op.attributes.get("mixed_array") {
        Some(Attribute::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!(matches!(arr[0], Attribute::Int(42)));
            assert!(matches!(arr[1], Attribute::Float(_)));
            assert!(matches!(arr[2], Attribute::String(_)));
            assert!(matches!(arr[3], Attribute::Bool(true)));
            assert!(matches!(arr[4], Attribute::Array(_)));
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 9: Module with operations having identical attributes
#[test]
fn test_module_operations_with_identical_attrs() {
    let mut module = Module::new("identical_attrs");

    for i in 0..3 {
        let mut op = Operation::new("same_type");
        op.attributes.insert("id".to_string(), Attribute::Int(i));
        op.attributes.insert("value".to_string(), Attribute::Float(1.0));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 3);
    // All operations should have the same attributes except id
    for op in &module.operations {
        assert_eq!(op.attributes.len(), 2);
        assert!(op.attributes.contains_key("id"));
        assert!(op.attributes.contains_key("value"));
    }
}

/// Test 10: IR utils - get_num_elements with zero dimensions
#[test]
fn test_get_num_elements_zero_dimensions() {
    // Tensor with zero in the middle
    let zero_middle = Value {
        name: "zero_middle".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(ir_utils::get_num_elements(&zero_middle), Some(0));

    // Tensor with zero at the beginning
    let zero_start = Value {
        name: "zero_start".to_string(),
        ty: Type::F32,
        shape: vec![0, 100, 100],
    };
    assert_eq!(ir_utils::get_num_elements(&zero_start), Some(0));

    // Tensor with zero at the end
    let zero_end = Value {
        name: "zero_end".to_string(),
        ty: Type::F32,
        shape: vec![100, 100, 0],
    };
    assert_eq!(ir_utils::get_num_elements(&zero_end), Some(0));
}