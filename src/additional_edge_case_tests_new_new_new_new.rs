//! Additional edge case tests for the Impulse compiler IR structures
//! This module provides comprehensive testing for boundary conditions and edge cases
//! using both standard assert macros and rstest for parameterized testing

use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
use std::collections::HashMap;
use rstest::rstest;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test creating module with maximum potential allocations
    #[test]
    fn test_module_with_extremely_large_allocation() {
        // Create a module name that is very long
        let long_name = "module_".repeat(10_000) + "end";
        let module = Module::new(long_name);
        assert!(module.name.len() > 60_000);
        assert!(module.operations.is_empty());
    }

    /// Test handling of usize::MAX dimensions in tensor shapes (won't actually multiply to overflow)
    #[test]
    fn test_tensor_with_max_usize_dimensions() {
        // We can't actually reach usize::MAX without causing real overflow,
        // but test the largest reasonable values that might approach limits
        let large_size = std::cmp::min(1_000_000, usize::MAX / 100);
        let value = Value {
            name: "max_tensor".to_string(),
            ty: Type::F32,
            shape: vec![large_size, large_size],
        };
        
        // This should not crash or panic
        match value.num_elements() {
            Some(elements) => {
                // Elements might be None if it overflows, or Some if it fits
                assert!(elements >= large_size); // At minimum, should be at least one dimension
            },
            None => {
                // This is acceptable if it overflows
                println!("Overflow correctly handled");
            }
        }
    }

    /// Test operations with invalid shapes (negative-like representation as usize)
    #[test]
    fn test_invalid_tensor_shapes_handling() {
        // Although we only accept positive integers, test edge behaviors
        let valid_small_shape = Value {
            name: "small_tensor".to_string(),
            ty: Type::F32,
            shape: vec![0, 1, 2, 3],  // Contains zero which should result in 0 elements
        };
        
        assert_eq!(valid_small_shape.num_elements(), Some(0));
        
        // Another case with multiple zeros
        let multi_zero = Value {
            name: "zero_multi".to_string(),
            ty: Type::I64,
            shape: vec![5, 0, 100, 0],
        };
        
        assert_eq!(multi_zero.num_elements(), Some(0));
    }

    /// Test deeply nested structures that could cause stack overflow
    #[test]
    fn test_extreme_deep_nesting_protection() {
        let mut current_type = Type::F32;
        
        // Create many levels of nesting to test recursion limits
        for i in 0..1000 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 10 + 1],  // Small, non-zero dimension to prevent immediate 0
            };
        }
        
        // Verify the nested type is still valid
        assert!(current_type.is_valid_type());
        
        // Test cloning of deeply nested structure
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test creation of operations with maximum possible attributes using HashMap
    #[test]
    fn test_operation_max_hashmap_entries() {
        let mut op = Operation::new("max_attr_op");
        let mut attributes = HashMap::new();
        
        // Add a huge number of attributes to test HashMap limits
        for i in 0..100_000 {
            attributes.insert(
                format!("key_{}", i),
                Attribute::String(format!("value_{}_{}", i, "x".repeat(10)))
            );
        }
        
        op.attributes = attributes;
        assert_eq!(op.attributes.len(), 100_000);
        
        // Verify we can access some random values
        assert!(op.attributes.contains_key("key_1"));
        assert!(op.attributes.contains_key("key_50000"));
        assert!(op.attributes.contains_key("key_99999"));
    }

    /// Test handling of floating point attributes with extreme values
    #[test]
    fn test_extreme_float_attribute_values() {
        let extreme_attrs = [
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(f64::NAN),
            Attribute::Float(f64::MIN),
            Attribute::Float(f64::MAX),
            Attribute::Float(std::f64::EPSILON),
            Attribute::Float(0.0),
        ];
        
        // All should be constructible
        assert_eq!(extreme_attrs.len(), 7);
        
        // Test comparison behavior for non-NaN values
        match extreme_attrs[0] {  // INFINITY
            Attribute::Float(f) if f.is_infinite() && f.is_sign_positive() => (),
            _ => panic!("Expected positive infinity"),
        }
        
        match extreme_attrs[1] {  // NEG_INFINITY
            Attribute::Float(f) if f.is_infinite() && f.is_sign_negative() => (),
            _ => panic!("Expected negative infinity"),
        }
    }

    /// Test recursive type equality with different nesting depths
    #[rstest]
    #[case(
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![1] },
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![1] },
        true
    )]
    #[case(
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2] },
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![1] },
        false
    )]
    #[case(
        Type::Tensor { element_type: Box::new(Type::I32), shape: vec![1] },
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![1] },
        false
    )]
    fn test_tensor_equality_parametrized(
        #[case] type1: Type,
        #[case] type2: Type,
        #[case] expected_equal: bool
    ) {
        assert_eq!(type1 == type2, expected_equal);
    }

    /// Test very large but valid tensor shapes
    #[test]
    fn test_extremely_large_but_valid_tensors() {
        let huge_value = Value {
            name: "huge_but_valid".to_string(),
            ty: Type::I32,
            shape: vec![1_000_000, 1_000],  // 1 billion elements
        };
        
        match huge_value.num_elements() {
            Some(num) => assert_eq!(num, 1_000_000_000),
            None => panic!("Unexpected overflow for valid large tensor")
        }
        
        // Test with many dimensions
        let multi_dim = Value {
            name: "multi_dim_huge".to_string(),
            ty: Type::F64,
            shape: vec![100, 100, 100, 100],  // 100M elements
        };
        
        match multi_dim.num_elements() {
            Some(num) => assert_eq!(num, 100_000_000),
            None => panic!("Unexpected overflow for valid multi-dim tensor")
        }
    }

    /// Test recursive type validation with complex nested structures
    #[test]
    fn test_complex_nested_validation() {
        // Create a complex nested structure: tensor<tensor<tensor<tensor<i32, [2]>, [3]>, [4]>, [5]>
        let level1 = Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![2],
        };
        assert!(level1.is_valid_type()); // Test first level right away before moving
        
        let level2 = Type::Tensor {
            element_type: Box::new(level1.clone()), // Clone to prevent moving
            shape: vec![3],
        };
        assert!(level2.is_valid_type()); // Test before moving
        
        let level3 = Type::Tensor {
            element_type: Box::new(level2.clone()), // Clone to prevent moving
            shape: vec![4],
        };
        assert!(level3.is_valid_type()); // Test before moving
        
        let level4 = Type::Tensor {
            element_type: Box::new(level3.clone()), // Clone to prevent moving
            shape: vec![5],
        };
        
        assert!(level4.is_valid_type());
        
        // Test deeply nested equality
        let level4_clone = level4.clone();
        assert_eq!(level4, level4_clone);
    }

    /// Test integer attribute boundaries (i64 min/max)
    #[rstest]
    #[case(i64::MAX, true)]
    #[case(i64::MIN, true)]
    #[case(0i64, true)]
    #[case(-1i64, true)]
    #[case(1i64, true)]
    fn test_integer_attribute_boundaries(#[case] value: i64, #[case] _should_be_valid: bool) {
        let attr = Attribute::Int(value);
        match attr {
            Attribute::Int(retrieved_value) => {
                assert_eq!(retrieved_value, value);
            },
            _ => panic!("Expected Int attribute")
        }
    }
}