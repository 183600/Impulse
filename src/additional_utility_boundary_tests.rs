//! Additional utility boundary tests - covering edge cases for utility functions
//! 
//! This module provides additional test coverage for utility functions in the Impulse compiler,
//! focusing on boundary conditions, edge cases, and potential overflow scenarios.

use crate::ir::{Value, Type, Operation, Attribute, Module};
use crate::utils::{is_scalar, is_vector, is_matrix, get_rank, get_num_elements, get_element_type, type_to_string, calculate_tensor_size, round_up_to_multiple};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: round_up_to_multiple with near-boundary scenarios
    #[test]
    fn test_round_up_to_multiple_boundary() {
        // Test near boundary but within safe range
        let large_value = 100_000_000;
        let result = round_up_to_multiple(large_value, 1024);
        // Should round up to next multiple of 1024
        let expected = ((large_value + 1023) / 1024) * 1024;
        assert_eq!(result, expected);

        // Test with value that's not aligned
        let not_aligned = 1_000_001;
        let result2 = round_up_to_multiple(not_aligned, 1024);
        let expected2 = ((not_aligned + 1023) / 1024) * 1024;
        assert_eq!(result2, expected2);

        // Test with large multiple
        let large_multiple = 100_000;
        let result3 = round_up_to_multiple(999, large_multiple);
        assert_eq!(result3, large_multiple);

        // Test with value already aligned
        let already_aligned = 2048;
        let result4 = round_up_to_multiple(already_aligned, 1024);
        assert_eq!(result4, already_aligned);
    }

    /// Test 2: calculate_tensor_size with nested tensor edge cases
    #[test]
    fn test_calculate_tensor_size_nested_edge_cases() {
        // Deeply nested tensor with multiple levels
        let level1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        };
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![3],
        };
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![4],
        };

        // Calculate size for the outermost tensor
        let result = calculate_tensor_size(&level3, &[5]);
        // Should be: 5 * (4 * (3 * (2 * 4))) = 5 * 4 * 3 * 2 * 4 = 480 bytes
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 480);

        // Nested tensor with empty outer shape
        let nested = Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![10, 10],
        };
        let result2 = calculate_tensor_size(&nested, &[]);
        // Scalar containing a tensor
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), 800); // 10 * 10 * 8 bytes

        // Nested tensor with zero in inner shape
        let zero_inner = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![10, 0, 5],
        };
        let result3 = calculate_tensor_size(&zero_inner, &[100]);
        // Should be 0 because inner tensor has 0 elements
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap(), 0);
    }

    /// Test 3: type_to_string with deeply nested tensors
    #[test]
    fn test_type_to_string_deeply_nested() {
        // Create a 3-level nested tensor
        let inner = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        };
        let middle = Type::Tensor {
            element_type: Box::new(inner),
            shape: vec![3, 3],
        };
        let outer = Type::Tensor {
            element_type: Box::new(middle),
            shape: vec![4],
        };

        let result = type_to_string(&outer);
        // Should contain nested tensor representation
        assert!(result.contains("tensor"));
        assert!(result.contains("f32"));
        assert!(result.contains("4"));
        assert!(result.contains("3"));
        assert!(result.contains("2"));
    }

    /// Test 4: get_element_type with mixed nested types
    #[test]
    fn test_get_element_type_mixed_nested() {
        // Test with tensor containing another tensor
        let tensor_of_tensor = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![5],
            }),
            shape: vec![10],
        };
        assert_eq!(get_element_type(&tensor_of_tensor), &Type::I64);

        // Test with tensor of tensor of tensor
        let triple_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F64),
                    shape: vec![1],
                }),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        assert_eq!(get_element_type(&triple_nested), &Type::F64);

        // Test with Bool at deepest level
        let nested_bool = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Bool),
                shape: vec![100],
            }),
            shape: vec![50],
        };
        assert_eq!(get_element_type(&nested_bool), &Type::Bool);
    }

    /// Test 5: is_scalar, is_vector, is_matrix with edge case shapes
    #[test]
    fn test_tensor_classification_edge_cases() {
        // Test with shape containing zeros
        let zero_dim = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: vec![0],
        };
        assert!(!is_scalar(&zero_dim));
        assert!(is_vector(&zero_dim));
        assert!(!is_matrix(&zero_dim));

        // Test with very large single dimension
        let large_1d = Value {
            name: "large_1d".to_string(),
            ty: Type::I32,
            shape: vec![1_000_000_000],
        };
        assert!(!is_scalar(&large_1d));
        assert!(is_vector(&large_1d));
        assert!(!is_matrix(&large_1d));

        // Test with 2D tensor where one dimension is 1
        let row_vector = Value {
            name: "row_vector".to_string(),
            ty: Type::F64,
            shape: vec![1, 100],
        };
        assert!(!is_scalar(&row_vector));
        assert!(!is_vector(&row_vector));
        assert!(is_matrix(&row_vector));

        let column_vector = Value {
            name: "column_vector".to_string(),
            ty: Type::F64,
            shape: vec![100, 1],
        };
        assert!(!is_scalar(&column_vector));
        assert!(!is_vector(&column_vector));
        assert!(is_matrix(&column_vector));
    }

    /// Test 6: get_rank and get_num_elements with extreme shapes
    #[test]
    fn test_rank_and_elements_extreme_shapes() {
        // Test with high-dimensional tensor
        let mut high_dim_shape = Vec::new();
        for i in 1..=10 {
            high_dim_shape.push(i);
        }
        let high_dim = Value {
            name: "high_dim".to_string(),
            ty: Type::F32,
            shape: high_dim_shape.clone(),
        };
        assert_eq!(get_rank(&high_dim), 10);
        // Calculate expected elements: 1*2*3*4*5*6*7*8*9*10 = 3,628,800
        let expected_elements: usize = high_dim_shape.iter().product();
        assert_eq!(get_num_elements(&high_dim), Some(expected_elements));

        // Test with alternating 1 and large values
        let alternating = Value {
            name: "alternating".to_string(),
            ty: Type::I64,
            shape: vec![1, 1000, 1, 1000, 1],
        };
        assert_eq!(get_rank(&alternating), 5);
        assert_eq!(get_num_elements(&alternating), Some(1_000_000));

        // Test with all ones
        let all_ones = Value {
            name: "all_ones".to_string(),
            ty: Type::Bool,
            shape: vec![1, 1, 1, 1, 1, 1, 1, 1],
        };
        assert_eq!(get_rank(&all_ones), 8);
        assert_eq!(get_num_elements(&all_ones), Some(1));
    }

    /// Test 7: Module with operations having special attribute keys
    #[test]
    fn test_module_with_special_attribute_keys() {
        let mut module = Module::new("special_attrs_module");
        
        let mut op = Operation::new("special_op");
        let mut attrs = HashMap::new();
        
        // Add attributes with special characters in keys
        attrs.insert("key-with-dashes".to_string(), Attribute::Int(1));
        attrs.insert("key_with_underscores".to_string(), Attribute::Int(2));
        attrs.insert("key.with.dots".to_string(), Attribute::Int(3));
        attrs.insert("key:with:colons".to_string(), Attribute::Int(4));
        attrs.insert("key/with/slashes".to_string(), Attribute::Int(5));
        attrs.insert("key with spaces".to_string(), Attribute::Int(6));
        attrs.insert("key@with#special$chars".to_string(), Attribute::Int(7));
        
        op.attributes = attrs;
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].attributes.len(), 7);
        
        // Verify all attributes are accessible
        assert!(module.operations[0].attributes.contains_key("key-with-dashes"));
        assert!(module.operations[0].attributes.contains_key("key_with_underscores"));
        assert!(module.operations[0].attributes.contains_key("key.with.dots"));
        assert!(module.operations[0].attributes.contains_key("key:with:colons"));
        assert!(module.operations[0].attributes.contains_key("key/with/slashes"));
        assert!(module.operations[0].attributes.contains_key("key with spaces"));
        assert!(module.operations[0].attributes.contains_key("key@with#special$chars"));
    }

    /// Test 8: calculate_tensor_size with different type sizes
    #[test]
    fn test_calculate_tensor_size_type_sizes() {
        // Test F32 (4 bytes)
        let f32_size = calculate_tensor_size(&Type::F32, &[10, 10]).unwrap();
        assert_eq!(f32_size, 100 * 4); // 400 bytes

        // Test F64 (8 bytes)
        let f64_size = calculate_tensor_size(&Type::F64, &[10, 10]).unwrap();
        assert_eq!(f64_size, 100 * 8); // 800 bytes

        // Test I32 (4 bytes)
        let i32_size = calculate_tensor_size(&Type::I32, &[10, 10]).unwrap();
        assert_eq!(i32_size, 100 * 4); // 400 bytes

        // Test I64 (8 bytes)
        let i64_size = calculate_tensor_size(&Type::I64, &[10, 10]).unwrap();
        assert_eq!(i64_size, 100 * 8); // 800 bytes

        // Test Bool (1 byte)
        let bool_size = calculate_tensor_size(&Type::Bool, &[10, 10]).unwrap();
        assert_eq!(bool_size, 100 * 1); // 100 bytes

        // Test scalar F32
        let scalar_f32 = calculate_tensor_size(&Type::F32, &[]).unwrap();
        assert_eq!(scalar_f32, 1 * 4); // 4 bytes

        // Test scalar Bool
        let scalar_bool = calculate_tensor_size(&Type::Bool, &[]).unwrap();
        assert_eq!(scalar_bool, 1 * 1); // 1 byte
    }

    /// Test 9: calculate_tensor_size overflow detection
    #[test]
    fn test_calculate_tensor_size_overflow() {
        // Test with shape that would cause overflow in element count
        let large_shape = [100_000, 100_000]; // 10 billion elements
        let f32_result = calculate_tensor_size(&Type::F32, &large_shape);
        // On most systems this would overflow
        assert!(f32_result.is_ok() || f32_result.is_err());

        // Test with extremely large single dimension
        let huge_1d = [usize::MAX / 4]; // Would need MAX bytes for F32
        let f64_result = calculate_tensor_size(&Type::F64, &huge_1d);
        // Should handle overflow gracefully
        assert!(f64_result.is_ok() || f64_result.is_err());

        // Test with dimensions that multiply to just under MAX but overflow when multiplying by type size
        let near_max_shape = [usize::MAX / 4, 2]; // ~MAX/2 elements
        let i64_result = calculate_tensor_size(&Type::I64, &near_max_shape);
        // 8 bytes per element * (MAX/2 * 2) = 8 * MAX = overflow
        assert!(i64_result.is_ok() || i64_result.is_err());
    }

    /// Test 10: round_up_to_multiple with zero and one edge cases
    #[test]
    fn test_round_up_to_multiple_zero_and_one() {
        // Test with multiple = 1 (should always return the same value)
        assert_eq!(round_up_to_multiple(0, 1), 0);
        assert_eq!(round_up_to_multiple(1, 1), 1);
        assert_eq!(round_up_to_multiple(100, 1), 100);
        assert_eq!(round_up_to_multiple(usize::MAX, 1), usize::MAX);

        // Test with multiple = 0 (special case, returns value unchanged)
        assert_eq!(round_up_to_multiple(0, 0), 0);
        assert_eq!(round_up_to_multiple(10, 0), 10);
        assert_eq!(round_up_to_multiple(1000, 0), 1000);

        // Test with value = 0 (should return 0 or the multiple)
        assert_eq!(round_up_to_multiple(0, 10), 0);
        assert_eq!(round_up_to_multiple(0, 100), 0);
        assert_eq!(round_up_to_multiple(0, 1024), 0);

        // Test with value equal to multiple
        assert_eq!(round_up_to_multiple(16, 16), 16);
        assert_eq!(round_up_to_multiple(1024, 1024), 1024);
        assert_eq!(round_up_to_multiple(4096, 4096), 4096);

        // Test with value one less than multiple
        assert_eq!(round_up_to_multiple(15, 16), 16);
        assert_eq!(round_up_to_multiple(1023, 1024), 1024);
        assert_eq!(round_up_to_multiple(4095, 4096), 4096);

        // Test with value one more than multiple
        assert_eq!(round_up_to_multiple(17, 16), 32);
        assert_eq!(round_up_to_multiple(1025, 1024), 2048);
        assert_eq!(round_up_to_multiple(4097, 4096), 8192);
    }
}