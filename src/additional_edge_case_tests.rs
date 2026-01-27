//! Additional edge case tests for the Impulse compiler IR structures
//! This module provides comprehensive testing for boundary conditions and edge cases

use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Test creating a module with maximum possible name length
    #[test]
    fn test_module_with_maximum_name_length() {
        let max_name = "A".repeat(10_000); // Very long name
        let module = Module::new(max_name.clone());
        assert_eq!(module.name, max_name);
        assert!(module.operations.is_empty());
    }

    /// Test Value with empty shape (scalar) and verify num_elements calculation
    #[test]
    fn test_scalar_value_num_elements() {
        let scalar = Value {
            name: "scalar_val".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape represents scalar
        };
        
        let num_el = scalar.num_elements();
        assert_eq!(num_el, Some(1));  // Scalar has 1 element
    }

    /// Test Value with shape containing zero dimension and verify num_elements calculation
    #[test]
    fn test_zero_dimension_value_num_elements() {
        let zero_dim = Value {
            name: "zero_dim_val".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5],  // Contains 0, so total is 0
        };
        
        let num_el = zero_dim.num_elements();
        assert_eq!(num_el, Some(0));  // Zero-dimensional tensor has 0 elements
    }

    /// Test Value with very large dimensions that could potentially cause overflow
    #[test]
    fn test_large_dimension_value_num_elements() {
        let large_dims = Value {
            name: "large_val".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],  // Might overflow multiplication
        };
        
        // This should handle overflow gracefully
        let num_el = large_dims.num_elements();
        // If it overflows, should return None; otherwise Some(actual_result)
        // checked_mul handles this properly
        match num_el {
            Some(result) => assert_eq!(result, 10_000_000_000),
            None => println!("Multiplication overflowed as expected"),
        }
    }

    /// Test deeply nested tensor types
    #[test]
    fn test_deeply_nested_tensor_types() {
        let mut current_type = Type::F32;
        
        // Create 50 levels of nesting
        for _ in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the type is still valid
        assert!(current_type.is_valid_type());
        
        // Deep clone to test memory management
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }

    /// Test operations with maximum possible attributes
    #[test]
    fn test_operation_with_many_attributes() {
        let mut op = Operation::new("test_op");
        let mut attributes = HashMap::new();
        
        // Add many attributes
        for i in 0..10_000 {
            attributes.insert(
                format!("key_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        op.attributes = attributes;
        assert_eq!(op.attributes.len(), 10_000);
    }

    /// Test operations with maximum inputs and outputs
    #[test]
    fn test_operation_with_max_io() {
        let mut op = Operation::new("io_test_op");
        
        // Add many inputs
        for i in 0..5_000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
        
        // Add many outputs
        for i in 0..2_500 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
        
        assert_eq!(op.inputs.len(), 5_000);
        assert_eq!(op.outputs.len(), 2_500);
    }

    /// Test parsing and validation of various type strings using rstest
    #[rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    fn test_basic_types_validation(#[case] type_val: Type) {
        assert!(type_val.is_valid_type());
    }

    /// Test nested tensor with different element types
    #[test]
    fn test_nested_tensor_with_different_element_types() {
        // Test tensor<f32, [10, 20]>
        let tensor_f32 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![10, 20],
        };
        assert!(tensor_f32.is_valid_type());
        
        // Test tensor<i64, [5, 5, 5]>
        let tensor_i64 = Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![5, 5, 5],
        };
        assert!(tensor_i64.is_valid_type());
        
        // Test nested tensor<tensor<bool, [2]>, [3]>
        let nested_bool = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Bool),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        assert!(nested_bool.is_valid_type());
    }

    /// Test attribute array edge cases
    #[test]
    fn test_attribute_array_edge_cases() {
        // Empty array
        let empty_array = Attribute::Array(vec![]);
        match empty_array {
            Attribute::Array(vec) => assert_eq!(vec.len(), 0),
            _ => panic!("Expected empty array"),
        }
        
        // Array with many nested elements
        let mut deep_array = Attribute::Array(vec![]);
        for _ in 0..1000 {
            deep_array = Attribute::Array(vec![deep_array]);
        }
        
        match deep_array {
            Attribute::Array(_) => (), // Success
            _ => panic!("Expected deeply nested array"),
        }
    }
}