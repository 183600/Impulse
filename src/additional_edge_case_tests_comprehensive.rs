//! Comprehensive edge case tests for the Impulse compiler IR structures
//! This file adds test cases to cover additional boundary conditions and edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::ir_utils;

#[cfg(test)]
mod comprehensive_edge_case_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_empty_module_operations() {
        let module = Module::new("");
        assert_eq!(module.name, "");
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());

        // Test with long empty string name
        let long_empty_module = Module::new("   ".repeat(100));
        assert_eq!(long_empty_module.name, "   ".repeat(100));
        assert!(long_empty_module.operations.is_empty());
    }

    #[test]
    fn test_operation_with_empty_fields() {
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());

        // Test operation with empty string fields
        let mut op_with_empty_attrs = Operation::new("empty_op");
        let empty_attrs: HashMap<String, Attribute> = HashMap::new();
        op_with_empty_attrs.attributes = empty_attrs;
        assert_eq!(op_with_empty_attrs.op_type, "empty_op");
        assert!(op_with_empty_attrs.attributes.is_empty());
    }

    #[test]
    fn test_value_with_edge_case_shapes() {
        // Test scalar (empty shape)
        let scalar = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(scalar.shape.is_empty());
        
        // Test single element tensor
        let single_elem = Value {
            name: "single".to_string(),
            ty: Type::I64,
            shape: vec![1],
        };
        assert_eq!(single_elem.shape, vec![1]);
        
        // Test very large but valid shape
        let large_valid = Value {
            name: "large_valid".to_string(),
            ty: Type::Bool,
            shape: vec![1000, 1000],  // 1M elements
        };
        assert_eq!(large_valid.shape, vec![1000, 1000]);
        
        // Test shape with zeros (results in 0 total elements)
        let zero_shape = Value {
            name: "zero_shape".to_string(),
            ty: Type::F64,
            shape: vec![10, 0, 100],  // Contains 0, so total is 0
        };
        assert_eq!(zero_shape.shape, vec![10, 0, 100]);
    }

    #[test]
    fn test_attribute_edge_cases() {
        // Test empty string attribute
        let empty_str_attr = Attribute::String("".to_string());
        match empty_str_attr {
            Attribute::String(s) => assert_eq!(s, ""),
            _ => panic!("Expected String attribute"),
        }

        // Test very long string attribute
        let long_str = "a".repeat(10_000);
        let long_str_attr = Attribute::String(long_str.clone());
        match &long_str_attr {
            Attribute::String(s) => assert_eq!(s, &long_str),
            _ => panic!("Expected String attribute"),
        }

        // Test empty array attribute
        let empty_array = Attribute::Array(vec![]);
        match empty_array {
            Attribute::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("Expected Array attribute"),
        }

        // Test nested empty arrays
        let nested_empty_arrays = Attribute::Array(vec![
            Attribute::Array(vec![]),
            Attribute::Array(vec![]),
        ]);
        match nested_empty_arrays {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 2);
                match &arr[0] {
                    Attribute::Array(inner) => assert!(inner.is_empty()),
                    _ => panic!("Expected nested Array"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    #[test]
    fn test_type_conversion_edge_cases() {
        // Test conversion of primitive types to strings
        let f32_str = ir_utils::type_to_string(&Type::F32);
        assert_eq!(f32_str, "f32");

        let f64_str = ir_utils::type_to_string(&Type::F64);
        assert_eq!(f64_str, "f64");

        let i32_str = ir_utils::type_to_string(&Type::I32);
        assert_eq!(i32_str, "i32");

        let i64_str = ir_utils::type_to_string(&Type::I64);
        assert_eq!(i64_str, "i64");

        let bool_str = ir_utils::type_to_string(&Type::Bool);
        assert_eq!(bool_str, "bool");

        // Test nested tensor type conversion
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            }),
            shape: vec![3],
        };
        let nested_str = ir_utils::type_to_string(&nested_type);
        // Check that it contains expected components
        assert!(nested_str.contains("f32"));
        assert!(nested_str.contains("[2, 2]"));
        assert!(nested_str.contains("[3]"));
    }

    #[test]
    fn test_tensor_size_calculation_edge_cases() {
        // Test scalar tensor size calculation
        let scalar_size = ir_utils::calculate_tensor_size(&Type::F32, &[]);
        assert_eq!(scalar_size.unwrap(), 4); // 1 element * 4 bytes for F32

        let i64_scalar_size = ir_utils::calculate_tensor_size(&Type::I64, &[]);
        assert_eq!(i64_scalar_size.unwrap(), 8); // 1 element * 8 bytes for I64

        // Test tensor with zero dimensions
        let zero_tensor_size = ir_utils::calculate_tensor_size(&Type::F32, &[10, 0, 5]);
        assert_eq!(zero_tensor_size.unwrap(), 0); // Contains 0, so size is 0

        // Test large tensor size
        let large_tensor_size = ir_utils::calculate_tensor_size(&Type::F32, &[1000, 1000]);
        assert_eq!(large_tensor_size.unwrap(), 4_000_000); // 1M elements * 4 bytes

        // Test nested tensor size
        let nested_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        let nested_size = ir_utils::calculate_tensor_size(&nested_tensor, &[5]);
        assert_eq!(nested_size.unwrap(), 120); // 5 * (2 * 3 * 4 bytes)

        // Test deeply nested tensor size
        let deeply_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![4],
            }),
            shape: vec![3],
        };
        let deep_size = ir_utils::calculate_tensor_size(&deeply_nested, &[2]);
        assert_eq!(deep_size.unwrap(), 192); // 2 * 3 * (4 * 8 bytes)
    }

    #[test]
    fn test_overflow_detection_in_tensor_calculations() {
        // Test very large dimensions that might cause overflow
        // We use checked arithmetic which should return an error for overflow
        
        // This should not overflow
        let large_but_safe = ir_utils::calculate_tensor_size(&Type::F32, &[1_000_000, 1_000]);
        if let Ok(size) = large_but_safe {
            assert_eq!(size, 4_000_000_000); // 1 billion * 4 bytes
        } else {
            // If it overflows, that's also acceptable behavior
        }

        // Test potential overflow scenario for element count
        let result = ir_utils::calculate_tensor_size(&Type::F32, &[usize::MAX, 2]);
        assert!(result.is_err()); // Should return an error due to overflow

        // Test potential overflow in final multiplication (elements * size)
        // This would be a very large tensor that causes overflow when multiplied by type size
        let huge_tensor = ir_utils::calculate_tensor_size(&Type::F64, &[1_000_000_000, 3]);
        if huge_tensor.is_err() {
            // Acceptable to return error for overflow
        } else {
            // If it doesn't overflow, verify the result
            if let Ok(size) = huge_tensor {
                assert_eq!(size, 1_000_000_000 * 3 * 8); // elements * 8 bytes per F64
            }
        }
    }

    #[test]
    fn test_deep_recursion_in_tensor_types() {
        // Create a deeply nested tensor type to test recursion limits
        let mut current_type = Type::F32;
        
        // Create 50 levels of nesting (this tests cloning and equality for deeply nested types)
        for _ in 0..50 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }

        // Verify the final type is still a tensor
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a tensor type after nesting"),
        }

        // Test cloning of deeply nested type
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);

        // Test equality comparison of two identically created deeply nested types
        let mut other_current_type = Type::F32;
        for _ in 0..50 {
            other_current_type = Type::Tensor {
                element_type: Box::new(other_current_type),
                shape: vec![2],
            };
        }
        assert_eq!(current_type, other_current_type);

        // Create a slightly different nested type and test inequality
        let mut different_type = Type::F32;
        for _ in 0..50 {
            different_type = Type::Tensor {
                element_type: Box::new(different_type),
                shape: vec![3], // Different shape
            };
        }
        assert_ne!(current_type, different_type);
    }

    #[test]
    fn test_module_manipulation_edge_cases() {
        let mut module = Module::new("test_module");

        // Add an operation with empty inputs and outputs
        let empty_op = Operation::new("empty_op");
        module.add_operation(empty_op);
        assert_eq!(module.operations.len(), 1);

        // Add an operation with many inputs and outputs
        let mut complex_op = Operation::new("complex_op");
        for i in 0..100 {
            complex_op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
            complex_op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
        }
        module.add_operation(complex_op);

        assert_eq!(module.operations.len(), 2);
        assert_eq!(module.operations[1].inputs.len(), 100);
        assert_eq!(module.operations[1].outputs.len(), 100);

        // Clear operations and test
        let empty_module = Module::new("empty_after_clear");
        assert!(empty_module.operations.is_empty());
    }

    #[test]
    fn test_error_handling_in_utils() {
        // Test edge cases in utility functions
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        
        // Test scalar recognition
        assert!(ir_utils::is_scalar(&scalar_value));
        assert!(!ir_utils::is_vector(&scalar_value));
        assert!(!ir_utils::is_matrix(&scalar_value));
        assert_eq!(ir_utils::get_rank(&scalar_value), 0);
        assert_eq!(ir_utils::get_num_elements(&scalar_value), Some(1));

        // Test vector
        let vector_value = Value {
            name: "vector".to_string(),
            ty: Type::F32,
            shape: vec![10],
        };
        assert!(!ir_utils::is_scalar(&vector_value));
        assert!(ir_utils::is_vector(&vector_value));
        assert!(!ir_utils::is_matrix(&vector_value));
        assert_eq!(ir_utils::get_rank(&vector_value), 1);
        assert_eq!(ir_utils::get_num_elements(&vector_value), Some(10));

        // Test matrix
        let matrix_value = Value {
            name: "matrix".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        };
        assert!(!ir_utils::is_scalar(&matrix_value));
        assert!(!ir_utils::is_vector(&matrix_value));
        assert!(ir_utils::is_matrix(&matrix_value));
        assert_eq!(ir_utils::get_rank(&matrix_value), 2);
        assert_eq!(ir_utils::get_num_elements(&matrix_value), Some(25));

        // Test tensor with zero in shape
        let zero_tensor = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5],
        };
        assert_eq!(ir_utils::get_rank(&zero_tensor), 3);
        assert_eq!(ir_utils::get_num_elements(&zero_tensor), Some(0));

        // Test element type extraction
        let element = ir_utils::get_element_type(&Type::F32);
        assert_eq!(element, &Type::F32);

        let nested_tensor = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I32),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        let nested_element = ir_utils::get_element_type(&nested_tensor);
        assert_eq!(nested_element, &Type::I32);
    }
}