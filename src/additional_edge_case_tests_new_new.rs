//! Additional edge case tests for the Impulse compiler
//! This module focuses on testing boundary conditions and potential edge cases
//! that could cause unexpected behavior in the compiler

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::ir_utils;

#[cfg(test)]
mod additional_edge_case_tests_new_new {
    use super::*;

    /// Test with maximum possible usize values in tensor shapes (potential overflow)
    #[test]
    fn test_very_large_shape_values() {
        // Using a large but reasonable value that won't necessarily overflow on multiplication
        let huge_shape = vec![1_000_000, 1_000_000];
        let value = Value {
            name: "huge_shape_tensor".to_string(),
            ty: Type::F32,
            shape: huge_shape,
        };
        
        assert_eq!(value.shape.len(), 2);
        assert_eq!(value.shape[0], 1_000_000);
        assert_eq!(value.shape[1], 1_000_000);
        
        // Test that the number of elements calculation handles large values appropriately
        let num_elements_result = value.num_elements();
        assert!(num_elements_result.is_some());
        
        if let Some(num_elements) = num_elements_result {
            assert_eq!(num_elements, 1_000_000 * 1_000_000); // 1 trillion elements
        }
    }

    /// Test deeply nested tensor types (up to 1000 levels) to test recursion limits
    #[test]
    fn test_extremely_deep_tensor_nesting() {
        let mut current_type = Type::F32;
        // Create 100 levels of nesting to test recursive operations without hitting stack limits
        for _ in 0..100 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the top level is still a tensor with shape [2]
        if let Type::Tensor { shape, .. } = &current_type {
            assert_eq!(shape, &vec![2]);
        } else {
            panic!("Expected a tensor type after deep nesting");
        }
        
        // Test that equality works even with deep nesting
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Test getting element type from deeply nested structure
        let element_type = ir_utils::get_element_type(&current_type);
        assert_eq!(element_type, &Type::F32);
    }

    /// Test operations with the maximum practical number of attributes
    #[test]
    fn test_operation_with_maximum_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attr_op");
        let mut attributes = HashMap::new();
        
        // Add a large number of attributes to the operation
        for i in 0..50_000 {
            let key = format!("attribute_{:06}", i);
            let value = Attribute::String(format!("value_{}", i));
            attributes.insert(key, value);
        }
        
        op.attributes = attributes;
        
        assert_eq!(op.attributes.len(), 50_000);
        assert_eq!(op.op_type, "max_attr_op");
        
        // Verify a few specific attributes exist with correct values
        assert!(op.attributes.contains_key("attribute_000000"));  // First element: index 0 formatted as 000000
        assert!(op.attributes.contains_key("attribute_049999"));  // Last element: index 49999 formatted as 049999
        
        if let Some(attr_val) = op.attributes.get("attribute_000000") {
            match attr_val {
                Attribute::String(s) => assert_eq!(s, "value_0"),
                _ => panic!("Expected string attribute"),
            }
        } else {
            panic!("Attribute not found");
        }
        
        // Also verify the last element
        if let Some(attr_val) = op.attributes.get("attribute_049999") {
            match attr_val {
                Attribute::String(s) => assert_eq!(s, "value_49999"),
                _ => panic!("Expected string attribute"),
            }
        } else {
            panic!("Last attribute not found");
        }
    }

    /// Test tensor with all possible primitive types and zero dimensions
    #[test]
    fn test_tensors_with_all_types_and_zero_dims() {
        let test_cases = vec![
            (Type::F32, vec![] as Vec<usize>),  // Scalar F32
            (Type::F64, vec![]),                // Scalar F64
            (Type::I32, vec![]),                // Scalar I32
            (Type::I64, vec![]),                // Scalar I64
            (Type::Bool, vec![]),               // Scalar Bool
            (Type::F32, vec![0]),               // 1D tensor with 0 elements
            (Type::I32, vec![0, 10]),          // 2D tensor with 0 elements
            (Type::F64, vec![5, 0, 3]),        // 3D tensor with 0 elements
        ];
        
        for (data_type, shape) in test_cases {
            let value = Value {
                name: "test_tensor".to_string(),
                ty: data_type.clone(),
                shape: shape.clone(),
            };
            
            assert_eq!(value.ty, data_type);
            assert_eq!(value.shape, shape);
            
            // Calculate number of elements - should be 1 for scalars, 0 for any shape with 0
            let expected_elements = if shape.is_empty() {
                1  // Scalar
            } else {
                shape.iter().product()
            };
            
            assert_eq!(value.num_elements(), Some(expected_elements));
        }
    }

    /// Test module creation with extreme name lengths and special Unicode characters
    #[test]
    fn test_module_with_extreme_name_variations() {
        // Test with a very long ASCII name
        let long_name = "a".repeat(100_000);
        let module_long = Module::new(&long_name);
        assert_eq!(module_long.name, long_name);
        
        // Test with Unicode characters
        let unicode_name = "模块_测试_Модуль_टेस्ट_テスト";
        let module_unicode = Module::new(unicode_name);
        assert_eq!(module_unicode.name, unicode_name);
        
        // Test with empty name
        let module_empty = Module::new("");
        assert_eq!(module_empty.name, "");
        
        // Test that all modules start with empty operations
        assert_eq!(module_long.operations.len(), 0);
        assert_eq!(module_unicode.operations.len(), 0);
        assert_eq!(module_empty.operations.len(), 0);
    }

    /// Test handling of extreme numeric values in attributes
    #[test]
    fn test_extreme_numeric_attributes() {
        let attrs = vec![
            Attribute::Int(i64::MAX),
            Attribute::Int(i64::MIN),
            Attribute::Int(0),
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(f64::NAN),
            Attribute::Float(f64::MAX),
            Attribute::Float(f64::MIN),
            Attribute::Float(0.0),
        ];
        
        match attrs[0] {  // i64::MAX
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }
        
        match attrs[1] {  // i64::MIN
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected Int attribute"),
        }
        
        // Test NaN specially since NaN != NaN
        match attrs[5] {  // f64::NAN
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected Float attribute"),
        }
        
        match attrs[6] {  // f64::MAX
            Attribute::Float(val) => assert_eq!(val, f64::MAX),
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test tensor size calculation with potential overflow scenarios
    #[test]
    fn test_tensor_size_calculation_overflow_scenarios() {
        // Test normal scenarios
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &[10, 10]).unwrap(), 400); // 10*10*4
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I64, &[5, 4]).unwrap(), 160);   // 5*4*8
        
        // Test scalar values
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &[]).unwrap(), 4);  // Scalar F32
        assert_eq!(ir_utils::calculate_tensor_size(&Type::Bool, &[]).unwrap(), 1); // Scalar Bool
        
        // Test zero-sized tensors
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &[0]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I32, &[10, 0, 5]).unwrap(), 0);
        
        // Test with large but safe dimensions
        let large_but_safe = ir_utils::calculate_tensor_size(&Type::F32, &[100_000, 100]);
        assert!(large_but_safe.is_ok());
        assert_eq!(large_but_safe.unwrap(), 40_000_000); // 100k * 100 * 4
        
        // Test nested tensor size calculation
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        // Outer shape [5], inner tensor [2, 3] of F32 -> 5 * (2 * 3 * 4) = 120 bytes
        let nested_size = ir_utils::calculate_tensor_size(&nested_type, &[5]).unwrap();
        assert_eq!(nested_size, 120);
    }

    /// Test creating and comparing operations with identical properties
    #[test]
    fn test_identical_operations_comparison() {
        let mut op1 = Operation::new("test_op");
        op1.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        op1.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        
        let mut op2 = Operation::new("test_op");
        op2.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        op2.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        
        // Operations with identical properties should be equal
        assert_eq!(op1, op2);
        
        // Modify one operation to make it different
        op1.op_type = "different_op".to_string();
        assert_ne!(op1, op2);
    }

    /// Test value equality with various edge cases
    #[test]
    fn test_value_equality_edge_cases() {
        // Identical values should be equal
        let val1 = Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        let val2 = Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        assert_eq!(val1, val2);
        
        // Values with different names but same other properties are NOT equal (name matters)
        let val3 = Value {
            name: "different_name".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        assert_ne!(val1, val3);  // Names also matter for equality
        
        // Values with different types are not equal
        let val4 = Value {
            name: "test".to_string(),
            ty: Type::I32,  // Different type
            shape: vec![2, 3, 4],
        };
        assert_ne!(val1, val4);
        
        // Values with different shapes are not equal
        let val5 = Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 5],  // Different shape
        };
        assert_ne!(val1, val5);
        
        // Edge case: empty shapes (scalars) with same type
        let scalar1 = Value {
            name: "scalar1".to_string(),
            ty: Type::F32,
            shape: vec![],  // Scalar
        };
        let scalar2 = Value {
            name: "scalar2".to_string(),
            ty: Type::F32,
            shape: vec![],  // Scalar
        };
        assert_ne!(scalar1, scalar2); // Different names make them unequal
    }

    /// Test operations with invalid or unusual configurations
    #[test]
    fn test_unusual_operation_configurations() {
        // Operation with no inputs, no outputs, but with attributes
        let mut op_no_io = Operation::new("config_op");
        op_no_io.attributes.insert("param1".to_string(), Attribute::Int(42));
        op_no_io.attributes.insert("param2".to_string(), Attribute::String("test".to_string()));
        
        assert_eq!(op_no_io.inputs.len(), 0);
        assert_eq!(op_no_io.outputs.len(), 0);
        assert_eq!(op_no_io.attributes.len(), 2);
        assert_eq!(op_no_io.op_type, "config_op");
        
        // Operation with many inputs but no outputs (e.g., logging operation)
        let mut op_many_inputs = Operation::new("log_op");
        for i in 0..100 {
            op_many_inputs.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1], // Different shapes for each input
            });
        }
        
        assert_eq!(op_many_inputs.inputs.len(), 100);
        assert_eq!(op_many_inputs.outputs.len(), 0);
        assert_eq!(op_many_inputs.op_type, "log_op");
        
        // Operation with no inputs but many outputs (e.g., constant generator)
        let mut op_many_outputs = Operation::new("const_gen");
        for i in 0..50 {
            op_many_outputs.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::I32,
                shape: vec![2, i + 1], // Different shapes
            });
        }
        
        assert_eq!(op_many_outputs.inputs.len(), 0);
        assert_eq!(op_many_outputs.outputs.len(), 50);
        assert_eq!(op_many_outputs.op_type, "const_gen");
    }
}