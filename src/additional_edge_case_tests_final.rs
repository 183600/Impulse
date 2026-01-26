//! Additional edge case tests for the Impulse compiler
//! This file contains extra tests covering boundary conditions and edge cases

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils,
};

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;

    /// Test 1: Operations with maximum possible attribute count
    #[test]
    fn test_operation_with_maximum_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_attr_op");
        let mut attrs = HashMap::new();
        
        // Add many different attribute types
        for i in 0..1000 {
            attrs.insert(
                format!("int_attr_{}", i),
                Attribute::Int(i as i64)
            );
            attrs.insert(
                format!("float_attr_{}", i),
                Attribute::Float(i as f64 * 0.5)
            );
            attrs.insert(
                format!("string_attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
            attrs.insert(
                format!("bool_attr_{}", i),
                Attribute::Bool(i % 2 == 0)
            );
        }
        
        op.attributes = attrs;
        
        // Verify we have many attributes
        assert_eq!(op.attributes.len(), 4000);
    }

    /// Test 2: Zero-dimensional tensors (scalars) with all data types
    #[test]
    fn test_scalar_tensors_all_types() {
        let scalar_types = [
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
        ];
        
        for scalar_type in &scalar_types {
            let value = Value {
                name: format!("{:?}", scalar_type),
                ty: scalar_type.clone(),
                shape: vec![],  // Scalar (zero-dimensional)
            };
            
            assert_eq!(value.shape.len(), 0);
            
            // Scalar tensors have exactly 1 element
            let element_count: usize = value.shape.iter().product();
            assert_eq!(element_count, 1);
            
            // Size calculation should work for scalars
            let size_result = ir_utils::calculate_tensor_size(&value.ty, &value.shape);
            assert!(size_result.is_ok());
        }
    }

    /// Test 3: Operations with empty inputs or outputs
    #[test]
    fn test_empty_io_operations() {
        // Operation with no inputs or outputs
        let op_no_io = Operation::new("constant_producer");
        assert_eq!(op_no_io.inputs.len(), 0);
        assert_eq!(op_no_io.outputs.len(), 0);
        assert_eq!(op_no_io.attributes.len(), 0);
        
        // Operation with inputs but no outputs (sink operation)
        let mut op_sink = Operation::new("data_sink");
        op_sink.inputs.push(Value {
            name: "data".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        assert_eq!(op_sink.inputs.len(), 1);
        assert_eq!(op_sink.outputs.len(), 0);
        
        // Operation with outputs but no inputs (source operation)
        let mut op_source = Operation::new("data_source");
        op_source.outputs.push(Value {
            name: "generated_data".to_string(),
            ty: Type::I32,
            shape: vec![5, 5],
        });
        assert_eq!(op_source.inputs.len(), 0);
        assert_eq!(op_source.outputs.len(), 1);
    }

    /// Test 4: Extreme tensor shapes and size calculations
    #[test]
    fn test_extreme_tensor_shapes() {
        // Extremely wide tensor (one very large dimension)
        let wide_tensor = Value {
            name: "wide_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000],  // 1M elements in 1D
        };
        
        let wide_size = ir_utils::calculate_tensor_size(&wide_tensor.ty, &wide_tensor.shape).unwrap();
        assert_eq!(wide_size, 1_000_000 * 4); // 4 bytes per F32
        
        // Extremely deep tensor (many dimensions, each small)
        let deep_tensor = Value {
            name: "deep_tensor".to_string(),
            ty: Type::F32,
            shape: vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  // 2^10 = 1024 elements
        };
        
        let deep_size = ir_utils::calculate_tensor_size(&deep_tensor.ty, &deep_tensor.shape).unwrap();
        assert_eq!(deep_size, 1024 * 4); // 1024 elements * 4 bytes per F32
        
        // Flat tensor (single dimension with moderate size)
        let flat_tensor = Value {
            name: "flat_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1024],
        };
        
        let flat_size = ir_utils::calculate_tensor_size(&flat_tensor.ty, &flat_tensor.shape).unwrap();
        assert_eq!(flat_size, 1024 * 4); // Same as deep tensor, but in 1D
    }

    /// Test 5: Unicode and special character handling in identifiers
    #[test]
    fn test_unicode_identifiers() {
        let unicode_test_cases = [
            ("tensor_特殊_字符", Type::F32),
            ("tensor_🚀_utf8", Type::I64),
            ("tensor_αβγ_delta", Type::F64),
            ("tensor_مختبر", Type::Bool),  // Arabic
            ("tensor_テスト", Type::I32),  // Japanese
        ];

        for (name, data_type) in &unicode_test_cases {
            // Test Value creation with unicode name
            let value = Value {
                name: name.to_string(),
                ty: data_type.clone(),
                shape: vec![1, 2, 3],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, *data_type);
            assert_eq!(value.shape, vec![1, 2, 3]);
            
            // Test Operation creation with unicode name
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
            
            // Test Module creation with unicode name
            let module = Module::new(*name);
            assert_eq!(module.name, *name);
        }
    }

    /// Test 6: Nested array attributes with maximum depth
    #[test]
    fn test_deeply_nested_array_attributes() {
        // Create a deeply nested array structure
        let mut nested_attr = Attribute::Int(0);
        
        // Build nested arrays to depth of 50
        for i in 1..50 {
            nested_attr = Attribute::Array(vec![
                Attribute::Int(i),
                nested_attr,
            ]);
        }
        
        // Verify the structure can be processed without stack overflow
        match &nested_attr {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 2);
                
                // Access the first element
                match &arr[0] {
                    Attribute::Int(val) => assert_eq!(*val, 49), // Should be the last value added
                    _ => panic!("Expected Int attribute at top level"),
                }
            },
            _ => panic!("Expected Array attribute at top level"),
        }
    }

    /// Test 7: Tensor size calculations with potential overflow prevention
    #[test]
    fn test_tensor_size_overflow_prevention() {
        
        
        // Create a tensor that could potentially cause overflow in size calculations
        // We'll use values that when multiplied together could exceed usize::MAX
        // We'll test with values that are large but not necessarily causing an overflow
        let large_but_safe_tensor = Value {
            name: "large_but_safe".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],  // This would theoretically be 10^10 elements
        };
        
        // This should not panic, and should either succeed or return an error gracefully
        let result = ir_utils::calculate_tensor_size(&large_but_safe_tensor.ty, &large_but_safe_tensor.shape);
        // Either the operation succeeds or it properly errors out, no panics allowed
        assert!(result.is_ok() || result.is_err());
        
        // Test with zero-dimension tensor (scalar)
        let scalar_tensor = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Scalar
        };
        let scalar_result = ir_utils::calculate_tensor_size(&scalar_tensor.ty, &scalar_tensor.shape);
        assert!(scalar_result.is_ok());
        assert_eq!(scalar_result.unwrap(), 4); // 4 bytes for F32 scalar
        
        // Test with tensor containing zero in dimensions (results in zero size)
        let zero_tensor = Value {
            name: "zero_in_dims".to_string(),
            ty: Type::F64,
            shape: vec![100, 0, 50],  // Contains zero, so total size is 0
        };
        let zero_result = ir_utils::calculate_tensor_size(&zero_tensor.ty, &zero_tensor.shape);
        assert!(zero_result.is_ok());
        assert_eq!(zero_result.unwrap(), 0); // Should be 0 due to zero in dimensions
    }

    /// Test 8: Operations with maximum possible input/output count
    #[test]
    fn test_operations_with_max_io() {
        // Create an operation with many inputs and outputs
        let mut op = Operation::new("max_io_operation");
        
        // Add many inputs
        for i in 0..500 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 10 + 1],  // Varying small shapes
            });
        }
        
        // Add many outputs
        for i in 0..300 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 5 + 1],  // Varying small shapes
            });
        }
        
        assert_eq!(op.inputs.len(), 500);
        assert_eq!(op.outputs.len(), 300);
        assert_eq!(op.op_type, "max_io_operation");
        
        // Verify we can access elements from different positions
        assert_eq!(op.inputs[0].name, "input_0");
        assert_eq!(op.inputs[499].name, "input_499");
        assert_eq!(op.outputs[0].name, "output_0");
        assert_eq!(op.outputs[299].name, "output_299");
    }

    /// Test 9: Module validation with edge case scenarios
    #[test]
    fn test_module_validation_edge_cases() {
        use crate::utils::validation_utils;
        
        // Test with extremely long module name
        let long_name = "module_".repeat(10_000) + "end";
        let module = Module::new(&long_name);
        assert_eq!(module.name.len(), long_name.len());
        
        // Test with duplicate input names (should fail validation)
        let mut bad_module = Module::new("bad_module");
        bad_module.inputs.push(Value {
            name: "duplicate_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        bad_module.inputs.push(Value {
            name: "duplicate_input".to_string(),  // Same name as above
            ty: Type::F32,
            shape: vec![20],
        });
        
        let validation_result = validation_utils::validate_module(&bad_module);
        assert!(validation_result.is_err());
        assert!(validation_result.unwrap_err().to_string().contains("Duplicate input name"));
        
        // Test with valid module
        let mut good_module = Module::new("good_module");
        good_module.inputs.push(Value {
            name: "unique_input1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        good_module.inputs.push(Value {
            name: "unique_input2".to_string(),
            ty: Type::I32,
            shape: vec![20],
        });
        
        let good_validation_result = validation_utils::validate_module(&good_module);
        assert!(good_validation_result.is_ok());
    }

    /// Test 10: Complex nested tensor types with multiple levels
    #[test]
    fn test_complex_nested_tensor_types() {
        // Create a complex nested tensor type: tensor<tensor<tensor<f32, [2]>, [3]>, [4]>
        // This is a 3-level nested tensor
        let complex_type = Type::Tensor {
            element_type: Box::new(
                Type::Tensor {
                    element_type: Box::new(
                        Type::Tensor {
                            element_type: Box::new(Type::F32),
                            shape: vec![2],  // Innermost tensor shape
                        }
                    ),
                    shape: vec![3],  // Middle tensor shape
                }
            ),
            shape: vec![4],  // Outermost tensor shape
        };
        
        // Calculate size with various outer shapes
        let size_with_empty = ir_utils::calculate_tensor_size(&complex_type, &[]).unwrap();
        // This should be: 1 (empty outer) * 4 (outermost shape) * 3 (middle shape) * 2 (inner shape) * 4 (F32 size)
        // = 1 * 4 * 3 * 2 * 4 = 96
        assert_eq!(size_with_empty, 96);
        
        let size_with_outer = ir_utils::calculate_tensor_size(&complex_type, &[5]).unwrap();
        // This should be: 5 (outer shape) * 4 * 3 * 2 * 4 = 480
        assert_eq!(size_with_outer, 480);
        
        // Test type equality
        let same_complex_type = Type::Tensor {
            element_type: Box::new(
                Type::Tensor {
                    element_type: Box::new(
                        Type::Tensor {
                            element_type: Box::new(Type::F32),
                            shape: vec![2],  
                        }
                    ),
                    shape: vec![3],  
                }
            ),
            shape: vec![4],  
        };
        
        assert_eq!(complex_type, same_complex_type);
        
        // Test inequality with slightly different shape
        let diff_complex_type = Type::Tensor {
            element_type: Box::new(
                Type::Tensor {
                    element_type: Box::new(
                        Type::Tensor {
                            element_type: Box::new(Type::F32),
                            shape: vec![3],  // Different from [2]
                        }
                    ),
                    shape: vec![3],  
                }
            ),
            shape: vec![4],  
        };
        
        assert_ne!(complex_type, diff_complex_type);
    }
}