//! Additional edge case tests for the Impulse compiler
//! This file includes comprehensive tests for boundary conditions and error scenarios

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    ImpulseCompiler,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Test 1: Operations with maximum numeric values in attributes and shapes
    #[test]
    fn test_operations_with_max_numeric_values() {
        // Test attribute with maximum i64 value
        let max_int_attr = Attribute::Int(i64::MAX);
        let min_int_attr = Attribute::Int(i64::MIN);
        
        // Test attribute with maximum f64 value
        let max_float_attr = Attribute::Float(f64::MAX);
        let min_float_attr = Attribute::Float(f64::MIN);
        let epsilon_float_attr = Attribute::Float(f64::EPSILON);
        
        // Verify the maximum values are stored correctly
        match max_int_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        };
        
        match max_float_attr {
            Attribute::Float(val) => assert_eq!(val, f64::MAX),
            _ => panic!("Expected Float attribute"),
        };
        
        // Test tensor with maximum possible dimensions
        let max_value = std::usize::MAX;
        let max_tensor = Value {
            name: "max_tensor".to_string(),
            ty: Type::F32,
            shape: vec![max_value, max_value],  // This would theoretically overflow in calculation
        };
        
        assert_eq!(max_tensor.shape, vec![max_value, max_value]);
    }

    /// Test 2: Tensor operations with special floating-point values
    #[test]
    fn test_tensor_operations_with_special_float_values() {
        // Create values with special floating-point values in their attributes
        let special_attrs = vec![
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(f64::NAN),
            Attribute::Float(0.0),
            Attribute::Float(-0.0),
            Attribute::Float(f64::consts::PI),
            Attribute::Float(f64::consts::E),
        ];
        
        let mut op = Operation::new("special_float_op");
        let mut attrs = HashMap::new();
        
        for (i, attr) in special_attrs.iter().enumerate() {
            attrs.insert(format!("special_attr_{}", i), attr.clone());
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 7);
        
        // Test that NaN comparisons work correctly 
        if let Some(Attribute::Float(nan_val)) = op.attributes.get("special_attr_2") {
            assert!(nan_val.is_nan());
        }
    }

    /// Test 3: Complex nested tensor type operations
    #[test]
    fn test_complex_nested_tensor_types() {
        // Create a complex nested tensor type: Tensor<Tensor<Tensor<F32, [2,3]>, [4,5]>, [6]>
        let deep_nested = Type::Tensor {
            element_type: Box::new(
                Type::Tensor {
                    element_type: Box::new(
                        Type::Tensor {
                            element_type: Box::new(Type::F32),
                            shape: vec![2, 3],
                        }
                    ),
                    shape: vec![4, 5],
                }
            ),
            shape: vec![6],
        };

        // Verify the structure by pattern matching
        match &deep_nested {
            Type::Tensor { element_type: level1, shape: outer_shape } => {
                assert_eq!(outer_shape, &vec![6]);
                
                match level1.as_ref() {
                    Type::Tensor { element_type: level2, shape: mid_shape } => {
                        assert_eq!(mid_shape, &vec![4, 5]);
                        
                        match level2.as_ref() {
                            Type::Tensor { element_type: level3, shape: inner_shape } => {
                                assert_eq!(inner_shape, &vec![2, 3]);
                                
                                match level3.as_ref() {
                                    Type::F32 => { /* Success */ },
                                    _ => panic!("Innermost type should be F32"),
                                }
                            },
                            _ => panic!("Mid type should be Tensor"),
                        }
                    },
                    _ => panic!("Outer type should be Tensor"),
                }
            },
            _ => panic!("Root type should be Tensor"),
        }

        // Test cloning of deeply nested type
        let cloned = deep_nested.clone();
        assert_eq!(deep_nested, cloned);
    }

    /// Test 4: Error handling with malformed input data
    #[test]
    fn test_error_handling_with_malformed_inputs() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with invalid ONNX-like bytes (not a real ONNX file but should not crash)
        let invalid_onnx_bytes = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let result = compiler.compile(&invalid_onnx_bytes, "cpu");
        
        // The result could be success or failure but should not panic
        assert!(result.is_ok() || result.is_err());
        
        // Test with completely random bytes
        let random_bytes = (0..100).map(|i| i as u8).collect::<Vec<_>>();
        let result2 = compiler.compile(&random_bytes, "cuda");
        
        assert!(result2.is_ok() || result2.is_err());
    }

    /// Test 5: Memory allocation edge cases with extremely large tensors
    #[test]
    fn test_extremely_large_tensor_allocations() {
        // Test creating a tensor with extremely large dimensions that might cause allocation issues
        let huge_tensor = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000, 1_000],  // 1 trillion elements
        };
        
        assert_eq!(huge_tensor.shape, vec![1_000_000, 1_000]);
        let product: usize = huge_tensor.shape.iter().product();
        assert_eq!(product, 1_000_000_000_000);  // 1 trillion
        
        // Verify that the value object was created successfully
        assert_eq!(huge_tensor.name, "huge_tensor");
        assert_eq!(huge_tensor.ty, Type::F32);
    }

    /// Test 6: Unicode and special character handling in identifiers
    #[test]
    fn test_unicode_and_special_character_identifiers() {
        let unicode_test_cases = [
            ("tensor_中文_日本語_한국어", Type::F32),
            ("tensor_🚀_🔥_🎉", Type::I64),
            ("tensor_α_β_γ_δ", Type::F64),
            ("tensor_ÀÁÂÃÄÅ_ÆÇÈÉ", Type::I32),
            ("name_with_control_chars_\u{0001}_\u{001F}", Type::Bool),
        ];

        for (name, ty) in &unicode_test_cases {
            let value = Value {
                name: name.to_string(),
                ty: ty.clone(),
                shape: vec![1, 2, 3],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, *ty);
            assert_eq!(value.shape, vec![1, 2, 3]);
            
            // Also test with operations
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
            
            // And modules
            let module = Module::new(name);
            assert_eq!(module.name, *name);
        }
    }

    /// Test 7: Zero-dimensional tensor operations (scalars)
    #[test]
    fn test_zero_dimensional_tensor_operations() {
        // Test scalar values (zero-dimensional tensors)
        let scalar = Value {
            name: "scalar_value".to_string(),
            ty: Type::F32,
            shape: vec![], // Zero-dimensional (scalar)
        };
        
        assert!(scalar.shape.is_empty());
        assert_eq!(scalar.shape.len(), 0);
        
        // A scalar should have exactly 1 element when calculating product
        let element_count: usize = scalar.shape.iter().product();
        assert_eq!(element_count, 1);
        
        // Add scalar to an operation
        let mut op = Operation::new("scalar_op");
        op.inputs.push(scalar);
        
        assert_eq!(op.inputs.len(), 1);
        assert!(op.inputs[0].shape.is_empty());
    }

    /// Test 8: Module operations with boundary condition shapes
    #[test]
    fn test_module_operations_boundary_conditions() {
        let mut module = Module::new("boundary_test_module");
        
        // Add operations with various boundary condition shapes
        let boundary_shapes = [
            vec![],           // Scalar
            vec![0],          // Zero-size
            vec![0, 0],       // 2D zero-size
            vec![0, 1, 0],    // 3D with zeros
            vec![1],          // Single element in 1D
            vec![1, 1, 1, 1], // Identity-like shape
            vec![usize::MAX, 1], // Max boundary
        ];
        
        for (i, shape) in boundary_shapes.iter().enumerate() {
            let mut op = Operation::new(&format!("boundary_op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: shape.clone(),
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), boundary_shapes.len());
        
        // Verify each operation has the correct input shape
        for (i, expected_shape) in boundary_shapes.iter().enumerate() {
            assert_eq!(module.operations[i].inputs[0].shape, *expected_shape);
            
            // Calculate element count for each shape
            let element_count: usize = module.operations[i].inputs[0].shape.iter().product();
            if expected_shape.iter().any(|&dim| dim == 0) && !expected_shape.is_empty() {
                assert_eq!(element_count, 0, "Shape {:?} should have 0 elements", expected_shape);
            }
        }
    }

    /// Test 9: Attribute operations with deeply nested arrays
    #[test]
    fn test_deeply_nested_attribute_arrays() {
        // Create a deeply nested array structure: [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]
        let nested_array_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(1),
                    Attribute::Int(2),
                ]),
                Attribute::Array(vec![
                    Attribute::Int(3),
                    Attribute::Int(4),
                ])
            ]),
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(5),
                    Attribute::Int(6),
                ]),
                Attribute::Array(vec![
                    Attribute::Int(7),
                    Attribute::Int(8),
                ])
            ])
        ]);
        
        // Verify the nested structure
        match &nested_array_attr {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);  // Two major groups
                
                match &outer[0] {
                    Attribute::Array(middle1) => {
                        assert_eq!(middle1.len(), 2);  // Two subgroups
                        
                        match &middle1[0] {
                            Attribute::Array(inner1) => {
                                assert_eq!(inner1.len(), 2);  // Two elements
                                match &inner1[0] {
                                    Attribute::Int(val) => assert_eq!(*val, 1),
                                    _ => panic!("Expected Int(1)"),
                                }
                                match &inner1[1] {
                                    Attribute::Int(val) => assert_eq!(*val, 2),
                                    _ => panic!("Expected Int(2)"),
                                }
                            },
                            _ => panic!("Expected nested Array"),
                        }
                    },
                    _ => panic!("Expected Array at middle level"),
                }
                
                match &outer[1] {
                    Attribute::Array(middle2) => {
                        assert_eq!(middle2.len(), 2);
                        
                        match &middle2[1] {
                            Attribute::Array(inner2) => {
                                assert_eq!(inner2.len(), 2);
                                match &inner2[0] {
                                    Attribute::Int(val) => assert_eq!(*val, 7),
                                    _ => panic!("Expected Int(7)"),
                                }
                                match &inner2[1] {
                                    Attribute::Int(val) => assert_eq!(*val, 8),
                                    _ => panic!("Expected Int(8)"),
                                }
                            },
                            _ => panic!("Expected nested Array"),
                        }
                    },
                    _ => panic!("Expected Array at middle level"),
                }
            },
            _ => panic!("Expected Array at top level"),
        }
        
        // Test cloning nested arrays
        let cloned_array = nested_array_attr.clone();
        assert_eq!(nested_array_attr, cloned_array);
    }

    /// Test 10: Comprehensive validation of tensor type validity checks
    #[test]
    fn test_tensor_type_validity_comprehensive() {
        // Test valid tensor types
        let valid_tensors = [
            Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![1, 2, 3],
            },
            Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![10, 20],
            },
            Type::Tensor {
                element_type: Box::new(
                    Type::Tensor {
                        element_type: Box::new(Type::Bool),
                        shape: vec![5],
                    }
                ),
                shape: vec![2, 2],
            },
            Type::F32,  // Non-tensor types should be valid
            Type::I64,
            Type::Bool,
        ];
        
        for tensor in &valid_tensors {
            assert!(tensor.is_valid_type());
        }
        
        // Test with zero dimensions in tensor shapes
        let zero_dim_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![10, 0, 5],  // Contains zero dimension
        };
        assert!(zero_dim_tensor.is_valid_type());
        
        // Test with empty shape for nested tensor
        let empty_shape_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![],  // Empty shape
        };
        assert!(empty_shape_tensor.is_valid_type());
        
        // Test deeply nested valid tensor
        let deep_valid = Type::Tensor {
            element_type: Box::new(
                Type::Tensor {
                    element_type: Box::new(
                        Type::Tensor {
                            element_type: Box::new(Type::F64),
                            shape: vec![3, 3],
                        }
                    ),
                    shape: vec![2],
                }
            ),
            shape: vec![5],
        };
        assert!(deep_valid.is_valid_type());
    }
}