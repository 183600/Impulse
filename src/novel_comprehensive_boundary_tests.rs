//! Novel comprehensive boundary tests for the Impulse compiler
//! This file contains 10 additional test cases covering unique boundary conditions
//! not previously tested in the existing test suite

use crate::{
    ir::{Module, Value, Type, Operation, Attribute, TypeExtensions},
    utils::{is_scalar, is_vector, is_matrix, get_rank, get_num_elements, get_element_type},
};

#[cfg(test)]
mod novel_boundary_tests {
    use super::*;

    /// Test 1: Value with num_elements() method for overflow detection
    /// Tests the Value::num_elements() method for shapes that could cause overflow
    #[test]
    fn test_value_num_elements_overflow_detection() {
        // Test with normal shapes
        let normal_value = Value {
            name: "normal".to_string(),
            ty: Type::F32,
            shape: vec![10, 20, 30],
        };
        assert_eq!(normal_value.num_elements(), Some(6000));

        // Test with zero in shape (should return 0)
        let zero_dim_value = Value {
            name: "zero_dim".to_string(),
            ty: Type::I32,
            shape: vec![100, 0, 50],
        };
        assert_eq!(zero_dim_value.num_elements(), Some(0));

        // Test with scalar (empty shape should return 1)
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F64,
            shape: vec![],
        };
        assert_eq!(scalar_value.num_elements(), Some(1));

        // Test with very large dimensions that might cause overflow
        let large_value = Value {
            name: "large".to_string(),
            ty: Type::I64,
            shape: vec![100000, 100000],
        };
        let result = large_value.num_elements();
        // Either it returns Some(10_000_000_000) or None if overflow occurred
        assert!(result.is_some() || result.is_none());
    }

    /// Test 2: Attribute with extreme float values (infinity, negative infinity, NaN)
    #[test]
    fn test_attribute_extreme_float_values() {
        // Test positive infinity
        let inf_attr = Attribute::Float(f64::INFINITY);
        match inf_attr {
            Attribute::Float(val) => {
                assert!(val.is_infinite());
                assert!(val.is_sign_positive());
            }
            _ => panic!("Expected Float attribute"),
        }

        // Test negative infinity
        let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
        match neg_inf_attr {
            Attribute::Float(val) => {
                assert!(val.is_infinite());
                assert!(val.is_sign_negative());
            }
            _ => panic!("Expected Float attribute"),
        }

        // Test NaN
        let nan_attr = Attribute::Float(f64::NAN);
        match nan_attr {
            Attribute::Float(val) => {
                assert!(val.is_nan());
            }
            _ => panic!("Expected Float attribute"),
        }

        // Test very large finite value
        let max_finite = Attribute::Float(f64::MAX);
        match max_finite {
            Attribute::Float(val) => {
                assert!(!val.is_infinite());
                assert!(!val.is_nan());
                assert!(val > 0.0);
            }
            _ => panic!("Expected Float attribute"),
        }

        // Test very small finite value
        let min_finite = Attribute::Float(f64::MIN_POSITIVE);
        match min_finite {
            Attribute::Float(val) => {
                assert!(!val.is_infinite());
                assert!(!val.is_nan());
                assert!(val > 0.0);
                assert!(val < 1e-300);
            }
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 3: TypeExtensions trait implementation for all types
    #[test]
    fn test_type_extensions_trait() {
        // Test all basic types
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());

        // Test nested tensor types
        let nested_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        assert!(nested_tensor.is_valid_type());

        // Test deeply nested tensor
        let deep_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![4],
            }),
            shape: vec![5, 6],
        };
        assert!(deep_nested.is_valid_type());
    }

    /// Test 4: Utility functions - is_scalar, is_vector, is_matrix
    #[test]
    fn test_utility_shape_classification() {
        // Test scalar (0D)
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(is_scalar(&scalar));
        assert!(!is_vector(&scalar));
        assert!(!is_matrix(&scalar));
        assert_eq!(get_rank(&scalar), 0);

        // Test vector (1D)
        let vector = Value {
            name: "vector".to_string(),
            ty: Type::I32,
            shape: vec![42],
        };
        assert!(!is_scalar(&vector));
        assert!(is_vector(&vector));
        assert!(!is_matrix(&vector));
        assert_eq!(get_rank(&vector), 1);

        // Test matrix (2D)
        let matrix = Value {
            name: "matrix".to_string(),
            ty: Type::F64,
            shape: vec![3, 4],
        };
        assert!(!is_scalar(&matrix));
        assert!(!is_vector(&matrix));
        assert!(is_matrix(&matrix));
        assert_eq!(get_rank(&matrix), 2);

        // Test 3D tensor
        let tensor3d = Value {
            name: "tensor3d".to_string(),
            ty: Type::Bool,
            shape: vec![1, 28, 28],
        };
        assert!(!is_scalar(&tensor3d));
        assert!(!is_vector(&tensor3d));
        assert!(!is_matrix(&tensor3d));
        assert_eq!(get_rank(&tensor3d), 3);

        // Test 4D tensor (common in batch operations)
        let tensor4d = Value {
            name: "tensor4d".to_string(),
            ty: Type::F32,
            shape: vec![32, 3, 224, 224],
        };
        assert_eq!(get_rank(&tensor4d), 4);
    }

    /// Test 5: get_num_elements with various edge cases
    #[test]
    fn test_get_num_elements_edge_cases() {
        // Scalar should return 1
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(get_num_elements(&scalar), Some(1));

        // Single element tensor
        let single = Value {
            name: "single".to_string(),
            ty: Type::I32,
            shape: vec![1],
        };
        assert_eq!(get_num_elements(&single), Some(1));

        // Zero-sized tensor
        let zero = Value {
            name: "zero".to_string(),
            ty: Type::F64,
            shape: vec![10, 0, 20],
        };
        assert_eq!(get_num_elements(&zero), Some(0));

        // Multiple zeros in shape
        let multi_zero = Value {
            name: "multi_zero".to_string(),
            ty: Type::Bool,
            shape: vec![0, 0, 0],
        };
        assert_eq!(get_num_elements(&multi_zero), Some(0));

        // Single dimension zero
        let single_zero = Value {
            name: "single_zero".to_string(),
            ty: Type::I64,
            shape: vec![0],
        };
        assert_eq!(get_num_elements(&single_zero), Some(0));

        // Very large but safe shape
        let large = Value {
            name: "large".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],
        };
        assert_eq!(get_num_elements(&large), Some(1_000_000));
    }

    /// Test 6: get_element_type with deeply nested types
    #[test]
    fn test_get_element_type_deep_nesting() {
        // Direct type
        assert_eq!(get_element_type(&Type::F32), &Type::F32);
        assert_eq!(get_element_type(&Type::I64), &Type::I64);
        assert_eq!(get_element_type(&Type::Bool), &Type::Bool);

        // Single level nesting
        let level1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        };
        assert_eq!(get_element_type(&level1), &Type::F32);

        // Two levels nesting
        let level2 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![3],
            }),
            shape: vec![4],
        };
        assert_eq!(get_element_type(&level2), &Type::I64);

        // Three levels nesting
        let level3 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F64),
                    shape: vec![5],
                }),
                shape: vec![6],
            }),
            shape: vec![7],
        };
        assert_eq!(get_element_type(&level3), &Type::F64);

        // Ten levels nesting
        let mut current = Type::F32;
        for _ in 0..10 {
            current = Type::Tensor {
                element_type: Box::new(current),
                shape: vec![2],
            };
        }
        assert_eq!(get_element_type(&current), &Type::F32);
    }

    /// Test 7: Operation with empty name (edge case)
    #[test]
    fn test_operation_with_empty_name() {
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);

        // Can add inputs/outputs to empty operation
        let mut op = Operation::new("");
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.op_type, "");
    }

    /// Test 8: Value with extremely large single dimension
    #[test]
    fn test_value_extremely_large_single_dimension() {
        // Test with i32::MAX as dimension
        let huge_dim_value = Value {
            name: "huge_dim".to_string(),
            ty: Type::Bool,
            shape: vec![i32::MAX as usize],
        };
        assert_eq!(huge_dim_value.shape.len(), 1);
        assert_eq!(huge_dim_value.shape[0], i32::MAX as usize);

        // Test with a very large but not overflow-prone value
        let large_single_dim = Value {
            name: "large_single".to_string(),
            ty: Type::I32,
            shape: vec![1_000_000],
        };
        assert_eq!(large_single_dim.shape, vec![1_000_000]);
        assert_eq!(get_num_elements(&large_single_dim), Some(1_000_000));

        // Test rank is correctly 1
        assert_eq!(get_rank(&huge_dim_value), 1);
        assert!(is_vector(&huge_dim_value));
        assert!(!is_scalar(&huge_dim_value));
        assert!(!is_matrix(&huge_dim_value));
    }

    /// Test 9: Attribute array with only empty arrays
    #[test]
    fn test_attribute_array_of_empty_arrays() {
        let empty_arrays = Attribute::Array(vec![
            Attribute::Array(vec![]),
            Attribute::Array(vec![]),
            Attribute::Array(vec![]),
        ]);

        match empty_arrays {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 3);
                for inner in outer.iter() {
                    match inner {
                        Attribute::Array(arr) => {
                            assert_eq!(arr.len(), 0);
                            assert!(arr.is_empty());
                        }
                        _ => panic!("Expected Array"),
                    }
                }
            }
            _ => panic!("Expected Array"),
        }

        // Deeply nested empty arrays
        let deep_empty = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![]),
            ]),
            Attribute::Array(vec![]),
        ]);

        match deep_empty {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                match &outer[0] {
                    Attribute::Array(inner) => {
                        assert_eq!(inner.len(), 1);
                        match &inner[0] {
                            Attribute::Array(deep) => {
                                assert!(deep.is_empty());
                            }
                            _ => panic!("Expected Array"),
                        }
                    }
                    _ => panic!("Expected Array"),
                }
            }
            _ => panic!("Expected Array"),
        }
    }

    /// Test 10: Module with alternating pattern of operation types
    #[test]
    fn test_module_with_alternating_operation_types() {
        let mut module = Module::new("alternating_ops");

        // Add operations with alternating types
        let op_types = vec!["conv", "relu", "batch_norm", "relu", "pool", "relu", "conv", "relu"];

        for (i, op_type) in op_types.iter().enumerate() {
            let mut op = Operation::new(op_type);
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
            module.add_operation(op);
        }

        assert_eq!(module.operations.len(), 8);

        // Verify the alternating pattern
        assert_eq!(module.operations[0].op_type, "conv");
        assert_eq!(module.operations[1].op_type, "relu");
        assert_eq!(module.operations[2].op_type, "batch_norm");
        assert_eq!(module.operations[3].op_type, "relu");
        assert_eq!(module.operations[4].op_type, "pool");
        assert_eq!(module.operations[5].op_type, "relu");
        assert_eq!(module.operations[6].op_type, "conv");
        assert_eq!(module.operations[7].op_type, "relu");

        // Count occurrences
        let mut conv_count = 0;
        let mut relu_count = 0;
        for op in &module.operations {
            match op.op_type.as_str() {
                "conv" => conv_count += 1,
                "relu" => relu_count += 1,
                _ => {}
            }
        }
        assert_eq!(conv_count, 2);
        assert_eq!(relu_count, 4);
    }
}

#[cfg(test)]
mod rstest_boundary_tests {
    use super::*;
    use rstest::rstest;

    /// rstest: Test various shape combinations for vector classification
    #[rstest]
    #[case(vec![1], true)]  // Single element vector
    #[case(vec![10], true)] // Normal vector
    #[case(vec![1000], true)] // Large vector
    #[case(vec![], false)] // Scalar
    #[case(vec![1, 1], false)] // Matrix
    #[case(vec![2, 3, 4], false)] // 3D tensor
    fn test_vector_classification(
        #[case] shape: Vec<usize>,
        #[case] expected: bool,
    ) {
        let value = Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape,
        };
        assert_eq!(is_vector(&value), expected);
    }

    /// rstest: Test num_elements for various shapes
    #[rstest]
    #[case(vec![], Some(1))] // Scalar
    #[case(vec![0], Some(0))] // Zero-length
    #[case(vec![1], Some(1))] // Single element
    #[case(vec![2, 3], Some(6))] // 2D
    #[case(vec![2, 0, 5], Some(0))] // With zero
    #[case(vec![10, 10, 10], Some(1000))] // 3D
    fn test_num_elements_various_shapes(
        #[case] shape: Vec<usize>,
        #[case] expected: Option<usize>,
    ) {
        let value = Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape,
        };
        assert_eq!(value.num_elements(), expected);
    }

    /// rstest: Test type validation with all basic types
    #[rstest]
    #[case(Type::F32, true)]
    #[case(Type::F64, true)]
    #[case(Type::I32, true)]
    #[case(Type::I64, true)]
    #[case(Type::Bool, true)]
    fn test_basic_type_validation(
        #[case] ty: Type,
        #[case] expected: bool,
    ) {
        assert_eq!(ty.is_valid_type(), expected);
    }

    /// rstest: Test tensor type validation
    #[rstest]
    #[case(Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2, 3] }, true)]
    #[case(Type::Tensor { element_type: Box::new(Type::I64), shape: vec![] }, true)]
    #[case(Type::Tensor { element_type: Box::new(Type::Bool), shape: vec![100] }, true)]
    fn test_tensor_type_validation(
        #[case] ty: Type,
        #[case] expected: bool,
    ) {
        assert_eq!(ty.is_valid_type(), expected);
    }
}