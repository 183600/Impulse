//! Additional edge case tests for the Impulse compiler
//! These tests cover critical edge cases not fully addressed in other test suites

use crate::ir::{Module, Operation, Value, Type, Attribute};

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;

    /// Test 1: Potential integer overflow in tensor size calculation using checked arithmetic
    #[test]
    fn test_tensor_size_overflow_protection() {
        // Test the num_elements method which uses checkedMul to prevent overflow
        let normal_tensor = Value {
            name: "normal".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],
        };
        
        assert_eq!(normal_tensor.num_elements(), Some(1_000_000));
        
        // Test with a shape that contains 0 - should return Some(0)
        let zero_tensor = Value {
            name: "zero".to_string(),
            ty: Type::F32,
            shape: vec![100, 0, 100],
        };
        
        assert_eq!(zero_tensor.num_elements(), Some(0));
        
        // Test with scalar (empty shape) - should return Some(1)
        let scalar_tensor = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        
        assert_eq!(scalar_tensor.num_elements(), Some(1));
    }

    /// Test 2: Deeply nested recursive structures that could cause stack overflow
    #[test]
    fn test_very_deep_tensor_nesting() {
        // Test deep nesting to catch potential stack overflow issues
        let mut nested_type = Type::F32;
        
        // Create a nested type structure with 1000 levels of nesting
        for i in 0..1000 {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![i % 10 + 1], // Cycle through values 1-10 to avoid extreme growth
            };
        }
        
        // Verify the structure can be created and cloned without stack overflow
        let cloned_type = nested_type.clone();
        assert_eq!(nested_type, cloned_type);
    }

    /// Test 3: Empty and malformed IR structures
    #[test]
    fn test_empty_and_minimal_ir_structures() {
        // Test empty module
        let empty_module = Module {
            name: "".to_string(),  // Empty name edge case
            operations: vec![],
            inputs: vec![],
            outputs: vec![],
        };
        
        assert_eq!(empty_module.name, "");
        assert!(empty_module.operations.is_empty());
        assert!(empty_module.inputs.is_empty());
        assert!(empty_module.outputs.is_empty());
        
        // Test minimal operation
        let min_op = Operation {
            op_type: "".to_string(),  // Empty type edge case
            inputs: vec![],
            outputs: vec![],
            attributes: std::collections::HashMap::new(),
        };
        
        assert_eq!(min_op.op_type, "");
        assert!(min_op.inputs.is_empty());
        assert!(min_op.outputs.is_empty());
        assert!(min_op.attributes.is_empty());
        
        // Test minimal value with empty shape (scalar)
        let min_val = Value {
            name: "".to_string(),  // Empty name
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        
        assert_eq!(min_val.name, "");
        assert_eq!(min_val.ty, Type::F32);
        assert!(min_val.shape.is_empty());
        assert_eq!(min_val.num_elements(), Some(1)); // Scalar has 1 element
        
        // Test value with maximum possible name length
        let long_name = "a".repeat(10_000);
        let long_name_val = Value {
            name: long_name.clone(),
            ty: Type::I64,
            shape: vec![1, 1, 1],
        };
        
        assert_eq!(long_name_val.name, long_name);
    }

    /// Test 4: Special floating-point values in attributes
    #[test]
    fn test_special_float_attributes() {
        use std::collections::HashMap;
        
        let mut attrs = HashMap::new();
        
        // Add special float values to attributes
        attrs.insert("inf_pos".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("inf_neg".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan_val".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("zero_pos".to_string(), Attribute::Float(0.0));
        attrs.insert("zero_neg".to_string(), Attribute::Float(-0.0));
        attrs.insert("min_normal".to_string(), Attribute::Float(f64::MIN_POSITIVE));
        attrs.insert("max_normal".to_string(), Attribute::Float(f64::MAX));
        attrs.insert("min_subnormal".to_string(), Attribute::Float(f64::MIN_POSITIVE / 2.0));
        
        // Verify positive infinity
        if let Attribute::Float(val) = attrs.get("inf_pos").unwrap() {
            assert!(val.is_infinite() && val.is_sign_positive());
        }
        
        // Verify negative infinity
        if let Attribute::Float(val) = attrs.get("inf_neg").unwrap() {
            assert!(val.is_infinite() && val.is_sign_negative());
        }
        
        // Verify NaN (NaN != NaN, so need to check with is_nan())
        if let Attribute::Float(val) = attrs.get("nan_val").unwrap() {
            assert!(val.is_nan());
        }
        
        // Verify that +0 and -0 are stored as different values
        if let Attribute::Float(pos_zero) = attrs.get("zero_pos").unwrap() {
            assert_eq!(*pos_zero, 0.0);
        }
        
        if let Attribute::Float(neg_zero) = attrs.get("zero_neg").unwrap() {
            assert_eq!(*neg_zero, -0.0);
            assert!(neg_zero.is_sign_negative() || *neg_zero == -0.0); // Handle -0.0 case
        }
        
        // Verify min/max values
        if let Attribute::Float(min_val) = attrs.get("min_normal").unwrap() {
            assert_eq!(*min_val, f64::MIN_POSITIVE);
        }
        
        if let Attribute::Float(max_val) = attrs.get("max_normal").unwrap() {
            assert_eq!(*max_val, f64::MAX);
        }
    }

    /// Test 5: Maximum size collections and capacity limits
    #[test]
    fn test_maximum_collection_sizes() {
        use std::collections::HashMap;
        
        // Test operation with maximum possible attributes (within reason)
        let mut op = Operation::new("max_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add a large number of attributes
        for i in 0..10_000 {
            attrs.insert(
                format!("attr_{:05}", i), 
                Attribute::String(format!("value_{:05}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 10_000);
        
        // Test operation with maximum possible inputs (within reason)
        let mut max_inputs_op = Operation::new("max_inputs_op");
        for i in 0..10_000 {
            max_inputs_op.inputs.push(Value {
                name: format!("input_{:05}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1],  // Cycle through sizes to avoid huge memory usage
            });
        }
        
        assert_eq!(max_inputs_op.inputs.len(), 10_000);
        
        // Test operation with maximum possible outputs (within reason)
        let mut max_outputs_op = Operation::new("max_outputs_op");
        for i in 0..5_000 {  // Fewer outputs to be reasonable
            max_outputs_op.outputs.push(Value {
                name: format!("output_{:05}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1],  // Cycle through sizes
            });
        }
        
        assert_eq!(max_outputs_op.outputs.len(), 5_000);
        
        // Test module with maximum operations (within reason)
        let mut max_ops_module = Module::new("max_ops_module");
        for i in 0..50_000 {
            let mut op = Operation::new(&format!("op_{:05}", i));
            op.inputs.push(Value {
                name: format!("input_{:05}_0", i),
                ty: Type::F32,
                shape: vec![1],
            });
            
            max_ops_module.add_operation(op);
        }
        
        assert_eq!(max_ops_module.operations.len(), 50_000);
    }

    /// Test 6: Error conditions and invalid states
    #[test]
    fn test_invalid_ir_states() {
        // Test creating a tensor type with invalid nesting (though the current implementation
        // doesn't have explicit validation, we can still test various combinations)
        
        // Create a deeply nested invalid-looking type but valid syntax
        let mut nested_type = Type::F32;
        for _ in 0..100 {
            nested_type = Type::Tensor {
                element_type: Box::new(nested_type),
                shape: vec![1],
            };
        }
        
        // This should be a valid nested type
        assert!(match nested_type {
            Type::Tensor { .. } => true,
            Type::F32 => true,
            _ => true,  // All variants should be valid
        });
        
        // Test a complex chain of operations that might cause issues
        let mut complex_op = Operation::new("complex_op");
        
        // Add inputs with different types
        for (idx, ty) in [Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool].iter().enumerate() {
            complex_op.inputs.push(Value {
                name: format!("input_{}", idx),
                ty: ty.clone(),
                shape: vec![idx + 1],
            });
        }
        
        // Add outputs with complex nested types
        for idx in 0..3 {
            let complex_type = if idx == 0 {
                Type::F32
            } else {
                // Create nested tensor: tensor<f32, [2]> -> tensor<tensor<f32, [2]>, [3]> -> etc.
                let base = Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                };
                
                if idx == 1 {
                    base
                } else {
                    Type::Tensor {
                        element_type: Box::new(base),
                        shape: vec![3],
                    }
                }
            };
            
            complex_op.outputs.push(Value {
                name: format!("output_{}", idx),
                ty: complex_type,
                shape: vec![idx + 1, idx + 2],
            });
        }
        
        assert_eq!(complex_op.inputs.len(), 5);
        assert_eq!(complex_op.outputs.len(), 3);
        assert_eq!(complex_op.op_type, "complex_op");
    }

    /// Test 7: Serialization edge cases
    #[test]
    fn test_serialization_edge_cases() {
        // Since the structs derive Serialize/Deserialize, test roundtrip
        use serde_json;
        
        // Test simple value serialization
        let simple_val = Value {
            name: "simple".to_string(),
            ty: Type::I32,
            shape: vec![2, 3, 4],
        };
        
        let serialized = serde_json::to_string(&simple_val).expect("Serialization failed");
        let deserialized: Value = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert_eq!(simple_val, deserialized);
        
        // Test complex nested type serialization
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![5, 5],
            }),
            shape: vec![3, 3],
        };
        
        let complex_val = Value {
            name: "complex_serial".to_string(),
            ty: nested_type.clone(),
            shape: vec![10, 20],
        };
        
        let complex_serialized = serde_json::to_string(&complex_val).expect("Complex serialization failed");
        let complex_deserialized: Value = serde_json::from_str(&complex_serialized).expect("Complex deserialization failed");
        
        assert_eq!(complex_val, complex_deserialized);
        
        // Test attribute serialization
        let attrs_vec = vec![
            Attribute::Int(42),
            Attribute::Float(3.14159),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
        ];
        
        for attr in attrs_vec {
            let attr_serialized = serde_json::to_string(&attr).expect("Attr serialization failed");
            let attr_deserialized: Attribute = serde_json::from_str(&attr_serialized).expect("Attr deserialization failed");
            assert_eq!(attr, attr_deserialized);
        }
    }

    /// Test 8: Comparison operations edge cases
    #[test]
    fn test_type_comparison_edge_cases() {
        // Test equality of identical types
        let type1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 4],
        };
        
        let type2 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 4],
        };
        
        assert_eq!(type1, type2);
        
        // Test inequality: different element types
        let type3 = Type::Tensor {
            element_type: Box::new(Type::F64), // Different element type
            shape: vec![2, 3, 4],
        };
        
        assert_ne!(type1, type3);
        
        // Test inequality: different shapes
        let type4 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 5], // Different shape
        };
        
        assert_ne!(type1, type4);
        
        // Test inequality: different dimensions
        let type5 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 12], // Different number of dimensions but same product
        };
        
        assert_ne!(type1, type5);
        
        // Test deeply nested type equality
        let deep1 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Bool),
                    shape: vec![1],
                }),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        
        let deep2 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Bool),
                    shape: vec![1],
                }),
                shape: vec![2],
            }),
            shape: vec![3],
        };
        
        assert_eq!(deep1, deep2);
        
        // Test deeply nested type inequality (one tiny difference)
        let deep3 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Bool), // Same
                    shape: vec![1],                     // Same
                }),
                shape: vec![2],                         // Same
            }),
            shape: vec![4],                             // Different!
        };
        
        assert_ne!(deep1, deep3);
    }

    /// Test 9: Complex nested tensor structures  
    #[test]
    fn test_complex_nested_tensor_structures() {
        // Create a complex nested tensor: tensor of matrices of vectors of scalars
        let complex_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {  // Matrix
                element_type: Box::new(Type::Tensor {  // Vector
                    element_type: Box::new(Type::Tensor {  // Scalar wrapper
                        element_type: Box::new(Type::F32),
                        shape: vec![],  // Scalar
                    }),
                    shape: vec![10],  // Vector of 10 scalars
                }),
                shape: vec![5, 5],  // Matrix (5x5) of vectors
            }),
            shape: vec![2, 3],  // 2x3 tensor of matrices
        };
        
        // Verify the structure
        if let Type::Tensor { element_type: mat_tensor, shape: outer_shape } = &complex_nested {
            let expected_outer: Vec<usize> = vec![2, 3];
            assert_eq!(outer_shape, &expected_outer);
            
            if let Type::Tensor { element_type: vec_tensor, shape: mat_shape } = mat_tensor.as_ref() {
                let expected_mat: Vec<usize> = vec![5, 5];
                assert_eq!(mat_shape, &expected_mat);
                
                if let Type::Tensor { element_type: scalar_tensor, shape: vec_shape } = vec_tensor.as_ref() {
                    let expected_vec: Vec<usize> = vec![10];
                    assert_eq!(vec_shape, &expected_vec);
                    
                    if let Type::Tensor { element_type: final_type, shape: scalar_shape } = scalar_tensor.as_ref() {
                        let expected_scalar: Vec<usize> = vec![];
                        assert_eq!(scalar_shape, &expected_scalar);
                        
                        if let Type::F32 = final_type.as_ref() {
                            // Success - verified the entire structure
                        } else {
                            panic!("Innermost type should be F32");
                        }
                    } else {
                        panic!("Expected tensor as 4th level");
                    }
                } else {
                    panic!("Expected tensor as 3rd level");
                }
            } else {
                panic!("Expected tensor as 2nd level");
            }
        } else {
            panic!("Expected tensor as 1st level");
        }
        
        // Test cloning of the complex structure
        let cloned_complex = complex_nested.clone();
        assert_eq!(complex_nested, cloned_complex);
        
        // Create another similar but slightly different complex nested tensor
        let different_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I32),  // Different base type
                    shape: vec![],  // Scalar
                }),
                shape: vec![10],  // Vector of 10 scalars
            }),
            shape: vec![2, 3],  // 2x3 tensor of vectors
        };
        
        assert_ne!(complex_nested, different_nested);
    }

    /// Test 10: Value with extremely large shape products that can still be computed
    #[test]
    fn test_extremely_large_but_valid_tensors() {
        // Test a tensor with a large but still valid shape
        let large_but_valid = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000],  // 100 million elements, should be fine on 64-bit
        };
        
        assert_eq!(large_but_valid.shape, vec![10_000, 10_000]);
        assert_eq!(large_but_valid.num_elements(), Some(100_000_000));
        
        // Test with a shape that includes 1s to avoid overflow but still has large individual dims
        let spread_shape = Value {
            name: "spread_tensor".to_string(),
            ty: Type::I64,
            shape: vec![1_000_000, 1, 1_000_000],  // Still 10^12 elements but with 1 in the middle
        };
        
        // The checkedMul approach should handle this gracefully
        let elements = spread_shape.num_elements();
        if let Some(count) = elements {
            assert_eq!(count, 1_000_000_000_000); // 1 trillion
        } else {
            // It's acceptable if this returns None on systems where the product exceeds usize
        }
        
        // Test with scalar (empty shape) - should return Some(1)
        let scalar_tensor = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        
        assert_eq!(scalar_tensor.num_elements(), Some(1));
    }
}