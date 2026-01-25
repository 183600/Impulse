//! New edge case tests for the Impulse compiler
//! This file contains additional test cases focusing on boundary conditions and error scenarios

#[cfg(test)]
mod new_edge_case_tests {
    use rstest::*;
    use crate::{ir::{Module, Value, Type, Operation, Attribute}, ImpulseCompiler};

    /// Test 1: Operations with maximum possible string lengths
    #[test]
    fn test_max_string_length_operations() {
        // Test creating an operation with a very long name
        let long_name = "x".repeat(1_000_000); // 1 million character name
        let op = Operation::new(&long_name);
        
        assert_eq!(op.op_type.len(), 1_000_000);
        assert_eq!(op.op_type.chars().count(), 1_000_000);
        
        // Test creating a value with a very long name
        let long_value_name = "v".repeat(1_000_000);
        let value = Value {
            name: long_value_name,
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name.len(), 1_000_000);
        
        // Test creating a module with a very long name
        let long_module_name = "m".repeat(1_000_000);
        let module = Module::new(&long_module_name);
        
        assert_eq!(module.name.len(), 1_000_000);
    }

    /// Test 2: Tensor shapes with prime numbers and unusual patterns
    #[rstest]
    #[case(vec![2, 3, 5, 7, 11], 2310)] // Product of first few primes
    #[case(vec![13, 17, 19], 4199)]     // Product of larger primes
    #[case(vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 1024)] // 2^10
    #[case(vec![1, 1000000], 1000000)]   // Large dimension with 1
    #[case(vec![9999991, 2], 19999982)]  // Large prime times 2
    fn test_prime_tensor_shapes(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
        let value = Value {
            name: "prime_shape_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        let actual_size: usize = value.shape.iter().product();
        assert_eq!(actual_size, expected_size);
    }

    /// Test 3: Special floating-point values in attributes
    #[test]
    fn test_special_floating_point_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("fp_special_op");
        let mut attrs = HashMap::new();
        
        // Test special floating point values that could appear in neural networks
        attrs.insert("positive_infinity".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("negative_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("epsilon_value".to_string(), Attribute::Float(f64::EPSILON));
        attrs.insert("min_positive".to_string(), Attribute::Float(f64::MIN_POSITIVE));
        attrs.insert("max_value".to_string(), Attribute::Float(f64::MAX));
        attrs.insert("min_value".to_string(), Attribute::Float(f64::MIN));
        attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 8);
        
        // Verify specific values (handling NaN specially)
        if let Attribute::Float(inf_val) = op.attributes.get("positive_infinity").unwrap() {
            assert!(inf_val.is_infinite() && inf_val.is_sign_positive());
        }
        
        if let Attribute::Float(neg_inf_val) = op.attributes.get("negative_infinity").unwrap() {
            assert!(neg_inf_val.is_infinite() && neg_inf_val.is_sign_negative());
        }
        
        if let Attribute::Float(nan_val) = op.attributes.get("nan_value").unwrap() {
            assert!(nan_val.is_nan());
        }
        
        if let Attribute::Float(eps_val) = op.attributes.get("epsilon_value").unwrap() {
            assert_eq!(*eps_val, f64::EPSILON);
        }
    }

    /// Test 4: Recursive nested tensor types with various depths
    #[test]
    fn test_recursive_tensor_depth_variations() {
        // Create tensor types with varying depths
        let base_type = Type::F32;
        
        // Depth 1: Tensor<F32, [2]>
        let tensor1 = Type::Tensor {
            element_type: Box::new(base_type.clone()),
            shape: vec![2],
        };
        
        // Depth 2: Tensor<Tensor<F32, [2]>, [3]>
        let tensor2 = Type::Tensor {
            element_type: Box::new(tensor1),
            shape: vec![3],
        };
        
        // Depth 3: Tensor<Tensor<Tensor<F32, [2]>, [3]>, [4]>
        let tensor3 = Type::Tensor {
            element_type: Box::new(tensor2),
            shape: vec![4],
        };
        
        // Depth 4: Go one level deeper
        let deep_tensor = Type::Tensor {
            element_type: Box::new(tensor3),
            shape: vec![5],
        };
        
        // Verify that deep nesting doesn't cause issues
        let cloned = deep_tensor.clone();
        assert_eq!(deep_tensor, cloned);
        
        // Attempt to serialize/deserialize to test stability
        // This tests internal structure without actually serializing
        match &deep_tensor {
            Type::Tensor { shape, element_type } => {
                assert_eq!(shape, &vec![5]);
                
                match element_type.as_ref() {
                    Type::Tensor { shape: _, element_type: inner_elem } => {
                        match inner_elem.as_ref() {
                            Type::Tensor { shape: _, element_type: deeper_elem } => {
                                match deeper_elem.as_ref() {
                                    Type::Tensor { shape: _, element_type: deepest_elem } => {
                                        match deepest_elem.as_ref() {
                                            Type::F32 => {}, // Successfully went 4 levels deep
                                            _ => panic!("Expected F32 at deepest level"),
                                        }
                                    },
                                    _ => panic!("Expected Tensor at 3rd level"),
                                }
                            },
                            _ => panic!("Expected Tensor at 2nd level"),
                        }
                    },
                    _ => panic!("Expected Tensor at 1st level"),
                }
            },
            _ => panic!("Expected Tensor at top level"),
        }
    }

    /// Test 5: Edge cases with tensor shapes containing 1s
    #[rstest]
    #[case(vec![1], 1)]                    // Singleton
    #[case(vec![1, 1], 1)]                 // 2D singleton
    #[case(vec![1, 1, 1], 1)]             // 3D singleton
    #[case(vec![1, 100], 100)]             // Leading singleton
    #[case(vec![100, 1], 100)]             // Trailing singleton  
    #[case(vec![1, 50, 1], 50)]            // Middle singleton
    #[case(vec![10, 1, 10], 100)]          // Middle singleton in 2D
    #[case(vec![1, 1, 100], 100)]          // Two leading singletons
    #[case(vec![100, 1, 1], 100)]          // Two trailing singletons
    #[case(vec![1, 2, 1, 3, 1], 6)]       // Alternating singletons
    fn test_singleton_dimension_tensor_shapes(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
        let value = Value {
            name: "singleton_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        
        let actual_size: usize = value.shape.iter().product();
        assert_eq!(actual_size, expected_size);
    }

    /// Test 6: Operations with mixed attribute types and complex structures
    #[test]
    fn test_complex_attribute_mix() {
        use std::collections::HashMap;
        
        let mut main_op = Operation::new("complex_mixed_op");
        let mut main_attrs = HashMap::new();
        
        // Mix of all attribute types in a complex structure
        main_attrs.insert("simple_int".to_string(), Attribute::Int(42));
        main_attrs.insert("simple_float".to_string(), Attribute::Float(3.14));
        main_attrs.insert("simple_bool".to_string(), Attribute::Bool(true));
        main_attrs.insert("simple_string".to_string(), Attribute::String("hello".to_string()));
        
        // Nested arrays with mixed types
        let nested_array1 = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::Bool(true),
            Attribute::String("nested1".to_string()),
        ]);
        
        let nested_array2 = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(10),
                Attribute::Float(20.5),
            ]),
            Attribute::Bool(false),
        ]);
        
        main_attrs.insert("mixed_array1".to_string(), nested_array1);
        main_attrs.insert("mixed_array2".to_string(), nested_array2);
        
        // Add even more complex nesting
        let very_complex = Attribute::Array(vec![
            Attribute::Int(100),
            Attribute::Array(vec![
                Attribute::String("deeply".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Float(99.99),
                ])
            ]),
            Attribute::String("outer".to_string()),
        ]);
        
        main_attrs.insert("very_complex_nesting".to_string(), very_complex);
        
        main_op.attributes = main_attrs;
        
        assert_eq!(main_op.attributes.len(), 7);
        
        // Verify complex nested structure
        match main_op.attributes.get("very_complex_nesting").unwrap() {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 3);
                
                // First element should be Int(100)
                match &outer[0] {
                    Attribute::Int(100) => {},
                    _ => panic!("Expected Int(100)"),
                }
                
                // Second element should be an Array
                match &outer[1] {
                    Attribute::Array(deep) => {
                        assert_eq!(deep.len(), 2);
                        
                        // First element of deep array should be String("deeply")
                        match &deep[0] {
                            Attribute::String(s) if s == "deeply" => {},
                            _ => panic!("Expected String('deeply')"),
                        }
                        
                        // Second element should be another Array
                        match &deep[1] {
                            Attribute::Array(deepest) => {
                                assert_eq!(deepest.len(), 2);
                                
                                // First element of deepest array should be Bool(true)
                                match &deepest[0] {
                                    Attribute::Bool(true) => {},
                                    _ => panic!("Expected Bool(true)"),
                                }
                            },
                            _ => panic!("Expected nested Array"),
                        }
                    },
                    _ => panic!("Expected Array"),
                }
            },
            _ => panic!("Expected Array at top level"),
        }
    }

    /// Test 7: Memory allocation stress with many small objects
    #[test]
    fn test_memory_allocation_stress() {
        let mut modules = Vec::new();
        
        // Create many small modules to stress allocation/deallocation
        for i in 0..10_000 {
            let mut module = Module::new(&format!("stress_module_{}", i));
            
            // Add a few operations to each module
            for j in 0..5 {
                let mut op = Operation::new(&format!("op_{}_{}", i, j));
                
                // Add a value to each operation
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![1],
                });
                
                module.add_operation(op);
            }
            
            modules.push(module);
        }
        
        assert_eq!(modules.len(), 10_000);
        assert_eq!(modules[0].operations.len(), 5);
        
        // Clean up to test deallocation
        drop(modules);
        
        // Create a new object to ensure memory operations worked correctly
        let new_module = Module::new("post_stress_test");
        assert_eq!(new_module.name, "post_stress_test");
    }

    /// Test 8: Empty collections and boundary conditions
    #[test]
    fn test_empty_collections_boundaries() {
        // Test empty module
        let empty_module = Module::new("");
        assert_eq!(empty_module.name, "");
        assert_eq!(empty_module.operations.len(), 0);
        assert_eq!(empty_module.inputs.len(), 0);
        assert_eq!(empty_module.outputs.len(), 0);
        
        // Test operation with all empty collections
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        assert_eq!(empty_op.inputs.len(), 0);
        assert_eq!(empty_op.outputs.len(), 0);
        assert_eq!(empty_op.attributes.len(), 0);
        
        // Test value with empty shape (scalar)
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty means scalar
        };
        assert_eq!(scalar_value.shape.len(), 0);
        assert!(scalar_value.shape.is_empty());
        
        // Test tensor product of empty shape (should be 1 for scalars)
        let size: usize = scalar_value.shape.iter().product();
        assert_eq!(size, 1);  // Product of empty iterator is 1
        
        // Test with empty strings everywhere
        let string_test_val = Value {
            name: "".to_string(),
            ty: Type::I32,
            shape: vec![],
        };
        assert_eq!(string_test_val.name, "");
        assert_eq!(string_test_val.shape.len(), 0);
    }

    /// Test 9: Compiler initialization under resource constraints simulation
    #[test]
    fn test_compiler_under_constraints_simulation() {
        // Test creating multiple compilers rapidly to simulate resource constraints
        let compilers: Vec<_> = (0..100).map(|_| ImpulseCompiler::new()).collect();
        
        // Verify all were created properly
        for (i, compiler) in compilers.iter().enumerate() {
            let _ = i; // Suppress unused variable warning
            assert_eq!(compiler.passes.passes.len(), 0);
            assert_eq!(compiler.frontend.name(), "Frontend");
        }
        
        assert_eq!(compilers.len(), 100);
        
        // Drop all to test cleanup
        drop(compilers);
        
        // Create one more to ensure resources were freed
        let final_compiler = ImpulseCompiler::new();
        assert_eq!(final_compiler.passes.passes.len(), 0);
    }

    /// Test 10: Boolean tensor operations edge cases
    #[rstest]
    #[case(vec![], true)]            // Scalar boolean (product of empty is true for "all" operation)
    #[case(vec![0], false)]          // Zero-sized boolean tensor
    #[case(vec![1], true)]           // Single-element true tensor 
    #[case(vec![2], true)]           // Two elements (would need actual values to determine result)
    #[case(vec![1, 1], true)]        // 2D scalar equivalent
    #[case(vec![2, 2], true)]        // 2x2 tensor (assumes all true for product)
    #[case(vec![0, 5], false)]       // Contains zero dimension
    #[case(vec![5, 0], false)]       // Contains zero dimension
    fn test_boolean_tensor_shapes(#[case] shape: Vec<usize>, #[case] expected_behavior: bool) {
        let bool_value = Value {
            name: "bool_tensor".to_string(),
            ty: Type::Bool,
            shape: shape.clone(),
        };
        
        assert_eq!(bool_value.shape, shape);
        assert_eq!(bool_value.ty, Type::Bool);
        
        // For boolean tensors, a zero dimension would result in "empty" tensor
        let has_zero_dim = shape.iter().any(|&dim| dim == 0);
        if has_zero_dim {
            // Tensors with zero dimensions have 0 elements
            let total_elements: usize = bool_value.shape.iter().product();
            assert_eq!(total_elements, 0);
        } else {
            // Non-zero dimensional tensors have positive element count
            let total_elements: usize = bool_value.shape.iter().product();
            assert!(total_elements > 0 || shape.is_empty()); // Empty shape is scalar (1 element)
        }
        
        // Use the expected_behavior parameter in validation
        let is_expected_to_have_elements = !shape.iter().any(|&dim| dim == 0) || shape.is_empty();
        assert_eq!(is_expected_to_have_elements, expected_behavior);
    }
}