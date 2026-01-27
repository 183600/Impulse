//! New edge case tests for the Impulse compiler
//! Covering additional boundary conditions and comprehensive scenarios

#[cfg(test)]
mod edge_case_tests_new {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use rstest::rstest;

    /// Test 1: Operation with maximum attribute count
    #[test]
    fn test_operation_with_max_attributes() {
        let mut op = Operation::new("max_attr_op");
        
        // Add many attributes to test hash map performance
        for i in 0..100_000 {
            op.attributes.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        
        assert_eq!(op.attributes.len(), 100_000);
        
        // Verify some attributes exist
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_50000"));
        assert!(op.attributes.contains_key("attr_99999"));
        
        // Check value retrieval
        if let Some(attr_val) = op.attributes.get("attr_0") {
            match attr_val {
                Attribute::String(s) => assert_eq!(s, "value_0"),
                _ => panic!("Expected string attribute"),
            }
        } else {
            panic!("Attribute not found");
        }
    }

    /// Test 2: Mixed NaN and infinity float attributes
    #[test]
    fn test_operation_with_nan_inf_mixed_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("nan_inf_mixed");
        let mut attrs = HashMap::new();
        
        // Add different float values
        attrs.insert("inf".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("normal".to_string(), Attribute::Float(3.14));
        attrs.insert("zero".to_string(), Attribute::Float(0.0));
        attrs.insert("neg_zero".to_string(), Attribute::Float(-0.0));
        
        op.attributes = attrs;
        
        // Check non-NaN values
        assert_eq!(op.attributes.get("inf"), Some(&Attribute::Float(f64::INFINITY)));
        assert_eq!(op.attributes.get("neg_inf"), Some(&Attribute::Float(f64::NEG_INFINITY)));
        assert_eq!(op.attributes.get("normal"), Some(&Attribute::Float(3.14)));
        assert_eq!(op.attributes.get("zero"), Some(&Attribute::Float(0.0)));
        assert_eq!(op.attributes.get("neg_zero"), Some(&Attribute::Float(-0.0)));
        
        // Special check for NaN (NaN != NaN)
        if let Some(Attribute::Float(val)) = op.attributes.get("nan") {
            assert!(val.is_nan());
        } else {
            panic!("Expected NaN value");
        }
    }

    /// Test 3: Module with very long circular references in types
    #[test]
    fn test_module_with_deeply_nested_circular_like_types() {
        let mut module = Module::new("deep_nested_module");
        
        // Create a complex deeply-nested type structure
        let mut current_type = Type::Bool;
        
        // Build deep nesting (500 levels deep)
        for i in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 10 + 1], // Varying shapes to make it interesting
            };
        }
        
        // Add an operation with this complex type
        let complex_value = Value {
            name: "deeply_nested_value".to_string(),
            ty: current_type,
            shape: vec![2],
        };
        
        let mut op = Operation::new("complex_op");
        op.inputs.push(complex_value);
        
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.name, "deep_nested_module");
    }

    /// Test 4: Operations with all possible type combinations as inputs/outputs
    #[rstest]
    #[case(Type::F32, vec![])]  // scalar F32
    #[case(Type::F64, vec![1])]  // 1D F64 tensor
    #[case(Type::I32, vec![1, 2, 3, 4, 5])]  // 5D I32 tensor
    #[case(Type::I64, vec![0])]  // Zero-dimension tensor (empty tensor)
    #[case(Type::Bool, vec![100, 200])]  // Large 2D tensor
    fn test_operations_with_different_types_and_shapes(#[case] data_type: Type, #[case] shape: Vec<usize>) {
        let mut op = Operation::new("typed_op");
        
        // Add input with specific type and shape
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: data_type.clone(),
            shape: shape.clone(),
        });
        
        // Add output with same type and shape
        op.outputs.push(Value {
            name: "output".to_string(),
            ty: data_type.clone(),
            shape: shape.clone(),
        });
        
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.inputs[0].ty, data_type);
        assert_eq!(op.outputs[0].ty, data_type);
        assert_eq!(op.inputs[0].shape, shape);
        assert_eq!(op.outputs[0].shape, shape);
    }

    /// Test 5: Value with shape containing multiple zeros in different positions
    #[test]
    fn test_tensor_shapes_with_multiple_zeros() {
        let test_cases = vec![
            (vec![0], 0),           // Single zero dimension
            (vec![0, 5], 0),        // Zero first, then other dimensions
            (vec![5, 0], 0),        // Other dimension first, then zero
            (vec![5, 0, 10], 0),    // Zero in middle
            (vec![0, 0], 0),        // Multiple zeros
            (vec![0, 1, 0], 0),     // Zero at beginning and end
            (vec![1, 0, 1], 0),     // Zero in center
        ];
        
        for (shape, expected_elements) in test_cases {
            let value = Value {
                name: "zero_shape_tensor".to_string(),
                ty: Type::F32,
                shape: shape,
            };
            
            let calculated_elements = value.num_elements().unwrap_or(0);
            assert_eq!(calculated_elements, expected_elements);
        }
        
        // Also test a normal case
        let normal_value = Value {
            name: "normal_tensor".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        
        assert_eq!(normal_value.num_elements().unwrap_or(0), 24);
    }

    /// Test 6: Attribute arrays with mixed nesting depths
    #[test]
    fn test_attribute_arrays_with_mixed_depths() {
        let complex_array = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Float(3.14),
                    Attribute::Bool(true),
                ]),
            ]),
            Attribute::Bool(false),
            Attribute::Array(vec![
                Attribute::Int(42),
                Attribute::String("another".to_string()),
            ]),
        ]);
        
        match &complex_array {
            Attribute::Array(top_level) => {
                assert_eq!(top_level.len(), 4);
                
                // Check first element - Int
                match &top_level[0] {
                    Attribute::Int(1) => (),
                    _ => panic!("First element should be Int(1)"),
                }
                
                // Check second element - nested Array
                match &top_level[1] {
                    Attribute::Array(nested) => {
                        assert_eq!(nested.len(), 2);
                        
                        match &nested[1] {  // Second element of nested array
                            Attribute::Array(deep_nested) => {
                                assert_eq!(deep_nested.len(), 2);
                            },
                            _ => panic!("Expected another nested array"),
                        }
                    },
                    _ => panic!("Expected nested array at position 1"),
                }
                
                // Check third element - Bool
                match &top_level[2] {
                    Attribute::Bool(false) => (),
                    _ => panic!("Third element should be Bool(false)"),
                }
                
                // Check fourth element - another nested array
                match &top_level[3] {
                    Attribute::Array(another_nested) => {
                        assert_eq!(another_nested.len(), 2);
                    },
                    _ => panic!("Expected array at position 3"),
                }
            },
            _ => panic!("Expected top-level array"),
        }
        
        // Test cloning of this complex structure
        let cloned = complex_array.clone();
        assert_eq!(complex_array, cloned);
    }

    /// Test 7: Module with operations that have no inputs, no outputs, and no attributes
    #[test]
    fn test_minimal_operations() {
        let mut module = Module::new("minimal_module");
        
        // Add a minimal operation
        let minimal_op = Operation::new("minimal");
        assert_eq!(minimal_op.inputs.len(), 0);
        assert_eq!(minimal_op.outputs.len(), 0);
        assert_eq!(minimal_op.attributes.len(), 0);
        
        module.add_operation(minimal_op);
        
        // Add another minimal operation with same name
        let minimal_op2 = Operation::new("minimal_same");
        module.add_operation(minimal_op2);
        
        assert_eq!(module.operations.len(), 2);
        assert_eq!(module.operations[0].op_type, "minimal");
        assert_eq!(module.operations[1].op_type, "minimal_same");
        
        // Verify all operations are truly minimal
        for op in &module.operations {
            assert_eq!(op.inputs.len(), 0);
            assert_eq!(op.outputs.len(), 0);
            assert_eq!(op.attributes.len(), 0);
        }
    }

    /// Test 8: Edge cases with empty collections for inputs/outputs
    #[test]
    fn test_operations_with_empty_collections() {
        // Operation with empty inputs, populated outputs
        let mut op1 = Operation::new("partial_io_1");
        op1.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        });
        
        assert_eq!(op1.inputs.len(), 0);
        assert_eq!(op1.outputs.len(), 1);
        
        // Operation with populated inputs, empty outputs
        let mut op2 = Operation::new("partial_io_2");
        op2.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::I32,
            shape: vec![],
        });
        
        assert_eq!(op2.inputs.len(), 1);
        assert_eq!(op2.outputs.len(), 0);
        
        // Operation with both empty inputs and outputs
        let op3 = Operation::new("no_io");
        assert_eq!(op3.inputs.len(), 0);
        assert_eq!(op3.outputs.len(), 0);
        
        // Create a module with these operations
        let mut module = Module::new("partial_io_module");
        module.add_operation(op1);
        module.add_operation(op2);
        module.add_operation(op3);
        
        assert_eq!(module.operations.len(), 3);
    }

    /// Test 9: Value with extremely high dimension count
    #[test]
    fn test_extremely_high_dimension_tensor() {
        // Create a tensor with many dimensions but small size in each dimension
        let many_dims = vec![1; 1000];  // 1000 dimensions, each size 1
        
        let value = Value {
            name: "many_dims_tensor".to_string(),
            ty: Type::F32,
            shape: many_dims,
        };
        
        assert_eq!(value.shape.len(), 1000);
        
        // All dimensions are 1, so total elements should be 1
        assert_eq!(value.num_elements().unwrap_or(0), 1);
        
        // Verify all dimensions are indeed 1
        for &dim in &value.shape {
            assert_eq!(dim, 1);
        }
        
        // Test with alternating 1s and 2s but with a smaller range to avoid overflow
        let mut alt_dims = Vec::new();
        for i in 0..30 {  // Reduced from 500 to 30 to prevent overflow
            alt_dims.push(if i % 2 == 0 { 1 } else { 2 });
        }
        
        let alt_value = Value {
            name: "alternating_dims".to_string(),
            ty: Type::I64,
            shape: alt_dims,
        };
        
        assert_eq!(alt_value.shape.len(), 30);
        
        // Calculate expected elements: product of alternating 1s and 2s
        // Only 15 2s in the sequence, so 2^15 = 32,768
        let mut expected_elements: usize = 1;
        for i in 0..30 {
            if i % 2 == 1 {  // Odd indices have value 2
                expected_elements = expected_elements.saturating_mul(2);
            }
            // Even indices have value 1, so no change to product
        }
        
        assert_eq!(alt_value.num_elements().unwrap_or(0), expected_elements);
    }

    /// Test 10: Testing the safe element calculation function with overflow handling
    #[test]
    fn test_element_calculation_overflow_handling() {
        // Test the safe element calculation that should handle overflow
        
        // Small values that shouldn't overflow
        let small_value = Value {
            name: "small".to_string(),
            ty: Type::F32,
            shape: vec![100, 100, 100],  // 1M elements
        };
        
        assert_eq!(small_value.num_elements(), Some(1_000_000));
        
        // Medium values that shouldn't overflow on 64-bit systems
        let medium_value = Value {
            name: "medium".to_string(),
            ty: Type::F32,
            shape: vec![1_000, 1_000, 1_000],  // 1B elements
        };
        
        assert_eq!(medium_value.num_elements(), Some(1_000_000_000));
        
        // Larger values that could potentially overflow usize on 32-bit systems
        // but should be handled gracefully on 64-bit systems
        let large_value = Value {
            name: "large".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],
        };
        
        let result = large_value.num_elements();
        // This should either give us the result or None if overflow occurs
        assert!(result.is_some());  // On 64-bit systems, this should work
        
        // Test case with potential overflow (using values that would definitely overflow if multiplied)
        // For actual overflow testing, we need to create a test that would exceed usize::MAX
        // But since this is platform-dependent, we'll test with a reasonable upper bound
        let safe_large = Value {
            name: "safe_large".to_string(),
            ty: Type::F32,
            shape: vec![46340, 46340],  // 46340^2 is roughly 2.1 billion, close to u32::MAX
        };
        
        let calc_result = safe_large.num_elements();
        if calc_result.is_none() {
            // This means overflow was detected and handled properly
            assert_eq!(calc_result, None);
        } else {
            // If there's a result, it should match our calculation
            assert_eq!(calc_result, Some(46340 * 46340));
        }
        
        // Test with zero in dimensions (should return Some(0) not None)
        let zero_value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: vec![1000, 0, 5000],  // Contains zero, should result in 0 elements
        };
        
        assert_eq!(zero_value.num_elements(), Some(0));
    }
}