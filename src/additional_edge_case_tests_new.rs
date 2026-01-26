//! Additional edge case tests for the Impulse compiler
//! These tests cover boundary conditions and extreme scenarios

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute, TypeExtensions};
    use rstest::rstest;
    use std::collections::HashMap;

    /// Test 1: Module creation with empty name
    #[test]
    fn test_module_with_empty_name() {
        let module = Module::new("");
        assert_eq!(module.name, "");
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }

    /// Test 2: Value with maximum possible dimensions array
    #[test]
    fn test_value_with_huge_shape_dimensions() {
        // Creates a shape with many dimensions
        let huge_shape = vec![1; 100_000];  // 100k dimensions with size 1
        let value = Value {
            name: "huge_dims".to_string(),
            ty: Type::F32,
            shape: huge_shape,
        };
        
        assert_eq!(value.shape.len(), 100_000);
        assert_eq!(value.name, "huge_dims");
        assert_eq!(value.ty, Type::F32);
        
        // Total elements would be 1^100000 = 1
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 1);
    }

    /// Test 3: Deeply nested recursive types using rstest
    #[rstest]
    #[case(1)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_deeply_nested_types_at_various_depths(#[case] depth: usize) {
        let mut current_type = Type::F32;
        
        for _ in 0..depth {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify we can create and check the type after deep nesting
        match &current_type {
            Type::Tensor { element_type: _, shape } => {
                assert_eq!(shape, &vec![2]);
            }
            _ => panic!("Expected tensor type"),
        }
        
        // Ensure we can clone the deeply nested type
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }

    /// Test 4: Operations with extreme string names
    #[test]
    fn test_operation_with_extreme_string_names() {
        // Test with maximum length string
        let long_op_name = "a".repeat(1_000_000); // 1 MB string
        let op = Operation::new(&long_op_name);
        assert_eq!(op.op_type, long_op_name);
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
        
        // Test with unicode and special characters
        let unicode_name = "🚀_tensor_operation_π_∑_∞_测试_🔥_🌟";
        let unicode_op = Operation::new(unicode_name);
        assert_eq!(unicode_op.op_type, unicode_name);
    }

    /// Test 5: Floating point values at extreme ranges
    #[rstest]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    #[case(f64::NAN)]
    #[case(f64::EPSILON)]
    #[case(f64::MIN_POSITIVE)]
    #[case(-f64::MAX)]
    #[case(f64::MAX)]
    fn test_extreme_float_attributes(#[case] value: f64) {
        let attr = Attribute::Float(value);
        
        match attr {
            Attribute::Float(retrieved_value) => {
                if value.is_nan() {
                    // NaN comparisons need special handling
                    assert!(retrieved_value.is_nan());
                } else {
                    // For all other values, they should match exactly
                    assert_eq!(retrieved_value, value);
                }
            },
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 6: Module with maximum possible operations (memory stress test)
    #[test]
    fn test_module_with_very_large_number_of_operations() {
        let mut module = Module::new("stress_test_module");
        
        // Add a very large number of operations to test memory management
        for i in 0..100_000 {
            let mut op = Operation::new(&format!("stress_op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1], // Cycle through shapes 1-100
            });
            
            // Add some outputs too
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![(i + 1) % 100 + 1], // Cycle through shapes 1-100
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 100_000);
        assert_eq!(module.name, "stress_test_module");
        
        // Verify the first and last operations still have correct data
        assert_eq!(module.operations[0].op_type, "stress_op_0");
        assert_eq!(module.operations[99999].op_type, "stress_op_99999");
    }

    /// Test 7: Edge cases with tensor shape calculations
    #[rstest]
    // Test zero-dimensional tensor (scalar)
    #[case(vec![], 1)]
    // Test tensor with one zero dimension
    #[case(vec![0], 0)]
    // Test tensor with multiple zero dimensions
    #[case(vec![0, 0, 0], 0)]
    // Test tensor with zero among other dimensions
    #[case(vec![5, 0, 10], 0)]
    // Test with large values that multiply to a reasonable result
    #[case(vec![100, 100, 100], 1_000_000)]
    // Test single dimension
    #[case(vec![42], 42)]
    // Test two dimensions
    #[case(vec![7, 6], 42)]
    fn test_tensor_shape_calculations(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let value = Value {
            name: "test_shape".to_string(),
            ty: Type::F32,
            shape,
        };
        
        let calculated_product: usize = value.shape.iter().product();
        assert_eq!(calculated_product, expected_product);
    }

    /// Test 8: Operations with maximum attribute complexity
    #[test]
    fn test_operation_with_extreme_attribute_complexity() {
        let mut op = Operation::new("complex_attr_op");
        
        // Create a hash map with many different attribute types
        let mut attrs = HashMap::new();
        
        // Add many simple attributes
        for i in 0..10_000 {
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
                Attribute::String(format!("string_value_{}", i))
            );
            attrs.insert(
                format!("bool_attr_{}", i),
                Attribute::Bool(i % 2 == 0)
            );
        }
        
        op.attributes = attrs;
        assert_eq!(op.attributes.len(), 40_000); // 10k * 4 attribute types
        assert_eq!(op.op_type, "complex_attr_op");
    }

    /// Test 9: Value with all possible type variants
    #[rstest]
    #[case(Type::F32)]
    #[case(Type::F64)]
    #[case(Type::I32)]
    #[case(Type::I64)]
    #[case(Type::Bool)]
    fn test_value_with_all_basic_types(#[case] data_type: Type) {
        let value = Value {
            name: "typed_value".to_string(),
            ty: data_type.clone(),
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name, "typed_value");
        assert_eq!(value.ty, data_type);
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    /// Test 10: Recursive tensor types with alternating patterns
    #[test]
    fn test_recursive_tensor_type_alternating_pattern() {
        // Create a complex recursive type pattern that alternates
        let mut current_type = Type::F32;
        
        // Alternate between different base types in the recursion
        for i in 0..20 {
            let next_base_type = if i % 3 == 0 {
                Type::F32
            } else if i % 3 == 1 {
                Type::I64
            } else {
                Type::Bool
            };
            
            current_type = Type::Tensor {
                element_type: Box::new(next_base_type),
                shape: vec![i + 1],
            };
        }
        
        // Verify we can create this complex recursive type
        assert!(current_type.is_valid_type());
        
        // Verify we can clone it without issues
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
        
        // Test with another complex pattern using nested tensors
        let nested_complex = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        };
        
        // Verify this also works
        assert!(nested_complex.is_valid_type());
    }
}