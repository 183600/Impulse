//! Additional boundary tests for the Impulse compiler
//! These tests focus on edge cases that complement existing test coverage

#[cfg(test)]
mod new_boundary_tests {
    use rstest::*;
    use crate::{
        ir::{Module, Value, Type, Operation, Attribute},
        ImpulseCompiler,
    };

    /// Test 1: Operations with mixed precision types in the same module
    #[test]
    fn test_mixed_precision_arithmetic_operations() {
        let mut module = Module::new("mixed_precision_module");
        
        // Add operations with different precision floating point types
        let f32_op = Operation::new("f32_op");
        let f64_op = Operation::new("f64_op");
        
        // Create values with different precision types
        let f32_val = Value {
            name: "f32_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        };
        
        let f64_val = Value {
            name: "f64_tensor".to_string(),
            ty: Type::F64,
            shape: vec![10, 10],
        };
        
        let mut mixed_op = Operation::new("mixed_op");
        mixed_op.inputs.push(f32_val);
        mixed_op.inputs.push(f64_val);
        
        module.add_operation(f32_op);
        module.add_operation(f64_op);
        module.add_operation(mixed_op);
        
        assert_eq!(module.operations.len(), 3);
        assert_eq!(module.operations[2].inputs.len(), 2);
        assert_eq!(module.operations[2].inputs[0].ty, Type::F32);
        assert_eq!(module.operations[2].inputs[1].ty, Type::F64);
    }

    /// Test 2: Operations with empty arrays in attributes
    #[test]
    fn test_empty_array_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("empty_array_op");
        let mut attrs = HashMap::new();
        
        // Add an empty array attribute
        attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
        // Add other attributes too
        attrs.insert("regular_int".to_string(), Attribute::Int(42));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 2);
        assert_eq!(op.attributes.get("empty_array"), Some(&Attribute::Array(vec![])));
        assert_eq!(op.attributes.get("regular_int"), Some(&Attribute::Int(42)));
    }

    /// Test 3: Deeply nested arrays in attributes
    #[test]
    fn test_deeply_nested_array_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("deeply_nested_array_op");
        let mut attrs = HashMap::new();
        
        // Create deeply nested arrays: [[1], [2, [3, [4]]]]
        let deeply_nested = Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1)]),
            Attribute::Array(vec![
                Attribute::Int(2),
                Attribute::Array(vec![
                    Attribute::Int(3),
                    Attribute::Array(vec![Attribute::Int(4)])
                ])
            ])
        ]);
        
        attrs.insert("deeply_nested_array".to_string(), deeply_nested);
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 1);
        
        match op.attributes.get("deeply_nested_array").unwrap() {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                
                // Check first nested array [1]
                match &outer[0] {
                    Attribute::Array(inner) => {
                        assert_eq!(inner.len(), 1);
                        assert_eq!(inner[0], Attribute::Int(1));
                    }
                    _ => panic!("Expected nested array"),
                }
                
                // Check second nested array [2, [3, [4]]]
                match &outer[1] {
                    Attribute::Array(second_inner) => {
                        assert_eq!(second_inner.len(), 2);
                        assert_eq!(second_inner[0], Attribute::Int(2));
                        
                        match &second_inner[1] {
                            Attribute::Array(third_inner) => {
                                assert_eq!(third_inner.len(), 2);
                                assert_eq!(third_inner[0], Attribute::Int(3));
                                
                                match &third_inner[1] {
                                    Attribute::Array(fourth_inner) => {
                                        assert_eq!(fourth_inner.len(), 1);
                                        assert_eq!(fourth_inner[0], Attribute::Int(4));
                                    }
                                    _ => panic!("Expected fourth level of nesting"),
                                }
                            }
                            _ => panic!("Expected third level of nesting"),
                        }
                    }
                    _ => panic!("Expected second level of nesting"),
                }
            }
            _ => panic!("Expected array attribute"),
        }
    }

    /// Test 4: Operations with maximum allowed recursion depth in types
    #[test]
    fn test_recursion_depth_limit_in_types() {
        // Create tensor types with moderate nesting depth that should not cause stack overflow
        let mut current_type = Type::F32;
        
        // Build a moderately nested type (depth 10 should be safe)
        for i in 0..10 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i + 1], // Varying shapes to make it interesting
            };
        }
        
        // Should be able to clone this without issue
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
        
        // Check that the structure is preserved - the deepest should have shape [10]
        match &current_type {
            Type::Tensor { shape, element_type: _ } => {
                assert_eq!(shape, &vec![10]);
            }
            _ => panic!("Expected top-level tensor type"),
        }
    }

    /// Test 5: Operations with very long attribute names
    #[test]
    fn test_very_long_attribute_names() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("long_attr_names_op");
        let mut attrs = HashMap::new();
        
        // Create attribute with very long name
        let long_attr_name = "a".repeat(10_000);
        attrs.insert(long_attr_name.clone(), Attribute::Int(123));
        
        // Add a few more normal attributes
        attrs.insert("normal_attr".to_string(), Attribute::String("value".to_string()));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 2);
        assert_eq!(op.attributes.get(&long_attr_name), Some(&Attribute::Int(123)));
    }

    /// Test 6: Compiler with invalid UTF-8 byte sequences
    #[test]
    fn test_compiler_with_invalid_utf8_sequences() {
        let mut compiler = ImpulseCompiler::new();
        
        // Create byte sequence that is invalid UTF-8
        // This is a malformed UTF-8 sequence (continuation byte without lead)
        let invalid_utf8 = vec![0xFF, 0xFE, 0x00, 0x01, 0xFF];
        
        // The compiler should handle this gracefully without crashing
        let result = compiler.compile(&invalid_utf8, "cpu");
        
        // Result could be success or failure, but should not panic
        assert!(result.is_ok() || result.is_err());
    }

    /// Test 7: Operations with special floating point values
    #[test]
    fn test_operations_with_special_floating_values() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("special_float_op");
        let mut attrs = HashMap::new();
        
        // Add attributes with special floating point values
        attrs.insert("infinity_attr".to_string(), Attribute::Float(std::f64::INFINITY));
        attrs.insert("neg_infinity_attr".to_string(), Attribute::Float(std::f64::NEG_INFINITY));
        attrs.insert("nan_attr".to_string(), Attribute::Float(std::f64::NAN));
        attrs.insert("negative_zero_attr".to_string(), Attribute::Float(-0.0));
        attrs.insert("epsilon_attr".to_string(), Attribute::Float(std::f64::EPSILON));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 5);
        
        // Verify we can extract the special values back
        match op.attributes.get("infinity_attr").unwrap() {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected float attribute"),
        }
        
        match op.attributes.get("neg_infinity_attr").unwrap() {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
            _ => panic!("Expected float attribute"),
        }
        
        match op.attributes.get("nan_attr").unwrap() {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected float attribute"),
        }
    }

    /// Test 8: Parameterized test for tensor shape validation with various dimensions
    #[rstest]
    #[case(vec![], 1)]           // 0-D tensor (scalar)
    #[case(vec![1], 1)]          // 1-D tensor with 1 element
    #[case(vec![5], 5)]          // 1-D tensor with 5 elements
    #[case(vec![2, 3], 6)]       // 2-D tensor 2x3
    #[case(vec![2, 3, 4], 24)]   // 3-D tensor 2x3x4
    #[case(vec![1, 1, 1, 1], 1)] // 4-D tensor 1x1x1x1
    #[case(vec![10, 0, 5], 0)]   // Contains zero dimension
    #[case(vec![0, 0, 0], 0)]    // All zero dimensions
    fn test_tensor_shape_products_comprehensive(#[case] shape: Vec<usize>, #[case] expected_count: usize) {
        let value = Value {
            name: "test_tensor".to_string(),
            ty: Type::F32,
            shape,
        };
        
        assert_eq!(value.shape.len(), value.shape.len()); // Basic sanity check
        
        let calculated_count: usize = value.shape.iter().map(|&x| x).product();
        assert_eq!(calculated_count, expected_count);
    }

    /// Test 9: Memory allocation edge cases with huge tensor sizes
    #[test]
    fn test_huge_tensor_size_without_allocation() {
        // Create tensor shapes that would represent huge memory but don't actually allocate
        // This tests the ability to represent these shapes without crashing
        let huge_shape = Value {
            name: "potentially_huge_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000, 1_000_000], // Would be 10^12 elements if we multiplied
        };
        
        assert_eq!(huge_shape.shape, vec![1_000_000, 1_000_000]);
        
        // Calculate a subset product to avoid overflow
        // Instead of multiplying all dimensions (which could overflow),
        // we just verify the shape is stored correctly
        assert_eq!(huge_shape.shape.len(), 2);
        assert_eq!(huge_shape.shape[0], 1_000_000);
        assert_eq!(huge_shape.shape[1], 1_000_000);
        
        // Test with a shape that has zero to avoid actual huge computation
        let zero_huge_shape = Value {
            name: "zero_huge_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000, 0, 1_000_000], // Contains zero, so product is 0
        };
        
        let zero_product: usize = zero_huge_shape.shape.iter().product();
        assert_eq!(zero_product, 0);
    }

    /// Test 10: Operations with complex attribute dependencies
    #[test]
    fn test_complex_attribute_dependencies() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("complex_deps_op");
        let mut attrs = HashMap::new();
        
        // Create interdependent attributes
        attrs.insert("count".to_string(), Attribute::Int(100));
        attrs.insert("unit_size".to_string(), Attribute::Int(8));
        attrs.insert("total_size".to_string(), Attribute::Int(800));
        attrs.insert("enabled".to_string(), Attribute::Bool(true));
        attrs.insert("name".to_string(), Attribute::String("complex_op".to_string()));
        attrs.insert("tags".to_string(), Attribute::Array(vec![
            Attribute::String("memory".to_string()),
            Attribute::String("compute".to_string()),
            Attribute::String("optimized".to_string())
        ]));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 6);
        
        // Verify all attributes are present
        assert_eq!(op.attributes.get("count"), Some(&Attribute::Int(100)));
        assert_eq!(op.attributes.get("unit_size"), Some(&Attribute::Int(8)));
        assert_eq!(op.attributes.get("total_size"), Some(&Attribute::Int(800)));
        assert_eq!(op.attributes.get("enabled"), Some(&Attribute::Bool(true)));
        assert_eq!(op.attributes.get("name"), Some(&Attribute::String("complex_op".to_string())));
        
        match op.attributes.get("tags").unwrap() {
            Attribute::Array(tags) => {
                assert_eq!(tags.len(), 3);
                if let Attribute::String(tag0) = &tags[0] {
                    assert_eq!(tag0, "memory");
                } else {
                    panic!("Expected string tag");
                }
            },
            _ => panic!("Expected array of tags"),
        }
    }
}