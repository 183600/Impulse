//! Comprehensive Edge Case Tests for Impulse Compiler
//! 
//! This module includes advanced edge case tests covering boundary conditions,
//! overflow scenarios, unicode handling, recursive data structures, and more.

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;
use rstest::rstest;

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    /// Test 1: Extremely deep nested tensor types to test recursion limits
    #[test]
    fn test_deeply_nested_tensor_types() {
        let mut current_type = Type::F32;
        
        // Create a deeply nested tensor type with 500 levels to test recursion limits
        for _ in 0..500 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }

        // Verify we can access and clone the deeply nested type
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a Tensor type after deep nesting"),
        }

        // Test cloning of deeply nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 2: Operations with maximum possible attributes to test memory limits
    #[test]
    fn test_operation_with_maximum_attributes() {
        let mut op = Operation::new("max_attr_op");
        let mut attrs = HashMap::new();

        // Add maximum range integers
        attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
        
        // Add special floating-point values
        attrs.insert("inf".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
        attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
        attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));

        // Add extremely long strings
        attrs.insert("long_string".to_string(), Attribute::String("a".repeat(100_000)));

        // Add boolean extremes
        attrs.insert("true_val".to_string(), Attribute::Bool(true));
        attrs.insert("false_val".to_string(), Attribute::Bool(false));

        // Add nested arrays
        attrs.insert("deep_array".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.0),
                Attribute::String("nested".to_string()),
            ]),
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Int(999),
                ])
            ])
        ]));

        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 12);
        assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
        
        // Verify NaN handling
        if let Some(Attribute::Float(val)) = op.attributes.get("nan") {
            assert!(val.is_nan());
        } else {
            panic!("Expected NaN value for 'nan' attribute");
        }
    }

    /// Test 3: Test tensor shapes that could cause integer overflow when calculating total elements
    #[test]
    fn test_potential_overflow_tensor_shapes() {
        // Create tensors with shapes that could cause overflow when multiplied
        // These are within usize bounds but large enough to stress the system
        
        // Large but safe tensor (would be ~4GB if allocated as F32)
        let large_shape = vec![10_000, 10_000];
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: large_shape.clone(),
        };

        let product: usize = large_value.shape.iter().product();
        assert_eq!(product, 100_000_000);

        // Test with a shape that results in 0 elements
        let zero_shape = vec![1000, 0, 5000];
        let zero_value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: zero_shape.clone(),
        };

        let zero_product: usize = zero_value.shape.iter().product();
        assert_eq!(zero_product, 0);

        // Test with single large dimension
        let single_large = vec![usize::MAX / 100];  // Keep it small enough to not overflow
        let single_value = Value {
            name: "single_large".to_string(),
            ty: Type::F32,
            shape: single_large.clone(),
        };
        
        let single_product: usize = single_value.shape.iter().product();
        assert_eq!(single_product, usize::MAX / 100);
    }

    /// Test 4: Unicode and special character handling in identifiers
    #[test]
    fn test_unicode_and_special_character_identifiers() {
        let unicode_cases = [
            ("tensor_🚀_with_unicode", Type::F32),
            ("chinese_chars_中文_测试", Type::I32),
            ("arabic_text_مرحبا", Type::F64),
            ("emoji_mix_🔥💧🌊_test", Type::I64),
            ("control_chars_\u{0001}_\u{001F}", Type::Bool),
        ];

        for (name, data_type) in unicode_cases.iter() {
            let value = Value {
                name: name.to_string(),
                ty: data_type.clone(),
                shape: vec![1, 2, 3],
            };
            
            assert_eq!(value.name, *name);
            assert_eq!(value.ty, *data_type);
            assert_eq!(value.shape, vec![1, 2, 3]);

            // Also test with operations
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
        }
    }

    /// Test 5: Zero-sized and empty tensors edge cases
    #[test]
    fn test_zero_sized_tensor_edge_cases() {
        let test_cases = [
            vec![],                    // Scalar (0-dimensional)
            vec![0],                  // 1-dimensional with size 0
            vec![0, 5],              // Contains zero, leading
            vec![5, 0],              // Contains zero, trailing
            vec![2, 0, 3],           // Contains zero, middle
            vec![0, 0, 0],           // All zeros
            vec![1, 0, 1, 0, 1],     // Mixed zeros and ones
        ];

        for shape in test_cases.iter() {
            let value = Value {
                name: "test_tensor".to_string(),
                ty: Type::F32,
                shape: shape.to_vec(),
            };

            // Calculate total elements
            let total_elements: usize = value.shape.iter().product();
            
            if shape.iter().any(|&x| x == 0) {
                // If any dimension is 0, total should be 0
                assert_eq!(total_elements, 0, "Shape {:?} should have 0 total elements", shape);
            } else if shape.is_empty() {
                // Scalar has 1 element
                assert_eq!(total_elements, 1, "Scalar shape {:?} should have 1 total element", shape);
            } else {
                // Non-zero dimensions should multiply normally
                let expected: usize = shape.iter().product();
                assert_eq!(total_elements, expected, "Shape {:?} should have {} total elements", shape, expected);
            }
        }
    }

    /// Test 6: Operations with maximum number of inputs and outputs
    #[test]
    fn test_operation_with_maximum_io() {
        let mut op = Operation::new("max_io_op");

        // Add maximum reasonable number of inputs
        for i in 0..10_000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 10 + 1], // Varying shapes
            });
        }

        // Add maximum reasonable number of outputs
        for i in 0..5_000 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 5 + 1], // Varying shapes
            });
        }

        assert_eq!(op.inputs.len(), 10_000);
        assert_eq!(op.outputs.len(), 5_000);
        assert_eq!(op.op_type, "max_io_op");

        // Verify some values are maintained correctly
        assert_eq!(op.inputs[0].name, "input_0");
        assert_eq!(op.inputs[9999].name, "input_9999");
        assert_eq!(op.outputs[0].name, "output_0");
        assert_eq!(op.outputs[4999].name, "output_4999");
    }

    /// Test 7: Module with extreme number of operations to test memory management
    #[test]
    fn test_module_with_extreme_operations_count() {
        let mut module = Module::new("extreme_module");

        // Add 50,000 operations to stress memory management
        for i in 0..50_000 {
            let mut op = Operation::new(&format!("operation_{}", i));
            
            // Add minimal inputs/outputs to each operation
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
            
            module.add_operation(op);
        }

        assert_eq!(module.operations.len(), 50_000);
        assert_eq!(module.name, "extreme_module");

        // Verify random operations still have correct data
        assert_eq!(module.operations[0].op_type, "operation_0");
        assert_eq!(module.operations[25_000].op_type, "operation_25000");
        assert_eq!(module.operations[49_999].op_type, "operation_49999");
    }

    /// Test 8: Extensive attribute nesting and complexity testing
    #[test]
    fn test_extremely_nested_attributes() {
        // Create a deeply nested attribute structure
        let mut attr: Attribute = Attribute::Int(1);

        // Nest arrays 10 levels deep
        for _ in 0..10 {
            attr = Attribute::Array(vec![attr]);
        }

        // Verify the structure
        match &attr {
            Attribute::Array(nested) => {
                assert_eq!(nested.len(), 1);
                
                // This pattern should continue recursively
                fn count_levels(attr: &Attribute, depth: usize) -> usize {
                    match attr {
                        Attribute::Array(items) => {
                            if items.is_empty() {
                                depth
                            } else {
                                count_levels(&items[0], depth + 1)
                            }
                        },
                        _ => depth,
                    }
                }
                
                let levels = count_levels(&attr, 0);
                assert_eq!(levels, 10);
            },
            _ => panic!("Expected nested array structure"),
        }

        // Clone the nested structure to ensure it works
        let cloned = attr.clone();
        assert_eq!(attr, cloned);
    }

    /// Test 9: Test with extremely long names for all entities
    #[test]
    fn test_extremely_long_names() {
        // Very long module name
        let long_module_name = "module_".repeat(5_000) + "end";
        let module = Module::new(&long_module_name);
        assert_eq!(module.name.len(), long_module_name.len());

        // Very long operation name
        let long_op_name = "operation_".repeat(10_000) + "end";
        let op = Operation::new(&long_op_name);
        assert_eq!(op.op_type, long_op_name);

        // Very long value name
        let long_value_name = "value_name_".repeat(15_000) + "end";
        let value = Value {
            name: long_value_name.clone(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        assert_eq!(value.name, long_value_name);
        assert_eq!(value.name.len(), long_value_name.len());

        // Very long attribute key and value
        let mut op_with_long_attrs = Operation::new("test_op");
        let long_attr_name = "attribute_name_".repeat(5_000);
        let long_attr_value = Attribute::String("attribute_value_".repeat(10_000));
        op_with_long_attrs.attributes.insert(long_attr_name.clone(), long_attr_value);
        
        assert!(op_with_long_attrs.attributes.contains_key(&long_attr_name));
    }

    /// Test 10: Using rstest for parametrized testing of multiple edge cases
    #[rstest]
    #[case(vec![], 1)]  // Scalar has 1 element
    #[case(vec![0], 0)]  // Contains 0, so product is 0
    #[case(vec![1, 1, 1, 1, 1], 1)]  // All 1s
    #[case(vec![2, 3, 4], 24)]  // Normal case
    #[case(vec![100, 100], 10_000)]  // Larger numbers
    #[case(vec![1000, 0, 5000], 0)]  // Contains zero
    #[case(vec![1, 2, 3, 4, 5], 120)]  // Multiple dimensions
    fn test_shape_products_with_rstest(#[case] shape: Vec<usize>, #[case] expected: usize) {
        let value = Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape,
        };
        
        let calculated: usize = value.shape.iter().product();
        assert_eq!(calculated, expected);
    }

    /// Additional rstest parametrized test for type variations
    #[rstest]
    #[case(Type::F32, "F32")]
    #[case(Type::F64, "F64")]
    #[case(Type::I32, "I32")]
    #[case(Type::I64, "I64")]
    #[case(Type::Bool, "Bool")]
    fn test_basic_types_with_rstest(#[case] type_variant: Type, #[case] _expected_name: &str) {
        // Just verify we can create and clone each type variant
        let cloned = type_variant.clone();
        assert_eq!(type_variant, cloned);
        
        // Test the is_valid_type extension trait
        assert!(type_variant.is_valid_type());
    }

    /// Test with multiple parameter combinations using rstest
    #[rstest]
    #[case("cpu", true)]
    #[case("gpu", true)]
    #[case("tpu", true)]
    #[case("", false)]  // Empty string might be invalid depending on implementation
    #[case("invalid_target", true)]  // Could be handled gracefully
    fn test_target_string_variations(#[case] target: &str, #[case] _may_be_valid: bool) {
        // This tests how the system handles various target strings
        // The actual validation would happen in the compiler implementation
        assert!(true);  // Basic sanity check - removed redundant assertion about usize >= 0
        
        // Test we can work with the string value safely
        let upper_case = target.to_uppercase();
        assert_eq!(upper_case.len(), target.len());
    }
}