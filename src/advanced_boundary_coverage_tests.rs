//! Advanced boundary coverage tests for the Impulse compiler
//! This file contains 10 unique edge case tests covering boundary conditions
//! not extensively tested in other test files

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

#[cfg(test)]
mod advanced_boundary_coverage_tests {
    use super::*;

    /// Test 1: num_elements() with shapes that multiply to values near usize boundaries
    #[test]
    fn test_num_elements_near_usize_boundaries() {
        // Test with shape that would produce a large but valid value
        let value1 = Value {
            name: "large_valid".to_string(),
            ty: Type::F32,
            shape: vec![46340, 46340], // 46340^2 = 2,147,395,600 (near u32::MAX)
        };
        assert_eq!(value1.num_elements(), Some(2_147_395_600));

        // Test with shape that would overflow (should return None)
        let value2 = Value {
            name: "overflow_case".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2],
        };
        assert_eq!(value2.num_elements(), None);

        // Test with single dimension near boundary
        let value3 = Value {
            name: "single_large_dim".to_string(),
            ty: Type::F64,
            shape: vec![usize::MAX],
        };
        assert_eq!(value3.num_elements(), Some(usize::MAX));

        // Test with zero dimension (should return 0)
        let value4 = Value {
            name: "zero_dim".to_string(),
            ty: Type::I32,
            shape: vec![100, 0, 200],
        };
        assert_eq!(value4.num_elements(), Some(0));
    }

    /// Test 2: String attributes with extensive whitespace and special characters
    #[test]
    fn test_string_attributes_with_special_chars() {
        let special_strings = vec![
            "  leading and trailing spaces  ",
            "\t\ttab\t\tseparated\t\t",
            "line\nbreaks\nhere",
            "carriage\rreturn\r",
            "mixed\t\n\rwhitespace",
            "emoji 😊 🚀 🎉 in strings",
            "unicode 中文 日本語 한글",
            "symbols: !@#$%^&*()_+-=[]{}|;':\".,/<>?",
            "null\x00byte\x00",
            "quotes \" ' inside",
        ];

        for (i, s) in special_strings.iter().enumerate() {
            let attr = Attribute::String(s.to_string());
            match attr {
                Attribute::String(ref string) => {
                    assert_eq!(string, *s);
                    assert_eq!(string.len(), s.len());
                },
                _ => panic!("Expected String attribute for test case {}", i),
            }
        }
    }

    /// Test 3: Operation types with spaces, punctuation, and unusual characters
    #[test]
    fn test_operation_types_with_unusual_chars() {
        let unusual_op_types = vec![
            "op with spaces",
            "op.with.dots",
            "op-with-dashes",
            "op/with/slashes",
            "op\\with\\backslashes",
            "op_with_underscores",
            "OP.WITH.MIXED.CASE",
            "op123with456numbers",
            "op!with@special#chars$",
            "日本語_操作",
        ];

        for op_type in unusual_op_types {
            let op = Operation::new(op_type);
            assert_eq!(op.op_type, op_type);
            assert_eq!(op.inputs.len(), 0);
            assert_eq!(op.outputs.len(), 0);
            assert_eq!(op.attributes.len(), 0);
        }
    }

    /// Test 4: Mixed attribute types in HashMap operations
    #[test]
    fn test_mixed_attributes_hashmap_operations() {
        let mut op = Operation::new("mixed_attrs");
        let mut attrs = HashMap::new();

        // Insert mixed attribute types
        attrs.insert("int_val".to_string(), Attribute::Int(-123));
        attrs.insert("float_val".to_string(), Attribute::Float(-3.14159));
        attrs.insert("string_val".to_string(), Attribute::String("test\n\t\r".to_string()));
        attrs.insert("bool_val".to_string(), Attribute::Bool(false));
        attrs.insert("array_val".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("three".to_string()),
        ]));

        op.attributes = attrs;

        // Verify all attributes are present
        assert_eq!(op.attributes.len(), 5);
        assert!(op.attributes.contains_key("int_val"));
        assert!(op.attributes.contains_key("float_val"));
        assert!(op.attributes.contains_key("string_val"));
        assert!(op.attributes.contains_key("bool_val"));
        assert!(op.attributes.contains_key("array_val"));

        // Verify attribute values
        match op.attributes.get("int_val") {
            Some(Attribute::Int(-123)) => {},
            _ => panic!("Expected Int(-123)"),
        }

        match op.attributes.get("float_val") {
            Some(Attribute::Float(val)) if (val - (-3.14159)).abs() < f64::EPSILON => {},
            _ => panic!("Expected Float(-3.14159)"),
        }

        match op.attributes.get("bool_val") {
            Some(Attribute::Bool(false)) => {},
            _ => panic!("Expected Bool(false)"),
        }
    }

    /// Test 5: Tensors with duplicate but valid dimension values
    #[test]
    fn test_tensors_with_duplicate_dimensions() {
        let shapes_with_duplicates = vec![
            vec![5, 5, 5],      // All dimensions same
            vec![10, 10],       // 2D with duplicates
            vec![3, 3, 3, 3],   // 4D with duplicates
            vec![100, 100, 100], // Large duplicates
            vec![1, 1, 1, 1, 1], // Many 1s
        ];

        for (i, shape) in shapes_with_duplicates.iter().enumerate() {
            let value = Value {
                name: format!("duplicate_dim_{}", i),
                ty: Type::F32,
                shape: shape.clone(),
            };

            assert_eq!(value.shape, *shape);

            // Calculate expected elements
            let expected: usize = shape.iter().product();
            assert_eq!(value.num_elements(), Some(expected));
        }
    }

    /// Test 6: Deep nesting equality comparison for tensor types
    #[test]
    fn test_deep_nested_tensor_equality() {
        // Create identical deeply nested tensors
        let deep1 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        };

        let deep2 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        };

        assert_eq!(deep1, deep2);

        // Different shape in middle level
        let deep3 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![5], // Different from deep1
            }),
            shape: vec![4],
        };

        assert_ne!(deep1, deep3);

        // Different innermost type
        let deep4 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I64), // Different from F32
                    shape: vec![2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        };

        assert_ne!(deep1, deep4);
    }

    /// Test 7: Array attributes with deeply mixed type nesting
    #[test]
    fn test_mixed_nested_array_attributes() {
        let nested_mixed = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(2.5),
                    Attribute::String("nested".to_string()),
                ]),
            ]),
            Attribute::Bool(true),
            Attribute::Array(vec![
                Attribute::Int(3),
                Attribute::Array(vec![
                    Attribute::Bool(false),
                    Attribute::Int(42),
                ]),
            ]),
        ]);

        match nested_mixed {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 3);

                // First element: deeply nested array
                match &outer[0] {
                    Attribute::Array(inner1) => {
                        assert_eq!(inner1.len(), 2);
                        match &inner1[0] {
                            Attribute::Int(1) => {},
                            _ => panic!("Expected Int(1)"),
                        }
                        match &inner1[1] {
                            Attribute::Array(deep) => {
                                assert_eq!(deep.len(), 2);
                                match &deep[0] {
                                    Attribute::Float(val) if (val - 2.5).abs() < f64::EPSILON => {},
                                    _ => panic!("Expected Float(2.5)"),
                                }
                                match &deep[1] {
                                    Attribute::String(s) if s == "nested" => {},
                                    _ => panic!("Expected String(\"nested\")"),
                                }
                            },
                            _ => panic!("Expected nested array"),
                        }
                    },
                    _ => panic!("Expected Array"),
                }

                // Second element: Bool
                match outer[1] {
                    Attribute::Bool(true) => {},
                    _ => panic!("Expected Bool(true)"),
                }

                // Third element: another nested array
                match &outer[2] {
                    Attribute::Array(inner2) => {
                        assert_eq!(inner2.len(), 2);
                        match &inner2[0] {
                            Attribute::Int(3) => {},
                            _ => panic!("Expected Int(3)"),
                        }
                        match &inner2[1] {
                            Attribute::Array(deep2) => {
                                assert_eq!(deep2.len(), 2);
                                match &deep2[0] {
                                    Attribute::Bool(false) => {},
                                    _ => panic!("Expected Bool(false)"),
                                }
                                match &deep2[1] {
                                    Attribute::Int(42) => {},
                                    _ => panic!("Expected Int(42)"),
                                }
                            },
                            _ => panic!("Expected nested array"),
                        }
                    },
                    _ => panic!("Expected Array"),
                }
            },
            _ => panic!("Expected outer Array"),
        }
    }

    /// Test 8: Module operations order and addition consistency
    #[test]
    fn test_module_operations_order_consistency() {
        let mut module = Module::new("order_test");

        // Add operations in specific order
        let op_names = vec!["op_first", "op_second", "op_third", "op_fourth", "op_fifth"];
        for name in &op_names {
            let op = Operation::new(name);
            module.add_operation(op);
        }

        // Verify order is preserved
        assert_eq!(module.operations.len(), 5);
        for (i, name) in op_names.iter().enumerate() {
            assert_eq!(module.operations[i].op_type, *name);
        }

        // Add more operations
        let more_names = vec!["op_sixth", "op_seventh"];
        for name in &more_names {
            let op = Operation::new(name);
            module.add_operation(op);
        }

        // Verify all operations including new ones
        assert_eq!(module.operations.len(), 7);
        let all_names = [&op_names[..], &more_names[..]].concat();
        for (i, name) in all_names.iter().enumerate() {
            assert_eq!(module.operations[i].op_type, *name);
        }
    }

    /// Test 9: Value names with null bytes and invisible characters
    #[test]
    fn test_value_names_with_invisible_chars() {
        let unusual_names = vec![
            "prefix\x00suffix",
            "tab\there",
            "new\nline",
            "carriage\rreturn",
            "vertical\t\x0btab",
            "form\x0cfeed",
            "mixed\x01\x02\x03bytes",
            "backspace\x08char",
            "delete\x7fchar",
            "unicode\u{200B}zero\u{200B}width", // Zero-width spaces
        ];

        for (i, name) in unusual_names.iter().enumerate() {
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1, 2, 3],
            };

            assert_eq!(value.name.len(), name.len());
            assert_eq!(value.name, *name);
            assert_eq!(value.shape, vec![1, 2, 3]);
        }
    }

    /// Test 10: Float attribute equality at extreme precision boundaries
    #[test]
    fn test_float_attribute_extreme_precision() {
        // Test very large positive numbers
        let large_pos1 = Attribute::Float(f64::MAX);
        let large_pos2 = Attribute::Float(f64::MAX);
        assert_eq!(large_pos1, large_pos2);

        // Test very large negative numbers
        let large_neg1 = Attribute::Float(f64::MIN);
        let large_neg2 = Attribute::Float(f64::MIN);
        assert_eq!(large_neg1, large_neg2);

        // Test very small positive numbers
        let small_pos1 = Attribute::Float(f64::MIN_POSITIVE);
        let small_pos2 = Attribute::Float(f64::MIN_POSITIVE);
        assert_eq!(small_pos1, small_pos2);

        // Test values near zero
        let near_zero_pos = Attribute::Float(1e-308);
        let near_zero_neg = Attribute::Float(-1e-308);
        assert_ne!(near_zero_pos, near_zero_neg);

        // Test infinity comparisons
        let pos_inf = Attribute::Float(f64::INFINITY);
        let neg_inf = Attribute::Float(f64::NEG_INFINITY);
        assert_ne!(pos_inf, neg_inf);

        // Test NaN - In Rust's f64, NaN != NaN (IEEE 754 semantics)
        // This is a boundary case: two NaN values are never equal
        let nan1 = Attribute::Float(f64::NAN);
        let nan2 = Attribute::Float(f64::NAN);
        assert_ne!(nan1, nan2);

        // Test that positive and negative zero are equal in f64's PartialEq
        let pos_zero = Attribute::Float(0.0);
        let neg_zero = Attribute::Float(-0.0);
        assert_eq!(pos_zero, neg_zero);

        // Test decimal precision
        let pi = Attribute::Float(std::f64::consts::PI);
        let pi_approx = Attribute::Float(3.141592653589793);
        assert_eq!(pi, pi_approx);

        // Test that very close but not equal values are not equal
        let val3 = Attribute::Float(1.0);
        let val4 = Attribute::Float(1.0 + f64::EPSILON);
        assert_ne!(val3, val4);
    }
}