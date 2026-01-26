//! Additional edge case tests for the Impulse compiler
//! Covering boundary conditions for tensor shapes, operations, and compiler functionality

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Value, Type, Operation, Attribute};
    use crate::ImpulseCompiler;
    use rstest::*;

    /// Test 1: Operations with minimum and maximum integer values in attributes
    #[test]
    fn test_operations_with_extreme_integer_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("extreme_int_test");
        let mut attrs = HashMap::new();
        
        attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("max_int".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("zero_int".to_string(), Attribute::Int(0));
        attrs.insert("neg_one".to_string(), Attribute::Int(-1));
        attrs.insert("pos_one".to_string(), Attribute::Int(1));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.get("min_int"), Some(&Attribute::Int(i64::MIN)));
        assert_eq!(op.attributes.get("max_int"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("zero_int"), Some(&Attribute::Int(0)));
        assert_eq!(op.attributes.get("neg_one"), Some(&Attribute::Int(-1)));
        assert_eq!(op.attributes.get("pos_one"), Some(&Attribute::Int(1)));
        
        assert_eq!(op.attributes.len(), 5);
    }

    /// Test 2: Tensor shapes with very large dimensions but not exceeding memory limits
    #[test]
    fn test_very_large_but_realistic_tensor_shapes() {
        // Test tensor shapes that are large but realistic
        let test_cases = vec![
            (vec![1000, 1000], 1_000_000),      // 1M elements
            (vec![100, 100, 100], 1_000_000),    // 1M elements in 3D
            (vec![10_000, 100], 1_000_000),      // 1M elements in 2D
            (vec![50_000, 20], 1_000_000),       // 1M elements, different ratio
            (vec![1, 1, 1_000_000], 1_000_000), // 1M elements in 3D, skinny
        ];

        for (shape, expected_size) in test_cases {
            let value = Value {
                name: "large_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };

            assert_eq!(value.shape, shape);
            
            // Calculate total elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, expected_size);
        }
    }

    /// Test 3: Tensor shapes that contain zeros (yielding 0 total elements)
    #[test]
    fn test_tensor_shapes_with_zeros() {
        let test_cases = vec![
            vec![0],              // Scalar-like but zero size
            vec![0, 5],           // Contains zero, total = 0
            vec![5, 0],           // Contains zero, total = 0
            vec![0, 0],           // Both dimensions zero
            vec![3, 0, 7],        // Zero in middle
            vec![1, 1, 0, 1],     // Zero somewhere in middle
            vec![0, 1000, 1000],  // Leading zero makes it 0
        ];

        for shape in test_cases {
            let value = Value {
                name: "zero_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };

            assert_eq!(value.shape, shape);
            
            // Any tensor with a zero dimension should have 0 elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, 0, "Shape {:?} should have 0 total elements", shape);
        }
    }

    /// Test 4: Edge cases for floating point values in attributes
    #[test]
    fn test_floating_point_attribute_edge_cases() {
        let test_cases = vec![
            (f64::INFINITY, "positive_infinity"),
            (f64::NEG_INFINITY, "negative_infinity"),
            (f64::NAN, "not_a_number"),
            (0.0, "positive_zero"),
            (-0.0, "negative_zero"),
            (f64::EPSILON, "epsilon"),
            (f64::MIN_POSITIVE, "min_positive"),
            (f64::MAX, "max_f64"),
            (f64::MIN, "min_f64"),
            (std::f64::consts::PI, "pi"),
            (std::f64::consts::E, "euler_constant"),
        ];

        for (value, name) in test_cases {
            let attr = Attribute::Float(value);
            match attr {
                Attribute::Float(retrieved_value) => {
                    if value.is_nan() {
                        assert!(retrieved_value.is_nan(), "Expected NaN for case: {}", name);
                    } else if value.is_infinite() {
                        assert!(retrieved_value.is_infinite(), "Expected infinity for case: {}", name);
                        assert_eq!(retrieved_value.is_sign_positive(), value.is_sign_positive(), 
                                  "Sign mismatch for infinity case: {}", name);
                    } else if value == 0.0 {
                        // Check for both positive and negative zero
                        assert!(retrieved_value == 0.0, "Expected zero for case: {}", name);
                    } else {
                        // For finite values, use approximate equality
                        let diff = (value - retrieved_value).abs();
                        assert!(diff < f64::EPSILON * 10.0, 
                               "Float mismatch for {}: expected {}, got {}, diff: {}", 
                               name, value, retrieved_value, diff);
                    }
                },
                _ => panic!("Expected Float attribute for case: {}", name),
            }
        }
    }

    /// Test 5: Operations with extremely long names and values
    #[test]
    fn test_extremely_long_operation_and_value_names() {
        let extremely_long_op_name = "very_long_operation_name_".repeat(500) + "_end"; // Very long name
        let extremely_long_value_name = "very_long_value_name_".repeat(500) + "_end"; // Very long name
        
        // Test operation with extremely long name
        let op = Operation::new(&extremely_long_op_name);
        assert_eq!(op.op_type, extremely_long_op_name);
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);

        // Test value with extremely long name
        let value = Value {
            name: extremely_long_value_name.clone(),
            ty: Type::F32,
            shape: vec![1, 1],
        };
        assert_eq!(value.name, extremely_long_value_name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1, 1]);
    }

    /// Test 6: String attributes with various Unicode and special characters
    #[test]
    fn test_string_attributes_with_unicode_and_special_chars() {
        let test_cases = vec![
            ("ascii_only", "Hello, World! 1234567890"),
            ("unicode_emojis", "Hello 🌍 World 🚀 Rust 🦀"),
            ("chinese_chars", "你好，世界！机器学习"),
            ("russian_chars", "Привет мир! Машинное обучение"),
            ("arabic_chars", "مرحبا بالعالم! تعلم الآلة"),
            ("mixed_scripts", "混合文本 Mixed Text ミックスされたテキスト"),
            ("control_chars", "\n\t\r\u{0000}\u{0001}\u{001F}"),
            ("special_symbols", "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
            ("programming_symbols", "~`\\|/?.>,<'\";:{[}]<tab><newline>"),
            ("mathematical_symbols", "∀∂∃∄∅∆∇∈∉∊∋∌∍∎∞"),
        ];

        for (case_name, content) in test_cases {
            let attr = Attribute::String(content.to_string());
            match attr {
                Attribute::String(retrieved) => {
                    assert_eq!(retrieved, content, "String mismatch in case: {}", case_name);
                },
                _ => panic!("Expected String attribute in case: {}", case_name),
            }
        }
    }

    /// Test 7: Parameterized test for empty collections and minimal structures
    #[rstest]
    #[case(vec![], 1)]      // Scalar (empty shape) has 1 element
    #[case(vec![0], 0)]     // Contains 0, so product is 0
    #[case(vec![1], 1)]     // Single dimension of 1
    #[case(vec![1, 1], 1)]  // Multiple 1s
    #[case(vec![1, 1, 1], 1)] // Multiple 1s in 3D
    #[case(vec![2, 3], 6)]  // Simple 2D
    #[case(vec![2, 3, 4], 24)] // 3D
    #[case(vec![10, 0, 5], 0)] // Contains 0, so product is 0
    fn test_tensor_shape_size_calculations(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
        let value = Value {
            name: "shape_test".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        assert_eq!(value.shape, shape);
        let calculated_size: usize = value.shape.iter().product();
        assert_eq!(calculated_size, expected_size);
    }

    /// Test 8: Testing deeply nested tensor types without causing stack overflow
    #[test]
    fn test_deeply_nested_tensor_types_safe() {
        // Create a moderately deep nested tensor type to test without stack overflow risk
        let mut current_type = Type::F32;
        const DEPTH: usize = 100; // Safe depth that won't cause issues
        
        for i in 0..DEPTH {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![i % 10 + 1], // Varying shape to make it interesting
            };
        }

        // Verify it's still a valid nested tensor type
        match &current_type {
            Type::Tensor { shape, .. } => {
                let expected_shape = vec![(DEPTH - 1) % 10 + 1]; // The last iteration's value
                assert_eq!(shape, &expected_shape);
            },
            _ => panic!("Expected a nested tensor type after {} levels", DEPTH),
        }

        // Test that we can clone this deeply nested type
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
        
        // Test equality works correctly
        assert!(current_type.eq(&cloned));
    }

    /// Test 9: Comprehensive test of operations with maximum diversity of attributes
    #[test]
    fn test_operation_with_maximum_attribute_diversity() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("max_diversity_op");
        
        // Populate with all different types of attributes
        let mut attrs = HashMap::new();
        
        // Integer attributes
        attrs.insert("int_positive".to_string(), Attribute::Int(42));
        attrs.insert("int_negative".to_string(), Attribute::Int(-17));
        attrs.insert("int_zero".to_string(), Attribute::Int(0));
        attrs.insert("int_min".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("int_max".to_string(), Attribute::Int(i64::MAX));
        
        // Float attributes
        attrs.insert("float_pi".to_string(), Attribute::Float(std::f64::consts::PI));
        attrs.insert("float_negative".to_string(), Attribute::Float(-3.14));
        attrs.insert("float_zero".to_string(), Attribute::Float(0.0));
        attrs.insert("float_inf".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("float_nan".to_string(), Attribute::Float(f64::NAN));
        
        // String attributes
        attrs.insert("string_short".to_string(), Attribute::String("short".to_string()));
        attrs.insert("string_long".to_string(), Attribute::String("long_string".repeat(1000)));
        attrs.insert("string_unicode".to_string(), Attribute::String("Unicode: 🚀🌍".to_string()));
        attrs.insert("string_empty".to_string(), Attribute::String("".to_string()));
        
        // Boolean attributes
        attrs.insert("bool_true".to_string(), Attribute::Bool(true));
        attrs.insert("bool_false".to_string(), Attribute::Bool(false));
        
        // Array attributes (mixed types)
        attrs.insert("array_mixed".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("three".to_string()),
            Attribute::Bool(true),
        ]));
        
        // Nested array attributes
        attrs.insert("array_nested".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(10),
                Attribute::Int(20),
            ]),
            Attribute::Array(vec![
                Attribute::Float(3.14),
                Attribute::Float(2.71),
            ]),
        ]));
        
        op.attributes = attrs;
        
        // Count the actual number of attributes that were added
        // Integer attributes: 5
        // Float attributes: 5  
        // String attributes: 4
        // Boolean attributes: 2
        // Array attributes: 2
        // Total: 18
        assert_eq!(op.attributes.len(), 18); // Count of all added attributes
        
        // Verify some key values
        assert_eq!(op.attributes.get("int_positive"), Some(&Attribute::Int(42)));
        assert_eq!(op.attributes.get("float_pi").unwrap(), &Attribute::Float(std::f64::consts::PI));
        assert_eq!(op.attributes.get("string_short"), Some(&Attribute::String("short".to_string())));
        assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
        
        // Verify nested array structure
        if let Some(Attribute::Array(outer_array)) = op.attributes.get("array_nested") {
            assert_eq!(outer_array.len(), 2);
            if let Attribute::Array(inner1) = &outer_array[0] {
                assert_eq!(inner1.len(), 2);
                assert_eq!(inner1[0], Attribute::Int(10));
            } else {
                panic!("Expected nested array structure for array_nested[0]");
            }
        } else {
            panic!("Expected array for array_nested attribute");
        }
    }

    /// Test 10: Test ImpulseCompiler with various edge case inputs
    #[test]
    fn test_impulse_compiler_edge_cases() {
        let mut compiler = ImpulseCompiler::new();
        
        // Test with empty inputs
        let empty_model = vec![];
        let _result = compiler.compile(&empty_model, "cpu");
        // Should not panic, regardless of result
        
        // Test with very large input (but not too large to be impractical)
        let large_model = vec![1u8; 1_000_000];  // 1MB model
        let _result2 = compiler.compile(&large_model, "cpu");
        // Should not panic
        
        // Test with single byte
        let single_byte_model = vec![42];
        let _result3 = compiler.compile(&single_byte_model, "cpu");
        // Should not panic
        
        // Test with various target strings
        let test_targets = vec!["cpu", "CPU", "CpU", "", "nonexistent_backend", "cpu_extra"];
        let mock_model = vec![1, 2, 3, 4, 5];
        
        for target in test_targets {
            let _result = compiler.compile(&mock_model, target);
            // All should not panic
        }
        
        // Verify compiler still works properly after these tests
        assert_eq!(compiler.passes.passes.len(), 0);
    }
}