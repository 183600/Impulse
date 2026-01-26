//! Additional edge case tests for attribute handling in the Impulse compiler
//! This file contains tests covering complex attribute structures and interactions

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils,
};
use std::collections::HashMap;

#[cfg(test)]
mod attribute_edge_case_tests {
    use super::*;

    #[test]
    fn test_deeply_nested_attribute_arrays() {
        // Test very deeply nested attribute arrays to test recursion limits
        let mut attr = Attribute::Int(0);
        
        // Create 100 levels of nesting with alternating types
        for i in 1..=100 {
            if i % 2 == 0 {
                attr = Attribute::Array(vec![
                    Attribute::String(format!("level_{}", i)),
                    attr,
                ]);
            } else {
                attr = Attribute::Array(vec![
                    Attribute::Int(i as i64),
                    attr,
                ]);
            }
        }
        
        // Verify the resulting structure
        match &attr {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 2);
                
                // The first element should be Int(100) since 100 is even
                match &arr[0] {
                    Attribute::String(s) => assert_eq!(s, "level_100"),
                    _ => panic!("Expected String at first position"),
                }
                
                // The second element should be the nested structure
                match &arr[1] {
                    Attribute::Array(_) => {}, // Expected
                    _ => panic!("Expected Array at second position"),
                }
            },
            _ => panic!("Expected final attribute to be Array"),
        }
    }

    #[test]
    fn test_attribute_equality_with_large_structures() {
        // Create two identical complex attribute structures
        let create_complex_attr = || Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(3.14),
                    Attribute::Array(vec![
                        Attribute::String("deep".to_string()),
                        Attribute::Array(vec![
                            Attribute::Bool(true),
                            Attribute::Int(42),
                        ])
                    ])
                ])
            ]),
            Attribute::Array(vec![
                Attribute::String("outer".to_string()),
                Attribute::Array(vec![
                    Attribute::Float(2.71),
                    Attribute::Array(vec![
                        Attribute::Bool(false),
                        Attribute::String("inner".to_string()),
                    ])
                ])
            ])
        ]);

        let attr1 = create_complex_attr();
        let attr2 = create_complex_attr();

        assert_eq!(attr1, attr2);

        // Modify one and verify inequality
        let attr3 = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(3.14),
                    Attribute::Array(vec![
                        Attribute::String("DEEP".to_string()), // Changed: "deep" -> "DEEP"
                        Attribute::Array(vec![
                            Attribute::Bool(true),
                            Attribute::Int(42),
                        ])
                    ])
                ])
            ]),
            Attribute::Array(vec![
                Attribute::String("outer".to_string()),
                Attribute::Array(vec![
                    Attribute::Float(2.71),
                    Attribute::Array(vec![
                        Attribute::Bool(false),
                        Attribute::String("inner".to_string()),
                    ])
                ])
            ])
        ]);

        assert_ne!(attr1, attr3);
    }

    #[test]
    fn test_operation_with_complex_attribute_matrices() {
        // Create an operation with many different types of attributes in a matrix-like structure
        
        let mut op = Operation::new("complex_matrix_op");
        
        let mut attrs = HashMap::new();
        
        // Add attributes arranged like a "matrix" of types
        // Row 1: All primitive types
        attrs.insert("int_attr".to_string(), Attribute::Int(123));
        attrs.insert("float_attr".to_string(), Attribute::Float(45.67));
        attrs.insert("string_attr".to_string(), Attribute::String("hello".to_string()));
        attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
        
        // Row 2: Single-element arrays
        attrs.insert("int_array".to_string(), Attribute::Array(vec![Attribute::Int(5)]));
        attrs.insert("float_array".to_string(), Attribute::Array(vec![Attribute::Float(3.14)]));
        attrs.insert("string_array".to_string(), Attribute::Array(vec![Attribute::String("world".to_string())]));
        attrs.insert("bool_array".to_string(), Attribute::Array(vec![Attribute::Bool(false)]));
        
        // Row 3: Mixed-type arrays
        attrs.insert("mixed_array_1".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.0),
            Attribute::String("three".to_string()),
        ]));
        attrs.insert("mixed_array_2".to_string(), Attribute::Array(vec![
            Attribute::Bool(true),
            Attribute::Int(42),
            Attribute::Float(1.23),
            Attribute::String("test".to_string()),
            Attribute::Bool(false),
        ]));
        
        // Row 4: Nested arrays
        attrs.insert("nested_array_1".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
            Attribute::Array(vec![Attribute::Float(3.0), Attribute::Float(4.0)]),
        ]));
        attrs.insert("nested_array_2".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![Attribute::Bool(true)])
            ]),
            Attribute::Array(vec![Attribute::Int(99)])
        ]));
        
        op.attributes = attrs;
        
        // Verify all attributes were added
        assert_eq!(op.attributes.len(), 12);
        
        // Verify specific values
        assert_eq!(op.attributes.get("int_attr"), Some(&Attribute::Int(123)));
        assert_eq!(op.attributes.get("float_attr"), Some(&Attribute::Float(45.67)));
        assert_eq!(op.attributes.get("string_attr"), Some(&Attribute::String("hello".to_string())));
        assert_eq!(op.attributes.get("bool_attr"), Some(&Attribute::Bool(true)));
        
        // Verify single-element arrays
        if let Some(Attribute::Array(single_arr)) = op.attributes.get("int_array") {
            assert_eq!(single_arr.len(), 1);
            assert_eq!(single_arr[0], Attribute::Int(5));
        } else {
            panic!("Expected int_array to be a single-element array");
        }
        
        // Verify nested arrays
        if let Some(Attribute::Array(nested_main)) = op.attributes.get("nested_array_2") {
            assert_eq!(nested_main.len(), 2);
            
            if let Attribute::Array(nested_inner1) = &nested_main[0] {
                assert_eq!(nested_inner1.len(), 2);
                match &nested_inner1[0] {
                    Attribute::String(s) => assert_eq!(s, "nested"),
                    _ => panic!("Expected string in nested array"),
                }
            } else {
                panic!("Expected nested array in first position");
            }
        } else {
            panic!("Expected nested_array_2 to be an array");
        }
    }

    #[test]
    fn test_attribute_serialization_simulation() {
        // Test how attributes would behave in serialization/deserialization patterns
        
        let original_attrs = vec![
            Attribute::Int(42),
            Attribute::Float(3.14159),
            Attribute::String("serialization_test".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(false),
                    Attribute::Int(99),
                ])
            ])
        ];
        
        // Simulate "serialization" by cloning
        let serialized_attrs = original_attrs.clone();
        
        // Simulate "deserialization" by comparing
        assert_eq!(original_attrs, serialized_attrs);
        
        // Verify the nested structure survived the "serialization/deserialization"
        match &serialized_attrs[4] {
            Attribute::Array(nested) => {
                assert_eq!(nested.len(), 4);
                
                match &nested[3] {
                    Attribute::Array(deep_nested) => {
                        assert_eq!(deep_nested.len(), 2);
                        assert_eq!(deep_nested[0], Attribute::Bool(false));
                        assert_eq!(deep_nested[1], Attribute::Int(99));
                    },
                    _ => panic!("Expected deep nested array"),
                }
            },
            _ => panic!("Expected top-level array attribute"),
        }
    }

    #[test]
    fn test_operation_with_varying_attribute_counts() {
        // Test operations with different numbers of attributes to test memory management
        
        // Operation with no attributes
        let op_no_attrs = Operation::new("no_attrs");
        assert_eq!(op_no_attrs.attributes.len(), 0);
        
        // Operation with one attribute
        let mut op_one_attr = Operation::new("one_attr");
        op_one_attr.attributes.insert("only_attr".to_string(), Attribute::Int(1));
        assert_eq!(op_one_attr.attributes.len(), 1);
        
        // Operation with many attributes (but all simple)
        let mut op_many_simple = Operation::new("many_simple");
        for i in 0..50 {
            op_many_simple.attributes.insert(
                format!("attr_{}", i),
                Attribute::Int(i as i64)
            );
        }
        assert_eq!(op_many_simple.attributes.len(), 50);
        
        // Operation with few but complex attributes
        let mut op_few_complex = Operation::new("few_complex");
        op_few_complex.attributes.insert("complex_1".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
            Attribute::Array(vec![Attribute::Float(3.14), Attribute::Float(2.71)]),
        ]));
        op_few_complex.attributes.insert("complex_2".to_string(), Attribute::Array(vec![
            Attribute::String("nested".to_string()),
            Attribute::Array(vec![Attribute::Bool(true), Attribute::Bool(false)]),
        ]));
        assert_eq!(op_few_complex.attributes.len(), 2);
        
        // Operation with mixture of simple and complex attributes
        let mut op_mixed = Operation::new("mixed_attrs");
        
        // Add simple attributes
        for i in 0..10 {
            op_mixed.attributes.insert(
                format!("simple_{}", i),
                Attribute::Int(i as i64)
            );
        }
        
        // Add complex attributes
        op_mixed.attributes.insert("complex_1".to_string(), Attribute::Array(vec![
            Attribute::String("complex_1".to_string()),
            Attribute::Int(123),
        ]));
        op_mixed.attributes.insert("complex_2".to_string(), Attribute::Array(vec![
            Attribute::Bool(true),
            Attribute::Array(vec![Attribute::Float(3.14)]),
        ]));
        
        assert_eq!(op_mixed.attributes.len(), 12); // 10 simple + 2 complex
    }

    #[test]
    fn test_attribute_value_ranges() {
        // Test attributes with extreme but valid values
        
        // Test INT min/max
        let int_min_attr = Attribute::Int(i64::MIN);
        let int_max_attr = Attribute::Int(i64::MAX);
        
        match int_min_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected Int attribute"),
        }
        
        match int_max_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }
        
        // Test FLOAT min/max
        let float_min_attr = Attribute::Float(f64::MIN);
        let float_max_attr = Attribute::Float(f64::MAX);
        
        match float_min_attr {
            Attribute::Float(val) => assert_eq!(val, f64::MIN),
            _ => panic!("Expected Float attribute"),
        }
        
        match float_max_attr {
            Attribute::Float(val) => assert_eq!(val, f64::MAX),
            _ => panic!("Expected Float attribute"),
        }
        
        // Test FLOAT special values
        let float_inf = Attribute::Float(f64::INFINITY);
        let float_neg_inf = Attribute::Float(f64::NEG_INFINITY);
        let float_nan = Attribute::Float(f64::NAN);
        
        match float_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected positive infinity"),
        }
        
        match float_neg_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
            _ => panic!("Expected negative infinity"),
        }
        
        match float_nan {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected NaN"),
        }
        
        // Test empty string attribute
        let empty_str_attr = Attribute::String("".to_string());
        match &empty_str_attr {
            Attribute::String(s) => assert!(s.is_empty()),
            _ => panic!("Expected empty string attribute"),
        }
        
        // Test very long string attribute
        let long_str = "a".repeat(1_000_000); // 1 million character string
        let long_str_attr = Attribute::String(long_str.clone());
        match &long_str_attr {
            Attribute::String(s) => assert_eq!(s.len(), 1_000_000),
            _ => panic!("Expected long string attribute"),
        }
    }

    #[test]
    fn test_complex_tensor_type_with_attributes() {
        // Test combining complex tensor types with complex attributes
        
        // Create a complex nested tensor type
        let complex_tensor_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![3, 3], // 3x3 filter
            }),
            shape: vec![64, 3], // 64 filters of 3 channels each
        };
        
        // Create a value using this complex type
        let complex_value = Value {
            name: "complex_conv_weight".to_string(),
            ty: complex_tensor_type,
            shape: vec![2], // Batch size of 2
        };
        
        // Create an operation using this complex value
        let mut op = Operation::new("complex_conv2d");
        op.inputs.push(complex_value);
        
        // Add complex attributes to the operation
        let mut attrs = HashMap::new();
        
        attrs.insert("kernel_size".to_string(), Attribute::Array(vec![
            Attribute::Int(3),
            Attribute::Int(3),
        ]));
        
        attrs.insert("strides".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(1),
        ]));
        
        attrs.insert("padding".to_string(), Attribute::String("same".to_string()));
        
        attrs.insert("groups".to_string(), Attribute::Int(1));
        
        attrs.insert("dilations".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(1),
        ]));
        
        // Add a complex nested attribute for activation function
        attrs.insert("activation_config".to_string(), Attribute::Array(vec![
            Attribute::String("relu".to_string()),
            Attribute::Array(vec![
                Attribute::Bool(true), // inplace
                Attribute::Float(0.0), // threshold
            ])
        ]));
        
        op.attributes = attrs;
        
        // Verify the operation structure
        assert_eq!(op.op_type, "complex_conv2d");
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.attributes.len(), 6); // kernel_size, strides, padding, groups, dilations, activation_config
        
        // Verify specific attributes
        assert!(op.attributes.contains_key("kernel_size"));
        assert!(op.attributes.contains_key("padding"));
        
        // Check the kernel_size array
        if let Some(Attribute::Array(kernel_arr)) = op.attributes.get("kernel_size") {
            assert_eq!(kernel_arr.len(), 2);
            if let Attribute::Int(3) = kernel_arr[0] { } else { panic!("Expected Int(3)"); }
            if let Attribute::Int(3) = kernel_arr[1] { } else { panic!("Expected Int(3)"); }
        } else {
            panic!("Expected kernel_size to be an array");
        }
        
        // Check the activation config
        if let Some(Attribute::Array(act_arr)) = op.attributes.get("activation_config") {
            assert_eq!(act_arr.len(), 2);
            if let Attribute::String(s) = &act_arr[0] {
                assert_eq!(s, "relu");
            } else {
                panic!("Expected 'relu' string");
            }
        } else {
            panic!("Expected activation_config to be an array");
        }
    }

    #[test]
    fn test_attribute_hashmap_operations() {
        // Test various HashMap operations with attributes to test hashing and equality
        
        let mut attr_map: HashMap<String, Attribute> = HashMap::new();
        
        // Insert various attribute types
        attr_map.insert("int_key".to_string(), Attribute::Int(100));
        attr_map.insert("float_key".to_string(), Attribute::Float(99.99));
        attr_map.insert("string_key".to_string(), Attribute::String("test_value".to_string()));
        attr_map.insert("bool_key".to_string(), Attribute::Bool(false));
        attr_map.insert("array_key".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
            Attribute::Int(3),
        ]));
        
        assert_eq!(attr_map.len(), 5);
        
        // Test retrieval
        assert_eq!(attr_map.get("int_key"), Some(&Attribute::Int(100)));
        assert_eq!(attr_map.get("float_key"), Some(&Attribute::Float(99.99)));
        assert_eq!(attr_map.get("nonexistent"), None);
        
        // Test removal
        let removed = attr_map.remove("bool_key");
        assert_eq!(removed, Some(Attribute::Bool(false)));
        assert_eq!(attr_map.len(), 4);
        assert_eq!(attr_map.get("bool_key"), None);
        
        // Test replacement
        attr_map.insert("int_key".to_string(), Attribute::Int(200));
        assert_eq!(attr_map.get("int_key"), Some(&Attribute::Int(200)));
        
        // Test iteration
        let mut keys_found = Vec::new();
        for (key, _value) in &attr_map {
            keys_found.push(key.clone());
        }
        
        keys_found.sort();
        let mut expected_keys = vec!["array_key", "float_key", "int_key", "string_key"];
        expected_keys.sort();
        
        assert_eq!(keys_found, expected_keys.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    }

    #[test]
    fn test_attribute_memory_efficiency_patterns() {
        // Test patterns that might reveal memory inefficiencies or problems
        
        // Create many operations with shared attribute patterns
        let mut operations = Vec::new();
        
        // Create a template for attributes
        let template_attrs = vec![
            ("precision".to_string(), Attribute::String("fp32".to_string())),
            ("device".to_string(), Attribute::String("gpu".to_string())),
            ("fusion".to_string(), Attribute::Bool(true)),
            ("tile_size".to_string(), Attribute::Int(1024)),
        ];
        
        for i in 0..1000 {
            let mut op = Operation::new(&format!("op_{}", i));
            let mut attrs = HashMap::new();
            
            // Add the template attributes
            for (key, attr) in &template_attrs {
                attrs.insert(key.clone(), attr.clone());
            }
            
            // Add operation-specific attribute
            attrs.insert("id".to_string(), Attribute::Int(i as i64));
            
            op.attributes = attrs;
            operations.push(op);
        }
        
        // Verify all operations have correct structure
        assert_eq!(operations.len(), 1000);
        
        // Check a few operations
        assert_eq!(operations[0].attributes.len(), 5); // 4 template + 1 id
        assert_eq!(operations[0].attributes.get("id"), Some(&Attribute::Int(0)));
        assert_eq!(operations[500].attributes.get("id"), Some(&Attribute::Int(500)));
        assert_eq!(operations[999].attributes.get("id"), Some(&Attribute::Int(999)));
        
        // Verify template attributes are consistent
        assert_eq!(operations[0].attributes.get("precision"), Some(&Attribute::String("fp32".to_string())));
        assert_eq!(operations[999].attributes.get("precision"), Some(&Attribute::String("fp32".to_string())));
        
        // Clean up to test memory de-allocation
        drop(operations);
    }

    #[test]
    fn test_attribute_pattern_matching_complex() {
        // Test complex pattern matching on attribute structures
        
        let complex_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(2.5),
                    Attribute::Bool(true),
                ])
            ]),
            Attribute::String("top_level".to_string()),
        ]);
        
        // Pattern match the complex structure
        match &complex_attr {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                
                match &outer[0] {
                    Attribute::Array(inner1) => {
                        assert_eq!(inner1.len(), 2);
                        
                        match &inner1[0] {
                            Attribute::Int(1) => {},
                            _ => panic!("Expected Int(1)"),
                        }
                        
                        match &inner1[1] {
                            Attribute::Array(deep_nested) => {
                                assert_eq!(deep_nested.len(), 2);
                                
                                match &deep_nested[0] {
                                    Attribute::Float(f) => assert!((f - 2.5).abs() < f64::EPSILON),
                                    _ => panic!("Expected Float(2.5)"),
                                }
                                
                                match &deep_nested[1] {
                                    Attribute::Bool(true) => {},
                                    _ => panic!("Expected Bool(true)"),
                                }
                            },
                            _ => panic!("Expected nested array in deep position"),
                        }
                    },
                    _ => panic!("Expected array in first position"),
                }
                
                match &outer[1] {
                    Attribute::String(s) => assert_eq!(s, "top_level"),
                    _ => panic!("Expected string in second position"),
                }
            },
            _ => panic!("Expected top-level array"),
        }
    }
}