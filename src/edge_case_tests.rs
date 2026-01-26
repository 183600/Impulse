//! Edge case tests for the Impulse compiler
//! Covers critical boundary conditions and edge cases

use crate::ir::{Module, Value, Type, Operation, Attribute};

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    use std::collections::HashMap;
    
    /// Test 1: Empty tensor shapes (scalar values)
    #[test]
    fn test_empty_tensor_shape() {
        let value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        
        assert!(value.shape.is_empty());
        assert_eq!(value.shape.len(), 0);
        
        // Scalar should have 1 element when calculating product
        let element_count: usize = value.shape.iter().product();
        assert_eq!(element_count, 1);
    }

    /// Test 2: Tensor shapes with zeros (zero-sized tensors)
    #[test]
    fn test_zero_sized_tensor_shapes() {
        let shapes_with_zeros = vec![
            vec![0],           // Single zero
            vec![0, 5],        // Zero followed by value
            vec![5, 0],        // Value followed by zero
            vec![2, 0, 4],     // Zero in middle
            vec![0, 0, 0],     // All zeros
        ];

        for shape in shapes_with_zeros {
            let value = Value {
                name: "zero_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            
            // Any tensor with a zero dimension should have zero elements
            let total_elements: usize = value.shape.iter().product();
            assert_eq!(total_elements, 0, "Shape {:?} should have 0 elements", shape);
        }
    }

    /// Test 3: Large numeric values in tensor shapes
    #[test]
    fn test_large_shape_values() {
        // Use values that are large but manageable for memory
        let large_shape = vec![1000, 1000];
        let value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: large_shape,
        };
        
        assert_eq!(value.shape, vec![1000, 1000]);
        let total_elements: usize = value.shape.iter().product();
        assert_eq!(total_elements, 1_000_000);  // 1 million elements
        
        // Test with single large dimension
        let single_large = Value {
            name: "single_large".to_string(),
            ty: Type::I64,
            shape: vec![100_000],
        };
        
        assert_eq!(single_large.shape, vec![100_000]);
        assert_eq!(single_large.shape[0], 100_000);
    }

    /// Test 4: Nested tensor types with moderate depth
    #[test]
    fn test_nested_tensor_types() {
        // Create a moderately nested tensor type to test recursion behavior
        let mut current_type = Type::F32;
        const NESTING_DEPTH: usize = 10;  // Moderate depth to avoid stack overflow
        
        for _ in 0..NESTING_DEPTH {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Verify the type was constructed properly
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected a nested tensor type"),
        }
        
        // Test that cloning works for moderate nesting
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 5: Operations with many inputs/outputs
    #[test]
    fn test_operations_with_many_inputs_outputs() {
        let mut op = Operation::new("many_io_op");
        
        // Add a moderate number of inputs
        for i in 0..100 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1],  // Small shape to conserve memory
            });
        }
        
        // Add a moderate number of outputs
        for i in 0..50 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],  // Small shape to conserve memory
            });
        }
        
        assert_eq!(op.inputs.len(), 100);
        assert_eq!(op.outputs.len(), 50);
        assert_eq!(op.op_type, "many_io_op");
    }

    /// Test 6: Operations with many attributes
    #[test]
    fn test_operations_with_many_attributes() {
        let mut op = Operation::new("many_attrs_op");
        let mut attrs = HashMap::new();
        
        // Add a moderate number of different types of attributes
        for i in 0..500 {
            match i % 5 {
                0 => { attrs.insert(format!("int_attr_{}", i), Attribute::Int(i as i64)); }
                1 => { attrs.insert(format!("float_attr_{}", i), Attribute::Float(i as f64)); }
                2 => { attrs.insert(format!("string_attr_{}", i), Attribute::String(format!("value_{}", i))); }
                3 => { attrs.insert(format!("bool_attr_{}", i), Attribute::Bool(i % 2 == 0)); }
                _ => { attrs.insert(format!("array_attr_{}", i), Attribute::Array(vec![
                    Attribute::Int(i as i64),
                    Attribute::String(format!("inner_{}", i))
                ])); }
            }
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 500);
        assert_eq!(op.op_type, "many_attrs_op");
        
        // Verify a few specific attributes
        assert!(op.attributes.contains_key("int_attr_0"));
        assert!(op.attributes.contains_key("string_attr_2"));  // index 2 % 5 = 2 => string_attr
        assert!(op.attributes.contains_key("array_attr_4"));   // index 4 % 5 = 4 => array_attr
    }

    /// Test 7: Unicode and special character handling in identifiers
    #[test]
    fn test_unicode_identifiers() {
        let test_cases = [
            ("tensor_名称", Type::F32),  // Chinese characters
            ("tensor_🚀_unicode", Type::I32),  // Emoji
            ("tensor_with_café", Type::F64),  // Accented characters
            ("tensor_مرحبا", Type::I64),  // Arabic characters
            ("tensor_🎉_more_emojis", Type::Bool),  // More emojis
        ];
        
        for (name, data_type) in &test_cases {
            let value = Value {
                name: name.to_string(),
                ty: data_type.clone(),
                shape: vec![1, 2, 3],
            };
            
            assert_eq!(value.name, *name);
            
            let op = Operation::new(name);
            assert_eq!(op.op_type, *name);
            
            let module = Module::new(name.to_string());
            assert_eq!(module.name, *name);
        }
    }

    /// Test 8: Special floating-point values in attributes
    #[test]
    fn test_special_float_values() {
        let special_values = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -0.0,  // Negative zero
            f64::EPSILON,
            std::f64::consts::PI,
            std::f64::consts::E,
        ];
        
        for (i, &val) in special_values.iter().enumerate() {
            let attr = Attribute::Float(val);
            
            // Extract the value and verify it matches (with special handling for NaN)
            if val.is_nan() {
                if let Attribute::Float(retrieved_val) = attr {
                    assert!(retrieved_val.is_nan(), "NaN test case {}", i);
                } else {
                    panic!("Expected Float attribute for NaN");
                }
            } else if val.is_infinite() {
                if let Attribute::Float(retrieved_val) = attr {
                    assert!(retrieved_val.is_infinite(), "Infinity test case {}", i);
                    assert_eq!(retrieved_val.is_sign_positive(), val.is_sign_positive());
                } else {
                    panic!("Expected Float attribute for infinity");
                }
            } else {
                match attr {
                    Attribute::Float(retrieved_val) => {
                        // For most special values, we can use approximate equality
                        // except for specific comparisons needed
                        if (val - retrieved_val).abs() < f64::EPSILON || 
                           (val.is_infinite() && retrieved_val.is_infinite() && 
                            val.is_sign_positive() == retrieved_val.is_sign_positive()) {
                            // Passes the test
                        } else {
                            assert_eq!(retrieved_val, val, "Special float test case {}", i);
                        }
                    },
                    _ => panic!("Expected Float attribute"),
                }
            }
        }
    }

    /// Test 9: Long string values
    #[test]
    fn test_long_strings() {
        let long_name = "a".repeat(10_000);  // 10k character string
        let value = Value {
            name: long_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name.len(), 10_000);
        assert_eq!(value.name, long_name);
        
        let long_op = Operation::new(&long_name);
        assert_eq!(long_op.op_type.len(), 10_000);
        assert_eq!(long_op.op_type, long_name);
        
        let long_module = Module::new(long_name.clone());
        assert_eq!(long_module.name.len(), 10_000);
        assert_eq!(long_module.name, long_name);
        
        // Test with long string attribute
        let long_attr = Attribute::String(long_name.clone());
        match &long_attr {
            Attribute::String(s) => assert_eq!(s.len(), 10_000),
            _ => panic!("Expected String attribute"),
        }
    }

    /// Test 10: Complex nested array attributes
    #[test]
    fn test_complex_nested_array_attributes() {
        // Create a complex nested array structure
        let complex_array = Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Float(3.14159),
            Attribute::String("nested".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(1),
                    Attribute::Int(2),
                    Attribute::Array(vec![
                        Attribute::String("deeply_nested".to_string())
                    ])
                ]),
                Attribute::Float(2.71828)
            ]),
            Attribute::Array(vec![
                Attribute::Bool(false),
                Attribute::Int(999)
            ])
        ]);
        
        // Validate the structure
        match &complex_array {
            Attribute::Array(top_level) => {
                assert_eq!(top_level.len(), 6);
                
                // Check first few elements
                match &top_level[0] {
                    Attribute::Int(42) => (),
                    _ => panic!("Expected Int(42) at index 0"),
                }
                
                match &top_level[2] {
                    Attribute::String(s) if s == "nested" => (),
                    _ => panic!("Expected String('nested') at index 2"),
                }
                
                match &top_level[3] {
                    Attribute::Bool(true) => (),
                    _ => panic!("Expected Bool(true) at index 3"),
                }
                
                // Check nested array
                match &top_level[4] {
                    Attribute::Array(nested) => {
                        assert_eq!(nested.len(), 2);
                        
                        match &nested[0] {
                            Attribute::Array(deeply_nested) => {
                                assert_eq!(deeply_nested.len(), 3);  // Int(1), Int(2), Array(...)
                                
                                match &deeply_nested[2] {
                                    Attribute::Array(deepest) => {
                                        assert_eq!(deepest.len(), 1);
                                        match &deepest[0] {
                                            Attribute::String(s) if s == "deeply_nested" => (),
                                            _ => panic!("Expected deeply nested string"),
                                        }
                                    },
                                    _ => panic!("Expected deepest nested array"),
                                }
                            },
                            _ => panic!("Expected deeply nested array at nested[0]"),
                        }
                    },
                    _ => panic!("Expected nested array at top_level[4]"),
                }
            },
            _ => panic!("Expected top-level array"),
        }
    }
}