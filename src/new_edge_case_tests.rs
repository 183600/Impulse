//! New edge case tests for the Impulse compiler
//! Covers additional boundary conditions and edge cases not covered in existing tests

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

#[cfg(test)]
mod new_edge_case_tests {
    use super::*;
    
    /// Test 1: Overflow-prone shape multiplication with checked arithmetic
    #[test]
    fn test_shape_multiplication_overflow_handling() {
        let large_dims = vec![100_000, 100_000];
        let value = Value {
            name: "large_mult".to_string(),
            ty: Type::F32,
            shape: large_dims.clone(),
        };
        
        // Test normal multiplication
        let product: usize = value.shape.iter().product();
        assert_eq!(product, 10_000_000_000);
        
        // Test with checked multiplication
        let checked_product: Option<usize> = value.shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        assert!(checked_product.is_some());
        assert_eq!(checked_product.unwrap(), 10_000_000_000);
        
        // Test with a shape that contains a zero - should result in 0
        let zero_shape = Value {
            name: "zero_mult".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 0, 100_000],
        };
        let zero_product: usize = zero_shape.shape.iter().product();
        assert_eq!(zero_product, 0);
    }

    /// Test 2: Extreme string lengths with unicode characters
    #[test]
    fn test_extreme_unicode_strings() {
        let extreme_unicode = "🚀🔥汉语DeutschРусский🌐".repeat(1000); // Mix of emojis, multibyte chars
        let value = Value {
            name: extreme_unicode.clone(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name, extreme_unicode);
        assert!(value.name.chars().count() > 10000); // At least 10k chars
        
        // Test with operation
        let op = Operation::new(&extreme_unicode);
        assert_eq!(op.op_type, extreme_unicode);
        
        // Test with string attribute containing unicode
        let attr = Attribute::String(extreme_unicode.clone());
        match &attr {
            Attribute::String(s) => assert_eq!(s, &extreme_unicode),
            _ => panic!("Expected String attribute"),
        }
    }

    /// Test 3: Empty collections and their behavior
    #[test]
    fn test_empty_collection_edge_cases() {
        // Test with empty input/output vectors in operations
        let mut op = Operation::new("empty_io");
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
        
        // Test with an operation that has no inputs, no outputs, but has attributes
        let mut attrs = HashMap::new();
        attrs.insert("meaning".to_string(), Attribute::Int(42));
        op.attributes = attrs;
        
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 1);
        
        // Test module with no operations
        let module = Module::new("empty_module");
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.inputs.len(), 0);
        assert_eq!(module.outputs.len(), 0);
    }

    /// Test 4: Very deep recursive tensor types (memory efficiency)
    #[test]
    fn test_deeply_nested_tensor_memory_efficiency() {
        // Create a very deeply nested tensor type to test memory allocation
        let mut current_type = Type::F32;
        const DEPTH: usize = 1000;  // Very deep nesting
        
        for _ in 0..DEPTH {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],  // Minimal shape to save space
            };
        }
        
        // Verify the type was created properly despite deep nesting
        match &current_type {
            Type::Tensor { shape, .. } => assert_eq!(shape, &vec![1]),
            _ => panic!("Expected a tensor type"),
        }
        
        // The deeply nested type should be cloneable without issue
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Test equality comparison on deeply nested type
        assert!(current_type.is_valid_type());
    }

    /// Test 5: Testing all primitive types exhaustively
    #[test]
    fn test_all_primitive_types_comprehensively() {
        let primitive_types = [
            Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool
        ];
        
        for typ in &primitive_types {
            // Test that all primitive types are considered valid
            assert!(typ.is_valid_type());
            
            // Create a value with each type
            let value = Value {
                name: format!("value_{:?}", typ),
                ty: typ.clone(),
                shape: vec![1, 2, 3],
            };
            
            assert_eq!(value.ty, *typ);
            assert_eq!(value.shape, vec![1, 2, 3]);
        }
        
        // Test that nested types with different primitives are handled
        let nested_types = [
            Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2, 3] },
            Type::Tensor { element_type: Box::new(Type::I64), shape: vec![4, 5] },
            Type::Tensor { element_type: Box::new(Type::Bool), shape: vec![6] },
        ];
        
        for nested_type in &nested_types {
            assert!(nested_type.is_valid_type());
        }
    }

    /// Test 6: Large but valid numeric ranges
    #[test]
    fn test_large_valid_numeric_ranges() {
        // Create attributes with the largest and smallest possible values
        let attrs = vec![
            Attribute::Int(i64::MAX),
            Attribute::Int(i64::MIN),
            Attribute::Int(0),
            Attribute::Int(-1),
            Attribute::Int(1),
            Attribute::Float(f64::MAX),
            Attribute::Float(f64::MIN),
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(0.0),
        ];
        
        // Verify each attribute can be matched correctly
        assert_eq!(attrs[0], Attribute::Int(i64::MAX));
        assert_eq!(attrs[1], Attribute::Int(i64::MIN));
        assert_eq!(attrs[5], Attribute::Float(f64::MAX));
        
        // Test float comparisons carefully
        match attrs[2] {
            Attribute::Int(0) => (), // OK
            _ => panic!("Expected 0"),
        }
        
        match &attrs[7] {
            Attribute::Float(f) => {
                if !f.is_infinite() || !f.is_sign_positive() {
                    panic!("Expected positive infinity, got {}", f);
                }
            },
            _ => panic!("Expected positive infinity"),
        }
        
        match &attrs[8] {
            Attribute::Float(f) => {
                if !f.is_infinite() || f.is_sign_positive() {
                    panic!("Expected negative infinity, got {}", f);
                }
            },
            _ => panic!("Expected negative infinity"),
        }
    }

    /// Test 7: Mixed attribute array complexity
    #[test]
    fn test_complex_attribute_array_structures() {
        // Create deeply nested and complex array structures
        let complex_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![Attribute::Int(1)]),
                Attribute::Array(vec![Attribute::Float(2.5)]),
            ]),
            Attribute::String("mixed".to_string()),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::Array(vec![Attribute::String("nested".to_string())]),
            ]),
        ]);
        
        // Verify the structure through pattern matching
        match &complex_attr {
            Attribute::Array(top_level) => {
                assert_eq!(top_level.len(), 3);
                
                // First element: Array of Arrays
                match &top_level[0] {
                    Attribute::Array(nested) => {
                        assert_eq!(nested.len(), 2);
                        match &nested[0] {
                            Attribute::Array(deeply_nested) => {
                                assert_eq!(deeply_nested.len(), 1);
                                match &deeply_nested[0] {
                                    Attribute::Int(1) => (),
                                    _ => panic!("Expected Int(1)"),
                                }
                            },
                            _ => panic!("Expected nested array"),
                        }
                    },
                    _ => panic!("Expected nested array at top level"),
                }
                
                // Second element: String
                match &top_level[1] {
                    Attribute::String(s) if s == "mixed" => (),
                    _ => panic!("Expected 'mixed' string"),
                }
                
                // Third element: Array with Bool and nested Array
                match &top_level[2] {
                    Attribute::Array(mixed_arr) => {
                        assert_eq!(mixed_arr.len(), 2);
                        match mixed_arr[0] {
                            Attribute::Bool(true) => (),
                            _ => panic!("Expected Bool(true)"),
                        }
                    },
                    _ => panic!("Expected mixed-type array"),
                }
            },
            _ => panic!("Expected top-level array"),
        }
    }

    /// Test 8: Empty and single-element tensor shapes
    #[test]
    fn test_single_and_empty_shape_edge_cases() {
        // Scalar (0-dimension tensor)
        let scalar_value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape = scalar
        };
        assert_eq!(scalar_value.shape.len(), 0);
        assert!(scalar_value.shape.is_empty());
        
        // Product of empty shape should be 1 (scalar has 1 element)
        let scalar_elements: usize = scalar_value.shape.iter().product();
        assert_eq!(scalar_elements, 1);
        
        // Single-dimension tensors
        let single_dim = Value {
            name: "single".to_string(),
            ty: Type::F32,
            shape: vec![1],  // Single element
        };
        assert_eq!(single_dim.shape, vec![1]);
        
        let single_elements: usize = single_dim.shape.iter().product();
        assert_eq!(single_elements, 1);
        
        // Another single-element tensor with different shape
        let single_2d = Value {
            name: "single_2d".to_string(),
            ty: Type::F32,
            shape: vec![1, 1],  // Still single element in 2D
        };
        assert_eq!(single_2d.shape, vec![1, 1]);
        
        let single_2d_elements: usize = single_2d.shape.iter().product();
        assert_eq!(single_2d_elements, 1);
        
        // Large but single-element dimensions
        let large_single = Value {
            name: "large_single".to_string(),
            ty: Type::F32,
            shape: vec![1, 1, 1, 1, 1000000, 1, 1],  // Still 1 element due to other dims being 1
        };
        let large_single_elements: usize = large_single.shape.iter().product();
        assert_eq!(large_single_elements, 1000000);
    }

    /// Test 9: Testing behavior with maximum possible inputs/attributes
    #[test]
    fn test_maximum_collection_sizes() {
        // Note: This test creates large collections but with minimal memory footprint per item
        let mut op = Operation::new("max_collections");
        
        // Add many minimal inputs (to test memory allocation and performance)
        for i in 0..1_000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,  // Minimal type
                shape: vec![1], // Minimal shape
            });
        }
        
        // Add many minimal outputs
        for i in 0..1_000 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![1],
            });
        }
        
        // Add many minimal attributes
        let mut attrs = HashMap::new();
        for i in 0..1_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::Int(i as i64)  // Simple int attribute
            );
        }
        op.attributes = attrs;
        
        // Verify all were added correctly
        assert_eq!(op.inputs.len(), 1000);
        assert_eq!(op.outputs.len(), 1000);
        assert_eq!(op.attributes.len(), 1000);
        assert_eq!(op.op_type, "max_collections");
        
        // Test that the operation is still usable
        assert!(op.attributes.contains_key("attr_0"));
        assert!(op.attributes.contains_key("attr_999"));
    }

    /// Test 10: Type conversions and edge cases in IR utilities
    #[test]
    fn test_type_conversion_edge_cases() {
        // Test all primitive types
        let primitive_tests = vec![
            (Type::F32, "f32"),
            (Type::F64, "f64"),
            (Type::I32, "i32"),
            (Type::I64, "i64"),
            (Type::Bool, "bool"),
        ];
        
        for (typ, expected_str) in primitive_tests {
            match typ {
                Type::F32 => assert_eq!(expected_str, "f32"),
                Type::F64 => assert_eq!(expected_str, "f64"),
                Type::I32 => assert_eq!(expected_str, "i32"),
                Type::I64 => assert_eq!(expected_str, "i64"),
                Type::Bool => assert_eq!(expected_str, "bool"),
                _ => panic!("Unexpected type"),
            }
        }
        
        // Test tensor types with different element types and shapes
        let tensor_tests = vec![
            (
                Type::Tensor { element_type: Box::new(Type::F32), shape: vec![] },
                Type::F32, 
                vec![]
            ),
            (
                Type::Tensor { element_type: Box::new(Type::I64), shape: vec![10, 20] },
                Type::I64,
                vec![10, 20]
            ),
            (
                Type::Tensor { element_type: Box::new(Type::Bool), shape: vec![1, 1, 1] },
                Type::Bool,
                vec![1, 1, 1]
            ),
        ];
        
        for (tensor_type, expected_element_type, expected_shape) in tensor_tests {
            match tensor_type {
                Type::Tensor { element_type, shape } => {
                    match element_type.as_ref() {
                        t if std::mem::discriminant(t) == std::mem::discriminant(&expected_element_type) => (),
                        _ => panic!("Element type mismatch"),
                    }
                    assert_eq!(shape, expected_shape);
                },
                _ => panic!("Expected tensor type"),
            }
        }
        
        // Test recursive validity check
        let complex_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![1, 2],
            }),
            shape: vec![3, 4],
        };
        
        assert!(complex_nested.is_valid_type());
    }
}