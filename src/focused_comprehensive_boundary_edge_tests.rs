//! Focused comprehensive boundary edge tests
//! Additional edge cases covering numerical boundaries, overflow scenarios, and edge conditions

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use std::collections::HashMap;

#[cfg(test)]
mod focused_boundary_edge_tests {
    use super::*;

    /// Test 1: Shape calculation with maximum safe values before overflow
    #[test]
    fn test_shape_max_safe_values() {
        // Test values that are close to overflow but still safe
        let safe_dims = [
            vec![100_000, 100],      // 10 million elements
            vec![10_000, 1_000],     // 10 million elements
            vec![1_000_000, 10],     // 10 million elements
        ];

        for shape in safe_dims.iter() {
            let value = Value {
                name: "safe_large".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            // num_elements uses checked_mul internally, so this should succeed
            assert_eq!(value.num_elements(), Some(shape.iter().product()));
        }
    }

    /// Test 2: Shape values at overflow boundary
    #[test]
    fn test_shape_overflow_boundary() {
        // Test with values that should cause overflow detection
        let overflow_shapes = [
            vec![usize::MAX, 2],
            vec![usize::MAX / 2 + 1, 2],
        ];

        for shape in overflow_shapes.iter() {
            let value = Value {
                name: "overflow_test".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            // Should return None due to overflow detection
            assert_eq!(value.num_elements(), None);
        }
    }

    /// Test 3: Negative zero and signed zero handling in floats
    #[test]
    fn test_float_signed_zero() {
        let positive_zero = Attribute::Float(0.0);
        let negative_zero = Attribute::Float(-0.0);

        match positive_zero {
            Attribute::Float(val) => {
                assert_eq!(val, 0.0);
                assert!(!val.is_sign_negative());
            }
            _ => panic!("Expected Float attribute"),
        }

        match negative_zero {
            Attribute::Float(val) => {
                assert_eq!(val, 0.0);
                assert!(val.is_sign_negative());
            }
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 4: Module with cyclic-like operation dependencies (names only)
    #[test]
    fn test_module_cyclic_operation_names() {
        let mut module = Module::new("cyclic_names");

        // Create operations with names that suggest a cycle
        let op1 = Operation::new("op_a");
        let op2 = Operation::new("op_b");
        let op3 = Operation::new("op_c");

        module.add_operation(op1);
        module.add_operation(op2);
        module.add_operation(op3);

        assert_eq!(module.operations.len(), 3);
        assert_eq!(module.operations[0].op_type, "op_a");
        assert_eq!(module.operations[1].op_type, "op_b");
        assert_eq!(module.operations[2].op_type, "op_c");
    }

    /// Test 5: Array attribute with single element (edge case of minimal array)
    #[test]
    fn test_single_element_array() {
        let single_int = Attribute::Array(vec![Attribute::Int(42)]);
        let single_float = Attribute::Array(vec![Attribute::Float(3.14)]);
        let single_string = Attribute::Array(vec![Attribute::String("test".to_string())]);

        match single_int {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
                match &arr[0] {
                    Attribute::Int(42) => {}
                    _ => panic!("Expected Int(42)"),
                }
            }
            _ => panic!("Expected Array attribute"),
        }

        match single_float {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
            }
            _ => panic!("Expected Array attribute"),
        }

        match single_string {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1);
            }
            _ => panic!("Expected Array attribute"),
        }
    }

    /// Test 6: String attribute with various special characters
    #[test]
    fn test_special_character_strings() {
        let special_strings = [
            "",                    // Empty string
            " ",                   // Single space
            "  ",                  // Multiple spaces
            "\t",                  // Tab
            "\n",                  // Newline
            "\r",                  // Carriage return
            "hello\r\nworld",      // CRLF
            "path/to/file",        // Slash
            "C:\\Windows\\Path",   // Backslash
            "a\"b\"c",             // Quotes
            "a\\b\\c",             // Backslash escapes
        ];

        for s in special_strings.iter() {
            let attr = Attribute::String(s.to_string());
            match attr {
                Attribute::String(val) => {
                    assert_eq!(val, *s);
                }
                _ => panic!("Expected String attribute"),
            }
        }
    }

    /// Test 7: Value with single dimension of 1 (vector vs scalar distinction)
    #[test]
    fn test_dimension_one_vector() {
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };

        let vector1d = Value {
            name: "vector1d".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };

        let vector1d_multi = Value {
            name: "vector1d_multi".to_string(),
            ty: Type::F32,
            shape: vec![1, 1],
        };

        // Scalar has empty shape
        assert_eq!(scalar.shape.len(), 0);
        assert_eq!(scalar.num_elements(), Some(1));

        // 1D vector of size 1
        assert_eq!(vector1d.shape.len(), 1);
        assert_eq!(vector1d.shape[0], 1);
        assert_eq!(vector1d.num_elements(), Some(1));

        // Multi-dimensional vector of ones
        assert_eq!(vector1d_multi.shape.len(), 2);
        assert_eq!(vector1d_multi.num_elements(), Some(1));
    }

    /// Test 8: Operation with empty attribute map vs None
    #[test]
    fn test_empty_vs_none_attributes() {
        let mut op1 = Operation::new("op_with_empty_attrs");
        op1.attributes = HashMap::new();

        let mut op2 = Operation::new("op_with_default_attrs");

        // Both should have empty attributes
        assert_eq!(op1.attributes.len(), 0);
        assert_eq!(op2.attributes.len(), 0);

        // Both should be able to add attributes
        op1.attributes.insert("key".to_string(), Attribute::Int(1));
        op2.attributes.insert("key".to_string(), Attribute::Int(2));

        assert_eq!(op1.attributes.len(), 1);
        assert_eq!(op2.attributes.len(), 1);
    }

    /// Test 9: Tensor type with single element shape (1x1 tensor)
    #[test]
    fn test_single_element_tensor() {
        let tensor1x1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![1, 1],
        };

        let tensor1x1x1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![1, 1, 1],
        };

        match tensor1x1 {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, vec![1, 1]);
            }
            _ => panic!("Expected Tensor type"),
        }

        match tensor1x1x1 {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, vec![1, 1, 1]);
            }
            _ => panic!("Expected Tensor type"),
        }
    }

    /// Test 10: Value with all type variants
    #[test]
    fn test_value_with_all_type_variants() {
        let types = [
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
            Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 3],
            },
        ];

        for (i, ty) in types.iter().enumerate() {
            let value = Value {
                name: format!("value_{}", i),
                ty: ty.clone(),
                shape: vec![10, 10],
            };

            assert_eq!(value.ty, *ty);
            assert!(value.ty.is_valid_type());
        }
    }
}
