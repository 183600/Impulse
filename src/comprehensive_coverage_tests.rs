//! Comprehensive coverage tests - boundary cases and edge scenarios
//! 
//! This module provides additional test coverage for various boundary conditions,
//! numeric edge cases, and system limits.

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

#[cfg(test)]
mod comprehensive_coverage_tests {
    use super::*;

    /// Test 1: NaN and Infinity handling in float attributes
    #[test]
    fn test_nan_infinity_attributes() {
        let nan_attr = Attribute::Float(f64::NAN);
        let inf_attr = Attribute::Float(f64::INFINITY);
        let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
        
        // Verify attributes are created (NaN != NaN is a property of floating point)
        match nan_attr {
            Attribute::Float(v) => assert!(v.is_nan()),
            _ => panic!("Expected Float(NAN)"),
        }
        
        match inf_attr {
            Attribute::Float(v) => assert!(v.is_infinite() && v > 0.0),
            _ => panic!("Expected Float(INFINITY)"),
        }
        
        match neg_inf_attr {
            Attribute::Float(v) => assert!(v.is_infinite() && v < 0.0),
            _ => panic!("Expected Float(NEG_INFINITY)"),
        }
    }

    /// Test 2: Integer boundary values (MIN and MAX)
    #[test]
    fn test_integer_boundaries() {
        let i32_max = Attribute::Int(i64::MAX);
        let i32_min = Attribute::Int(i64::MIN);
        let i32_neg_one = Attribute::Int(-1);
        let i32_pos_one = Attribute::Int(1);
        
        assert_eq!(i32_max, Attribute::Int(i64::MAX));
        assert_eq!(i32_min, Attribute::Int(i64::MIN));
        assert_ne!(i32_neg_one, i32_pos_one);
        assert_ne!(i32_max, i32_min);
    }

    /// Test 3: Unicode and special characters in names
    #[test]
    fn test_unicode_and_special_characters() {
        let unicode_module = Module::new("module_模块_test🚀");
        assert_eq!(unicode_module.name, "module_模块_test🚀");
        
        let special_value = Value {
            name: "test/value!@#$%^&*()".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(special_value.name, "test/value!@#$%^&*()");
    }

    /// Test 4: Module with all types of input-output shapes
    #[test]
    fn test_all_shape_patterns() {
        let patterns = vec![
            vec![],                    // Scalar (0-d)
            vec![0],                   // Empty 1D
            vec![1],                   // Single element 1D
            vec![1, 1],                // Single element 2D
            vec![2, 3],                // Small 2D
            vec![100, 100, 100],       // Large 3D
            vec![1, 3, 224, 224],      // Image-like (NCHW)
            vec![224, 224, 3],         // Image-like (HWC)
            vec![0, 10, 10],           // Mixed with zero
        ];
        
        for (i, shape) in patterns.iter().enumerate() {
            let value = Value {
                name: format!("shape_test_{}", i),
                ty: Type::F32,
                shape: shape.clone(),
            };
            assert_eq!(value.shape, *shape);
        }
    }

    /// Test 5: Attribute with very large numbers and precision
    #[test]
    fn test_large_numbers_and_precision() {
        let huge_int = Attribute::Int(999999999999999999i64);
        let huge_float = Attribute::Float(1e308);
        let tiny_float = Attribute::Float(1e-150);
        
        match huge_int {
            Attribute::Int(v) => assert_eq!(v, 999999999999999999),
            _ => panic!("Expected huge int"),
        }
        
        match huge_float {
            Attribute::Float(v) => assert!(v.is_normal() && v > 1e300),
            _ => panic!("Expected huge float"),
        }
        
        match tiny_float {
            Attribute::Float(v) => assert!(v.is_finite() && v < 1e-100),
            _ => panic!("Expected tiny float"),
        }
    }

    /// Test 6: Module clone and equality operations
    #[test]
    fn test_module_clone_equality() {
        let mut original = Module::new("test_module");
        
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: "x".to_string(),
            ty: Type::F32,
            shape: vec![2, 2],
        });
        original.add_operation(op);
        
        let cloned = original.clone();
        
        assert_eq!(original.name, cloned.name);
        assert_eq!(original.operations.len(), cloned.operations.len());
        assert_eq!(original.operations[0].op_type, cloned.operations[0].op_type);
    }

    /// Test 7: Operation with all possible data types in attributes
    #[test]
    fn test_operation_all_data_type_attrs() {
        let mut op = Operation::new("type_test_op");
        let mut attrs = HashMap::new();
        
        // All primitive types
        attrs.insert("i32_val".to_string(), Attribute::Int(42));
        attrs.insert("f32_val".to_string(), Attribute::Float(3.14));
        attrs.insert("bool_val".to_string(), Attribute::Bool(true));
        attrs.insert("str_val".to_string(), Attribute::String("test".to_string()));
        attrs.insert("arr_val".to_string(), Attribute::Array(vec![
            Attribute::Int(1), Attribute::Int(2),
        ]));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 5);
        assert!(op.attributes.contains_key("i32_val"));
        assert!(op.attributes.contains_key("arr_val"));
    }

    /// Test 8: Value with maximum reasonable element count
    #[test]
    fn test_max_element_count() {
        // Test shape that would result in a large but manageable element count
        let value = Value {
            name: "max_elements".to_string(),
            ty: Type::F32,
            shape: vec![100, 100, 100],  // 1 million elements
        };
        
        assert_eq!(value.num_elements(), Some(1_000_000));
    }

    /// Test 9: Module with empty string names
    #[test]
    fn test_empty_string_names() {
        let module = Module::new("");
        assert_eq!(module.name, "");
        
        let value = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, "");
        
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
    }

    /// Test 10: Nested tensor with all primitive types
    #[test]
    fn test_nested_tensors_all_types() {
        let types = vec![
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
        ];
        
        for base_type in types {
            let nested = Type::Tensor {
                element_type: Box::new(base_type.clone()),
                shape: vec![2, 2],
            };
            
            match nested {
                Type::Tensor { element_type, shape } => {
                    assert_eq!(shape, vec![2, 2]);
                    assert_eq!(*element_type, base_type);
                }
                _ => panic!("Expected Tensor type"),
            }
        }
    }
}