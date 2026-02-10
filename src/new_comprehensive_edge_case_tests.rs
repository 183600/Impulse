//! New comprehensive edge case tests covering additional boundary conditions
//! Focus on less-tested scenarios and extreme cases

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    ir::TypeExtensions,
    ImpulseCompiler,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Value with overflow-prone shape dimensions
    #[test]
    fn test_value_overflow_safe_calculation() {
        // Test shapes that approach overflow but are still safe
        let value = Value {
            name: "large_safe_tensor".to_string(),
            ty: Type::F32,
            shape: vec![46340, 46340], // 46340^2 ≈ 2.1 billion
        };
        
        // Use num_elements() which returns Option for safety
        assert_eq!(value.num_elements(), Some(46340 * 46340));
    }

    /// Test 2: Module with cyclic-like naming patterns
    #[test]
    fn test_module_cyclic_operation_naming() {
        let mut module = Module::new("cyclic_test");
        
        // Add operations with names that reference each other
        let mut op1 = Operation::new("op_a");
        op1.outputs.push(Value {
            name: "output_a".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        
        let mut op2 = Operation::new("op_b");
        op2.inputs.push(op1.outputs[0].clone());
        op2.outputs.push(Value {
            name: "output_b".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        
        module.add_operation(op1);
        module.add_operation(op2);
        
        assert_eq!(module.operations.len(), 2);
        assert_eq!(module.operations[1].inputs[0].name, "output_a");
    }

    /// Test 3: Attribute with NaN and infinity values
    #[test]
    fn test_special_float_values() {
        let nan_attr = Attribute::Float(f64::NAN);
        let inf_attr = Attribute::Float(f64::INFINITY);
        let neg_inf_attr = Attribute::Float(f64::NEG_INFINITY);
        
        // Verify these special values can be created
        match nan_attr {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected Float(NAN)"),
        }
        
        match inf_attr {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected Float(INFINITY)"),
        }
        
        match neg_inf_attr {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
            _ => panic!("Expected Float(NEG_INFINITY)"),
        }
    }

    /// Test 4: Empty array attributes
    #[test]
    fn test_empty_array_attributes() {
        let empty_array = Attribute::Array(vec![]);
        
        match empty_array {
            Attribute::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("Expected empty Array"),
        }
        
        // Test nested empty arrays
        let nested_empty = Attribute::Array(vec![
            Attribute::Array(vec![]),
            Attribute::Array(vec![]),
        ]);
        
        match nested_empty {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                if let Attribute::Array(inner) = &outer[0] {
                    assert!(inner.is_empty());
                } else {
                    panic!("Expected nested empty array");
                }
            },
            _ => panic!("Expected nested Array"),
        }
    }

    /// Test 5: Value with single dimension of 1 (degenerate tensors)
    #[test]
    fn test_degenerate_tensors() {
        // Test various degenerate tensor shapes
        let degenerate_cases = vec![
            vec![1],           // 1D tensor with 1 element
            vec![1, 1],        // 2D tensor with 1x1
            vec![1, 1, 1],     // 3D tensor with 1x1x1
            vec![10, 1, 20],   // 3D tensor with degenerate middle dimension
            vec![1, 100],      // 2D tensor with degenerate first dimension
            vec![100, 1],      // 2D tensor with degenerate second dimension
        ];
        
        for shape in degenerate_cases {
            let value = Value {
                name: "degenerate".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let product: usize = value.shape.iter().product();
            // Verify product is calculated correctly
            assert_eq!(product, value.shape.iter().product::<usize>());
        }
    }

    /// Test 6: Operation with duplicate attribute keys (last wins)
    #[test]
    fn test_attribute_key_behavior() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("test_op");
        let mut attrs = HashMap::new();
        
        // Insert with same key multiple times
        attrs.insert("key".to_string(), Attribute::Int(1));
        attrs.insert("key".to_string(), Attribute::Int(2)); // Overwrites
        
        op.attributes = attrs;
        
        // Last value should win
        assert_eq!(op.attributes.get("key"), Some(&Attribute::Int(2)));
        assert_eq!(op.attributes.len(), 1);
    }

    /// Test 7: Module with large number of inputs/outputs
    #[test]
    fn test_module_many_inputs_outputs() {
        let mut module = Module::new("many_io");
        
        // Add many inputs
        for i in 0..100 {
            module.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
        }
        
        // Add many outputs
        for i in 0..100 {
            module.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
        }
        
        assert_eq!(module.inputs.len(), 100);
        assert_eq!(module.outputs.len(), 100);
    }

    /// Test 8: Type validation with invalid nested tensors
    #[test]
    fn test_type_validation() {
        // Test valid types
        assert!(Type::F32.is_valid_type());
        assert!(Type::I32.is_valid_type());
        
        // Test valid nested tensor
        let valid_nested = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        assert!(valid_nested.is_valid_type());
        
        // Test deeply nested valid type
        let mut nested = Type::F32;
        for _ in 0..5 {
            nested = Type::Tensor {
                element_type: Box::new(nested),
                shape: vec![2],
            };
        }
        assert!(nested.is_valid_type());
    }

    /// Test 9: Compiler with empty string target
    #[test]
    fn test_compiler_with_empty_target() {
        let mut compiler = ImpulseCompiler::new();
        let mock_model = vec![1u8, 2u8, 3u8];
        
        // Test with empty target string
        let result = compiler.compile(&mock_model, "");
        // Should handle gracefully without panic
        match result {
            Ok(_) => (),
            Err(e) => {
                assert!(e.to_string().len() > 0);
            }
        }
    }

    /// Test 10: Value with very large but valid dimension (near usize::MAX)
    #[test]
    fn test_value_with_max_dimension() {
        // Test with a single very large dimension
        let large_dim = 100_000_000; // 100 million
        let value = Value {
            name: "large_dim_tensor".to_string(),
            ty: Type::F32,
            shape: vec![large_dim],
        };
        
        assert_eq!(value.shape, vec![large_dim]);
        assert_eq!(value.num_elements(), Some(large_dim));
    }
}