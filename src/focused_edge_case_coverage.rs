//! Focused edge case tests covering specific boundary conditions
//! Targeting less-tested scenarios across IR, compiler, and runtime

#[cfg(test)]
mod tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use crate::runtime::{Device, ExecutionContext};
    use crate::compiler::Compiler;
    use std::collections::HashMap;

    /// Test 1: Operation with identical input and output values
    #[test]
    fn test_operation_identity_same_input_output() {
        let mut op = Operation::new("identity");
        let value = Value {
            name: "same_value".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        };
        op.inputs.push(value.clone());
        op.outputs.push(value);
        
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.inputs[0].name, op.outputs[0].name);
    }

    /// Test 2: Module with operations that reference non-existent values
    #[test]
    fn test_module_orphaned_operations() {
        let mut module = Module::new("orphan_test");
        
        // Add operation with inputs that don't exist in module.inputs
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: "non_existent_input".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        module.add_operation(op);
        
        // Module should still be valid even with orphaned inputs
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.inputs.len(), 0); // Empty, no registered inputs
    }

    /// Test 3: Tensor with shape containing extremely large dimension followed by zero
    #[test]
    fn test_tensor_shape_large_then_zero() {
        let value = Value {
            name: "large_then_zero".to_string(),
            ty: Type::I64,
            shape: vec![usize::MAX / 2, 0, 10],
        };
        
        // Should have 0 total elements due to zero dimension
        assert_eq!(value.num_elements(), Some(0));
        assert_eq!(value.shape.len(), 3);
    }

    /// Test 4: Compiler state persistence across multiple operations
    #[test]
    fn test_compiler_state_persistence() {
        let compiler = Compiler::new();
        
        // Verify compiler can be used multiple times
        let _compiler1 = Compiler::new();
        let _compiler2 = Compiler::new();
        
        // Multiple instances should be independent
        drop(compiler);
        
        // Creating new compiler after dropping should work
        let _compiler3 = Compiler::new();
    }

    /// Test 5: Execution context with zero-sized tensor allocation
    #[test]
    fn test_context_zero_allocation() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        
        // Allocate zero bytes - should handle gracefully
        let handle = ctx.allocate_tensor(0).unwrap();
        assert_eq!(handle.size, 0);
        assert_eq!(handle.device, Device::Cpu);
    }

    /// Test 6: Nested tensor type with single-element dimensions
    #[test]
    fn test_nested_tensor_single_element_dimensions() {
        // tensor<tensor<tensor<f32, [1]>, [1]>, [1]>
        let level1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![1],
        };
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![1],
        };
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![1],
        };
        
        match &level3 {
            Type::Tensor { element_type, shape } => {
                assert_eq!(shape, &vec![1]);
                match element_type.as_ref() {
                    Type::Tensor { element_type: inner, shape: inner_shape } => {
                        assert_eq!(inner_shape, &vec![1]);
                        match inner.as_ref() {
                            Type::Tensor { shape: innermost_shape, .. } => {
                                assert_eq!(innermost_shape, &vec![1]);
                            },
                            _ => panic!("Expected nested tensor at level 2"),
                        }
                    },
                    _ => panic!("Expected nested tensor at level 1"),
                }
            },
            _ => panic!("Expected tensor type"),
        }
    }

    /// Test 7: Attribute array with deep nesting and alternating types
    #[test]
    fn test_attribute_array_alternating_deep_nesting() {
        let nested = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Array(vec![
                Attribute::String("test".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Int(2),
                ]),
            ]),
            Attribute::Float(3.14),
        ]);
        
        match nested {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 3);
                
                match arr[0] {
                    Attribute::Int(1) => {},
                    _ => panic!("Expected Int at index 0"),
                }
                
                match &arr[1] {
                    Attribute::Array(inner) => {
                        assert_eq!(inner.len(), 2);
                        match &inner[1] {
                            Attribute::Array(deep_inner) => {
                                assert_eq!(deep_inner.len(), 2);
                            },
                            _ => panic!("Expected nested array"),
                        }
                    },
                    _ => panic!("Expected Array at index 1"),
                }
                
                match arr[2] {
                    Attribute::Float(val) if (val - 3.14).abs() < f64::EPSILON => {},
                    _ => panic!("Expected Float at index 2"),
                }
            },
            _ => panic!("Expected outer Array"),
        }
    }

    /// Test 8: Module with operations having no inputs but multiple outputs
    #[test]
    fn test_operation_no_inputs_multiple_outputs() {
        let mut module = Module::new("generator_ops");
        
        // Like a constant or random generator operation
        let mut op = Operation::new("generate");
        op.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![100],
        });
        op.outputs.push(Value {
            name: "output2".to_string(),
            ty: Type::I32,
            shape: vec![50],
        });
        op.outputs.push(Value {
            name: "output3".to_string(),
            ty: Type::Bool,
            shape: vec![25],
        });
        
        module.add_operation(op);
        
        assert_eq!(module.operations[0].inputs.len(), 0);
        assert_eq!(module.operations[0].outputs.len(), 3);
    }

    /// Test 9: Value with shape containing repeated identical dimensions
    #[test]
    fn test_tensor_repeated_identical_dimensions() {
        let test_cases = vec![
            (vec![5, 5, 5], 125),       // All same
            (vec![2, 2, 2, 2, 2], 32),  // 5D all same
            (vec![10, 10, 10, 10], 10000), // 4D all same
        ];
        
        for (shape, expected_elements) in test_cases {
            let value = Value {
                name: "repeated_dims".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.num_elements(), Some(expected_elements));
            
            // Verify all dimensions are the same
            if !shape.is_empty() {
                let first_dim = shape[0];
                assert!(shape.iter().all(|&dim| dim == first_dim));
            }
        }
    }

    /// Test 10: Module operations with Unicode attribute keys and values
    #[test]
    fn test_operation_unicode_attributes() {
        let mut op = Operation::new("unicode_attrs");
        
        let mut attrs = HashMap::new();
        attrs.insert("属性_1".to_string(), Attribute::Int(42));
        attrs.insert("名前".to_string(), Attribute::String("テスト".to_string()));
        attrs.insert("имя".to_string(), Attribute::Float(2.718));
        attrs.insert("🔥".to_string(), Attribute::Bool(true));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 4);
        assert!(op.attributes.contains_key("属性_1"));
        assert!(op.attributes.contains_key("名前"));
        assert!(op.attributes.contains_key("имя"));
        assert!(op.attributes.contains_key("🔥"));
    }
}