//! Additional edge case tests for the Impulse compiler
//! Covers boundary conditions and error scenarios using standard library macros

#[cfg(test)]
mod tests {
    use crate::ir::{Value, Type, Operation, Module, Attribute};
    
    use rstest::rstest;

    #[test]
    fn test_empty_value_shape() {
        // Test scalar value with empty shape
        let value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        
        assert_eq!(value.name, "scalar");
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape.len(), 0);
        assert!(value.shape.is_empty());
        
        // Scalar has 1 element
        match value.num_elements() {
            Some(elements) => assert_eq!(elements, 1),
            None => panic!("num_elements should return Some(1) for scalar"),
        }
    }

    #[test]
    fn test_zero_in_shape() {
        // Test tensor with zero in shape dimensions (results in zero elements)
        let value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100, 0, 50],
        };
        
        assert_eq!(value.shape, vec![100, 0, 50]);
        
        match value.num_elements() {
            Some(elements) => assert_eq!(elements, 0),
            None => panic!("num_elements should return Some(0) for tensor with zero dimension"),
        }
    }

    #[test]
    fn test_extremely_large_shape_no_overflow() {
        // Test very large but valid tensor shapes
        let value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000], // 100 million elements
        };
        
        assert_eq!(value.shape, vec![10_000, 10_000]);
        
        // This should not overflow for reasonable values
        match value.num_elements() {
            Some(elements) => assert_eq!(elements, 100_000_000),
            None => panic!("num_elements should handle large but valid shapes"),
        }
    }

    #[test]
    fn test_shape_that_causes_overflow() {
        // Test a shape that would cause overflow in multiplication
        let value = Value {
            name: "overflow_tensor".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2], // This will definitely overflow
        };
        
        assert_eq!(value.shape, vec![usize::MAX, 2]);
        
        // The num_elements method should handle overflow gracefully
        match value.num_elements() {
            Some(_) => panic!("num_elements should return None for overflow conditions"),
            None => assert!(true), // Expected behavior
        }
    }

    #[test]
    fn test_operation_with_empty_inputs_outputs() {
        // Test operation with no inputs or outputs
        let op = Operation::new("noop");
        
        assert_eq!(op.op_type, "noop");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    #[test]
    fn test_operation_with_many_inputs_outputs() {
        // Test operation with many inputs and outputs to test memory limits
        let mut op = Operation::new("multi_io_op");
        
        // Add 1000 inputs
        for i in 0..1000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 10 + 1], // Varying shape sizes
            });
        }
        
        // Add 500 outputs
        for i in 0..500 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 5 + 1], // Varying shape sizes
            });
        }
        
        assert_eq!(op.inputs.len(), 1000);
        assert_eq!(op.outputs.len(), 500);
        assert_eq!(op.op_type, "multi_io_op");
    }

    #[test]
    fn test_module_with_empty_operations() {
        // Test module with no operations
        let module = Module::new("empty_module");
        
        assert_eq!(module.name, "empty_module");
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }

    #[test]
    fn test_module_with_many_operations() {
        // Test module with many operations
        let mut module = Module::new("large_module");
        
        for i in 0..5000 {
            let mut op = Operation::new(&format!("operation_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1, 1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 5000);
        assert_eq!(module.name, "large_module");
        
        // Check first and last operations to ensure all were added correctly
        assert_eq!(module.operations[0].op_type, "operation_0");
        assert_eq!(module.operations[4999].op_type, "operation_4999");
    }

    #[test]
    fn test_attribute_array_nested_depth() {
        // Test deeply nested array attributes
        let deep_nested = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(42),
                    Attribute::Int(43),
                ]),
                Attribute::Array(vec![
                    Attribute::Int(44),
                    Attribute::Int(45),
                ]),
            ]),
        ]);
        
        match &deep_nested {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 1);
                match &outer[0] {
                    Attribute::Array(middle) => {
                        assert_eq!(middle.len(), 2);
                        match &middle[0] {
                            Attribute::Array(inner) => {
                                assert_eq!(inner.len(), 2);
                                match inner[0] {
                                    Attribute::Int(42) => (),
                                    _ => panic!("Expected Int(42)"),
                                }
                            },
                            _ => panic!("Expected nested Array"),
                        }
                    },
                    _ => panic!("Expected nested Array"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    #[rstest]
    #[case(vec![], 1)]           // Scalar
    #[case(vec![0], 0)]          // Zero-sized tensor
    #[case(vec![1], 1)]          // Single element
    #[case(vec![2, 3], 6)]       // Simple 2D
    #[case(vec![2, 3, 4], 24)]   // 3D
    #[case(vec![5, 1, 10], 50)]  // With ones
    fn test_num_elements_with_cases(#[case] shape: Vec<usize>, #[case] expected: usize) {
        let value = Value {
            name: "test_tensor".to_string(),
            ty: Type::F32,
            shape,
        };
        
        match value.num_elements() {
            Some(result) => assert_eq!(result, expected),
            None => panic!("num_elements returned None for valid shape"),
        }
    }
}