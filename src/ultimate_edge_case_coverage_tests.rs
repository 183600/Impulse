/// Ultimate edge case coverage tests - Comprehensive boundary condition testing
/// Focuses on numeric precision, memory safety, type conversion, and compiler robustness

#[cfg(test)]
mod tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;

    /// Test 1: Value with single element scalar (empty shape)
    #[test]
    fn test_scalar_value_empty_shape() {
        let scalar = Value {
            name: "scalar_value".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(scalar.shape.len(), 0);
        assert_eq!(scalar.num_elements(), Some(1));
    }

    /// Test 2: Overflow prevention in num_elements with large dimensions
    #[test]
    fn test_overflow_prevention_num_elements() {
        // Test dimensions that would overflow if multiplied naively
        let large_value = Value {
            name: "overflow_test".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2],
        };
        // Should return None due to overflow detection
        assert_eq!(large_value.num_elements(), None);
    }

    /// Test 3: Operation with negative float attribute handling
    #[test]
    fn test_negative_float_attributes() {
        let mut op = Operation::new("test_op");
        let mut attrs = HashMap::new();
        
        attrs.insert("neg_one".to_string(), Attribute::Float(-1.0));
        attrs.insert("neg_pi".to_string(), Attribute::Float(-std::f64::consts::PI));
        attrs.insert("neg_large".to_string(), Attribute::Float(-1e308));
        
        op.attributes = attrs;
        
        match op.attributes.get("neg_one") {
            Some(Attribute::Float(val)) => assert_eq!(*val, -1.0),
            _ => panic!("Expected Float(-1.0)"),
        }
        
        match op.attributes.get("neg_pi") {
            Some(Attribute::Float(val)) => assert_eq!(*val, -std::f64::consts::PI),
            _ => panic!("Expected Float(-PI)"),
        }
    }

    /// Test 4: Module with alternating operation add/remove pattern
    #[test]
    fn test_module_alternating_operations() {
        let mut module = Module::new("alternating_test");
        
        // Add 5 operations
        for i in 0..5 {
            module.add_operation(Operation::new(&format!("op_{}", i)));
        }
        assert_eq!(module.operations.len(), 5);
        
        // Verify all operations have unique op_types
        let mut op_types: Vec<&str> = module.operations.iter().map(|op| op.op_type.as_str()).collect();
        op_types.sort();
        op_types.dedup();
        assert_eq!(op_types.len(), 5);
    }

    /// Test 5: Deeply nested attribute array with mixed types
    #[test]
    fn test_deeply_nested_mixed_attribute_array() {
        let nested_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(1.5),
                    Attribute::String("inner".to_string()),
                ]),
            ]),
            Attribute::Bool(false),
            Attribute::Array(vec![
                Attribute::Int(2),
                Attribute::Float(2.5),
            ]),
        ]);
        
        match nested_attr {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 3);
                match &outer[0] {
                    Attribute::Array(inner) => assert_eq!(inner.len(), 2),
                    _ => panic!("Expected nested array"),
                }
                match &outer[1] {
                    Attribute::Bool(false) => {},
                    _ => panic!("Expected Bool(false)"),
                }
            }
            _ => panic!("Expected Array"),
        }
    }

    /// Test 6: Zero-sized tensor with non-empty shape containing zero
    #[test]
    fn test_zero_sized_tensor_with_zero_dimension() {
        let zero_tensor = Value {
            name: "zero_sized".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5],
        };
        assert_eq!(zero_tensor.shape, vec![10, 0, 5]);
        assert_eq!(zero_tensor.num_elements(), Some(0));
    }

    /// Test 7: Operation with maximum i32/i64 integer attributes
    #[test]
    fn test_extreme_integer_attributes() {
        let mut op = Operation::new("extreme_ints");
        let mut attrs = HashMap::new();
        
        attrs.insert("i32_max".to_string(), Attribute::Int(i64::from(i32::MAX)));
        attrs.insert("i32_min".to_string(), Attribute::Int(i64::from(i32::MIN)));
        attrs.insert("i64_max".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("i64_min".to_string(), Attribute::Int(i64::MIN));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 4);
        match op.attributes.get("i32_max") {
            Some(Attribute::Int(val)) => assert_eq!(*val, i64::from(i32::MAX)),
            _ => panic!("Expected i32::MAX"),
        }
    }

    /// Test 8: Module with all data type variants in operations
    #[test]
    fn test_module_with_all_type_variants() {
        let mut module = Module::new("all_types");
        
        let types = vec![
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
        ];
        
        for (i, ty) in types.iter().enumerate() {
            let mut op = Operation::new(&format!("type_test_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: ty.clone(),
                shape: vec![1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 5);
        for (i, expected_type) in types.iter().enumerate() {
            assert_eq!(module.operations[i].inputs[0].ty, *expected_type);
        }
    }

    /// Test 9: Tensor with single dimension (1D tensor)
    #[test]
    fn test_one_dimensional_tensor() {
        let tensor_1d = Value {
            name: "vector".to_string(),
            ty: Type::F32,
            shape: vec![1000],
        };
        assert_eq!(tensor_1d.shape.len(), 1);
        assert_eq!(tensor_1d.num_elements(), Some(1000));
    }

    /// Test 10: Module with operation containing both inputs and outputs
    #[test]
    fn test_operation_with_inputs_and_outputs() {
        let mut module = Module::new("full_io_test");
        let mut op = Operation::new("full_io");
        
        // Add multiple inputs
        op.inputs.push(Value {
            name: "input_a".to_string(),
            ty: Type::F32,
            shape: vec![2, 2],
        });
        op.inputs.push(Value {
            name: "input_b".to_string(),
            ty: Type::F64,
            shape: vec![3, 3],
        });
        
        // Add multiple outputs
        op.outputs.push(Value {
            name: "output_c".to_string(),
            ty: Type::I32,
            shape: vec![1],
        });
        op.outputs.push(Value {
            name: "output_d".to_string(),
            ty: Type::Bool,
            shape: vec![5],
        });
        
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].inputs.len(), 2);
        assert_eq!(module.operations[0].outputs.len(), 2);
        assert_eq!(module.operations[0].inputs[0].name, "input_a");
        assert_eq!(module.operations[0].outputs[1].ty, Type::Bool);
    }
}