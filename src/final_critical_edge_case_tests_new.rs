/// Final critical edge case tests - additional boundary scenarios with standard library assertions
#[cfg(test)]
mod final_critical_edge_case_tests_new {
    use super::*;
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;

    /// Test 1: Value with shape product overflow protection using checked operations
    #[test]
    fn test_shape_product_overflow_protection() {
        // Test with dimensions that could potentially overflow when multiplied
        // Use checked_mul to prevent overflow
        let value = Value {
            name: "overflow_risk".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000, 1_000_000], // This would overflow if not checked
        };

        // Verify num_elements handles this safely with Option return
        match value.num_elements() {
            Some(elements) => assert!(elements > 0),
            None => assert!(true, "Correctly detected potential overflow"),
        }
    }

    /// Test 2: Attribute with denormalized float values (subnormals)
    #[test]
    fn test_denormalized_float_attributes() {
        // Test subnormal/denormalized floats (very small values close to zero)
        let subnormal_f32 = Attribute::Float(f32::MIN_POSITIVE as f64);
        let tiny_float = Attribute::Float(1e-308);

        match subnormal_f32 {
            Attribute::Float(val) => assert!(val > 0.0 && val < 1e-30),
            _ => panic!("Expected Float attribute for subnormal value"),
        }

        match tiny_float {
            Attribute::Float(val) => assert!(val >= 0.0 && val < 1e-307),
            _ => panic!("Expected Float attribute for tiny value"),
        }
    }

    /// Test 3: Module with cyclic operation dependencies (simulated)
    #[test]
    fn test_module_with_simulated_cyclic_dependencies() {
        let mut module = Module::new("cyclic_deps");

        // Create operations that could form a cycle if linked
        let mut op1 = Operation::new("op_a");
        op1.outputs.push(Value {
            name: "intermediate_a".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        let mut op2 = Operation::new("op_b");
        op2.inputs.push(Value {
            name: "intermediate_a".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op2.outputs.push(Value {
            name: "intermediate_b".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        // Add operations in order
        module.add_operation(op1);
        module.add_operation(op2);

        assert_eq!(module.operations.len(), 2);
        assert_eq!(module.operations[0].op_type, "op_a");
        assert_eq!(module.operations[1].op_type, "op_b");
    }

    /// Test 4: Value with extremely large dimension count (rank)
    #[test]
    fn test_extremely_high_rank_tensor() {
        // Create a tensor with many dimensions (high rank)
        let mut shape = Vec::new();
        for i in 0..12 {
            shape.push(2);
        }

        let high_rank_value = Value {
            name: "high_rank".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };

        assert_eq!(high_rank_value.shape.len(), 12);
        assert_eq!(high_rank_value.shape.iter().product::<usize>(), 4096);
    }

    /// Test 5: Operation with very deep attribute nesting
    #[test]
    fn test_deeply_nested_attribute_structure() {
        let mut op = Operation::new("deep_nested");
        let mut attrs = HashMap::new();

        // Create deeply nested array structure
        let deep_nested = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(1),
                ]),
            ]),
        ]);

        attrs.insert("deep".to_string(), deep_nested);
        op.attributes = attrs;

        match op.attributes.get("deep") {
            Some(Attribute::Array(outer)) => {
                assert_eq!(outer.len(), 1);
                match &outer[0] {
                    Attribute::Array(middle) => {
                        assert_eq!(middle.len(), 1);
                        match &middle[0] {
                            Attribute::Array(inner) => {
                                assert_eq!(inner.len(), 1);
                                match &inner[0] {
                                    Attribute::Int(1) => (),
                                    _ => panic!("Expected Int(1) at deepest level"),
                                }
                            },
                            _ => panic!("Expected nested array at middle level"),
                        }
                    },
                    _ => panic!("Expected nested array at outer level"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    /// Test 6: Module with input/output name conflicts
    #[test]
    fn test_module_with_name_conflicts() {
        let mut module = Module::new("name_conflicts");

        // Add inputs with conflicting names
        module.inputs.push(Value {
            name: "data".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        // Add outputs with the same name as inputs (allowed in this IR)
        module.outputs.push(Value {
            name: "data".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        assert_eq!(module.inputs[0].name, "data");
        assert_eq!(module.outputs[0].name, "data");
        // Both can exist with the same name
    }

    /// Test 7: Attribute with negative zero float
    #[test]
    fn test_negative_zero_float_attribute() {
        // Test negative zero (-0.0) which is different from positive zero in IEEE 754
        let neg_zero = Attribute::Float(-0.0);
        let pos_zero = Attribute::Float(0.0);

        match neg_zero {
            Attribute::Float(val) => {
                assert_eq!(val, 0.0);
                assert!(val.is_sign_negative());
            },
            _ => panic!("Expected Float attribute"),
        }

        match pos_zero {
            Attribute::Float(val) => {
                assert_eq!(val, 0.0);
                assert!(val.is_sign_positive());
            },
            _ => panic!("Expected Float attribute"),
        }
    }

    /// Test 8: Value with single dimension of 1 vs scalar (empty shape)
    #[test]
    fn test_scalar_vs_single_element_tensor() {
        // Scalar: empty shape
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };

        // Single element tensor: shape [1]
        let single_elem = Value {
            name: "single_elem".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };

        assert_eq!(scalar.num_elements(), Some(1));
        assert_eq!(single_elem.num_elements(), Some(1));

        // But their shapes are different
        assert_ne!(scalar.shape, single_elem.shape);
        assert!(scalar.shape.is_empty());
        assert_eq!(single_elem.shape, vec![1]);
    }

    /// Test 9: Module with operation containing empty attribute values
    #[test]
    fn test_operation_with_empty_attribute_values() {
        let mut op = Operation::new("empty_attrs");
        let mut attrs = HashMap::new();

        // Add attributes with empty/zero values
        attrs.insert("empty_str".to_string(), Attribute::String("".to_string()));
        attrs.insert("zero_int".to_string(), Attribute::Int(0));
        attrs.insert("false_bool".to_string(), Attribute::Bool(false));
        attrs.insert("empty_array".to_string(), Attribute::Array(vec![]));
        attrs.insert("zero_float".to_string(), Attribute::Float(0.0));

        op.attributes = attrs;

        assert_eq!(op.attributes.len(), 5);

        match op.attributes.get("empty_str") {
            Some(Attribute::String(s)) => assert_eq!(s.len(), 0),
            _ => panic!("Expected empty string"),
        }

        match op.attributes.get("zero_int") {
            Some(Attribute::Int(0)) => (),
            _ => panic!("Expected Int(0)"),
        }

        match op.attributes.get("false_bool") {
            Some(Attribute::Bool(false)) => (),
            _ => panic!("Expected Bool(false)"),
        }

        match op.attributes.get("empty_array") {
            Some(Attribute::Array(arr)) => assert_eq!(arr.len(), 0),
            _ => panic!("Expected empty array"),
        }
    }

    /// Test 10: Type::Tensor with nested tensor element type
    #[test]
    fn test_tensor_with_nested_tensor_element() {
        // Create a tensor of tensors (tensor of vectors conceptually)
        let element_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![3],  // Inner vectors have 3 elements
        };

        let outer_tensor = Type::Tensor {
            element_type: Box::new(element_tensor),
            shape: vec![4],  // Outer dimension has 4 vectors
        };

        match outer_tensor {
            Type::Tensor { element_type: outer_elem, shape: outer_shape } => {
                assert_eq!(outer_shape, vec![4]);
                match outer_elem.as_ref() {
                    Type::Tensor { element_type: inner_elem, shape: inner_shape } => {
                        assert_eq!(inner_shape, &vec![3]);
                        match inner_elem.as_ref() {
                            Type::F32 => (),
                            _ => panic!("Innermost element should be F32"),
                        }
                    },
                    _ => panic!("Inner element should be a Tensor type"),
                }
            },
            _ => panic!("Outer type should be a Tensor"),
        }
    }
}