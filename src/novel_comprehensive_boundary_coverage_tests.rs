//! Novel comprehensive boundary coverage tests
//! This module provides additional edge case testing with focus on:
//! - Memory safety and overflow prevention
//! - Type conversion edge cases
//! - Module consistency validation
//! - Operation attribute validation
//! - Shape calculation edge cases

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::{validation_utils, ir_utils},
};

#[cfg(test)]
mod novel_comprehensive_boundary_coverage_tests {
    use super::*;

    /// Test 1: Module with name containing only whitespace characters
    #[test]
    fn test_module_whitespace_only_name() {
        let whitespace_names = vec![" ", "\t", "\n", "  \t\n  "];
        
        for name in whitespace_names {
            let module = Module::new(name);
            assert_eq!(module.name, name);
            assert!(module.operations.is_empty());
            assert!(module.inputs.is_empty());
            assert!(module.outputs.is_empty());
        }
    }

    /// Test 2: Value with shape containing very large single dimension that fits in usize
    #[test]
    fn test_value_large_single_dimension_safe() {
        let large_dim = usize::MAX / 2; // Safe single dimension
        
        let value = Value {
            name: "large_single_dim".to_string(),
            ty: Type::Bool,
            shape: vec![large_dim],
        };
        
        assert_eq!(value.shape.len(), 1);
        assert_eq!(value.shape[0], large_dim);
        assert_eq!(value.num_elements(), Some(large_dim));
    }

    /// Test 3: Operation with attribute values that are edge case numbers
    #[test]
    fn test_operation_edge_case_numeric_attributes() {
        let mut op = Operation::new("edge_attrs");
        
        let edge_ints = vec![
            (i64::MAX, "max_int"),
            (i64::MIN, "min_int"),
            (0, "zero"),
            (1, "one"),
            (-1, "neg_one"),
        ];
        
        for (val, key) in edge_ints {
            op.attributes.insert(key.to_string(), Attribute::Int(val));
        }
        
        let edge_floats = vec![
            (f64::MIN, "min_float"),
            (f64::MAX, "max_float"),
            (f64::MIN_POSITIVE, "min_pos_float"),
            (-f64::MIN_POSITIVE, "neg_min_pos_float"),
        ];
        
        for (val, key) in edge_floats {
            op.attributes.insert(key.to_string(), Attribute::Float(val));
        }
        
        assert_eq!(op.attributes.len(), 9);
        
        // Verify each edge case attribute
        assert_eq!(op.attributes.get("max_int"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("min_int"), Some(&Attribute::Int(i64::MIN)));
        assert_eq!(op.attributes.get("zero"), Some(&Attribute::Int(0)));
        assert_eq!(op.attributes.get("one"), Some(&Attribute::Int(1)));
        assert_eq!(op.attributes.get("neg_one"), Some(&Attribute::Int(-1)));
        assert_eq!(op.attributes.get("min_float"), Some(&Attribute::Float(f64::MIN)));
        assert_eq!(op.attributes.get("max_float"), Some(&Attribute::Float(f64::MAX)));
        assert_eq!(op.attributes.get("min_pos_float"), Some(&Attribute::Float(f64::MIN_POSITIVE)));
    }

    /// Test 4: Nested tensor type with deeply nested structure (3 levels deep)
    #[test]
    fn test_deeply_nested_tensor_type() {
        // Create a 3-level nested tensor type
        let level1 = Type::F32;
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![2, 2],
        };
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![3],
        };
        let level4 = Type::Tensor {
            element_type: Box::new(level3),
            shape: vec![4],
        };
        
        match level4 {
            Type::Tensor { element_type, shape } => {
                assert_eq!(shape, vec![4]);
                match *element_type {
                    Type::Tensor { element_type: inner1, shape: inner_shape1 } => {
                        assert_eq!(inner_shape1, vec![3]);
                        match *inner1 {
                            Type::Tensor { element_type: inner2, shape: inner_shape2 } => {
                                assert_eq!(inner_shape2, vec![2, 2]);
                                assert_eq!(*inner2, Type::F32);
                            }
                            _ => panic!("Expected 3-level nesting"),
                        }
                    }
                    _ => panic!("Expected 2-level nesting"),
                }
            }
            _ => panic!("Expected Tensor type"),
        }
    }

    /// Test 5: Module with operations having cyclic naming pattern
    #[test]
    fn test_module_cyclic_operation_naming() {
        let mut module = Module::new("cyclic_ops");
        
        let op_names = vec!["op_a", "op_b", "op_c", "op_a", "op_b", "op_c"];
        
        for name in op_names {
            let op = Operation::new(name);
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 6);
        assert_eq!(module.operations[0].op_type, "op_a");
        assert_eq!(module.operations[1].op_type, "op_b");
        assert_eq!(module.operations[2].op_type, "op_c");
        assert_eq!(module.operations[3].op_type, "op_a");
        assert_eq!(module.operations[4].op_type, "op_b");
        assert_eq!(module.operations[5].op_type, "op_c");
    }

    /// Test 6: Validation of module with inputs/outputs having same types but different shapes
    #[test]
    fn test_validation_same_type_different_shapes() {
        let mut module = Module::new("same_type_diff_shape");
        
        // Add inputs with same type but different shapes
        module.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.inputs.push(Value {
            name: "input2".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        module.inputs.push(Value {
            name: "input3".to_string(),
            ty: Type::F32,
            shape: vec![10, 10, 10],
        });
        
        // Add outputs with same type but different shapes
        module.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        module.outputs.push(Value {
            name: "output2".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        
        // Validation should pass
        assert!(validation_utils::validate_module(&module).is_ok());
    }

    /// Test 7: Value with shape containing alternating small and large dimensions
    #[test]
    fn test_value_alternating_dimension_pattern() {
        let patterns = vec![
            vec![1, 1000, 1, 1000],
            vec![2, 500, 2, 500, 2],
            vec![10, 100, 10, 100, 10, 100],
        ];
        
        for shape in patterns {
            let value = Value {
                name: "alternating_dim".to_string(),
                ty: Type::I32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, shape);
            let total_elements: usize = shape.iter().product();
            assert_eq!(value.num_elements(), Some(total_elements));
        }
    }

    /// Test 8: Module with operations that have inputs/outputs but no attributes
    #[test]
    fn test_operations_with_io_no_attributes() {
        let mut module = Module::new("io_only_ops");
        
        // Create operations with inputs and outputs but no attributes
        for i in 0..3 {
            let mut op = Operation::new(&format!("io_op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F64,
                shape: vec![5],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F64,
                shape: vec![5],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 3);
        
        for op in &module.operations {
            assert!(!op.inputs.is_empty());
            assert!(!op.outputs.is_empty());
            assert!(op.attributes.is_empty());
        }
    }

    /// Test 9: Tensor size calculation with overflow detection for very large dimensions
    #[test]
    fn test_tensor_size_overflow_detection() {
        // Test shapes that would cause overflow if not handled
        let overflow_shapes = vec![
            vec![usize::MAX, 2],  // Would overflow
            vec![usize::MAX / 2 + 1, 2],  // Would overflow
        ];
        
        for shape in overflow_shapes {
            let result = ir_utils::calculate_tensor_size(&Type::F32, &shape);
            assert!(result.is_err(), "Expected overflow for shape {:?}", shape);
        }
        
        // Test shapes that should NOT overflow (these are within safe bounds)
        let safe_shapes = vec![
            vec![10000, 10000],  // 100M elements
            vec![1000, 1000, 1000],  // 1B elements
            vec![100, 100, 100, 100],  // 100M elements
        ];
        
        for shape in safe_shapes {
            let result = ir_utils::calculate_tensor_size(&Type::F32, &shape);
            assert!(result.is_ok(), "Expected success for shape {:?}", shape);
        }
        
        // Test with a shape that causes overflow in the size calculation (not just element count)
        // The element count might not overflow, but the size calculation (elements * bytes) might
        let overflow_size_shape = vec![usize::MAX / 8 + 1];  // Elements that overflow when multiplied by 8 bytes
        let result = ir_utils::calculate_tensor_size(&Type::F64, &overflow_size_shape);
        assert!(result.is_err(), "Expected overflow for F64 size calculation");
    }

    /// Test 10: Module validation with operation having duplicate input and output names
    #[test]
    fn test_validation_duplicate_io_names_in_operation() {
        // Create an operation with conflicting names
        let mut op = Operation::new("conflict_op");
        op.inputs.push(Value {
            name: "shared_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: "shared_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        // Validation should fail due to conflicting names
        let result = validation_utils::validate_operation(&op);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Input and output share the same name"));
        
        // Now add it to a module to confirm module creation works
        let mut module = Module::new("duplicate_io_names");
        module.add_operation(op);
        assert_eq!(module.operations.len(), 1);
    }
}
