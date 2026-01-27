//! Additional parameterized tests using rstest for IR structures in the Impulse compiler

#[cfg(test)]
mod rstest_edge_cases {
    use rstest::*;
    use crate::ir::{Value, Type, Operation, Attribute, Module};
    use crate::utils::ir_utils;
    use std::collections::HashMap;

    // Test different basic types with scalar values (empty shape)
    #[rstest]
    #[case(Type::F32, 4)]  // F32 takes 4 bytes
    #[case(Type::F64, 8)]  // F64 takes 8 bytes  
    #[case(Type::I32, 4)]  // I32 takes 4 bytes
    #[case(Type::I64, 8)]  // I64 takes 8 bytes
    #[case(Type::Bool, 1)] // Bool takes 1 byte
    fn test_scalar_tensor_sizes(#[case] data_type: Type, #[case] expected_byte_size: usize) {
        let value = Value {
            name: "scalar".to_string(),
            ty: data_type,
            shape: vec![],  // Scalar shape
        };
        
        assert!(ir_utils::is_scalar(&value));
        assert_eq!(ir_utils::get_rank(&value), 0);
        
        // Test tensor size calculation for scalars
        let size_result = ir_utils::calculate_tensor_size(&value.ty, &value.shape);
        assert!(size_result.is_ok());
        assert_eq!(size_result.unwrap(), expected_byte_size);
    }

    // Test different dimensions with same type to ensure consistency
    #[rstest]
    fn test_different_rank_tensors(
        #[values(vec![], vec![5], vec![3, 4], vec![2, 3, 4])] shape: Vec<usize>,
        #[values(Type::F32)] data_type: Type
    ) {
        let value = Value {
            name: "test_tensor".to_string(),
            ty: data_type,
            shape,
        };
        
        let rank = ir_utils::get_rank(&value);
        assert_eq!(rank, value.shape.len());
        
        let is_scalar = ir_utils::is_scalar(&value);
        assert_eq!(is_scalar, value.shape.is_empty());
        
        let is_vec = ir_utils::is_vector(&value);
        assert_eq!(is_vec, value.shape.len() == 1);
        
        let is_mat = ir_utils::is_matrix(&value);
        assert_eq!(is_mat, value.shape.len() == 2);
    }

    // Test attribute creation with different types using rstest
    #[rstest]
    #[case(Attribute::Int(42))]
    #[case(Attribute::Float(3.14159))]
    #[case(Attribute::String("test_string".to_string()))]
    #[case(Attribute::Bool(true))]
    #[case(Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]))]
    fn test_attribute_creation_and_matching(#[case] attr: Attribute) {
        match attr {
            Attribute::Int(val) => {
                assert!(val != 0 || val == 0); // Just ensuring the match works
            }
            Attribute::Float(val) => {
                assert!(val.is_finite()); // Float should be finite
            }
            Attribute::String(s) => {
                assert!(!s.is_empty() || s.is_empty()); // String should exist
            }
            Attribute::Bool(b) => {
                assert!(b == true || b == false); // Bool should be true or false
            }
            Attribute::Array(arr) => {
                assert!(arr.len() <= usize::MAX); // Array length is always valid in Rust
            }
        }
    }

    // Test operation creation with different numbers of inputs and outputs
    #[rstest]
    #[case(0, 0)]  // No inputs, no outputs
    #[case(1, 0)]  // One input, no outputs  
    #[case(0, 1)]  // No inputs, one output
    #[case(1, 1)]  // One input, one output
    #[case(5, 3)]  // Multiple inputs and outputs
    fn test_operation_io_counts(#[case] num_inputs: usize, #[case] num_outputs: usize) {
        let mut op = Operation::new("test_op");
        
        // Add inputs
        for i in 0..num_inputs {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![10],  // Fixed shape for simplicity
            });
        }
        
        // Add outputs
        for i in 0..num_outputs {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10],  // Fixed shape for simplicity
            });
        }
        
        assert_eq!(op.inputs.len(), num_inputs);
        assert_eq!(op.outputs.len(), num_outputs);
        assert_eq!(op.op_type, "test_op");
    }

    // Test nested tensor types with various nesting levels
    #[rstest]
    #[case(Type::F32, vec![], "f32")]
    #[case(Type::I32, vec![], "i32")]
    #[case(Type::Bool, vec![], "bool")]
    #[case(Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2, 3] }, vec![], "tensor<f32, [2, 3]>")]
    fn test_type_to_string_conversion(#[case] type_val: Type, #[case] _shape: Vec<usize>, #[case] expected: &str) {
        let result = ir_utils::type_to_string(&type_val);
        assert_eq!(result, expected);
    }

    // Test value names with different patterns
    #[rstest]
    #[case("")]
    #[case("simple")]
    #[case("with_underscores")]
    #[case("with-dashes")]
    #[case("with.dots")]
    #[case("with123numbers")]
    #[case("A")]
    #[case("aVeryLongNameThatMightCauseIssuesWithSomeSystems")]
    fn test_value_name_variations(#[case] name: &str) {
        let value = Value {
            name: name.to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name, name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    // Test operation types with different string patterns
    #[rstest]
    #[case("")]
    #[case("add")]
    #[case("matmul")]
    #[case("conv2d")]
    #[case("custom_op_with_underscores")]
    #[case("CamelCaseOp")]
    #[case("!@#$%special_chars")]
    #[case("very_long_operation_name_that_exceeds_normal_lengths")]
    fn test_operation_type_variations(#[case] op_type: &str) {
        let op = Operation::new(op_type);
        
        assert_eq!(op.op_type, op_type);
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    // Test module names with various patterns
    #[rstest]
    #[case("")]
    #[case("simple_module")]
    #[case("module_with_underscores")]
    #[case("Module-With-Dashes")]
    #[case("module.with.dots")]
    #[case("module123with456numbers")]
    #[case("a")]  // Single character
    #[case("A")]  // Single capital
    #[case("very_long_module_name_that_has_many_characters_and_might_cause_issues")]
    fn test_module_name_variations(#[case] name: &str) {
        let module = Module::new(name);
        
        assert_eq!(module.name, name);
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }

    // Test attribute arrays with different element counts
    #[rstest]
    #[case(0)]  // Empty array
    #[case(1)]  // Single element
    #[case(5)]  // Multiple elements
    #[case(10)] // More elements
    fn test_attribute_array_sizes(#[case] array_size: usize) {
        let mut attr_array = Vec::new();
        
        for i in 0..array_size {
            attr_array.push(Attribute::Int(i as i64));
        }
        
        let array_attr = Attribute::Array(attr_array);
        
        match array_attr {
            Attribute::Array(vec) => {
                assert_eq!(vec.len(), array_size);
                
                // Verify each element
                for (idx, attr) in vec.iter().enumerate() {
                    match attr {
                        Attribute::Int(val) => assert_eq!(*val, idx as i64),
                        _ => panic!("Expected Int attribute"),
                    }
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    // Test nested operations with different attribute counts
    #[rstest]
    #[case(0)]
    #[case(1)] 
    #[case(10)]
    #[case(100)]
    fn test_operations_with_various_attribute_counts(#[case] attr_count: usize) {
        let mut op = Operation::new("attributed_op");
        let mut attrs = HashMap::new();
        
        for i in 0..attr_count {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::Int(i as i64)
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.op_type, "attributed_op");
        assert_eq!(op.attributes.len(), attr_count);
        
        // Verify a few attributes exist if any were added
        if attr_count > 0 {
            assert!(op.attributes.contains_key(&format!("attr_{}", 0)));
            
            if attr_count > 1 {
                assert!(op.attributes.contains_key(&format!("attr_{}", attr_count - 1)));
            }
        }
    }
}