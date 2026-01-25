//! Additional edge case tests for the Impulse compiler
//! This file contains new test cases focusing on boundary conditions and error scenarios

#[cfg(test)]
mod more_edge_case_tests {
    use rstest::*;
    use crate::{ir::{Module, Value, Type, Operation, Attribute}, ImpulseCompiler};

    /// Test 1: Operations with extremely large integer values as attributes
    #[test]
    fn test_operations_extreme_integer_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("extreme_int_op");
        let mut attrs = HashMap::new();
        
        // Test with maximum and minimum integer values
        attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("zero_i64".to_string(), Attribute::Int(0));
        attrs.insert("negative_i64".to_string(), Attribute::Int(-999999999999));
        attrs.insert("positive_large_i64".to_string(), Attribute::Int(999999999999));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 5);
        assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
    }

    /// Test 2: Tensor shapes with potential overflow in multiplication
    #[test]
    fn test_tensor_shape_overflow_scenarios() {
        // Test shapes that could potentially cause overflow in calculations
        let safe_but_large = Value {
            name: "safe_large".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000],  // May cause overflow but within usize range
        };
        
        let size_product: usize = safe_but_large.shape.iter().product();
        assert_eq!(size_product, 10_000_000_000); // 10 billion
        
        // Test with a vector that would cause actual overflow
        let huge_shape = vec![usize::MAX, 2];
        let overflow_check: Option<usize> = huge_shape.iter()
            .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
        assert_eq!(overflow_check, None); // Should overflow
    }

    /// Test 3: Recursive types with complex nesting patterns
    #[test]
    fn test_complex_recursive_types() {
        // Create a complex recursive type structure
        let mut current_type = Type::I32;
        
        // Alternate between different types in the recursion
        for i in 0..10 {
            if i % 2 == 0 {
                current_type = Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![i + 1],
                };
            } else {
                current_type = Type::Tensor {
                    element_type: Box::new(current_type),
                    shape: vec![i % 3 + 1], // Varying shape sizes
                };
            }
        }
        
        // Test that we can clone the complex type without issues
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }

    /// Test 4: Operations with mixed boolean and numeric attributes
    #[test]
    fn test_operations_mixed_bool_numeric_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("mixed_types_op");
        let mut attrs = HashMap::new();
        
        // Mix different types of attributes
        attrs.insert("bool_true".to_string(), Attribute::Bool(true));
        attrs.insert("bool_false".to_string(), Attribute::Bool(false));
        attrs.insert("int_value".to_string(), Attribute::Int(42));
        attrs.insert("float_value".to_string(), Attribute::Float(3.14159));
        attrs.insert("string_value".to_string(), Attribute::String("test".to_string()));
        attrs.insert("zero_int".to_string(), Attribute::Int(0));
        attrs.insert("negative_float".to_string(), Attribute::Float(-2.71828));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 7);
        assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
        assert_eq!(op.attributes.get("float_value"), Some(&Attribute::Float(3.14159)));
        assert_eq!(op.attributes.get("string_value"), Some(&Attribute::String("test".to_string())));
    }

    /// Test 5: Handling null bytes and special escape sequences in string attributes
    #[test]
    fn test_string_attributes_special_sequences() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("special_str_op");
        let mut attrs = HashMap::new();
        
        // Test various special string sequences
        attrs.insert("null_byte_str".to_string(), Attribute::String("hello\0world".to_string()));
        attrs.insert("tab_str".to_string(), Attribute::String("col1\tcol2".to_string()));
        attrs.insert("newline_str".to_string(), Attribute::String("line1\nline2".to_string()));
        attrs.insert("carriage_return_str".to_string(), Attribute::String("before\rafter".to_string()));
        attrs.insert("escape_quotes".to_string(), Attribute::String("\"quoted\"".to_string()));
        attrs.insert("backslash_str".to_string(), Attribute::String("path\\to\\file".to_string()));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 6);
        assert_eq!(op.attributes.get("null_byte_str"), Some(&Attribute::String("hello\0world".to_string())));
        assert_eq!(op.attributes.get("tab_str"), Some(&Attribute::String("col1\tcol2".to_string())));
        assert_eq!(op.attributes.get("escape_quotes"), Some(&Attribute::String("\"quoted\"".to_string())));
    }

    /// Test 6: Values with all different primitive types
    #[test]
    fn test_all_primitive_types_in_values() {
        // Test each primitive type in the Type enum
        let f32_val = Value {
            name: "f32_val".to_string(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        let f64_val = Value {
            name: "f64_val".to_string(),
            ty: Type::F64,
            shape: vec![4, 5],
        };
        
        let i32_val = Value {
            name: "i32_val".to_string(),
            ty: Type::I32,
            shape: vec![6],
        };
        
        let i64_val = Value {
            name: "i64_val".to_string(),
            ty: Type::I64,
            shape: vec![7, 8, 9, 10],
        };
        
        let bool_val = Value {
            name: "bool_val".to_string(),
            ty: Type::Bool,
            shape: vec![2, 2],
        };
        
        assert_eq!(f32_val.ty, Type::F32);
        assert_eq!(f64_val.ty, Type::F64);
        assert_eq!(i32_val.ty, Type::I32);
        assert_eq!(i64_val.ty, Type::I64);
        assert_eq!(bool_val.ty, Type::Bool);
        
        assert_eq!(f32_val.shape, vec![1, 2, 3]);
        assert_eq!(bool_val.shape, vec![2, 2]);
    }

    /// Test 7: Operations with maximum length names
    #[test]
    fn test_operations_maximum_length_names() {
        // Test with a very long operation name
        let long_name = "a".repeat(100_000); // 100k character operation name
        let op = Operation::new(&long_name);
        assert_eq!(op.op_type, long_name);
        assert_eq!(op.op_type.len(), 100_000);
        
        // Test with a very long value name
        let long_value_name = "v".repeat(100_000);
        let value = Value {
            name: long_value_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name, long_value_name);
        assert_eq!(value.name.len(), 100_000);
        
        // Test with a very long module name
        let long_module_name = "m".repeat(100_000);
        let module = Module::new(&long_module_name);
        assert_eq!(module.name, long_module_name);
        assert_eq!(module.name.len(), 100_000);
    }

    /// Test 8: Special floating-point values in tensor computations
    #[test]
    fn test_special_floating_point_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("fp_edge_op");
        let mut attrs = HashMap::new();
        
        // Add special floating-point values
        attrs.insert("infinity".to_string(), Attribute::Float(f64::INFINITY));
        attrs.insert("neg_infinity".to_string(), Attribute::Float(f64::NEG_INFINITY));
        attrs.insert("nan_value".to_string(), Attribute::Float(f64::NAN));
        attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
        attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
        attrs.insert("subnormal".to_string(), Attribute::Float(f64::MIN_POSITIVE / 2.0));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 6);
        
        // Check special values
        if let Some(Attribute::Float(inf_val)) = op.attributes.get("infinity") {
            assert!(inf_val.is_infinite() && inf_val.is_sign_positive());
        } else {
            panic!("Expected positive infinity");
        }
        
        if let Some(Attribute::Float(neginf_val)) = op.attributes.get("neg_infinity") {
            assert!(neginf_val.is_infinite() && neginf_val.is_sign_negative());
        } else {
            panic!("Expected negative infinity");
        }
        
        // Check for NaN - need special handling since NaN != NaN
        if let Some(Attribute::Float(nan_val)) = op.attributes.get("nan_value") {
            assert!(nan_val.is_nan());
        } else {
            panic!("Expected NaN value");
        }
    }

    /// Test 9: Nested operations with complex dependency chains
    #[test]
    fn test_nested_operations_complex_dependencies() {
        let mut module = Module::new("complex_deps");
        
        // Create a chain of operations where each depends on the previous
        for i in 0..50 {
            let mut op = Operation::new(&format!("op_{}", i));
            
            // Connect to previous operation output if not the first
            if i > 0 {
                op.inputs.push(Value {
                    name: format!("output_{}", i - 1),
                    ty: Type::F32,
                    shape: vec![i.min(5)], // Use varying shapes
                });
            }
            
            // Add an output
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i.min(5)],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 50);
        
        // Verify the connections
        for i in 0..50 {
            if i > 0 {
                assert_eq!(module.operations[i].inputs.len(), 1);
                assert_eq!(module.operations[i].inputs[0].name, format!("output_{}", i - 1));
            } else {
                assert_eq!(module.operations[i].inputs.len(), 0); // First op has no inputs
            }
            assert_eq!(module.operations[i].outputs.len(), 1);
            assert_eq!(module.operations[i].outputs[0].name, format!("output_{}", i));
        }
    }

    /// Test 10: Modules with extreme combinations of operations, values, and attributes
    #[test]
    fn test_module_extreme_combination() {
        let mut module = Module::new("extreme_combo");
        
        // Create an operation with many different aspects combined
        for op_idx in 0..10 {
            let mut op = Operation::new(&format!("extreme_op_{}", op_idx));
            
            // Add multiple inputs with different types
            for input_idx in 0..5 {
                op.inputs.push(Value {
                    name: format!("input_{}_{}", op_idx, input_idx),
                    ty: match input_idx % 5 {
                        0 => Type::F32,
                        1 => Type::F64,
                        2 => Type::I32,
                        3 => Type::I64,
                        _ => Type::Bool,
                    },
                    shape: vec![(input_idx + 1) * 2, (op_idx + 1) * 3],
                });
            }
            
            // Add multiple outputs with different types
            for output_idx in 0..3 {
                op.outputs.push(Value {
                    name: format!("output_{}_{}", op_idx, output_idx),
                    ty: match output_idx % 3 {
                        0 => Type::F32,
                        1 => Type::I64,
                        _ => Type::Bool,
                    },
                    shape: vec![(output_idx + 1) * 4, (op_idx + 1) * 2],
                });
            }
            
            // Add various attribute types
            use std::collections::HashMap;
            let mut attrs = HashMap::new();
            attrs.insert(
                format!("int_attr_{}", op_idx),
                Attribute::Int((op_idx * 1000) as i64)
            );
            attrs.insert(
                format!("float_attr_{}", op_idx),
                Attribute::Float(op_idx as f64 * 3.14)
            );
            attrs.insert(
                format!("bool_attr_{}", op_idx),
                Attribute::Bool(op_idx % 2 == 0)
            );
            attrs.insert(
                format!("str_attr_{}", op_idx),
                Attribute::String(format!("value_{}", op_idx))
            );
            
            op.attributes = attrs;
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 10);
        
        // Verify characteristics of first operation
        let first_op = &module.operations[0];
        assert_eq!(first_op.inputs.len(), 5);
        assert_eq!(first_op.outputs.len(), 3);
        assert_eq!(first_op.attributes.len(), 4);
        
        // Verify that types are distributed correctly
        assert_eq!(first_op.inputs[0].ty, Type::F32);
        assert_eq!(first_op.inputs[1].ty, Type::F64);
        assert_eq!(first_op.inputs[2].ty, Type::I32);
        assert_eq!(first_op.inputs[3].ty, Type::I64);
        assert_eq!(first_op.inputs[4].ty, Type::Bool);
    }
}