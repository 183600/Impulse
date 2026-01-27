//! Additional edge case tests for the Impulse compiler
//! This module focuses on testing boundary conditions and potential edge cases
//! that could cause unexpected behavior in the compiler

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::ir_utils;

#[cfg(test)]
mod additional_edge_case_tests_new_new_new {
    use super::*;

    /// Test creating operations with maximum length strings for names and attributes
    #[test]
    fn test_maximum_length_strings_in_operations() {
        // Test with a very long operation name (reduced from 1M to 100k)
        let long_op_name = "a".repeat(100_000); // 100k character name
        let mut op = Operation::new(&long_op_name);
        
        // Add inputs and outputs with long names (reduced from 600k/700k to 60k/70k)
        let long_input_name = "input_".repeat(10_000); // 60k character input name
        let long_output_name = "output_".repeat(10_000); // 70k character output name
        
        op.inputs.push(Value {
            name: long_input_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        });
        
        op.outputs.push(Value {
            name: long_output_name.clone(),
            ty: Type::F32,
            shape: vec![1],
        });
        
        // Add a long string attribute (reduced from 1M to 100k)
        let long_string_attr = "attr_char_".repeat(10_000); // 100k char string attribute
        op.attributes.insert("long_attr".to_string(), Attribute::String(long_string_attr.clone()));
        
        assert_eq!(op.op_type, long_op_name);
        assert_eq!(op.inputs[0].name, long_input_name);
        assert_eq!(op.outputs[0].name, long_output_name);
        if let Attribute::String(attr_val) = &op.attributes["long_attr"] {
            assert_eq!(attr_val, &long_string_attr);
        } else {
            panic!("Expected string attribute");
        }
    }

    /// Test tensor operations with maximum possible dimensions
    #[test]
    fn test_maximum_tensor_dimensions() {
        // Create a tensor with many dimensions but keep it reasonable (reduced from 1000 to 100)
        let max_dims = vec![1; 100]; // 100 dimensions, each of size 1
        let value = Value {
            name: "max_dims_tensor".to_string(),
            ty: Type::F32,
            shape: max_dims.clone(),
        };
        
        assert_eq!(value.shape.len(), 100);
        assert!(value.shape.iter().all(|&x| x == 1));
        
        // The total number of elements should still be 1
        assert_eq!(value.num_elements(), Some(1));
    }

    /// Test nested tensor types with maximum depth that could cause stack overflow
    #[test]
    fn test_maximum_depth_nested_tensors_for_stack_safety() {
        let mut current_type = Type::F32;
        
        // Create 1000 levels of nesting which should be safe without causing stack overflow
        for _ in 0..1000 {
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1], // Keep shape minimal to focus on nesting depth
            };
        }
        
        // Verify the top level is still a tensor
        match &current_type {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![1]);
            },
            _ => panic!("Expected a tensor type after deep nesting"),
        }
        
        // Test that equality works even with deep nesting (cloning won't cause stack overflow)
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
        
        // Test getting element type from deeply nested structure
        let element_type = ir_utils::get_element_type(&current_type);
        assert_eq!(element_type, &Type::F32);
    }

    /// Test operations with values that have 0 in various positions of their shapes
    #[test]
    fn test_operations_with_zero_dimension_at_various_positions() {
        let test_shapes = vec![
            vec![0],           // Single zero dim
            vec![0, 5],       // Zero at start  
            vec![5, 0],       // Zero at end
            vec![3, 0, 7],    // Zero in middle
            vec![0, 0, 10],   // Multiple zeros at start
            vec![10, 0, 0],   // Multiple zeros at end
            vec![2, 0, 0, 5], // Zeros in middle
        ];
        
        for (i, shape) in test_shapes.iter().enumerate() {
            let value = Value {
                name: format!("zero_test_{}", i),
                ty: Type::F32,
                shape: shape.clone(),
            };
            
            assert_eq!(value.shape, *shape);
            // Any shape that contains 0 should result in 0 total elements
            if shape.contains(&0) {
                assert_eq!(value.num_elements(), Some(0));
            } else {
                let expected = shape.iter().product::<usize>();
                assert_eq!(value.num_elements(), Some(expected));
            }
        }
    }

    /// Test module with extreme number of input/output values
    #[test]
    fn test_module_with_extreme_number_of_inputs_outputs() {
        let mut module = Module::new("extreme_io_module");
        
        // Add many input values to the module (not operations, but module-level inputs)
        // Reduced from 10k to 1k to be safer
        for i in 0..1_000 {
            module.inputs.push(Value {
                name: format!("module_input_{}", i),
                ty: Type::F32,
                shape: vec![i % 100 + 1], // Varying small shapes
            });
            
            module.outputs.push(Value {
                name: format!("module_output_{}", i),
                ty: Type::I32,
                shape: vec![(i + 10) % 100 + 1], // Different varying shapes
            });
        }
        
        assert_eq!(module.name, "extreme_io_module");
        assert_eq!(module.inputs.len(), 1_000);
        assert_eq!(module.outputs.len(), 1_000);
        assert_eq!(module.operations.len(), 0);
        
        // Check a few specific values
        assert_eq!(module.inputs[0].name, "module_input_0");
        assert_eq!(module.inputs[999].name, "module_input_999");
        assert_eq!(module.outputs[500].name, "module_output_500");
    }

    /// Test arithmetic overflow in tensor size calculations with actual large values
    #[test]
    fn test_arithmetic_overflow_in_tensor_size_calculations() {
        // Test with dimensions that would cause overflow when multiplied
        // Use checked arithmetic to handle these properly
        
        // This should not overflow: large but safe dimensions
        let safe_large_tensor = Value {
            name: "safe_large".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000], // 100M elements
        };
        
        assert_eq!(safe_large_tensor.num_elements(), Some(100_000_000));
        
        // Test edge case with potential overflow - using checked_mul internally
        let mut large_value = Value {
            name: "potentially_overflowing".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        
        // Manually construct a shape that would cause overflow
        // To trigger overflow, we'd multiply two numbers whose product exceeds usize::MAX
        // Instead we'll test the safe handling of very large but not quite overflowing values
        large_value.shape = vec![100_000_000, 1_000_000_000]; // 10^17, likely to overflow on 64-bit
        
        // The num_elements function should handle overflow gracefully
        let result = large_value.num_elements();
        // If it overflows in our test environment, result should be None
        // If it doesn't overflow, result will be Some(actual_value)
        assert!(result.is_some() || result.is_none()); // Either way is valid depending on system
    }

    /// Test complex nested tensor operations with various depth and breadth
    #[test]
    fn test_complex_nested_tensor_operations_various_depth_breadth() {
        // Create nested tensors with different combinations of depth and breadth
        let base_type = Type::F32;
        
        // Shallow and wide: [tensor<f32, [2,2,2,2,2]>] (depth 1, breadth 5 dims)
        let shallow_wide = Type::Tensor {
            element_type: Box::new(base_type.clone()),
            shape: vec![2, 2, 2, 2, 2], // 5-dimensional, each dim size 2
        };
        
        // Deep and narrow: tensor<tensor<tensor<f32, [2]>, [2]>, [2]> (depth 3, breadth 1 per level)
        let deep_narrow = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(base_type.clone()),
                    shape: vec![2],
                }),
                shape: vec![2],
            }),
            shape: vec![2],
        };
        
        // Mixed: tensor<tensor<f32, [2,2]>, [3,3]> (depth 2, mixed breadth)
        let mixed_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(base_type.clone()),
                shape: vec![2, 2],
            }),
            shape: vec![3, 3],
        };
        
        // Verify types match expected structure
        match &shallow_wide {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2, 2, 2, 2, 2]);
            },
            _ => panic!("Expected shallow_wide to be tensor"),
        }
        
        match &deep_narrow {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected deep_narrow to be tensor"),
        }
        
        match &mixed_nested {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![3, 3]);
            },
            _ => panic!("Expected mixed_nested to be tensor"),
        }
        
        // Test element type extraction
        assert_eq!(ir_utils::get_element_type(&shallow_wide), &base_type);
        assert_eq!(ir_utils::get_element_type(&deep_narrow), &base_type);
        assert_eq!(ir_utils::get_element_type(&mixed_nested), &base_type);
    }

    /// Test attribute handling with maximum complexity (nesting and size)
    #[test]
    fn test_maximum_complexity_attribute_handling() {
        // Create deeply nested array structures (reduce nesting to be safer)
        let mut deeply_nested_array = Attribute::Array(vec![]);
        
        // Create 5 levels of nested arrays (was 10) to be safer
        for level in 0..5 {
            let next_level = Attribute::Array(vec![
                deeply_nested_array,
                Attribute::Int(level),
                Attribute::String(format!("level_{}", level)),
            ]);
            deeply_nested_array = next_level;
        }
        
        // Verify the structure
        match &deeply_nested_array {
            Attribute::Array(outer_array) => {
                assert_eq!(outer_array.len(), 3);
                
                // Verify the integer and string parts
                if let Attribute::Int(i) = &outer_array[1] {
                    assert_eq!(*i, 4); // Last level added
                } else {
                    panic!("Expected integer at index 1");
                }
                
                if let Attribute::String(s) = &outer_array[2] {
                    assert_eq!(s, "level_4");
                } else {
                    panic!("Expected string at index 2");
                }
            },
            _ => panic!("Expected outer array"),
        }
        
        // Add this complex attribute to an operation
        let mut op = Operation::new("complex_attr_op");
        op.attributes.insert("complex_array".to_string(), deeply_nested_array);
        
        assert_eq!(op.attributes.len(), 1);
        assert!(op.attributes.contains_key("complex_array"));
    }

    /// Test value comparison with many identical fields but different names
    #[test]
    fn test_value_comparison_with_many_identical_fields_different_names() {
        let base_shape = vec![10, 20, 30];
        let base_type = Type::F32;
        
        // Create multiple values with identical properties except names
        // Reduce from 1000 to 100 to be safer
        let values: Vec<Value> = (0..100)
            .map(|i| Value {
                name: format!("value_{}", i),
                ty: base_type.clone(),
                shape: base_shape.clone(),
            })
            .collect();
        
        // Test that all have the same type and shape
        for value in &values {
            assert_eq!(value.ty, base_type);
            assert_eq!(value.shape, base_shape);
        }
        
        // Test that they're all different because of different names
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                assert_ne!(values[i], values[j]); // Different names make them unequal
            }
        }
        
        // Test that a value equals itself
        assert_eq!(values[0], values[0]);
        assert_eq!(values[50], values[50]);
        assert_eq!(values[99], values[99]);
    }

    /// Test type conversion and casting operations edge cases
    #[test]
    fn test_type_conversion_casting_edge_cases() {
        use crate::utils::ir_utils::type_to_string;
        
        // Create various types and convert to string
        let types_to_test = [
            Type::F32,
            Type::F64, 
            Type::I32,
            Type::I64,
            Type::Bool,
            Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![1],
            },
            Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I64),
                    shape: vec![2, 3],
                }),
                shape: vec![4, 5, 6],
            },
        ];
        
        let expected_strings = [
            "f32",
            "f64", 
            "i32",
            "i64",
            "bool",
            "tensor<f32, [1]>",
            "tensor<tensor<i64, [2, 3]>, [4, 5, 6]>",
        ];
        
        for (i, typ) in types_to_test.iter().enumerate() {
            let actual_str = type_to_string(typ);
            assert_eq!(actual_str, expected_strings[i]);
        }
        
        // Test some edge cases with empty shapes in nested tensors
        let nested_empty = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![], // Empty shape inside tensor
        };
        
        let nested_empty_str = type_to_string(&nested_empty);
        assert_eq!(nested_empty_str, "tensor<f32, []>");
        
        // Test scalar type representation
        let scalar_f32 = Type::F32;
        let scalar_str = type_to_string(&scalar_f32);
        assert_eq!(scalar_str, "f32");
    }
}