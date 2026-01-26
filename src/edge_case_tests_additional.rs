//! Additional edge case tests for the Impulse compiler
//! This file contains tests covering boundary conditions and error scenarios
//! not covered in the main test suite

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::ir_utils,
};
use crate::ir::TypeExtensions;

#[cfg(test)]
mod additional_edge_case_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_extreme_tensor_shape_values() {
        // Test tensor with the largest possible dimensions that don't cause overflow
        let max_dim = 10_000;
        let extreme_value = Value {
            name: "extreme_tensor".to_string(),
            ty: Type::F32,
            shape: vec![max_dim, max_dim],
        };

        // Calculate expected size
        let expected_size = max_dim * max_dim * 4; // 4 bytes per F32
        let calculated_size = ir_utils::calculate_tensor_size(&extreme_value.ty, &extreme_value.shape);
        
        assert!(calculated_size.is_ok());
        if let Ok(size) = calculated_size {
            assert_eq!(size, expected_size);
        }

        // Test with different types and extreme shapes
        let extreme_i64 = Value {
            name: "extreme_i64".to_string(),
            ty: Type::I64,
            shape: vec![max_dim, max_dim / 2], // Smaller second dimension to prevent overflow
        };
        
        let expected_i64_size = max_dim * (max_dim / 2) * 8; // 8 bytes per I64
        let calculated_i64_size = ir_utils::calculate_tensor_size(&extreme_i64.ty, &extreme_i64.shape);
        assert!(calculated_i64_size.is_ok());
        if let Ok(size) = calculated_i64_size {
            assert_eq!(size, expected_i64_size);
        }
    }

    #[test]
    fn test_single_element_tensors() {
        // Test scalar tensor (0-dimensions)
        let scalar = Value {
            name: "scalar_tensor".to_string(),
            ty: Type::F32,
            shape: vec![],  // scalar has 0 dimensions
        };

        // A scalar tensor has 1 element
        assert_eq!(scalar.shape.len(), 0);
        let num_elements: usize = scalar.shape.iter().product();
        assert_eq!(num_elements, 1);

        // Test single-element 1-dimensional tensor
        let single_elem = Value {
            name: "single_element".to_string(),
            ty: Type::F32,
            shape: vec![1],
        };

        assert_eq!(single_elem.shape, vec![1]);
        let num_elements: usize = single_elem.shape.iter().product();
        assert_eq!(num_elements, 1);

        // Test single-element multi-dimensional tensor
        let single_multi = Value {
            name: "single_multi".to_string(),
            ty: Type::F64,
            shape: vec![1, 1, 1],
        };

        assert_eq!(single_multi.shape, vec![1, 1, 1]);
        let num_elements: usize = single_multi.shape.iter().product();
        assert_eq!(num_elements, 1);

        // Calculate size
        assert_eq!(ir_utils::calculate_tensor_size(&single_multi.ty, &single_multi.shape).unwrap(), 8); // 8 bytes for F64
    }

    #[test]
    fn test_tensor_with_negative_logic() {
        // This test verifies that our tensor shape product calculation
        // correctly handles cases where one might expect negative values
        
        // Since we're using usize, we can't have negative dimensions,
        // but we can test the boundaries and edge cases
        
        // Test tensor with 0 in various positions
        let cases = vec![
            vec![0],              // [0]
            vec![0, 10],          // [0, 10]
            vec![10, 0],          // [10, 0]
            vec![5, 0, 10],       // [5, 0, 10]
            vec![0, 0, 0],        // [0, 0, 0]
            vec![1, 2, 0, 4, 5],  // [1, 2, 0, 4, 5]
        ];

        for shape in cases {
            let value = Value {
                name: "test_tensor".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };

            let product: usize = value.shape.iter().product();
            assert_eq!(product, 0, "Shape {:?} should have 0 elements", shape);
            
            // Size should be 0 when there are 0 total elements
            let size = ir_utils::calculate_tensor_size(&value.ty, &value.shape).unwrap();
            assert_eq!(size, 0);
        }
    }

    #[test]
    fn test_operation_with_extreme_attribute_counts() {
        // Test operation with many attributes of different types
        let mut op = Operation::new("complex_op");
        
        // Add many different types of attributes
        let mut attrs = HashMap::new();
        
        // Add many integer attributes
        for i in 0..100 {
            attrs.insert(format!("int_attr_{}", i), Attribute::Int(i as i64));
        }
        
        // Add many float attributes
        for i in 0..100 {
            attrs.insert(
                format!("float_attr_{}", i),
                Attribute::Float((i as f64) * 0.5),
            );
        }
        
        // Add many string attributes
        for i in 0..100 {
            attrs.insert(
                format!("string_attr_{}", i),
                Attribute::String(format!("value_{}", i)),
            );
        }
        
        // Add many boolean attributes
        for i in 0..100 {
            attrs.insert(
                format!("bool_attr_{}", i),
                Attribute::Bool(i % 2 == 0),
            );
        }
        
        // Add some array attributes
        for i in 0..50 {
            attrs.insert(
                format!("array_attr_{}", i),
                Attribute::Array(vec![
                    Attribute::Int(i as i64),
                    Attribute::String(format!("array_item_{}", i)),
                ]),
            );
        }
        
        op.attributes = attrs;
        
        // Verify all attributes were added
        assert_eq!(op.attributes.len(), 450); // 100 + 100 + 100 + 100 + 50
        
        // Verify we can access some attributes
        assert_eq!(
            op.attributes.get("int_attr_50"),
            Some(&Attribute::Int(50))
        );
        
        assert_eq!(
            op.attributes.get("float_attr_25"),
            Some(&Attribute::Float(12.5))
        );
        
        assert_eq!(
            op.attributes.get("string_attr_99"),
            Some(&Attribute::String("value_99".to_string()))
        );
        
        assert_eq!(
            op.attributes.get("bool_attr_42"),
            Some(&Attribute::Bool(true))  // 42 % 2 == 0
        );
        
        assert_eq!(
            op.attributes.get("bool_attr_43"),
            Some(&Attribute::Bool(false)) // 43 % 2 != 0
        );
    }

    #[test]
    fn test_complex_nested_tensor_types() {
        // Test deeply nested tensor types with different configurations
        let nested_1 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        };

        // Calculate expected size: outer [4] * mid [3] * inner [2] * element F32 (4 bytes) = 96
        let size_1 = ir_utils::calculate_tensor_size(&nested_1, &[]).unwrap();
        assert_eq!(size_1, 4 * 3 * 2 * 4); // 96 bytes

        // Another complex nesting
        let nested_2 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![5, 5],
            }),
            shape: vec![2, 2],
        };

        // Calculate expected size: outer [2,2]=4 * mid [5,5]=25 * element I64 (8 bytes) = 800
        let size_2 = ir_utils::calculate_tensor_size(&nested_2, &[]).unwrap();
        assert_eq!(size_2, 4 * 25 * 8); // 800 bytes

        // Test with non-empty outer shape
        let size_2_with_outer = ir_utils::calculate_tensor_size(&nested_2, &[3]).unwrap();
        assert_eq!(size_2_with_outer, 3 * 4 * 25 * 8); // 2400 bytes

        // Test equivalence and non-equivalence
        let nested_3 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![5, 5],
            }),
            shape: vec![2, 2],
        };

        assert_eq!(nested_2, nested_3); // Should be equal

        let nested_4 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32), // Different element type
                shape: vec![5, 5],
            }),
            shape: vec![2, 2],
        };

        assert_ne!(nested_2, nested_4); // Should not be equal due to different element type
    }

    #[test]
    fn test_value_name_edge_cases() {
        // Test values with various problematic names
        let binding = "a".repeat(1000);
        let test_cases = vec![
            ("", Type::F32),  // Empty name
            (" ", Type::I32), // Single space
            ("\t\n\r", Type::F64), // Control characters
            (&binding, Type::Bool), // Very long name
            ("normal_name", Type::I64), // Normal name
            ("name_with_numbers_123", Type::F32), // Name with numbers
            ("name_with_symbols_!@#$%^&*()", Type::Bool), // Name with symbols
            ("name_with_unicode_测试_🚀", Type::F64), // Name with unicode
        ];

        for (name, data_type) in test_cases {
            let value = Value {
                name: name.to_string(),
                ty: data_type.clone(),
                shape: vec![1, 2, 3],
            };

            assert_eq!(value.name, name);
            assert_eq!(value.ty, data_type);
            assert_eq!(value.shape, vec![1, 2, 3]);
        }
    }

    #[test]
    fn test_operation_with_extreme_name_lengths() {
        // Test operation with extremely long name
        let long_name = "a".repeat(1_000_000); // 1 million character name
        let op = Operation::new(&long_name);
        
        assert_eq!(op.op_type, long_name);
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());

        // Test operation with empty name
        let empty_op = Operation::new("");
        assert_eq!(empty_op.op_type, "");
        assert!(empty_op.inputs.is_empty());
        assert!(empty_op.outputs.is_empty());
        assert!(empty_op.attributes.is_empty());
    }

    #[test]
    fn test_module_extreme_scenarios() {
        // Test module with extremely long name
        let long_name = "module_".repeat(100_000); // 700k+ character name
        let module = Module::new(long_name.clone());
        
        assert_eq!(module.name, long_name);
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());

        // Test module with minimal name
        let empty_name_module = Module::new("");
        assert_eq!(empty_name_module.name, "");
        assert!(empty_name_module.operations.is_empty());
        assert!(empty_name_module.inputs.is_empty());
        assert!(empty_name_module.outputs.is_empty());

        // Test creating and dropping a large module multiple times
        for _ in 0..10 {
            let mut large_module = Module::new("test_large_module");
            
            // Add many operations
            for i in 0..1000 {
                let mut op = Operation::new(&format!("op_{}", i));
                op.inputs.push(Value {
                    name: format!("input_{}", i),
                    ty: Type::F32,
                    shape: vec![1],
                });
                large_module.add_operation(op);
            }
            
            assert_eq!(large_module.operations.len(), 1000);
            
            // Drop the module and let it deallocate
            drop(large_module);
        }
    }

    #[test]
    fn test_tensor_size_overflow_protection() {
        // Test tensor size calculations that might cause overflow
        // Using safe multiplication to avoid panics
        
        // Create a shape that would cause overflow in naive multiplication
        // but our implementation should handle it gracefully
        let huge_shape = vec![100_000, 100_000];
        let value = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: huge_shape,
        };

        // Use the safe calculation method that was implemented
        let result = ir_utils::calculate_tensor_size(&value.ty, &value.shape);
        
        // This should either succeed with a large number or fail gracefully
        // depending on whether the multiplication causes an overflow during calculation
        match result {
            Ok(size) => {
                // If it succeeds, verify the size is expected
                assert_eq!(size, 100_000 * 100_000 * 4);
            },
            Err(_) => {
                // If it fails, that's also acceptable (better than panicking)
                // The important thing is that it didn't panic
            }
        }
    }

    #[test]
    fn test_attribute_array_operations() {
        // Test operations with complex array attributes
        
        // Create an operation with nested arrays
        let mut op = Operation::new("array_op");
        
        // Create a complex nested array structure
        let complex_array = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
                Attribute::Array(vec![
                    Attribute::Float(3.14),
                    Attribute::Float(2.71),
                ]),
            ]),
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Bool(false),
                ]),
            ]),
        ]);
        
        op.attributes.insert("complex_array".to_string(), complex_array);
        
        // Verify the structure
        assert_eq!(op.attributes.len(), 1);
        
        // Access the complex array
        if let Some(Attribute::Array(ref outer_arr)) = op.attributes.get("complex_array") {
            assert_eq!(outer_arr.len(), 2);
            
            // Check the first nested array
            if let Attribute::Array(ref inner1) = outer_arr[0] {
                assert_eq!(inner1.len(), 3);
                
                // Check first element is Int(1)
                if let Attribute::Int(1) = inner1[0] {
                    // OK
                } else {
                    panic!("Expected Int(1)");
                }
                
                // Check third element is another array
                if let Attribute::Array(ref deeper) = inner1[2] {
                    assert_eq!(deeper.len(), 2);
                    
                    if let Attribute::Float(val) = deeper[0] {
                        assert!((val - 3.14).abs() < f64::EPSILON);
                    } else {
                        panic!("Expected Float(3.14)");
                    }
                } else {
                    panic!("Expected nested array");
                }
            } else {
                panic!("Expected array as first element");
            }
        } else {
            panic!("Expected complex_array to be an array attribute");
        }
    }

    #[test]
    fn test_rstest_style_combinations() {
        // Test some parameterized edge cases using manual loops (similar to rstest)

        // Test different primitive types
        let types = [
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
        ];

        for data_type in &types {
            let value = Value {
                name: "param_test".to_string(),
                ty: data_type.clone(),
                shape: vec![1],
            };

            // Calculate expected size based on type
            let expected_size = match data_type {
                Type::F32 => 4,
                Type::F64 => 8,
                Type::I32 => 4,
                Type::I64 => 8,
                Type::Bool => 1,
                _ => panic!("Unexpected type"),
            };

            let calculated = ir_utils::calculate_tensor_size(data_type, &[1]).unwrap();
            assert_eq!(calculated, expected_size);
        }

        // Test different shapes for each type
        let shapes = vec![
            vec![],        // scalar
            vec![1],       // single element
            vec![10],      // 1D
            vec![2, 3],    // 2D
            vec![2, 3, 4], // 3D
            vec![0],       // zero size
            vec![0, 5],    // zero in first dim
            vec![5, 0],    // zero in second dim
        ];

        for shape in &shapes {
            let value = Value {
                name: "shape_test".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };

            let expected_elements: usize = shape.iter().product();
            let expected_size = expected_elements * 4; // 4 bytes for F32

            let calculated = ir_utils::calculate_tensor_size(&Type::F32, shape).unwrap();
            assert_eq!(calculated, expected_size, "Failed for shape {:?}", shape);
        }
    }

    #[test]
    fn test_tensor_shape_edge_cases() {
        // Test tensor shapes with very large numbers in certain dimensions
        // but keeping total product manageable to avoid overflow
        
        let test_cases = vec![
            // (dimensions, expected_elements, description)
            (vec![1000, 1000], 1_000_000, "square large tensor"),
            (vec![1, 1_000_000], 1_000_000, "very wide tensor"),
            (vec![1_000_000, 1], 1_000_000, "very tall tensor"),
            (vec![10_000, 100, 10], 10_000_000, "multi dimensional large tensor"),
            (vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 1024, "deeply factorized"),
            (vec![0, 1000, 1000], 0, "contains zero dimension"),
            (vec![1000, 0, 1000], 0, "zero in middle dimension"),
            (vec![1000, 1000, 0], 0, "zero at end"),
        ];

        for (shape, expected_elements, description) in test_cases {
            let value = Value {
                name: format!("shape_test_{}", description.replace(" ", "_")),
                ty: Type::F32,
                shape: shape.clone(),
            };

            let actual_elements: usize = value.shape.iter().product();
            assert_eq!(actual_elements, expected_elements, "Failed for: {}", description);

            let calculated_size = ir_utils::calculate_tensor_size(&value.ty, &value.shape).unwrap();
            let expected_size = expected_elements * 4; // 4 bytes for F32
            assert_eq!(calculated_size, expected_size, "Size failed for: {}", description);
        }
    }

    #[test]
    fn test_value_num_elements_method() {
        // Test the new num_elements method on Value
        let test_values = vec![
            // (shape, expected_num_elements, test_name)
            (vec![], 1, "scalar"),
            (vec![1], 1, "single_element"),
            (vec![5], 5, "1d_tensor"),
            (vec![2, 3], 6, "2d_tensor"),
            (vec![2, 3, 4], 24, "3d_tensor"),
            (vec![0], 0, "zero_sized_1d"),
            (vec![0, 5], 0, "zero_first_dim"),
            (vec![5, 0], 0, "zero_second_dim"),
            (vec![10, 0, 20], 0, "zero_middle_dim"),
            (vec![0, 0, 0], 0, "all_zeros"),
        ];

        for (shape, expected_elements, name) in test_values {
            let value = Value {
                name: format!("num_elem_test_{}", name),
                ty: Type::F32,
                shape: shape.clone(),
            };

            let actual_elements = value.num_elements();
            assert_eq!(actual_elements, Some(expected_elements), "num_elements failed for test: {}", name);

            // Also verify using manual calculation
            let manual_elements: usize = value.shape.iter().product();
            assert_eq!(manual_elements, expected_elements, "Manual calc failed for test: {}", name);
        }

        // Test a case that might overflow if calculated naively
        // Using a shape that would overflow a 32-bit calculation but fits in 64-bit
        let large_value = Value {
            name: "large_num_elements".to_string(),
            ty: Type::F32,
            shape: vec![50_000, 50_000], // Would be 2.5 billion elements, over 32-bit limit
        };

        let num_elements_opt = large_value.num_elements();
        assert!(num_elements_opt.is_some()); // Should return Some despite large number
        
        if let Some(num_elements) = num_elements_opt {
            assert_eq!(num_elements, 50_000 * 50_000);
        }
    }
    
    #[test]
    fn test_type_extensions_validation() {
        // Test the TypeExtensions trait implementation
        use crate::ir::TypeExtensions;
        
        // Test primitive types
        assert!(Type::F32.is_valid_type());
        assert!(Type::F64.is_valid_type());
        assert!(Type::I32.is_valid_type());
        assert!(Type::I64.is_valid_type());
        assert!(Type::Bool.is_valid_type());
        
        // Test nested tensor types
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        assert!(nested_type.is_valid_type());
        
        // Test deeply nested type
        let deep_nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![5],
            }),
            shape: vec![3],
        };
        assert!(deep_nested.is_valid_type());
        
        // Test invalid case (though our enum doesn't have invalid cases)
        // The type system ensures all enum variants are valid by construction
    }
}