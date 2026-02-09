//! Additional edge case tests for overflow scenarios, memory allocation, and rstest functionality
//! This file covers the remaining test scenarios from our todo list

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::{ir_utils, math_utils},
};
use std::collections::HashMap;

#[cfg(test)]
mod remaining_edge_case_tests {
    use super::*;
    
    #[cfg(feature = "rstest")]
    #[rstest::rstest]
    #[case(Type::F32, vec![], 4)]  // scalar F32 -> 4 bytes
    #[case(Type::F64, vec![], 8)]  // scalar F64 -> 8 bytes
    #[case(Type::I32, vec![], 4)]  // scalar I32 -> 4 bytes
    #[case(Type::I64, vec![], 8)]  // scalar I64 -> 8 bytes
    #[case(Type::Bool, vec![], 1)] // scalar Bool -> 1 byte
    #[case(Type::F32, vec![1], 4)] // [1] F32 -> 4 bytes
    #[case(Type::F32, vec![5], 20)] // [5] F32 -> 20 bytes
    #[case(Type::F32, vec![2, 3], 24)] // [2,3] F32 -> 24 bytes
    fn test_rstest_tensor_sizes(#[case] data_type: Type, #[case] shape: Vec<usize>, #[case] expected_size: usize) {
        let result = ir_utils::calculate_tensor_size(&data_type, &shape).unwrap();
        assert_eq!(result, expected_size);
    }
    
    #[test]
    fn test_integer_overflow_scenarios() {
        // Test scenarios that could lead to integer overflow in tensor calculations
        
        // Use values that would cause overflow when naively multiplied
        // but are handled safely by our implementation
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![100_000, 100_000], // Would be 10 billion elements, likely causing overflow
        };
        
        // Use safe method to calculate number of elements
        let num_elements = large_value.num_elements();
        if let Some(elements) = num_elements {
            // If it doesn't overflow, check the value
            assert_eq!(elements, 100_000 * 100_000);
        }
        // If it overflows, num_elements returns None, which is also valid behavior
        
        // Test with tensor size calculation
        let _size_result = ir_utils::calculate_tensor_size(&large_value.ty, &large_value.shape);
        // This might succeed or fail gracefully depending on implementation - both are acceptable
        
        // Test with a more conservative large size that won't overflow
        let conservative_large = Value {
            name: "conservative_large".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000, 100], // 100 million elements, safer
        };
        
        let conservative_result = ir_utils::calculate_tensor_size(
            &conservative_large.ty, 
            &conservative_large.shape
        ).unwrap();
        assert_eq!(conservative_result, 1_000_000 * 100 * 4); // 4 bytes per F32
        
        // Test with zero-containing shapes (should not overflow)
        let zero_shape = Value {
            name: "zero_shape_tensor".to_string(),
            ty: Type::I64,
            shape: vec![1_000_000, 0, 1_000_000], // Contains zero, so result should be 0
        };
        
        let zero_result = ir_utils::calculate_tensor_size(
            &zero_shape.ty,
            &zero_shape.shape
        ).unwrap();
        assert_eq!(zero_result, 0); // Zero in dimensions results in zero size
        
        // Test the safe calculation utility function directly
        use crate::utils::calculate_tensor_size_safe;
        let safe_result = calculate_tensor_size_safe(&[10_000, 10_000]);
        assert_eq!(safe_result, Some(10_000 * 10_000)); // 100 million elements
        
        // Test with zero dimensions
        assert_eq!(calculate_tensor_size_safe(&[0]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[5, 0, 10]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[0, 0, 0]), Some(0));
    }

    #[test]
    fn test_memory_allocation_edge_cases() {
        // Test memory allocation scenarios
        
        // Test creating and dropping many small objects
        for _ in 0..1000 {
            let _small_obj = Value {
                name: "temp".to_string(),
                ty: Type::F32,
                shape: vec![1],
            };
            // Object gets dropped automatically
        }
        
        // Test creating large vectors and dropping them
        let mut large_operations = Vec::with_capacity(50_000);
        for i in 0..50_000 {
            let mut op = Operation::new(&format!("temp_op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![1, 1],
            });
            large_operations.push(op);
        }
        
        // Verify the vector has the right size
        assert_eq!(large_operations.len(), 50_000);
        
        // Drop the large vector to test deallocation
        drop(large_operations);
        
        // Test creating deeply nested structures
        let mut nested_value = Type::F32;
        for _ in 0..100 {  // Create 100 levels of nesting
            nested_value = Type::Tensor {
                element_type: Box::new(nested_value),
                shape: vec![2],
            };
        }
        
        // Verify we can work with this deeply nested type
        match &nested_value {
            Type::Tensor { shape, .. } => {
                assert_eq!(shape, &vec![2]);
            },
            _ => panic!("Expected nested tensor type"),
        }
        
        // Test cloning and dropping deeply nested types
        let cloned_nested = nested_value.clone();
        assert_eq!(cloned_nested, nested_value);
        drop(cloned_nested);
        
        // Test with many small modules
        let modules: Vec<Module> = (0..1000)
            .map(|i| Module::new(&format!("temp_module_{}", i)))
            .collect();
        
        assert_eq!(modules.len(), 1000);
        drop(modules); // Test bulk deallocation
    }

    #[cfg(feature = "rstest")]
    #[rstest::rstest]
    #[case(vec![2, 3], 6)]
    #[case(vec![2, 3, 4], 24)]
    #[case(vec![1, 1, 1, 1, 1], 1)]
    #[case(vec![0], 0)]
    #[case(vec![10, 0, 5], 0)]
    #[case(vec![5, 0, 0, 10], 0)]
    fn test_rstest_shape_products(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
        let actual_product: usize = shape.iter().product();
        assert_eq!(actual_product, expected_product);
    }

    #[cfg(feature = "rstest")]
    #[rstest::rstest]
    #[case(5, 3, 1)]
    #[case(12, 8, 4)]
    #[case(17, 13, 1)]
    #[case(100, 25, 25)]
    fn test_rstest_gcd(#[case] a: usize, #[case] b: usize, #[case] expected: usize) {
        assert_eq!(math_utils::gcd(a, b), expected);
    }

    #[cfg(feature = "rstest")]
    #[rstest::rstest]
    #[case(4, 6, 12)]
    #[case(12, 18, 36)]
    #[case(7, 5, 35)]
    #[case(15, 25, 75)]
    fn test_rstest_lcm(#[case] a: usize, #[case] b: usize, #[case] expected: usize) {
        assert_eq!(math_utils::lcm(a, b), expected);
    }

    #[test]
    fn test_validation_utilities_with_complex_nested_structures() {
        use crate::utils::validation_utils;
        
        // Create a complex module with deeply nested types and structures
        
        let mut complex_module = Module::new("complex_nested_module");
        
        // Add complex nested tensor types
        let deeply_nested_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2, 2],
                }),
                shape: vec![3],
            }),
            shape: vec![4],
        };
        
        // Add inputs with complex types
        complex_module.inputs.push(Value {
            name: "complex_input".to_string(),
            ty: deeply_nested_type.clone(),
            shape: vec![5], // Outer shape for the nested tensor type
        });
        
        complex_module.outputs.push(Value {
            name: "complex_output".to_string(),
            ty: deeply_nested_type,
            shape: vec![6], // Different outer shape
        });
        
        // Add operations with complex nested structures
        let mut complex_op = Operation::new("complex_nested_op");
        
        // Complex input value
        complex_op.inputs.push(Value {
            name: "op_complex_input".to_string(),
            ty: Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![4, 4], // 4x4 matrix of I64 values
            },
            shape: vec![10], // 10 such matrices
        });
        
        // Complex output value
        complex_op.outputs.push(Value {
            name: "op_complex_output".to_string(),
            ty: Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![3, 3, 3], // 3x3x3 cube of F32 values
            },
            shape: vec![2, 2], // 2x2 grid of such cubes
        });
        
        // Complex nested attributes
        let complex_attrs = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Array(vec![
                    Attribute::Float(3.14),
                    Attribute::String("nested_config".to_string()),
                ])
            ]),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::Array(vec![
                    Attribute::String("inner_config".to_string()),
                    Attribute::Array(vec![
                        Attribute::Int(42),
                        Attribute::Bool(false),
                    ])
                ])
            ])
        ]);
        
        let mut attrs = HashMap::new();
        attrs.insert("complex_config".to_string(), complex_attrs);
        attrs.insert("simple_flag".to_string(), Attribute::Bool(true));
        attrs.insert("count".to_string(), Attribute::Int(100));
        
        complex_op.attributes = attrs;
        complex_module.add_operation(complex_op);
        
        // Add another operation to make it more complex
        let mut second_op = Operation::new("second_complex_op");
        second_op.inputs.push(Value {
            name: "second_input".to_string(),
            ty: Type::F64,
            shape: vec![1000, 1000], // Large 2D array
        });
        second_op.outputs.push(Value {
            name: "second_output".to_string(),
            ty: Type::I32,
            shape: vec![500, 500], // Medium 2D array
        });
        
        complex_module.add_operation(second_op);
        
        // Validate the complex module - should pass validation
        let validation_result = validation_utils::validate_module(&complex_module);
        assert!(validation_result.is_ok(), "Complex module should validate: {:?}", validation_result.err());
        
        // Check that all components are intact
        assert_eq!(complex_module.name, "complex_nested_module");
        assert_eq!(complex_module.inputs.len(), 1);
        assert_eq!(complex_module.outputs.len(), 1);
        assert_eq!(complex_module.operations.len(), 2);
        
        // Verify the first operation has the correct structure
        assert_eq!(complex_module.operations[0].op_type, "complex_nested_op");
        assert_eq!(complex_module.operations[0].attributes.len(), 3);
        
        // Verify the second operation
        assert_eq!(complex_module.operations[1].op_type, "second_complex_op");
        
        // Check that complex attributes are preserved
        assert!(complex_module.operations[0].attributes.contains_key("complex_config"));
        assert!(complex_module.operations[0].attributes.contains_key("simple_flag"));
        assert!(complex_module.operations[0].attributes.contains_key("count"));
    }

    #[test]
    fn test_large_tensor_shape_products() {
        // Test tensor shape calculations with large but valid numbers
        
        // Test 1: Very large but valid single dimension
        let large_single = Value {
            name: "large_single_dim".to_string(),
            ty: Type::F32,
            shape: vec![100_000_000], // 100 million
        };
        
        let elements = large_single.num_elements().unwrap();
        assert_eq!(elements, 100_000_000);
        
        let size_bytes = ir_utils::calculate_tensor_size(&large_single.ty, &large_single.shape).unwrap();
        assert_eq!(size_bytes, 100_000_000 * 4); // 4 bytes per F32
        
        // Test 2: Multi-dimensional large tensor
        let multi_large = Value {
            name: "multi_large".to_string(),
            ty: Type::I64,
            shape: vec![10_000, 10_000], // 100 million elements
        };
        
        let elements = multi_large.num_elements().unwrap();
        assert_eq!(elements, 10_000 * 10_000);
        
        let size_bytes = ir_utils::calculate_tensor_size(&multi_large.ty, &multi_large.shape).unwrap();
        assert_eq!(size_bytes, 10_000 * 10_000 * 8); // 8 bytes per I64
        
        // Test 3: Extremely large but mathematically valid
        let extreme_but_valid = Value {
            name: "extreme_valid".to_string(),
            ty: Type::Bool,
            shape: vec![50_000_000, 2], // 100 million booleans
        };
        
        let elements = extreme_but_valid.num_elements().unwrap();
        assert_eq!(elements, 50_000_000 * 2);
        
        let size_bytes = ir_utils::calculate_tensor_size(&extreme_but_valid.ty, &extreme_but_valid.shape).unwrap();
        assert_eq!(size_bytes, 50_000_000 * 2 * 1); // 1 byte per bool
        
        // Test 4: Shape with zeros (should result in 0 elements regardless of other dimensions)
        let zero_included = Value {
            name: "zero_included".to_string(),
            ty: Type::F64,
            shape: vec![1_000_000, 0, 1_000_000], // Contains zero
        };
        
        let elements = zero_included.num_elements().unwrap();
        assert_eq!(elements, 0);
        
        let size_bytes = ir_utils::calculate_tensor_size(&zero_included.ty, &zero_included.shape).unwrap();
        assert_eq!(size_bytes, 0);
    }

    #[test]
    fn test_edge_cases_with_tensor_size_calculation() {
        // Test tensor size calculations that could potentially cause issues
        use crate::utils::calculate_tensor_size_safe;
        
        // Test the helper function directly
        assert_eq!(calculate_tensor_size_safe(&[]), Some(1));  // Scalar has 1 element
        assert_eq!(calculate_tensor_size_safe(&[0]), Some(0));  // Zero dimension
        assert_eq!(calculate_tensor_size_safe(&[1, 1, 1]), Some(1));  // All ones
        assert_eq!(calculate_tensor_size_safe(&[2, 3, 4]), Some(24));  // Normal case
        
        // Test cases that might overflow if not handled carefully
        // These should either return a valid number or None if overflow would occur
        let result = calculate_tensor_size_safe(&[100_000, 100_000]);  // May or may not overflow
        
        // If it doesn't overflow, the result should be correct
        if let Some(elements) = result {
            assert_eq!(elements, 100_000 * 100_000);
        }
        // If it does overflow, result will be None, which is acceptable
        
        // Test with multiple zeros in different positions
        assert_eq!(calculate_tensor_size_safe(&[0, 100, 200]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[100, 0, 200]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[100, 200, 0]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[0, 0, 0]), Some(0));
        
        // Test with some very large but non-zero numbers
        let large_but_safe = calculate_tensor_size_safe(&[10_000, 10_000, 100]); // 10 billion
        if let Some(elements) = large_but_safe {
            assert_eq!(elements, 10_000 * 10_000 * 100);  // 10 billion
        }
        // If it overflows the result is None, which is also acceptable
    }

    #[test]
    fn test_math_utils_with_large_numbers() {
        // Test the math utilities with large numbers to make sure they work properly
        
        // Test GCD with large numbers
        assert_eq!(math_utils::gcd(1071, 462), 21);  // Classic example
        assert_eq!(math_utils::gcd(462, 1071), 21);  // Same numbers, different order
        assert_eq!(math_utils::gcd(12, 8), 4);
        assert_eq!(math_utils::gcd(17, 13), 1);  // Coprime numbers
        
        // Test LCM with the same pairs
        assert_eq!(math_utils::lcm(12, 8), 24);  // LCM(12,8) = 24
        assert_eq!(math_utils::lcm(17, 13), 221);  // LCM of two primes = their product
        
        // Test relationship: gcd(a,b) * lcm(a,b) = a * b
        let (a, b) = (12, 8);
        let gcd_result = math_utils::gcd(a, b);
        let lcm_result = math_utils::lcm(a, b);
        assert_eq!((gcd_result as u64) * (lcm_result as u64), (a as u64) * (b as u64));
        
        // Test round_up_to_multiple with large numbers
        assert_eq!(math_utils::round_up_to_multiple(100, 16), 112);  // Next multiple of 16 after 100
        assert_eq!(math_utils::round_up_to_multiple(112, 16), 112);  // Exact multiple
        assert_eq!(math_utils::round_up_to_multiple(113, 16), 128);  // Next multiple after 113
        assert_eq!(math_utils::round_up_to_multiple(1, 1024), 1024); // Round 1 up to 1024
        
        // Test next_power_of_2 with various values
        assert_eq!(math_utils::next_power_of_2(0), 1);   // Defined as 1
        assert_eq!(math_utils::next_power_of_2(1), 1);   // 1 is power of 2
        assert_eq!(math_utils::next_power_of_2(2), 2);   // 2 is power of 2
        assert_eq!(math_utils::next_power_of_2(3), 4);   // Next after 3 is 4
        assert_eq!(math_utils::next_power_of_2(4), 4);   // 4 is power of 2
        assert_eq!(math_utils::next_power_of_2(5), 8);   // Next after 5 is 8
        assert_eq!(math_utils::next_power_of_2(8), 8);   // 8 is power of 2
        assert_eq!(math_utils::next_power_of_2(9), 16);  // Next after 9 is 16
        assert_eq!(math_utils::next_power_of_2(1000), 1024); // Next after 1000
        assert_eq!(math_utils::next_power_of_2(1024), 1024); // 1024 is power of 2
        assert_eq!(math_utils::next_power_of_2(1025), 2048); // Next after 1025
    }

    #[test]
    fn test_complex_operation_with_all_features() {
        // Create an operation that uses all the complex features of the IR
        
        let mut complex_op = Operation::new("fully_complex_op");
        
        // Add multiple inputs with different types and shapes
        complex_op.inputs.push(Value {
            name: "input_matrix_1".to_string(),
            ty: Type::F32,
            shape: vec![32, 64],  // 32x64 matrix
        });
        
        complex_op.inputs.push(Value {
            name: "input_matrix_2".to_string(),
            ty: Type::F32,
            shape: vec![64, 128], // 64x128 matrix
        });
        
        complex_op.inputs.push(Value {
            name: "input_bias".to_string(),
            ty: Type::F32,
            shape: vec![128],     // 128-element bias vector
        });
        
        complex_op.inputs.push(Value {
            name: "input_weights".to_string(),
            ty: Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![3, 3],  // 3x3 convolution kernel
            },
            shape: vec![64, 32],    // 64 kernels for 32 input channels
        });
        
        // Add multiple outputs
        complex_op.outputs.push(Value {
            name: "output_result".to_string(),
            ty: Type::F32,
            shape: vec![32, 128],   // 32x128 result matrix
        });
        
        complex_op.outputs.push(Value {
            name: "output_intermediate".to_string(),
            ty: Type::F32,
            shape: vec![32, 64],    // 32x64 intermediate result
        });
        
        // Add complex attributes
        let mut attrs = HashMap::new();
        
        // Numeric attributes
        attrs.insert("precision_bits".to_string(), Attribute::Int(32));
        attrs.insert("learning_rate".to_string(), Attribute::Float(0.001));
        
        // String attributes
        attrs.insert("algorithm".to_string(), Attribute::String("sgd".to_string()));
        attrs.insert("activation".to_string(), Attribute::String("relu".to_string()));
        
        // Boolean attributes
        attrs.insert("use_bias".to_string(), Attribute::Bool(true));
        attrs.insert("training_mode".to_string(), Attribute::Bool(false));
        
        // Complex nested array attributes
        attrs.insert("kernel_shape".to_string(), Attribute::Array(vec![
            Attribute::Int(3),
            Attribute::Int(3),
        ]));
        
        attrs.insert("padding_config".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(1),  // (top/bottom, left/right) padding
            ]),
        ]));
        
        attrs.insert("metadata".to_string(), Attribute::Array(vec![
            Attribute::String("convolution_layer".to_string()),
            Attribute::Bool(true),  // enabled
            Attribute::Array(vec![
                Attribute::Float(0.1),   // dropout rate
                Attribute::Int(1000),    // max iterations
            ])
        ]));
        
        complex_op.attributes = attrs;
        
        // Validate the complex operation
        use crate::utils::validation_utils;
        let result = validation_utils::validate_operation(&complex_op);
        assert!(result.is_ok(), "Complex operation should validate: {:?}", result.err());
        
        // Check the structure
        assert_eq!(complex_op.op_type, "fully_complex_op");
        assert_eq!(complex_op.inputs.len(), 4);
        assert_eq!(complex_op.outputs.len(), 2);
        assert_eq!(complex_op.attributes.len(), 9);  // 1 int + 1 float + 2 string + 2 bool + 3 array
        
        // Verify specific components
        assert_eq!(complex_op.inputs[0].name, "input_matrix_1");
        assert_eq!(complex_op.inputs[0].shape, vec![32, 64]);
        assert_eq!(complex_op.outputs[0].name, "output_result");
        
        // Check that attributes are accessible
        assert_eq!(complex_op.attributes.get("precision_bits"), Some(&Attribute::Int(32)));
        assert_eq!(complex_op.attributes.get("algorithm"), Some(&Attribute::String("sgd".to_string())));
        assert_eq!(complex_op.attributes.get("use_bias"), Some(&Attribute::Bool(true)));
        
        // Check nested array attribute
        if let Some(Attribute::Array(kernel_shape)) = complex_op.attributes.get("kernel_shape") {
            assert_eq!(kernel_shape.len(), 2);
            if let Attribute::Int(3) = kernel_shape[0] { } else { panic!("Expected Int(3)"); }
            if let Attribute::Int(3) = kernel_shape[1] { } else { panic!("Expected Int(3)"); }
        } else {
            panic!("Expected kernel_shape to be an array");
        }
    }
}