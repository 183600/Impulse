//! Mathematical edge case tests for the Impulse compiler
//! This file contains tests covering mathematical operations with extreme values,
//! precision issues, and boundary conditions in tensor computations

use crate::{
    ir::{Module, Value, Type, Operation},
    utils::{math_utils, ir_utils},
};

#[cfg(test)]
mod mathematical_edge_case_tests {
    use super::*;

    /// Test 1: GCD and LCM with extreme values
    #[test]
    fn test_gcd_lcm_extreme_values() {
        // Test GCD with large co-prime numbers
        assert_eq!(math_utils::gcd(1_000_000_007, 1_000_000_009), 1);
        
        // Test GCD with one being a multiple of the other
        assert_eq!(math_utils::gcd(2_147_483_648, 1), 1);
        assert_eq!(math_utils::gcd(2_147_483_648, 2_147_483_648), 2_147_483_648);
        
        // Test LCM with large numbers
        assert_eq!(math_utils::lcm(999_983, 999_979), 999_983 * 999_979); // Both are prime
        
        // Test LCM edge cases
        assert_eq!(math_utils::lcm(0, 12345), 0);
        assert_eq!(math_utils::lcm(12345, 0), 0);
    }

    /// Test 2: Round up to multiple with boundary values
    #[test]
    fn test_round_up_to_multiple_boundaries() {
        // Test with value equal to multiple
        assert_eq!(math_utils::round_up_to_multiple(1024, 1024), 1024);
        
        // Test with value just one less than multiple
        assert_eq!(math_utils::round_up_to_multiple(1023, 1024), 1024);
        
        // Test with very large multiples
        assert_eq!(math_utils::round_up_to_multiple(1, usize::MAX), usize::MAX);
        
        // Test with value greater than multiple
        assert_eq!(math_utils::round_up_to_multiple(2049, 1024), 3072);
        
        // Test with multiple of 1
        assert_eq!(math_utils::round_up_to_multiple(123456, 1), 123456);
    }

    /// Test 3: Next power of 2 with extreme values
    #[test]
    fn test_next_power_of_2_extreme_values() {
        // Test with powers of 2
        assert_eq!(math_utils::next_power_of_2(1), 1);
        assert_eq!(math_utils::next_power_of_2(2), 2);
        assert_eq!(math_utils::next_power_of_2(1024), 1024);
        assert_eq!(math_utils::next_power_of_2(65536), 65536);
        
        // Test with one less than power of 2
        assert_eq!(math_utils::next_power_of_2(1023), 1024);
        assert_eq!(math_utils::next_power_of_2(65535), 65536);
        
        // Test with one more than power of 2
        assert_eq!(math_utils::next_power_of_2(1025), 2048);
        assert_eq!(math_utils::next_power_of_2(65537), 131072);
        
        // Test with large values (avoid overflow)
        assert_eq!(math_utils::next_power_of_2(100_000_000), 134_217_728);
    }

    /// Test 4: Tensor size calculation with byte alignment considerations
    #[test]
    fn test_tensor_size_byte_alignment() {
        // F32 tensors (1024*1024*4 = 4194304 bytes)
        let f32_value = Value {
            name: "f32_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1024, 1024], // 1M elements
        };
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &f32_value.shape), Ok(4194304));
        
        // F64 tensors (double the size: 1024*1024*8 = 8388608 bytes)
        let f64_value = Value {
            name: "f64_tensor".to_string(),
            ty: Type::F64,
            shape: vec![1024, 1024],
        };
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F64, &f64_value.shape), Ok(8388608));
        
        // Bool tensors (1 byte per element)
        let bool_value = Value {
            name: "bool_tensor".to_string(),
            ty: Type::Bool,
            shape: vec![100, 100],
        };
        assert_eq!(ir_utils::calculate_tensor_size(&Type::Bool, &bool_value.shape), Ok(10_000));
        
        // Mixed precision comparison
        let mixed_ops = vec![
            (Type::I32, vec![1000, 1000], 4_000_000),
            (Type::I64, vec![1000, 1000], 8_000_000),
            (Type::F32, vec![1000, 1000], 4_000_000),
            (Type::F64, vec![1000, 1000], 8_000_000),
        ];
        
        for (ty, shape, expected_bytes) in mixed_ops {
            assert_eq!(ir_utils::calculate_tensor_size(&ty, &shape), Ok(expected_bytes));
        }
    }

    /// Test 5: Nested tensor size calculations
    #[test]
    fn test_nested_tensor_size_calculations() {
        // Test basic tensor size calculation
        let basic_size = ir_utils::calculate_tensor_size(&Type::F32, &vec![3, 3]).unwrap();
        assert_eq!(basic_size, 36); // 3 * 3 * 4 bytes
        
        // Test nested tensor: tensor<tensor<f32, [3, 3]>, [4]>
        // This represents a batch of 4 tensors, each being 3x3 F32
        let nested_inner = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![3, 3],
        };
        let nested_outer = Type::Tensor {
            element_type: Box::new(nested_inner),
            shape: vec![4],
        };
        
        // 传入空的外部 shape 以避免重复计算内部 shape
        // 计算结果 = 外部 shape [4] (来自类型定义) * 内部张量大小
        // 内部张量大小 = [3, 3] shape * F32 size = 36 bytes
        // 总大小 = 4 * 36 = 144 bytes
        let nested_size = ir_utils::calculate_tensor_size(&nested_outer, &vec![]).unwrap();
        assert_eq!(nested_size, 144); // 4 * 36 bytes
        
        // Test deep nesting: 4 levels deep
        // Level 1: F32 (4 bytes per element)
        // Level 2: tensor<F32, [2, 2]> = 4 elements * 4 bytes = 16 bytes
        // Level 3: tensor<Level2, [3]> = 3 * 16 = 48 bytes  
        // Level 4: tensor<Level3, [4]> = 4 * 48 = 192 bytes
        let level1 = Type::F32;
        let level2 = Type::Tensor { element_type: Box::new(level1), shape: vec![2, 2] };
        let level3 = Type::Tensor { element_type: Box::new(level2), shape: vec![3] };
        let level4 = Type::Tensor { element_type: Box::new(level3), shape: vec![4] };
        
        // 计算时: 使用空外部 shape，让类型内部的 shape 生效
        // = 4 (level4 shape) * (3 (level3 shape) * (4 (level2 shape 2*2) * 4 bytes))
        // = 4 * (3 * 16) = 4 * 48 = 192 bytes
        assert_eq!(ir_utils::calculate_tensor_size(&level4, &vec![]), Ok(192)); // 4 * 3 * 2 * 2 * 4
        
        // Test with different element types in nested structure
        let nested_i64 = Type::Tensor {
            element_type: Box::new(Type::I64),
            shape: vec![2, 2], // 4 elements * 8 bytes = 32 bytes
        };
        let nested_i64_outer = Type::Tensor {
            element_type: Box::new(nested_i64),
            shape: vec![5], // 5 * 32 = 160 bytes
        };
        
        assert_eq!(ir_utils::calculate_tensor_size(&nested_i64_outer, &vec![]), Ok(160));
        
        // Test nested bool tensors
        let nested_bool = Type::Tensor {
            element_type: Box::new(Type::Bool),
            shape: vec![10, 10], // 100 elements * 1 byte = 100 bytes
        };
        let nested_bool_outer = Type::Tensor {
            element_type: Box::new(nested_bool),
            shape: vec![3], // 3 * 100 = 300 bytes
        };
        
        assert_eq!(ir_utils::calculate_tensor_size(&nested_bool_outer, &vec![]), Ok(300));
        
        // Test scalar nested tensor (edge case)
        let scalar_nested = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![], // Scalar (1 element)
        };
        let scalar_outer = Type::Tensor {
            element_type: Box::new(scalar_nested),
            shape: vec![10],
        };
        // 外部 [10] * 内部 scalar (1 * 4 = 4 bytes) = 40 bytes
        assert_eq!(ir_utils::calculate_tensor_size(&scalar_outer, &vec![]), Ok(40));
        
        // Test with explicit external shape (combining with internal shape)
        // 当传入非空外部 shape 时，会与内部 shape 相乘
        let combined_size = ir_utils::calculate_tensor_size(&nested_outer, &vec![2]).unwrap();
        // 外部 [2] * 内部 [4] * 内部 [3,3] * 4 bytes = 2 * 4 * 36 = 288 bytes
        assert_eq!(combined_size, 288);
    }

    /// Test 6: Rank detection with various tensor shapes
    #[test]
    fn test_rank_detection_various_shapes() {
        // Scalar (rank 0)
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(ir_utils::get_rank(&scalar), 0);
        assert!(ir_utils::is_scalar(&scalar));
        
        // 1D tensors of various sizes
        for size in [1, 10, 100, 1000] {
            let vec_val = Value {
                name: format!("vec_{}", size),
                ty: Type::F32,
                shape: vec![size],
            };
            assert_eq!(ir_utils::get_rank(&vec_val), 1);
            assert!(ir_utils::is_vector(&vec_val));
        }
        
        // 2D tensors
        let matrix_shapes = [
            vec![1, 1],   // 1x1 matrix
            vec![1, 100], // Row vector
            vec![100, 1], // Column vector
            vec![10, 10], // Square matrix
            vec![100, 50], // Rectangular matrix
        ];
        
        for shape in matrix_shapes {
            let matrix = Value {
                name: "matrix".to_string(),
                ty: Type::F32,
                shape: shape.clone(),
            };
            assert_eq!(ir_utils::get_rank(&matrix), 2);
            assert!(ir_utils::is_matrix(&matrix));
        }
        
        // Higher rank tensors
        for rank in 3..=10 {
            let shape = vec![2; rank]; // All dimensions are 2
            let tensor = Value {
                name: format!("rank_{}", rank),
                ty: Type::F32,
                shape: shape.clone(),
            };
            assert_eq!(ir_utils::get_rank(&tensor), rank);
            assert!(!ir_utils::is_scalar(&tensor));
            assert!(!ir_utils::is_vector(&tensor));
            assert!(!ir_utils::is_matrix(&tensor));
        }
    }

    /// Test 7: Element count with overflow prevention
    #[test]
    fn test_element_count_overflow_prevention() {
        // Normal cases
        assert_eq!(ir_utils::get_num_elements(&Value {
            name: "test".to_string(),
            ty: Type::F32,
            shape: vec![10, 10, 10],
        }), Some(1000));
        
        // Zero dimensions
        assert_eq!(ir_utils::get_num_elements(&Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 10],
        }), Some(0));
        
        // Large but safe dimensions
        assert_eq!(ir_utils::get_num_elements(&Value {
            name: "large_safe".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],
        }), Some(1_000_000));
        
        // Potentially overflow - should return None
        let potentially_overflow = Value {
            name: "overflow".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2],
        };
        assert_eq!(ir_utils::get_num_elements(&potentially_overflow), None);
        
        // Scalar case
        assert_eq!(ir_utils::get_num_elements(&Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        }), Some(1));
    }

    /// Test 8: Type string representations
    #[test]
    fn test_type_string_representations() {
        // Basic types
        assert_eq!(ir_utils::type_to_string(&Type::F32), "f32");
        assert_eq!(ir_utils::type_to_string(&Type::F64), "f64");
        assert_eq!(ir_utils::type_to_string(&Type::I32), "i32");
        assert_eq!(ir_utils::type_to_string(&Type::I64), "i64");
        assert_eq!(ir_utils::type_to_string(&Type::Bool), "bool");
        
        // Tensor types
        assert_eq!(
            ir_utils::type_to_string(&Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 3]
            }),
            "tensor<f32, [2, 3]>"
        );
        
        assert_eq!(
            ir_utils::type_to_string(&Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![10]
            }),
            "tensor<i64, [10]>"
        );
        
        // Nested tensor types
        let nested = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![3, 3],
            }),
            shape: vec![4],
        };
        let nested_str = ir_utils::type_to_string(&nested);
        assert!(nested_str.contains("tensor"));
        assert!(nested_str.contains("f32"));
    }

    /// Test 9: Operation counting and finding
    #[test]
    fn test_operation_counting_and_finding() {
        let mut module = Module::new("count_test");
        
        // Add various operations
        let op_types = vec!["conv2d", "relu", "maxpool", "conv2d", "relu", "batchnorm", "conv2d"];
        
        for op_type in &op_types {
            let mut op = Operation::new(op_type);
            op.inputs.push(Value {
                name: "input".to_string(),
                ty: Type::F32,
                shape: vec![10, 10],
            });
            module.add_operation(op);
        }
        
        // Count operations
        let counts = ir_utils::count_operations_by_type(&module);
        assert_eq!(counts.get("conv2d"), Some(&3));
        assert_eq!(counts.get("relu"), Some(&2));
        assert_eq!(counts.get("maxpool"), Some(&1));
        assert_eq!(counts.get("batchnorm"), Some(&1));
        
        // Find operations
        let conv_ops = ir_utils::find_operations_by_type(&module, "conv2d");
        assert_eq!(conv_ops.len(), 3);
        
        let relu_ops = ir_utils::find_operations_by_type(&module, "relu");
        assert_eq!(relu_ops.len(), 2);
        
        // Find non-existent operation type
        let nonexistent = ir_utils::find_operations_by_type(&module, "softmax");
        assert_eq!(nonexistent.len(), 0);
    }

    /// Test 10: Element type extraction from nested structures
    #[test]
    fn test_element_type_extraction_nested() {
        // Direct types
        assert_eq!(ir_utils::get_element_type(&Type::F32), &Type::F32);
        assert_eq!(ir_utils::get_element_type(&Type::I32), &Type::I32);
        assert_eq!(ir_utils::get_element_type(&Type::Bool), &Type::Bool);
        
        // Single level nesting
        let level1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![10, 10],
        };
        assert_eq!(ir_utils::get_element_type(&level1), &Type::F32);
        
        // Double level nesting
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![5],
        };
        assert_eq!(ir_utils::get_element_type(&level2), &Type::F32);
        
        // Triple level nesting
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![3],
        };
        assert_eq!(ir_utils::get_element_type(&level3), &Type::F32);
        
        // Different element types in nesting
        let nested_i64 = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::I64),
                shape: vec![2, 2],
            }),
            shape: vec![10],
        };
        assert_eq!(ir_utils::get_element_type(&nested_i64), &Type::I64);
        
        let nested_bool = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Bool),
                    shape: vec![1],
                }),
                shape: vec![5],
            }),
            shape: vec![100],
        };
        assert_eq!(ir_utils::get_element_type(&nested_bool), &Type::Bool);
    }
}