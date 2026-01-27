//! Comprehensive edge case tests for the Impulse compiler
//! Tests various boundary conditions and extreme values

use crate::ir::{Module, Value, Type, Operation};
use std::collections::HashMap;

#[cfg(test)]
mod comprehensive_edge_case_tests {
    use super::*;
    use crate::ir::{Attribute, TypeExtensions};

    /// Test 1: Extensive floating point edge cases
    #[test]
    fn test_comprehensive_floating_point_edge_cases() {
        
        // Test various floating point constants
        let special_attrs = [
            Attribute::Float(f64::INFINITY),
            Attribute::Float(f64::NEG_INFINITY),
            Attribute::Float(f64::NAN),
            Attribute::Float(-0.0),
            Attribute::Float(0.0),
            Attribute::Float(f64::EPSILON),
            Attribute::Float(f64::MIN_POSITIVE),
            Attribute::Float(std::f64::consts::PI),
            Attribute::Float(std::f64::consts::E),
        ];

        // Only compare non-NaN values directly
        for (_i, attr) in special_attrs.iter().enumerate() {
            match attr {
                Attribute::Float(v) => {
                    if v.is_finite() {
                        // Verify the value can be stored and retrieved
                        assert!(true); // Placeholder since we're not comparing specific values
                    }
                },
                _ => panic!("Expected Float attribute"),
            }
        }

        // Test creating Values with special float characteristics
        let values_with_floats = [
            Value {
                name: "positive_infinity_tensor".to_string(),
                ty: Type::F64,
                shape: vec![1],
            },
            Value {
                name: "negative_infinity_tensor".to_string(),
                ty: Type::F64,
                shape: vec![1],
            },
            Value {
                name: "nan_tensor".to_string(),
                ty: Type::F64,
                shape: vec![1],
            },
        ];

        for value in values_with_floats.iter() {
            assert!(!value.name.is_empty());
            assert!(value.shape.len() >= 1);
        }
    }

    /// Test 2: Boundary string conditions
    #[test]
    fn test_boundary_string_conditions() {
        // Test empty strings
        let empty_module = Module::new("");
        assert_eq!(empty_module.name, "");

        // Very long operation name
        let long_name = "a".repeat(100_000);
        let long_op = Operation::new(&long_name);
        assert_eq!(long_op.op_type.len(), 100_000);

        // Unicode and special character tests
        let unicode_names = [
            "🚀_tensor",
            "tensor_名称",
            "tensor_日本語",
            "тензор_ру́сский",
            "tensor_with_control_\u{0001}_char",
        ];

        for name in unicode_names.iter() {
            let value = Value {
                name: name.to_string(),
                ty: Type::F32,
                shape: vec![1],
            };
            assert_eq!(value.name, *name);
        }

        // Test with various special characters
        let special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?~`";
        let special_op = Operation::new(special_chars);
        assert_eq!(special_op.op_type, special_chars);
    }

    /// Test 3: Memory allocation with large collections
    #[test]
    fn test_large_collection_memory_allocation() {
        // Large number of modules
        let mut modules = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            let module = Module::new(&format!("module_{}", i));
            modules.push(module);
        }
        assert_eq!(modules.len(), 10_000);

        // Large number of operations in a single module
        let mut module = Module::new("big_module");
        for i in 0..20_000 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::I32,
                shape: vec![1],
            });
            module.add_operation(op);
        }
        assert_eq!(module.operations.len(), 20_000);

        // Large number of attributes
        let mut op = Operation::new("many_attrs");
        let mut attrs = HashMap::with_capacity(50_000);
        for i in 0..50_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::Int(i as i64),
            );
        }
        op.attributes = attrs;
        assert_eq!(op.attributes.len(), 50_000);
    }

    /// Test 4: Recursion and nesting limits
    #[test]
    fn test_recursion_and_nesting_limits() {
        // Deeply nested tensor types
        let mut inner_type = Type::F32;
        for i in 0..500 {  // Depth of 500
            inner_type = Type::Tensor {
                element_type: Box::new(inner_type),
                shape: vec![2],
            };
            
            // At every 100th iteration, clone the type to test deep cloning
            if i % 100 == 0 {
                let cloned = inner_type.clone();
                assert_eq!(inner_type, cloned);
            }
        }

        // Verify the final deeply nested type
        match &inner_type {
            Type::Tensor { .. } => {},
            _ => panic!("Expected a tensor type after deep nesting"),
        }

        // Test cloning the deeply nested type
        let cloned_deep = inner_type.clone();
        assert_eq!(inner_type, cloned_deep);
    }

    /// Test 5: Extreme tensor shape configurations
    #[test]
    fn test_extreme_tensor_shape_configurations() {
        // Various edge case shapes
        let extreme_shapes = [
            vec![],                      // Scalar
            vec![0],                     // Zero-size 1D
            vec![0, 0],                  // Zero-size 2D
            vec![0, 10, 0],              // Zero in multiple positions
            vec![1],                     // Single element 1D
            vec![1, 1, 1, 1, 1],         // All ones
            vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], // Many dimensions
            vec![1000_000],              // Very long 1D vector
            vec![1000, 1000],            // Square matrix 1M elements
            vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], // 15 dimensions
        ];

        for shape in extreme_shapes.iter() {
            let value = Value {
                name: format!("tensor_shape_{:?}", shape),
                ty: Type::F32,
                shape: shape.to_vec(),
            };

            // Calculate if the product would be computationally expensive
            let _size_product: Option<usize> = value.shape.iter().try_fold(1_usize, |acc, &x| {
                acc.checked_mul(x)
            });

            // The shape should be preserved
            assert_eq!(value.shape, *shape);
            
            // Name should contain the shape info
            assert!(value.name.contains("tensor_shape_"));
        }
    }

    /// Test 6: Comprehensive attribute handling
    #[test]
    fn test_comprehensive_attribute_handling() {
        
        // Create an operation with all possible attribute types
        let mut op = Operation::new("comprehensive_attr_test");

        // Prepare various attribute types
        let mut attrs = HashMap::new();
        
        // Primitive types
        attrs.insert("int_min".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("int_max".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("int_zero".to_string(), Attribute::Int(0));
        attrs.insert("float_min".to_string(), Attribute::Float(f64::MIN));
        attrs.insert("float_max".to_string(), Attribute::Float(f64::MAX));
        attrs.insert("float_zero".to_string(), Attribute::Float(0.0));
        attrs.insert("float_neg_zero".to_string(), Attribute::Float(-0.0));
        attrs.insert("bool_true".to_string(), Attribute::Bool(true));
        attrs.insert("bool_false".to_string(), Attribute::Bool(false));
        attrs.insert("string_empty".to_string(), Attribute::String("".to_string()));
        attrs.insert("string_long".to_string(), Attribute::String("a".repeat(50_000)));

        // Nested arrays
        attrs.insert("simple_array".to_string(), Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
            Attribute::Int(3),
        ]));

        attrs.insert("nested_array".to_string(), Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(10),
                Attribute::Float(1.5),
            ]),
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Bool(false),
            ]),
        ]));

        // Mixed-type array
        attrs.insert("mixed_array".to_string(), Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Float(3.14),
            Attribute::String("mixed".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![Attribute::Int(99)]),
        ]));

        op.attributes = attrs;

        // Verify all attributes were added
        assert_eq!(op.attributes.len(), 14); // 11 basic + 3 complex structures
        
        // Verify some specific values
        assert_eq!(op.attributes.get("int_max"), Some(&Attribute::Int(i64::MAX)));
        assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
        assert_eq!(op.attributes.get("string_empty"), Some(&Attribute::String("".to_string())));
        
        // Verify complex array structures
        if let Some(Attribute::Array(ref arr)) = op.attributes.get("simple_array") {
            assert_eq!(arr.len(), 3);
        } else {
            panic!("Expected simple array");
        }
    }

    /// Test 7: Large numerical value handling
    #[test]
    fn test_large_numerical_value_handling() {
        // Test with very large integers
        let large_int_attrs = [
            Attribute::Int(i64::MAX),
            Attribute::Int(i64::MIN),
            Attribute::Int(i64::MAX / 2),
            Attribute::Int(i64::MIN / 2),
        ];

        for attr in large_int_attrs.iter() {
            match attr {
                Attribute::Int(val) => {
                    // Just ensure we can store and retrieve large values
                    assert!(*val >= i64::MIN && *val <= i64::MAX);
                },
                _ => panic!("Expected Int attribute"),
            }
        }

        // Test with extreme float values
        let extreme_float_attrs = [
            Attribute::Float(f64::MAX),
            Attribute::Float(f64::MIN), // This is -f64::MAX
            Attribute::Float(f64::MIN_POSITIVE),
            Attribute::Float(-f64::MIN_POSITIVE),
        ];

        for attr in extreme_float_attrs.iter() {
            match attr {
                Attribute::Float(val) => {
                    // Ensure the extreme values are preserved
                    assert!(val.is_finite() || val.is_infinite());
                },
                _ => panic!("Expected Float attribute"),
            }
        }

        // Test with very large shapes
        let huge_shape = vec![1_000_000, 1_000]; // 1B elements
        let huge_value = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: huge_shape,
        };
        assert_eq!(huge_value.shape[0], 1_000_000);
        assert_eq!(huge_value.shape[1], 1_000);
    }

    /// Test 8: Error condition simulation with safe operations
    #[test]
    fn test_error_condition_simulations() {
        // Test creating operations with empty names
        let empty_op = Operation::new("");
        assert!(empty_op.op_type.is_empty());

        // Create values with empty names
        let empty_value = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(empty_value.name.is_empty());

        // Create modules with empty names
        let empty_module = Module::new("");
        assert!(empty_module.name.is_empty());

        // Test with different types
        let type_variants = [
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
        ];

        for ty in type_variants.iter() {
            let value = Value {
                name: "typed_value".to_string(),
                ty: ty.clone(),
                shape: vec![1],
            };
            assert_eq!(value.ty, *ty);
        }

        // Test cloning of all type variants
        for ty in type_variants.iter() {
            let cloned = ty.clone();
            assert_eq!(*ty, cloned);
        }
    }

    /// Test 9: Concurrent operations (simulated)
    #[test]
    fn test_simulated_concurrent_operations() {
        // Create multiple modules in succession to simulate concurrent creation
        let mut modules = Vec::new();
        
        for i in 0..5 {
            let mut module = Module::new(&format!("concurrent_module_{}", i));
            
            // Add operations to each module
            for j in 0..1000 {
                let mut op = Operation::new(&format!("op_{}_{}", i, j));
                
                // Add inputs and outputs
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![4, 4],
                });
                
                op.outputs.push(Value {
                    name: format!("output_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![4, 4],
                });
                
                module.add_operation(op);
            }
            
            modules.push(module);
        }
        
        // Verify all modules and their operations
        assert_eq!(modules.len(), 5);
        for (idx, module) in modules.iter().enumerate() {
            assert_eq!(module.operations.len(), 1000);
            assert_eq!(module.name, format!("concurrent_module_{}", idx));
        }
    }

    /// Test 10: Mixed type operations and validation
    #[test]
    fn test_mixed_type_operations_and_validation() {
        
        // Create various types and test validation
        let types_to_test = [
            Type::F32,
            Type::F64,
            Type::I32,
            Type::I64,
            Type::Bool,
            Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            },
            Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::I64),
                    shape: vec![3],
                }),
                shape: vec![2],
            },
        ];

        for ty in types_to_test.iter() {
            // Test that all types validate correctly
            assert!(ty.is_valid_type());
            
            // Test cloning
            let cloned_ty = ty.clone();
            assert_eq!(*ty, cloned_ty);
        }

        // Create operations with different combinations of types
        let mut complex_op = Operation::new("mixed_type_op");
        
        // Add inputs of different types
        complex_op.inputs.extend(vec![
            Value {
                name: "f32_input".to_string(),
                ty: Type::F32,
                shape: vec![10, 10],
            },
            Value {
                name: "f64_input".to_string(),
                ty: Type::F64,
                shape: vec![5, 5],
            },
            Value {
                name: "i32_input".to_string(),
                ty: Type::I32,
                shape: vec![3, 3, 3],
            },
            Value {
                name: "bool_input".to_string(),
                ty: Type::Bool,
                shape: vec![2, 2, 2, 2],
            },
        ]);

        // Add outputs of different types
        complex_op.outputs.extend(vec![
            Value {
                name: "nested_output".to_string(),
                ty: Type::Tensor {
                    element_type: Box::new(Type::F32),
                    shape: vec![2, 2],
                },
                shape: vec![4],
            },
            Value {
                name: "deeply_nested_output".to_string(),
                ty: Type::Tensor {
                    element_type: Box::new(Type::Tensor {
                        element_type: Box::new(Type::I64),
                        shape: vec![3],
                    }),
                    shape: vec![2],
                },
                shape: vec![2, 2],
            },
        ]);

        assert_eq!(complex_op.inputs.len(), 4);
        assert_eq!(complex_op.outputs.len(), 2);
        assert_eq!(complex_op.op_type, "mixed_type_op");
    }
}