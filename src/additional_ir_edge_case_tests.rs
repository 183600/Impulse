//! Additional edge case tests for IR structures in the Impulse compiler
//! Covers boundary conditions not addressed in existing tests

#[cfg(test)]
mod tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;

    #[test]
    fn test_maximum_dimensionality_tensor() {
        // Test tensor with maximum allowed dimensions to ensure no limitations
        let max_dims = vec![1; 100];  // 100 dimensions, each of size 1
        let value = Value {
            name: "max_dims".to_string(),
            ty: Type::F32,
            shape: max_dims,
        };
        
        assert_eq!(value.shape.len(), 100);
        assert_eq!(value.ty, Type::F32);
        // All dimensions are 1, so total elements should be 1
        let product: usize = value.shape.iter().product();
        assert_eq!(product, 1);
    }

    #[test]
    fn test_empty_module_operations() {
        // Test handling of empty modules with no operations but with inputs/outputs
        let mut module = Module::new("");
        module.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        
        assert_eq!(module.name, "");
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.inputs.len(), 1);
        assert_eq!(module.outputs.len(), 1);
        assert_eq!(module.inputs[0].name, "input1");
        assert_eq!(module.outputs[0].name, "output1");
    }

    #[test]
    fn test_operation_with_empty_inputs_outputs() {
        // Test operation with empty inputs and outputs but with attributes
        let mut op = Operation::new("noop_with_attrs");
        let mut attrs = HashMap::new();
        attrs.insert("debug".to_string(), Attribute::Bool(true));
        attrs.insert("count".to_string(), Attribute::Int(42));
        op.attributes = attrs;
        
        assert_eq!(op.op_type, "noop_with_attrs");
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 2);
        assert!(op.attributes.contains_key("debug"));
        assert!(op.attributes.contains_key("count"));
    }

    #[test]
    fn test_deeply_nested_tensor_with_empty_inner_shape() {
        // Test nested tensor with empty inner shape: tensor<tensor<f32, []>, [5]>
        let inner_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),  // Scalar of type f32
            shape: vec![],                     // Empty shape means scalar
        };
        let outer_tensor = Type::Tensor {
            element_type: Box::new(inner_tensor),
            shape: vec![5],                   // Array of 5 scalars
        };

        match &outer_tensor {
            Type::Tensor { element_type: boxed_inner, shape: outer_shape } => {
                assert_eq!(outer_shape, &vec![5]);
                
                match boxed_inner.as_ref() {
                    Type::Tensor { element_type: final_type, shape: inner_shape } => {
                        assert!(inner_shape.is_empty());
                        
                        match final_type.as_ref() {
                            Type::F32 => {}, // Success
                            _ => panic!("Expected F32 as innermost type"),
                        }
                    },
                    _ => panic!("Expected nested tensor"),
                }
            },
            _ => panic!("Expected outer tensor"),
        }
    }

    #[test]
    fn test_all_basic_types_equivalence() {
        // Test that each basic type is equal to itself
        let f32_a = Type::F32;
        let f32_b = Type::F32;
        assert_eq!(f32_a, f32_b);

        let f64_a = Type::F64;
        let f64_b = Type::F64;
        assert_eq!(f64_a, f64_b);

        let i32_a = Type::I32;
        let i32_b = Type::I32;
        assert_eq!(i32_a, i32_b);

        let i64_a = Type::I64;
        let i64_b = Type::I64;
        assert_eq!(i64_a, i64_b);

        let bool_a = Type::Bool;
        let bool_b = Type::Bool;
        assert_eq!(bool_a, bool_b);
    }

    #[test]
    fn test_value_with_special_unicode_name() {
        // Test with unicode characters in value names
        let unicode_value = Value {
            name: "test_🚀_tensor_中文".to_string(),  // Includes emoji and Chinese characters
            ty: Type::I64,
            shape: vec![2, 3],
        };
        
        assert_eq!(unicode_value.name, "test_🚀_tensor_中文");
        assert_eq!(unicode_value.ty, Type::I64);
        assert_eq!(unicode_value.shape, vec![2, 3]);
    }

    #[test]
    fn test_attribute_array_with_different_nested_depths() {
        // Test attribute arrays with varying nested depths
        let complex_attr = Attribute::Array(vec![
            Attribute::Array(vec![  // Nested array of depth 2
                Attribute::Int(10),
                Attribute::Array(vec![  // Nested array of depth 3
                    Attribute::Float(1.5),
                    Attribute::Float(2.5),
                ]),
                Attribute::Int(20),
            ]),
            Attribute::String("deep".to_string()),
        ]);

        match complex_attr {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                
                // First element is a nested array
                if let Attribute::Array(middle) = &outer[0] {
                    assert_eq!(middle.len(), 3);
                    
                    if let Attribute::Array(deep) = &middle[1] {
                        assert_eq!(deep.len(), 2);
                    } else {
                        panic!("Expected deep array at middle[1]");
                    }
                } else {
                    panic!("Expected middle array at outer[0]");
                }
                
                // Second element is a string
                if let Attribute::String(s) = &outer[1] {
                    assert_eq!(s, "deep");
                } else {
                    panic!("Expected string at outer[1]");
                }
            },
            _ => panic!("Expected outer array"),
        }
    }

    #[test]
    fn test_large_number_of_attributes_in_operation() {
        // Test operation with a large number of attributes to test map handling
        let mut op = Operation::new("high_attr_op");
        let mut attrs = HashMap::new();
        
        // Add 1000 attributes to test map capacity
        for i in 0..1000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::Int(i as i64)
            );
        }
        
        op.attributes = attrs;
        
        assert_eq!(op.op_type, "high_attr_op");
        assert_eq!(op.attributes.len(), 1000);
        
        // Verify a few specific attributes exist
        assert_eq!(op.attributes.get("attr_0"), Some(&Attribute::Int(0)));
        assert_eq!(op.attributes.get("attr_100"), Some(&Attribute::Int(100)));
        assert_eq!(op.attributes.get("attr_999"), Some(&Attribute::Int(999)));
    }

    #[test]
    fn test_value_with_very_large_dimension_values() {
        // Test tensor with dimension values that are very large but still valid
        let huge_dims = Value {
            name: "huge_dims".to_string(),
            ty: Type::Bool,
            shape: vec![2_000_000_000, 2_000_000_000],  // Each dimension is 2 billion
        };
        
        assert_eq!(huge_dims.shape[0], 2_000_000_000);
        assert_eq!(huge_dims.shape[1], 2_000_000_000);
        assert_eq!(huge_dims.ty, Type::Bool);
        
        // Note: This product would overflow for actual computation but is valid as a shape
        let product_check = huge_dims.shape[0] as u128 * huge_dims.shape[1] as u128;
        assert_eq!(product_check, 4_000_000_000_000_000_000u128);
    }

    #[test]
    fn test_module_name_boundaries() {
        // Test modules with various name lengths (empty, short, long)
        let empty_name_module = Module::new("");
        assert_eq!(empty_name_module.name, "");

        let single_char_module = Module::new("X");
        assert_eq!(single_char_module.name, "X");

        let long_name_module = Module::new(&"A".repeat(10000));  // 10k character name
        assert_eq!(long_name_module.name.len(), 10000);
        
        // Verify all have empty operations initially
        assert!(empty_name_module.operations.is_empty());
        assert!(single_char_module.operations.is_empty());
        assert!(long_name_module.operations.is_empty());
    }
}