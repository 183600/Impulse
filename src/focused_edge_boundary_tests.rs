//! Focused edge boundary tests - covering critical edge cases with assert! and assert_eq!

#[cfg(test)]
mod tests {
    use crate::ir::{Module, Value, Type, Operation, Attribute};
    use std::collections::HashMap;

    /// Test 1: Value with extremely large shape dimensions that could cause overflow
    #[test]
    fn test_value_with_overflow_prone_dimensions() {
        let value = Value {
            name: "overflow_risk".to_string(),
            ty: Type::F32,
            shape: vec![usize::MAX, 2],
        };
        
        // The num_elements method should handle potential overflow gracefully
        let num_elements = value.num_elements();
        assert!(num_elements.is_none(), "Should return None for overflow-prone dimensions");
    }

    /// Test 2: Attribute with special floating-point values (NaN, Infinity, -Infinity)
    #[test]
    fn test_special_float_values_in_attributes() {
        let nan_attr = Attribute::Float(f64::NAN);
        let pos_inf = Attribute::Float(f64::INFINITY);
        let neg_inf = Attribute::Float(f64::NEG_INFINITY);
        
        // Verify attributes are created
        match nan_attr {
            Attribute::Float(val) => assert!(val.is_nan(), "NaN should be NaN"),
            _ => panic!("Expected Float attribute for NaN"),
        }
        
        match pos_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive(), "Should be positive infinity"),
            _ => panic!("Expected Float attribute for positive infinity"),
        }
        
        match neg_inf {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative(), "Should be negative infinity"),
            _ => panic!("Expected Float attribute for negative infinity"),
        }
    }

    /// Test 3: Module with empty operation names
    #[test]
    fn test_module_with_empty_operation_names() {
        let mut module = Module::new("empty_op_names");
        
        let mut op = Operation::new("");
        op.inputs.push(Value {
            name: "input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].op_type, "");
    }

    /// Test 4: Value with empty name
    #[test]
    fn test_value_with_empty_name() {
        let value = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![5],
        };
        
        assert_eq!(value.name, "");
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape.len(), 1);
    }

    /// Test 5: Operation with extremely large integer attributes
    #[test]
    fn test_operation_with_extreme_int_attributes() {
        let mut op = Operation::new("extreme_ints");
        let mut attrs = HashMap::new();
        
        attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
        attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
        attrs.insert("zero".to_string(), Attribute::Int(0));
        attrs.insert("negative_one".to_string(), Attribute::Int(-1));
        attrs.insert("positive_one".to_string(), Attribute::Int(1));
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 5);
        
        // Verify extreme values
        match op.attributes.get("max_i64") {
            Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
            _ => panic!("Expected max_i64 attribute"),
        }
        
        match op.attributes.get("min_i64") {
            Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
            _ => panic!("Expected min_i64 attribute"),
        }
    }

    /// Test 6: Module with very long names (stress test for string handling)
    #[test]
    fn test_module_with_very_long_names() {
        let long_name = "a".repeat(10000);
        let module = Module::new(&long_name);
        
        assert_eq!(module.name.len(), 10000);
        assert!(module.name.chars().all(|c| c == 'a'));
    }

    /// Test 7: Value with single dimension equal to 1 (broadcast-like shapes)
    #[test]
    fn test_value_with_unit_dimensions() {
        let test_shapes = [
            vec![1],           // Single element
            vec![1, 1],        // 1x1 tensor
            vec![1, 100],      // 1x100 tensor
            vec![100, 1],      // 100x1 tensor
            vec![1, 1, 1],     // 1x1x1 tensor
            vec![1, 5, 1],     // 1x5x1 tensor
        ];
        
        for shape in test_shapes.iter() {
            let value = Value {
                name: "unit_dim".to_string(),
                ty: Type::F32,
                shape: shape.to_vec(),
            };
            
            // Calculate total elements
            let product: usize = value.shape.iter().product();
            let expected = shape.iter().product::<usize>();
            assert_eq!(product, expected);
        }
    }

    /// Test 8: Nested tensor with empty inner shape
    #[test]
    fn test_nested_tensor_with_empty_inner_shape() {
        let inner_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![],
        };
        
        let outer_type = Type::Tensor {
            element_type: Box::new(inner_type),
            shape: vec![3],
        };
        
        match outer_type {
            Type::Tensor { element_type, shape } => {
                assert_eq!(shape, vec![3]);
                match element_type.as_ref() {
                    Type::Tensor { shape: inner_shape, .. } => {
                        assert!(inner_shape.is_empty());
                    }
                    _ => panic!("Expected inner Tensor type"),
                }
            }
            _ => panic!("Expected outer Tensor type"),
        }
    }

    /// Test 9: Module with operations that have no inputs or outputs
    #[test]
    fn test_module_with_no_io_operations() {
        let mut module = Module::new("no_io_ops");
        
        // Add operations with no inputs or outputs
        for i in 0..5 {
            let op = Operation::new(&format!("noop_{}", i));
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 5);
        
        // Verify all operations have no inputs or outputs
        for op in &module.operations {
            assert_eq!(op.inputs.len(), 0);
            assert_eq!(op.outputs.len(), 0);
        }
    }

    /// Test 10: Attribute array with mixed types
    #[test]
    fn test_mixed_type_attribute_array() {
        let mixed_array = Attribute::Array(vec![
            Attribute::Int(42),
            Attribute::Float(3.14),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
            Attribute::Int(-100),
            Attribute::Float(-2.71),
            Attribute::String("".to_string()),
            Attribute::Bool(false),
        ]);
        
        match mixed_array {
            Attribute::Array(elements) => {
                assert_eq!(elements.len(), 8);
                
                // Verify each element type
                match &elements[0] {
                    Attribute::Int(42) => {},
                    _ => panic!("Expected Int(42)"),
                }
                
                match &elements[1] {
                    Attribute::Float(val) => assert!((val - 3.14).abs() < f64::EPSILON),
                    _ => panic!("Expected Float(3.14)"),
                }
                
                match &elements[2] {
                    Attribute::String(s) => assert_eq!(s, "test"),
                    _ => panic!("Expected String(\"test\")"),
                }
                
                match &elements[3] {
                    Attribute::Bool(true) => {},
                    _ => panic!("Expected Bool(true)"),
                }
                
                match &elements[4] {
                    Attribute::Int(-100) => {},
                    _ => panic!("Expected Int(-100)"),
                }
                
                match &elements[5] {
                    Attribute::Float(val) => assert!((val - (-2.71)).abs() < f64::EPSILON),
                    _ => panic!("Expected Float(-2.71)"),
                }
                
                match &elements[6] {
                    Attribute::String(s) => assert_eq!(s, ""),
                    _ => panic!("Expected empty String"),
                }
                
                match &elements[7] {
                    Attribute::Bool(false) => {},
                    _ => panic!("Expected Bool(false)"),
                }
            }
            _ => panic!("Expected Array attribute"),
        }
    }
}
