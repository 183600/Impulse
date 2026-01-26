//! Additional focused edge case tests for the Impulse compiler
//! Covers boundary conditions, error handling, and extreme values

#[cfg(test)]
mod additional_edge_case_tests {
    use crate::ir::{Module, Operation, Value, Type, Attribute};
    use crate::ImpulseCompiler;
    use std::collections::HashMap;

    /// Test 1: Operations with maximum length names and Unicode characters
    #[test]
    fn test_operations_with_extreme_names() {
        // Test with very long operation and value names
        let long_name = "a".repeat(1_000_000); // 1MB string
        let op = Operation::new(&long_name);
        assert_eq!(op.op_type, long_name);

        // Test value with long name
        let value = Value {
            name: "b".repeat(1_000_000),
            ty: Type::F32,
            shape: vec![1],
        };
        assert_eq!(value.name.len(), 1_000_000);
    }

    /// Test 2: Empty collections and zero-sized containers
    #[test]
    fn test_empty_collections_edge_cases() {
        // Test with empty string as module name
        let module = Module::new("");
        assert_eq!(module.name, "");

        // Test operation with empty string type
        let op = Operation::new("");
        assert_eq!(op.op_type, "");
        
        // Test value with empty name
        let value = Value {
            name: "".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(value.name, "");
        assert_eq!(value.shape.len(), 0);
    }

    /// Test 3: Extreme numerical values in attributes and shapes
    #[test]
    fn test_extreme_numerical_values() {
        // Test with maximum integer values
        let int_attr = Attribute::Int(i64::MAX);
        match int_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MAX),
            _ => panic!("Expected Int attribute"),
        }

        // Test with minimum integer values
        let min_int_attr = Attribute::Int(i64::MIN);
        match min_int_attr {
            Attribute::Int(val) => assert_eq!(val, i64::MIN),
            _ => panic!("Expected Int attribute"),
        }

        // Test with extreme float values
        let max_float_attr = Attribute::Float(f64::MAX);
        match max_float_attr {
            Attribute::Float(val) => assert_eq!(val, f64::MAX),
            _ => panic!("Expected Float attribute"),
        }

        let min_float_attr = Attribute::Float(f64::MIN);
        match min_float_attr {
            Attribute::Float(val) => assert_eq!(val, f64::MIN),
            _ => panic!("Expected Float attribute"),
        }

        // Test with NaN and infinity
        let nan_attr = Attribute::Float(f64::NAN);
        match nan_attr {
            Attribute::Float(val) => assert!(val.is_nan()),
            _ => panic!("Expected NaN Float attribute"),
        }

        let inf_attr = Attribute::Float(f64::INFINITY);
        match inf_attr {
            Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
            _ => panic!("Expected positive infinity Float attribute"),
        }
    }

    /// Test 4: Extremely deep nested tensor types
    #[test]
    fn test_extremely_deep_nested_tensors() {
        // Create a very deeply nested tensor type
        let mut current_type = Type::F32;
        for _ in 0..1000 {  // Very deep recursion
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![1],  // Keep shape simple to avoid overflow
            };
        }

        // Verify the structure
        match &current_type {
            Type::Tensor { element_type: _, shape } => {
                assert_eq!(shape, &vec![1]);
            },
            _ => panic!("Expected tensor type"),
        }

        // Ensure it can be cloned without stack overflow
        let cloned = current_type.clone();
        assert_eq!(current_type, cloned);
    }

    /// Test 5: Shape calculations that could cause overflow
    #[test]
    fn test_potential_overflow_in_shape_calculations() {
        // Test with a shape that would cause overflow when calculating total size
        // On a 64-bit system, the maximum product would exceed usize::MAX
        // Using smaller values that still stress the calculation
        
        // Use dimensions that when multiplied would exceed typical limits
        // But using checked arithmetic to prevent actual overflow
        let large_shape = vec![100_000, 100_000]; // 10^10 elements
        let value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: large_shape,
        };

        // Calculate with checked multiplication to avoid overflow
        let mut product: Option<usize> = Some(1);
        for dim in &value.shape {
            product = product.and_then(|p| p.checked_mul(*dim));
        }
        
        assert!(product.is_some()); // Should not overflow in this example

        // Test with a shape that definitely contains zero (causing 0 product)
        let zero_shape = vec![usize::MAX, 0, usize::MAX];
        let zero_value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::I32,
            shape: zero_shape,
        };
        
        let zero_product: usize = zero_value.shape.iter().product();
        assert_eq!(zero_product, 0);
    }

    /// Test 6: Operations with maximum possible inputs, outputs, and attributes
    #[test]
    fn test_max_complexity_operation() {
        let mut op = Operation::new("max_complexity_op");
        
        // Add maximum possible inputs (testing memory limits)
        for i in 0..100_000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 1000 + 1], // Cycling through different shapes to avoid patterns
            });
        }
        
        // Add maximum possible outputs
        for i in 0..50_000 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 1000 + 1],
            });
        }
        
        // Add maximum possible attributes
        let mut attrs = HashMap::new();
        for i in 0..10_000 {
            attrs.insert(
                format!("attr_{}", i),
                Attribute::String(format!("value_{}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.inputs.len(), 100_000);
        assert_eq!(op.outputs.len(), 50_000);
        assert_eq!(op.attributes.len(), 10_000);
        assert_eq!(op.op_type, "max_complexity_op");
    }

    /// Test 7: String operations with special characters and control codes
    #[test]
    fn test_string_attributes_with_control_codes() {
        // Test string attributes with various special characters
        let special_strings = [
            "\0",               // Null byte
            "\n\t\r",          // Whitespace controls
            "\x01\x02\x1F",    // Control characters
            "🚀🔥中文",         // Emoji and Unicode
            "a".repeat(100_000), // Very long string
        ];

        for (i, special_str) in special_strings.iter().enumerate() {
            let attr = Attribute::String(special_str.clone());
            
            match attr {
                Attribute::String(s) => {
                    assert_eq!(&s, special_str);
                    
                    // Test creating a value with special string name
                    let value = Value {
                        name: format!("val_{}_", i) + special_str,
                        ty: Type::F32,
                        shape: vec![1],
                    };
                    
                    assert!(value.name.contains(special_str));
                },
                _ => panic!("Expected String attribute for special string test {}", i),
            }
        }
    }

    /// Test 8: Boolean and array attribute edge cases
    #[test]
    fn test_boolean_and_array_attribute_edge_cases() {
        // Test boolean attributes
        let true_attr = Attribute::Bool(true);
        let false_attr = Attribute::Bool(false);
        
        match (true_attr, false_attr) {
            (Attribute::Bool(t), Attribute::Bool(f)) => {
                assert_eq!(t, true);
                assert_eq!(f, false);
            },
            _ => panic!("Expected Bool attributes"),
        }

        // Test empty array attribute
        let empty_array = Attribute::Array(vec![]);
        match empty_array {
            Attribute::Array(arr) => assert_eq!(arr.len(), 0),
            _ => panic!("Expected empty Array attribute"),
        }

        // Test deeply nested array attribute
        let mut nested_array = Attribute::Int(42);
        for i in 0..100 {  // Create 100 levels of nesting
            nested_array = Attribute::Array(vec![nested_array]);
        }
        
        // Verify we can still handle this deeply nested structure
        match &nested_array {
            Attribute::Array(_) => (),
            _ => panic!("Expected nested array structure"),
        }
        
        // Test cloning deeply nested array
        let cloned_nested = nested_array.clone();
        assert_eq!(nested_array, cloned_nested);
    }

    /// Test 9: Module creation with unusual patterns
    #[test]
    fn test_unusual_module_patterns() {
        // Create a module with empty name
        let empty_module = Module::new("");
        assert_eq!(empty_module.name, "");
        assert_eq!(empty_module.operations.len(), 0);
        assert_eq!(empty_module.inputs.len(), 0);
        assert_eq!(empty_module.outputs.len(), 0);

        // Create a module with Unicode name
        let unicode_module = Module::new("模块_测试_🚀");
        assert_eq!(unicode_module.name, "模块_测试_🚀");

        // Create a module with very long name
        let long_name = "module_".repeat(100_000); // Very long module name
        let long_module = Module::new(long_name.clone());
        assert_eq!(long_module.name, long_name);
    }

    /// Test 10: Comprehensive error handling and resource management
    #[test]
    fn test_resource_management_under_stress() {
        // Create many objects to test memory management
        let modules: Vec<_> = (0..1000)
            .map(|i| {
                let mut module = Module::new(&format!("stress_test_{}", i));
                
                for j in 0..100 {
                    let mut op = Operation::new(&format!("op_{}_{}", i, j));
                    op.inputs.push(Value {
                        name: format!("input_{}_{}", i, j),
                        ty: Type::F32,
                        shape: vec![j % 10 + 1],
                    });
                    module.add_operation(op);
                }
                module
            })
            .collect();

        // Verify we created the expected number of modules
        assert_eq!(modules.len(), 1000);
        
        // Check some modules still have correct data
        assert_eq!(modules[0].name, "stress_test_0");
        assert_eq!(modules[999].name, "stress_test_999");
        assert_eq!(modules[500].operations.len(), 100);
        
        // Clean up to test deallocation
        drop(modules);
        
        // Create a compiler and test it works after the stress test
        let compiler = ImpulseCompiler::new();
        assert_eq!(compiler.passes.passes.len(), 0);
    }
}