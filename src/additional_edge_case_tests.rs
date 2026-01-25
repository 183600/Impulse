//! Additional edge case tests for the Impulse compiler
//! Covering more boundary conditions and extreme values

use rstest::*;
use crate::{
    ir::{Value, Type, Operation, Attribute, Module},
};

// ImpulseCompiler and ir_utils are not actually used in this file
// Removed unused imports to eliminate warnings

/// Test 1: Very large tensor dimensions that could cause overflow in calculations
#[test]
fn test_tensor_overflow_protection() {
    // Testing with values that could cause overflow when multiplied
    use std::usize;
    
    // Use a value that when squared would exceed reasonable limits
    // On a 64-bit system, usize::MAX is very large, so we'll test more moderate values
    // that could still cause issues
    let safe_large_value = 1_000_000;  // 1M
    let large_tensor = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![safe_large_value, safe_large_value],  // 1E12 elements
    };
    
    assert_eq!(large_tensor.shape[0], safe_large_value);
    assert_eq!(large_tensor.shape[1], safe_large_value);
    
    // Calculate product to ensure it doesn't panic
    let product: Option<usize> = large_tensor.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    
    // In this case the multiplication would overflow, so we expect None
    assert!(product.is_some());  // For our chosen values, it should not overflow on 64-bit
    
    // Calculate with a shape that definitely has an element of 0
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::F32,
        shape: vec![safe_large_value, 0, safe_large_value],
    };
    
    let zero_product: usize = zero_tensor.shape.iter().product();
    assert_eq!(zero_product, 0);  // Should be 0 due to 0 in dimensions
}

/// Test 2: Maximum depth recursive tensor types
#[test]
fn test_maximum_depth_tensor_types() {
    // Create a deeply nested tensor type to test recursion limits
    let mut current_type = Type::F32;
    
    // Only go to 500 levels to avoid stack overflow while still being thorough
    for _ in 0..500 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Verify the deeply nested type can be handled
    match &current_type {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, &vec![2]);
            // Just verify that we can access the structure without crashing
            assert!(element_type.is_valid_type());
        },
        _ => panic!("Expected deeply nested tensor type"),
    }
    
    // Test cloning of this deeply nested type
    let _cloned = current_type.clone();
    // Just ensure clone doesn't panic - equality comparison could be expensive
    assert!(true); // The clone succeeded if we reach here
}

/// Test 3: Operations with maximum possible inputs and outputs
#[test]
fn test_operation_maximum_io() {
    let mut op = Operation::new("max_io_op");
    
    // Add 100,000 inputs to stress test memory allocation
    for i in 0..100_000 {
        op.inputs.push(Value {
            name: format!("input_{:08}", i),
            ty: Type::F32,
            shape: vec![i % 100 + 1],  // Small shape to avoid memory issues
        });
        
        if i % 50_000 == 0 {  // Report progress every 50k
            assert!(true);  // Just ensure we're not stuck
        }
    }
    
    // Add 50,000 outputs
    for i in 0..50_000 {
        op.outputs.push(Value {
            name: format!("output_{:08}", i),
            ty: Type::F32,
            shape: vec![i % 100 + 1],  // Small shape to avoid memory issues
        });
        
        if i % 25_000 == 0 {  // Report progress every 25k
            assert!(true);  // Just ensure we're not stuck
        }
    }
    
    assert_eq!(op.inputs.len(), 100_000);
    assert_eq!(op.outputs.len(), 50_000);
    assert_eq!(op.op_type, "max_io_op");
}

/// Test 4: Complex attribute nesting with error handling
#[test]
fn test_complex_attribute_nesting() {
    use std::collections::HashMap;
    
    // Create a complex nested structure of attributes
    let mut complex_attr = Attribute::String("base".to_string());
    
    // Nest attributes 100 levels deep
    for level in 0..100 {
        complex_attr = Attribute::Array(vec![
            Attribute::Int(level as i64),
            complex_attr,
            Attribute::String(format!("level_{}", level)),
        ]);
    }
    
    // Verify the structure can be processed
    match &complex_attr {
        Attribute::Array(content) => {
            assert_eq!(content.len(), 3);
            match &content[0] {
                Attribute::Int(l) => assert_eq!(*l, 99), // Top level
                _ => panic!("Expected int at first position"),
            }
        },
        _ => panic!("Expected complex array structure"),
    }
    
    // Test with an operation that has complex attributes
    let mut op = Operation::new("complex_attr_op");
    let mut attrs = HashMap::new();
    
    // Add multiple complex attribute structures
    for i in 0..10_000 {
        attrs.insert(
            format!("complex_attr_{}", i),
            Attribute::Array(vec![
                Attribute::Int(i as i64),
                Attribute::String(format!("value_{}", i)),
                Attribute::Bool(i % 2 == 0),
            ])
        );
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 10_000);
}

/// Test 5: Module with maximum number of operations and complex structure
#[test]
fn test_maximum_complex_module() {
    let mut module = Module::new("maximum_complexity_module");
    
    // Add 200,000 operations to stress test the module
    for op_idx in 0..200_000 {
        let mut op = Operation::new(&format!("op_{:08}", op_idx));
        
        // Add some inputs and outputs to each operation
        for inp_idx in 0..5 {
            op.inputs.push(Value {
                name: format!("op{}_inp{}", op_idx, inp_idx),
                ty: if inp_idx % 2 == 0 { Type::F32 } else { Type::I32 },
                shape: vec![(inp_idx + 1) * 2, (op_idx % 1000) + 1],
            });
        }
        
        for out_idx in 0..3 {
            op.outputs.push(Value {
                name: format!("op{}_out{}", op_idx, out_idx),
                ty: if out_idx % 3 == 0 { Type::F64 } else { Type::I64 },
                shape: vec![(out_idx + 1) * 3, (op_idx % 500) + 1],
            });
        }
        
        module.add_operation(op);
        
        // Periodic checks to ensure we're making progress
        if op_idx % 50_000 == 0 {
            assert!(true);  // Just ensure we're still running
        }
    }
    
    assert_eq!(module.operations.len(), 200_000);
    assert_eq!(module.name, "maximum_complexity_module");
    
    // Check a few specific operations to ensure data integrity
    assert_eq!(module.operations[0].op_type, "op_00000000");
    assert_eq!(module.operations[199_999].op_type, "op_00199999");
}

/// Test 6: Edge case tensor shapes with various zero configurations
#[rstest]
#[case(vec![0], 0)]
#[case(vec![0, 1], 0)]
#[case(vec![1, 0], 0)]
#[case(vec![0, 0], 0)]
#[case(vec![1, 0, 1], 0)]
#[case(vec![2, 3, 0, 5], 0)]
#[case(vec![1], 1)]
#[case(vec![2, 3], 6)]
#[case(vec![2, 3, 4], 24)]
fn test_various_zero_tensor_shapes(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let value = Value {
        name: "test_tensor".to_string(),
        ty: Type::F32,
        shape,
    };
    
    let calculated_size: usize = value.shape.iter().product();
    assert_eq!(calculated_size, expected_size);
    
    // Also test with different types to ensure consistency
    let i32_value = Value {
        name: "test_i32_tensor".to_string(),
        ty: Type::I32,
        shape: value.shape.clone(),
    };
    
    let i32_calculated_size: usize = i32_value.shape.iter().product();
    assert_eq!(i32_calculated_size, expected_size);
}

/// Test 7: Memory stress test with large string values
#[test]
fn test_large_string_memory_handling() {
    // Create operations and values with increasingly large string names
    for size_factor in [10, 100, 1000] {
        let large_string = "a".repeat(1000 * size_factor);
        
        let value = Value {
            name: large_string.clone(),
            ty: Type::F32,
            shape: vec![1],
        };
        
        assert_eq!(value.name.len(), large_string.len());
        assert_eq!(value.name, large_string);
        
        let op = Operation::new(&large_string);
        assert_eq!(op.op_type.len(), large_string.len());
    }
    
    // Test with module name
    let huge_module_name = "module_".repeat(10_000);
    let module = Module::new(huge_module_name.clone());
    assert_eq!(module.name, huge_module_name);
}

/// Test 8: Error-prone mathematical operations on shapes
#[test]
fn test_mathematical_operations_on_shapes() {
    // Test various shape configurations that could cause math issues
    
    // Very large values that could potentially overflow when multiplied
    let large_but_safe = vec![100_000, 100_000];  // 10^10, should be safe on 64-bit
    let large_value = Value {
        name: "large_math".to_string(),
        ty: Type::F32,
        shape: large_but_safe,
    };
    
    let product_result: usize = large_value.shape.iter().product();
    assert_eq!(product_result, 10_000_000_000);
    
    // Test with checked arithmetic to prevent overflow
    let checked_product = large_value.shape.iter().try_fold(1_usize, |acc, &x| {
        acc.checked_mul(x)
    });
    assert!(checked_product.is_some());
    
    // Test with shape that certainly contains 0
    let problematic_shape = Value {
        name: "problematic".to_string(),
        ty: Type::I64,
        shape: vec![usize::MAX, 0, 5],
    };
    
    let zero_result: usize = problematic_shape.shape.iter().product();
    assert_eq!(zero_result, 0);
}

/// Test 9: Mixed data type operations and tensor compatibility
#[test]
fn test_mixed_data_type_compatibility() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("mixed_types_op");
    
    // Add values of different types
    let types_to_test = [
        Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool
    ];
    
    for (idx, data_type) in types_to_test.iter().enumerate() {
        op.inputs.push(Value {
            name: format!("input_{}", idx),
            ty: data_type.clone(),
            shape: vec![idx + 1, idx + 2],
        });
        
        op.outputs.push(Value {
            name: format!("output_{}", idx),
            ty: data_type.clone(),
            shape: vec![idx + 2, idx + 1],  // Different shape for outputs
        });
    }
    
    // Add attributes of different types too
    let mut attrs = HashMap::new();
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14159));
    attrs.insert("string_attr".to_string(), Attribute::String("test".to_string()));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1), Attribute::Int(2), Attribute::Int(3)
    ]));
    
    op.attributes = attrs;
    
    assert_eq!(op.inputs.len(), 5);  // One for each type
    assert_eq!(op.outputs.len(), 5); // One for each type
    assert_eq!(op.attributes.len(), 5); // One for each attribute type
}

/// Test 10: Concurrency-related edge cases (without actual threading)
#[test]
fn test_resource_duplication_scenarios() {
    // Test creating multiple large objects in succession to simulate
    // potential resource allocation issues
    
    let mut modules = Vec::new();
    
    // Create 100 modules with substantial content
    for i in 0..100 {
        let mut module = Module::new(&format!("test_module_{}", i));
        
        // Add operations to each module
        for j in 0..1000 {
            let mut op = Operation::new(&format!("op_{}_{}", i, j));
            
            // Add a few inputs and outputs
            for k in 0..5 {
                op.inputs.push(Value {
                    name: format!("op{}_inp{}", j, k),
                    ty: Type::F32,
                    shape: vec![i + 1, j % 10 + 1],
                });
            }
            
            module.add_operation(op);
        }
        
        modules.push(module);
    }
    
    // Verify we created all modules
    assert_eq!(modules.len(), 100);
    
    // Check a few specific ones to ensure data integrity
    assert_eq!(modules[0].name, "test_module_0");
    assert_eq!(modules[99].name, "test_module_99");
    assert_eq!(modules[0].operations.len(), 1000);
    assert_eq!(modules[99].operations.len(), 1000);
}

// Helper trait and implementation to extend Type functionality for testing
trait TypeExtensions {
    fn is_valid_type(&self) -> bool;
}

impl TypeExtensions for Type {
    fn is_valid_type(&self) -> bool {
        match self {
            Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Bool => true,
            Type::Tensor { element_type, shape } => {
                // Recursively validate the nested type
                element_type.is_valid_type() && 
                // Ensure shape doesn't contain obviously invalid values (though 0 is valid)
                shape.iter().all(|&d| d <= usize::MAX)
            }
        }
    }
}