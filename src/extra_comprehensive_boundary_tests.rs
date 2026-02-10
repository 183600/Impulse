//! Extra comprehensive boundary tests covering additional edge cases
//! Focus on numerical precision, overflow scenarios, and special value handling

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ImpulseCompiler;
use std::collections::HashMap;

/// Test 1: Value with extremely large dimension near overflow boundary
#[test]
fn test_value_near_overflow_boundary() {
    // Test with dimensions that approach usize::MAX on 64-bit systems
    // Using dimensions that multiply to a large but safe value
    let value = Value {
        name: "boundary_tensor".to_string(),
        ty: Type::F32,
        shape: vec![100_000, 100_000], // 10 billion elements
    };
    assert_eq!(value.shape, vec![100_000, 100_000]);
    
    // Verify num_elements handles large values correctly
    let num = value.num_elements();
    assert!(num.is_some());
    assert_eq!(num.unwrap(), 10_000_000_000);
}

/// Test 2: Compiler with models containing all possible byte values
#[test]
fn test_compiler_with_all_byte_values() {
    let mut compiler = ImpulseCompiler::new();
    
    // Create a model containing all possible byte values 0-255
    let all_bytes_model: Vec<u8> = (0..=255).collect();
    
    let result = compiler.compile(&all_bytes_model, "cpu");
    // Should handle without panicking
    assert!(result.is_ok() || result.is_err());
}

/// Test 3: Operation with very negative integer attributes
#[test]
fn test_very_negative_integer_attributes() {
    let mut op = Operation::new("negative_attrs");
    let mut attrs = HashMap::new();
    
    // Test with very negative integers
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("very_negative".to_string(), Attribute::Int(-999_999_999_999));
    attrs.insert("negative_one".to_string(), Attribute::Int(-1));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 3);
    match op.attributes.get("min_i64") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected MIN_I64"),
    }
}

/// Test 4: Module with operations having no inputs or outputs
#[test]
fn test_module_with_empty_operations() {
    let mut module = Module::new("empty_ops_module");
    
    // Add operations with no inputs and no outputs
    for i in 0..5 {
        let op = Operation::new(&format!("empty_op_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
    }
}

/// Test 5: Value with shape containing many ones (identity dimensions)
#[test]
fn test_value_with_many_ones_in_shape() {
    // Test shape: [1, 1, 1, 1, 100, 1, 1, 1]
    let value = Value {
        name: "ones_tensor".to_string(),
        ty: Type::F64,
        shape: vec![1, 1, 1, 1, 100, 1, 1, 1],
    };
    
    assert_eq!(value.shape.len(), 8);
    assert_eq!(value.num_elements(), Some(100));
}

/// Test 6: Attribute with string containing unicode characters
#[test]
fn test_unicode_string_attributes() {
    let unicode_strings = vec![
        "Hello 世界",
        "Привет мир",
        "مرحبا",
        "🚀🌟✨",
        "日本語",
        "한국어",
        "العربية",
        "😀😁😂",
    ];
    
    for s in unicode_strings {
        let attr = Attribute::String(s.to_string());
        match attr {
            Attribute::String(val) => assert_eq!(val, s),
            _ => panic!("Expected String attribute"),
        }
    }
}

/// Test 7: Value with all data types including nested tensors
#[test]
fn test_value_with_all_type_combinations() {
    let types = vec![
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
            element_type: Box::new(Type::I64),
            shape: vec![10],
        },
    ];
    
    for ty in types {
        let value = Value {
            name: "test_value".to_string(),
            ty: ty.clone(),
            shape: vec![2, 3],
        };
        assert_eq!(value.ty, ty);
    }
}

/// Test 8: Module with alternating input/output patterns
#[test]
fn test_module_alternating_io_pattern() {
    let mut module = Module::new("alternating_module");
    
    // Add inputs and outputs in alternating pattern
    for i in 0..3 {
        module.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
    }
    
    assert_eq!(module.inputs.len(), 3);
    assert_eq!(module.outputs.len(), 3);
}

/// Test 9: Compiler with empty target string
#[test]
fn test_compiler_with_empty_target() {
    let mut compiler = ImpulseCompiler::new();
    let model = vec![1u8, 2u8, 3u8];
    
    // Test with empty target string
    let result = compiler.compile(&model, "");
    // Should handle gracefully without panic
    assert!(result.is_ok() || result.is_err());
}

/// Test 10: Operation with attribute key containing special characters
#[test]
fn test_operation_with_special_attribute_keys() {
    let mut op = Operation::new("special_keys");
    let mut attrs = HashMap::new();
    
    let special_keys = vec![
        "key-with-dashes",
        "key_with_underscores",
        "key.with.dots",
        "key:with:colons",
        "key/with/slashes",
        "key with spaces",
    ];
    
    for key in special_keys {
        attrs.insert(key.to_string(), Attribute::Int(42));
    }
    
    op.attributes = attrs;
    assert_eq!(op.attributes.len(), 6);
}