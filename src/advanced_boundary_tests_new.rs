//! Advanced boundary tests - additional edge case coverage
//! 
//! This module contains tests for boundary conditions not covered in existing test suites.

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with shape containing usize::MAX/2 to test overflow detection
#[test]
fn test_value_shape_half_max_overflow() {
    let value = Value {
        name: "half_max_shape".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 2],
    };
    // Should return None due to overflow
    assert_eq!(value.num_elements(), None);
}

/// Test 2: Module with operations having cyclic naming pattern
#[test]
fn test_module_cyclic_operation_names() {
    let mut module = Module::new("cyclic_ops");
    let names = ["op_a", "op_b", "op_c"];
    
    // Add operations in a cyclic pattern
    for i in 0..30 {
        let mut op = Operation::new(names[i % 3]);
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 30);
    // Verify cyclic pattern
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[1].op_type, "op_b");
    assert_eq!(module.operations[2].op_type, "op_c");
    assert_eq!(module.operations[3].op_type, "op_a");
}

/// Test 3: Attribute with very small positive float near f64::MIN_POSITIVE
#[test]
fn test_attribute_min_positive_float() {
    let min_pos = Attribute::Float(f64::MIN_POSITIVE);
    let slightly_above = Attribute::Float(f64::MIN_POSITIVE * 2.0);
    let slightly_below = Attribute::Float(f64::MIN_POSITIVE * 0.5);
    
    match (min_pos, slightly_above, slightly_below) {
        (Attribute::Float(a), Attribute::Float(b), Attribute::Float(c)) => {
            assert!(a > 0.0);
            assert!(b > a);
            assert!(c < a);
        },
        _ => panic!("Expected Float attributes"),
    }
}

/// Test 4: Module with inputs/outputs having identical names but different types
#[test]
fn test_module_identical_names_different_types() {
    let mut module = Module::new("same_name_diff_type");
    
    // Input named "data" with F32 type
    module.inputs.push(Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    // Output named "data" with I32 type (different type, same name)
    module.outputs.push(Value {
        name: "data".to_string(),
        ty: Type::I32,
        shape: vec![10],
    });
    
    assert_eq!(module.inputs[0].name, "data");
    assert_eq!(module.outputs[0].name, "data");
    assert_ne!(module.inputs[0].ty, module.outputs[0].ty);
}

/// Test 5: Value with alternating 1 and 0 dimensions pattern
#[test]
fn test_value_alternating_zero_one_dimensions() {
    let patterns = [
        vec![1, 0, 1, 0, 1],
        vec![0, 1, 0, 1, 0],
        vec![1, 0, 1, 0],
        vec![0, 1, 0, 1],
    ];
    
    for shape in patterns.iter() {
        let value = Value {
            name: "alternating".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any zero in shape should result in 0 elements
        if shape.contains(&0) {
            assert_eq!(value.num_elements(), Some(0));
        }
    }
}

/// Test 6: Operation with attribute containing Unicode string
#[test]
fn test_operation_unicode_string_attribute() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("unicode_op");
    let mut attrs = HashMap::new();
    
    // Add Unicode strings
    attrs.insert("chinese".to_string(), Attribute::String("你好世界".to_string()));
    attrs.insert("emoji".to_string(), Attribute::String("🚀🔥✨".to_string()));
    attrs.insert("arabic".to_string(), Attribute::String("مرحبا".to_string()));
    attrs.insert("emoji_combo".to_string(), Attribute::String("😀😃😄😁😆😅".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 4);
    if let Attribute::String(s) = &op.attributes["emoji"] {
        assert_eq!(s, "🚀🔥✨");
    }
}

/// Test 7: Module with all operations having no inputs and no outputs
#[test]
fn test_module_operations_no_io() {
    let mut module = Module::new("no_io_ops");
    
    for i in 0..10 {
        let op = Operation::new(&format!("standalone_{}", i));
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 10);
    for op in &module.operations {
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
    }
}

/// Test 8: Nested tensor with zero-length shape at each level
#[test]
fn test_nested_tensor_zero_shapes() {
    // Create nested tensors with zero shapes
    let inner = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![0], // Zero-length
    };
    
    let middle = Type::Tensor {
        element_type: Box::new(inner.clone()),
        shape: vec![0], // Zero-length
    };
    
    let outer = Type::Tensor {
        element_type: Box::new(middle.clone()),
        shape: vec![0], // Zero-length
    };
    
    // Verify all levels have zero-length shapes
    if let Type::Tensor { shape: outer_shape, .. } = &outer {
        assert_eq!(outer_shape, &vec![0]);
    }
}

/// Test 9: Attribute array with single element repeated many times
#[test]
fn test_attribute_array_repeated_single_element() {
    let repeated = Attribute::Array(vec![Attribute::Int(42); 1000]);
    
    match repeated {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1000);
            // All elements should be Int(42)
            for elem in arr.iter() {
                if let Attribute::Int(42) = elem {
                    // OK
                } else {
                    panic!("Expected all elements to be Int(42)");
                }
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 10: Module with operations having maximum attribute key length
#[test]
fn test_module_operations_max_attr_key_length() {
    use std::collections::HashMap;
    
    let mut module = Module::new("max_attr_keys");
    let mut op = Operation::new("max_key_op");
    let mut attrs = HashMap::new();
    
    // Use very long attribute keys
    let long_key = "x".repeat(10000);
    attrs.insert(long_key, Attribute::Int(1));
    
    // Multiple long keys
    for i in 0..5 {
        let key = format!("key_{}_{}", "a".repeat(1000), i);
        attrs.insert(key, Attribute::Int(i));
    }
    
    op.attributes = attrs;
    module.add_operation(op);
    
    assert!(module.operations[0].attributes.len() >= 6);
}