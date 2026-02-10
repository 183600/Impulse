//! Advanced boundary tests covering edge cases and extreme scenarios
use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};

/// Test 1: Value with overflow-prone dimension calculation
#[test]
fn test_value_overflow_dimension_product() {
    // Test a shape that will definitely overflow usize multiplication
    // usize is typically 64-bit, so using values that exceed u64::MAX
    let value = Value {
        name: "overflow_product".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX / 2 + 1, 3], // Will overflow
    };

    // num_elements should return None for overflow cases
    let result = value.num_elements();
    // The multiplication would overflow in checked arithmetic
    assert!(result.is_none());
}

/// Test 2: Attribute with special float values (NaN, Infinity)
#[test]
fn test_special_float_attributes() {
    let nan_attr = Attribute::Float(f64::NAN);
    let pos_inf = Attribute::Float(f64::INFINITY);
    let neg_inf = Attribute::Float(f64::NEG_INFINITY);
    
    // Verify these can be created without panic
    match nan_attr {
        Attribute::Float(val) => assert!(val.is_nan()),
        _ => panic!("Expected Float(NaN)"),
    }
    
    match pos_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_positive()),
        _ => panic!("Expected Float(INFINITY)"),
    }
    
    match neg_inf {
        Attribute::Float(val) => assert!(val.is_infinite() && val.is_sign_negative()),
        _ => panic!("Expected Float(NEG_INFINITY)"),
    }
}

/// Test 3: Type validation for deeply nested tensor types
#[test]
fn test_deeply_nested_tensor_validation() {
    // Create a deeply nested tensor type (5 levels deep)
    let level5 = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::Tensor {
                    element_type: Box::new(Type::Tensor {
                        element_type: Box::new(Type::F32),
                        shape: vec![2],
                    }),
                    shape: vec![3],
                }),
                shape: vec![4],
            }),
            shape: vec![5],
        }),
        shape: vec![6],
    };

    // Create separate instances to test validation at each level
    let test_level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2],
    };
    let test_level2 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![3],
    };
    let test_level3 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![4],
    };
    let test_level4 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![5],
    };

    // All should be valid types
    assert!(test_level1.is_valid_type());
    assert!(test_level2.is_valid_type());
    assert!(test_level3.is_valid_type());
    assert!(test_level4.is_valid_type());
    assert!(level5.is_valid_type());
}

/// Test 4: Operation with extremely long operation type name
#[test]
fn test_operation_with_very_long_name() {
    let long_name = "a".repeat(1000);
    let op = Operation::new(&long_name);
    
    assert_eq!(op.op_type.len(), 1000);
    assert!(op.op_type.chars().all(|c| c == 'a'));
}

/// Test 5: Module with cyclic input-output naming pattern
#[test]
fn test_module_cyclic_io_naming() {
    let mut module = Module::new("cyclic_io");
    
    // Create inputs and outputs with cyclic naming
    let v1 = Value {
        name: "x".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    let v2 = Value {
        name: "y".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    let v3 = Value {
        name: "z".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    module.inputs.push(v1.clone());
    module.inputs.push(v2.clone());
    module.outputs.push(v2.clone());
    module.outputs.push(v3.clone());
    
    assert_eq!(module.inputs.len(), 2);
    assert_eq!(module.outputs.len(), 2);
    // v2 appears in both inputs and outputs
    assert_eq!(module.inputs[1].name, module.outputs[0].name);
}

/// Test 6: Value with zero-dimensional tensor and edge case types
#[test]
fn test_zero_dimensional_tensors_all_types() {
    let types = [Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool];
    
    for ty in types.iter() {
        let value = Value {
            name: "scalar".to_string(),
            ty: ty.clone(),
            shape: vec![], // Scalar - 0-dimensional tensor
        };
        
        assert_eq!(value.shape.len(), 0);
        assert!(value.ty.is_valid_type());
    }
}

/// Test 7: Attribute with array containing mixed types and null-like values
#[test]
fn test_mixed_type_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(0),
        Attribute::Float(0.0),
        Attribute::String("".to_string()),
        Attribute::Bool(false),
        Attribute::Array(vec![]), // Empty array
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            match &arr[0] {
                Attribute::Int(0) => {},
                _ => panic!("Expected Int(0)"),
            }
            match &arr[1] {
                Attribute::Float(val) if *val == 0.0 => {},
                _ => panic!("Expected Float(0.0)"),
            }
            match &arr[2] {
                Attribute::String(s) if s.is_empty() => {},
                _ => panic!("Expected empty string"),
            }
            match &arr[3] {
                Attribute::Bool(false) => {},
                _ => panic!("Expected Bool(false)"),
            }
            match &arr[4] {
                Attribute::Array(empty) if empty.is_empty() => {},
                _ => panic!("Expected empty array"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 8: Module with all dimensions being 1 (single element path)
#[test]
fn test_all_ones_dimension_module() {
    let mut module = Module::new("all_ones");
    
    // Add multiple operations with all-ones shapes
    for i in 0..5 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![1, 1, 1, 1, 1], // 5 dimensions, all 1s
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![1], // Single output
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    // All input shapes should have product of 1
    for op in &module.operations {
        let product: usize = op.inputs[0].shape.iter().product();
        assert_eq!(product, 1);
    }
}

/// Test 9: Attribute with extreme integer boundary values
#[test]
fn test_extreme_integer_boundary_attributes() {
    let extreme_ints = vec![
        (Attribute::Int(i64::MAX), i64::MAX),
        (Attribute::Int(i64::MIN), i64::MIN),
        (Attribute::Int(0), 0),
        (Attribute::Int(-1), -1),
        (Attribute::Int(1), 1),
    ];
    
    for (attr, expected) in extreme_ints {
        match attr {
            Attribute::Int(val) => assert_eq!(val, expected),
            _ => panic!("Expected Int attribute"),
        }
    }
}

/// Test 10: Operation with attributes using all possible Unicode characters in strings
#[test]
fn test_unicode_attribute_strings() {
    let mut op = Operation::new("unicode_op");
    op.attributes.insert(
        "emoji".to_string(),
        Attribute::String("🚀🌟✨".to_string()),
    );
    op.attributes.insert(
        "chinese".to_string(),
        Attribute::String("中文测试".to_string()),
    );
    op.attributes.insert(
        "emoji_chinese_mixed".to_string(),
        Attribute::String("🇨🇳 China 中国 🇺🇸".to_string()),
    );
    op.attributes.insert(
        "arabic".to_string(),
        Attribute::String("مرحبا بالعالم".to_string()),
    );
    op.attributes.insert(
        "cyrillic".to_string(),
        Attribute::String("Привет мир".to_string()),
    );
    
    assert_eq!(op.attributes.len(), 5);
    
    match op.attributes.get("emoji") {
        Some(Attribute::String(s)) => assert!(s.contains('🚀')),
        _ => panic!("Expected emoji string"),
    }
    
    match op.attributes.get("chinese") {
        Some(Attribute::String(s)) => assert_eq!(s, "中文测试"),
        _ => panic!("Expected Chinese string"),
    }
}