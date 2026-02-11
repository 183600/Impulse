//! Focused comprehensive edge tests - New boundary cases with standard assertions
//! 
//! This module provides additional test coverage for edge cases and boundary conditions
//! using standard library assert! and assert_eq! macros.

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Value with overflow protection using checked arithmetic
#[test]
fn test_value_overflow_protection() {
    // Safe value that doesn't overflow
    let safe_value = Value {
        name: "safe".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000],
    };
    assert_eq!(safe_value.num_elements(), Some(1_000_000));
    
    // Scalar with empty shape (product of empty iterator is 1)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));
    
    // Zero dimension results in 0 elements
    let zero_dim = Value {
        name: "zero".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));
}

/// Test 2: Type validation for all supported types
#[test]
fn test_type_validation_all_types() {
    use crate::ir::TypeExtensions;
    
    // All basic types should be valid
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
    
    // Nested tensor types should also be valid
    let nested = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(nested.is_valid_type());
    
    // Deeply nested tensor
    let deep_nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![5],
        }),
        shape: vec![3],
    };
    assert!(deep_nested.is_valid_type());
}

/// Test 3: Module with maximum reasonable dimension count
#[test]
fn test_module_max_dimensions() {
    // Create a value with many dimensions
    let high_dim_value = Value {
        name: "high_dim".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3, 4, 5, 6, 7, 8],
    };
    assert_eq!(high_dim_value.shape.len(), 8);
    assert_eq!(high_dim_value.num_elements(), Some(40320));
}

/// Test 4: Attribute array with mixed element types
#[test]
fn test_mixed_attribute_array() {
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(42),
        Attribute::Float(3.14),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Array(vec![Attribute::Int(1), Attribute::Int(2)]),
    ]);
    
    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 5);
            match &arr[0] {
                Attribute::Int(42) => {},
                _ => panic!("Expected Int(42)"),
            }
            match &arr[4] {
                Attribute::Array(nested) => assert_eq!(nested.len(), 2),
                _ => panic!("Expected nested array"),
            }
        },
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 5: Operation with extreme integer attribute values
#[test]
fn test_extreme_integer_attributes() {
    let mut op = Operation::new("extreme_ints");
    op.attributes.insert("max".to_string(), Attribute::Int(i64::MAX));
    op.attributes.insert("min".to_string(), Attribute::Int(i64::MIN));
    op.attributes.insert("zero".to_string(), Attribute::Int(0));
    op.attributes.insert("negative".to_string(), Attribute::Int(-1));
    
    assert_eq!(op.attributes.len(), 4);
    
    match op.attributes.get("max") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MAX),
        _ => panic!("Expected max int"),
    }
    match op.attributes.get("min") {
        Some(Attribute::Int(val)) => assert_eq!(*val, i64::MIN),
        _ => panic!("Expected min int"),
    }
}

/// Test 6: Module with all possible tensor type combinations
#[test]
fn test_all_tensor_type_combinations() {
    let combinations = vec![
        (Type::F32, vec![1]),
        (Type::F64, vec![2, 2]),
        (Type::I32, vec![3, 3, 3]),
        (Type::I64, vec![4, 4, 4, 4]),
        (Type::Bool, vec![5, 5, 5, 5, 5]),
    ];
    
    for (base_type, shape) in combinations {
        let tensor = Type::Tensor {
            element_type: Box::new(base_type.clone()),
            shape: shape.clone(),
        };
        
        match tensor {
            Type::Tensor { element_type, shape: s } => {
                assert_eq!(s, shape);
                assert_eq!(*element_type, base_type);
            },
            _ => panic!("Expected Tensor type"),
        }
    }
}

/// Test 7: Value with alternating dimension pattern
#[test]
fn test_alternating_dimension_pattern() {
    let alternating = Value {
        name: "alternating".to_string(),
        ty: Type::F32,
        shape: vec![1, 0, 1, 0, 1],
    };
    
    // Any zero in dimensions results in 0 total elements
    assert_eq!(alternating.num_elements(), Some(0));
    assert_eq!(alternating.shape, vec![1, 0, 1, 0, 1]);
}

/// Test 8: Module with operations having no explicit attributes
#[test]
fn test_operations_without_attributes() {
    let mut module = Module::new("no_attrs_module");
    
    for i in 0..3 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 3);
    for op in &module.operations {
        assert!(op.attributes.is_empty());
    }
}

/// Test 9: Attribute with empty string and special characters
#[test]
fn test_special_string_attributes() {
    let special_strings = vec![
        ("".to_string(), "empty"),
        (" ".to_string(), "space"),
        ("\t".to_string(), "tab"),
        ("\n".to_string(), "newline"),
        ("test@email.com".to_string(), "email"),
        ("path/to/file".to_string(), "path"),
    ];
    
    for (s, desc) in special_strings {
        let attr = Attribute::String(s.clone());
        match attr {
            Attribute::String(val) => assert_eq!(val, s),
            _ => panic!("Expected string for {}", desc),
        }
    }
}

/// Test 10: Module with single element tensors of various types
#[test]
fn test_single_element_various_types() {
    let types = vec![Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool];
    
    for ty in types {
        let single_elem = Value {
            name: "single".to_string(),
            ty: ty.clone(),
            shape: vec![1, 1, 1],
        };
        
        assert_eq!(single_elem.num_elements(), Some(1));
        assert_eq!(single_elem.ty, ty);
        assert_eq!(single_elem.shape.iter().product::<usize>(), 1);
    }
}