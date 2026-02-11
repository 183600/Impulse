//! Focused new edge case tests - unique boundary scenarios not covered by existing tests
//! Using standard library assertions (assert!, assert_eq!)

use crate::ir::{Module, Value, Type, Operation, Attribute};

/// Test 1: Subnormal floating point values in attributes
#[test]
fn test_subnormal_float_attributes() {
    // f64::MIN_POSITIVE is the smallest positive normalized float
    let min_positive = Attribute::Float(f64::MIN_POSITIVE);
    
    // Subnormal floats (denormals) are smaller than MIN_POSITIVE
    let subnormal1 = Attribute::Float(1e-320);
    let _subnormal2 = Attribute::Float(f64::MIN_POSITIVE / 2.0);
    
    // Negative subnormals
    let neg_subnormal = Attribute::Float(-1e-320);
    
    // Verify attributes can be created
    match min_positive {
        Attribute::Float(val) => assert!(val > 0.0 && val.is_normal()),
        _ => panic!("Expected Float attribute"),
    }
    
    // Subnormals may underflow to zero on some systems
    match subnormal1 {
        Attribute::Float(val) => assert!(val >= 0.0), // May be 0.0 due to underflow
        _ => panic!("Expected Float attribute"),
    }
    
    match neg_subnormal {
        Attribute::Float(val) => assert!(val <= 0.0), // May be -0.0 due to underflow
        _ => panic!("Expected Float attribute"),
    }
}

/// Test 2: Module with all operations having identical names
#[test]
fn test_module_identical_operation_names() {
    let mut module = Module::new("identical_names");
    
    // Add multiple operations with the same name
    for i in 0..5 {
        let mut op = Operation::new("same_name");
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.op_type, "same_name");
    }
}

/// Test 3: Value with shape dimensions that are powers of two
#[test]
fn test_power_of_two_shapes() {
    let shapes = vec![
        vec![1],          // 2^0
        vec![2],          // 2^1
        vec![4],          // 2^2
        vec![8],          // 2^3
        vec![16],         // 2^4
        vec![32],         // 2^5
        vec![64, 64],     // 2^6, 2^6
        vec![128, 128],   // 2^7, 2^7
        vec![256, 256],   // 2^8, 2^8
        vec![512, 512],   // 2^9, 2^9
        vec![1024, 1024], // 2^10, 2^10
    ];
    
    for shape in shapes {
        let value = Value {
            name: "power_of_two".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Verify each dimension is a power of two
        for dim in &value.shape {
            assert!(dim.is_power_of_two());
        }
        
        // Verify num_elements works correctly
        assert_eq!(value.num_elements(), Some(shape.iter().product()));
    }
}

/// Test 4: Operation with attributes containing null byte in string
#[test]
fn test_attribute_with_null_byte_string() {
    let mut op = Operation::new("null_byte_test");
    
    // String containing null byte
    let string_with_null = Attribute::String("hello\0world".to_string());
    op.attributes.insert("null_str".to_string(), string_with_null);
    
    match op.attributes.get("null_str") {
        Some(Attribute::String(s)) => {
            // The string should contain the null byte
            assert!(s.contains('\0'));
            assert_eq!(s.len(), 11); // "hello\0world" = 11 bytes
        }
        _ => panic!("Expected String attribute with null byte"),
    }
}

/// Test 5: Value with alternating zero and non-zero dimensions
#[test]
fn test_alternating_zero_dimensions() {
    let patterns = vec![
        vec![0, 1, 0, 1, 0],       // Alternating 0 and 1
        vec![1, 0, 2, 0, 3, 0],    // Alternating with increasing values
        vec![10, 0, 10, 0, 10],    // Alternating larger values
        vec![0, 0, 1, 0, 0],       // Multiple consecutive zeros
    ];
    
    for shape in patterns {
        let value = Value {
            name: "alternating_dims".to_string(),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // Any shape containing zero should have 0 total elements
        assert_eq!(value.num_elements(), Some(0));
        assert_eq!(value.shape, shape);
    }
}

/// Test 6: Module with input names matching output names
#[test]
fn test_module_input_output_name_collision() {
    let mut module = Module::new("name_collision");
    
    // Add input and output with same name
    let same_name_value = Value {
        name: "shared_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };
    
    module.inputs.push(same_name_value.clone());
    module.outputs.push(same_name_value.clone());
    
    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 1);
    assert_eq!(module.inputs[0].name, module.outputs[0].name);
    assert_eq!(module.inputs[0].name, "shared_name");
}

/// Test 7: Tensor type with empty element type shape
#[test]
fn test_tensor_with_empty_element_shape() {
    let tensor_type = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![],  // Empty shape for element tensor
    };
    
    match tensor_type {
        Type::Tensor { element_type, shape } => {
            assert_eq!(shape, Vec::<usize>::new());
            assert_eq!(*element_type, Type::F32);
        }
        _ => panic!("Expected Tensor type with empty shape"),
    }
}

/// Test 8: Value with prime number dimensions
#[test]
fn test_prime_number_dimensions() {
    let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    
    for (i, &prime) in primes.iter().enumerate() {
        let value = Value {
            name: format!("prime_dim_{}", i),
            ty: Type::F32,
            shape: vec![prime],
        };
        
        assert_eq!(value.shape[0], prime);
        assert_eq!(value.num_elements(), Some(prime));
        
        // Verify it's actually prime
        let is_prime = prime > 1 && 
            (2..=(prime as f64).sqrt() as usize)
                .all(|d| prime % d != 0);
        assert!(is_prime);
    }
}

/// Test 9: Operation with deeply nested array attributes
#[test]
fn test_deeply_nested_array_attributes() {
    // Create a 4-level nested array: [[[[1]]]]
    let deep_nested = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(42),
                ]),
            ]),
        ]),
    ]);
    
    match deep_nested {
        Attribute::Array(level1) => {
            assert_eq!(level1.len(), 1);
            match &level1[0] {
                Attribute::Array(level2) => {
                    assert_eq!(level2.len(), 1);
                    match &level2[0] {
                        Attribute::Array(level3) => {
                            assert_eq!(level3.len(), 1);
                            match &level3[0] {
                                Attribute::Array(level4) => {
                                    assert_eq!(level4.len(), 1);
                                    match level4[0] {
                                        Attribute::Int(42) => (),
                                        _ => panic!("Expected Int(42) at deepest level"),
                                    }
                                }
                                _ => panic!("Expected Array at level 3"),
                            }
                        }
                        _ => panic!("Expected Array at level 2"),
                    }
                }
                _ => panic!("Expected Array at level 1"),
            }
        }
        _ => panic!("Expected outer Array"),
    }
}

/// Test 10: Module with operation containing only empty string attributes
#[test]
fn test_module_empty_string_attributes() {
    let mut module = Module::new("empty_attrs");
    let mut op = Operation::new("empty_strings");
    
    // Add multiple empty string attributes
    for i in 0..5 {
        op.attributes.insert(
            format!("empty_attr_{}", i),
            Attribute::String("".to_string()),
        );
    }
    
    module.add_operation(op);
    
    assert_eq!(module.operations.len(), 1);
    assert_eq!(module.operations[0].attributes.len(), 5);
    
    // Verify all attributes are empty strings
    for (key, attr) in &module.operations[0].attributes {
        match attr {
            Attribute::String(s) => {
                assert_eq!(s, "");
                assert!(s.is_empty());
            }
            _ => panic!("Expected String attribute"),
        }
        assert!(key.starts_with("empty_attr_"));
    }
}