//! Exclusive boundary edge tests - unique edge cases not covered by other test files
//! Tests cover specialized boundary scenarios, extreme value combinations, and unique patterns

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::ir::TypeExtensions;
use std::collections::HashMap;

/// Test 1: Module with alternating types in inputs/outputs chain
#[test]
fn test_module_alternating_type_chain() {
    let mut module = Module::new("alternating_type_chain");

    // Create operations with alternating types in their outputs
    let types = vec![Type::F32, Type::I32, Type::F64, Type::I64, Type::Bool];

    for (i, ty) in types.iter().enumerate() {
        let mut op = Operation::new(&format!("alternating_{}", i));
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: ty.clone(),
            shape: vec![10],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);
    // Verify types alternate correctly
    assert_eq!(module.operations[0].outputs[0].ty, Type::F32);
    assert_eq!(module.operations[1].outputs[0].ty, Type::I32);
    assert_eq!(module.operations[2].outputs[0].ty, Type::F64);
    assert_eq!(module.operations[3].outputs[0].ty, Type::I64);
    assert_eq!(module.operations[4].outputs[0].ty, Type::Bool);
}

/// Test 2: Attribute with Fibonacci sequence in integer array
#[test]
fn test_fibonacci_array_attribute() {
    let fib_numbers: Vec<i64> = (0..20).scan((0, 1), |(a, b), _| {
        let result = *a;
        *a = *b;
        *b = result + *b;
        Some(result)
    }).collect();
    
    let fib_attrs: Vec<Attribute> = fib_numbers.iter().map(|&n| Attribute::Int(n)).collect();
    let fib_array = Attribute::Array(fib_attrs);
    
    match fib_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 20);
            // Verify first few Fibonacci numbers
            if let Attribute::Int(n) = arr[0] { assert_eq!(n, 0); }
            if let Attribute::Int(n) = arr[1] { assert_eq!(n, 1); }
            if let Attribute::Int(n) = arr[2] { assert_eq!(n, 1); }
            if let Attribute::Int(n) = arr[3] { assert_eq!(n, 2); }
            if let Attribute::Int(n) = arr[4] { assert_eq!(n, 3); }
            if let Attribute::Int(n) = arr[19] { assert_eq!(n, 4181); }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 3: Value with factorial-based dimensions
#[test]
fn test_factorial_based_dimensions() {
    let mut factorial = 1usize;
    let factorials: Vec<usize> = (1..=7).map(|n| {
        factorial *= n;
        factorial
    }).collect();

    for (i, &fact) in factorials.iter().enumerate() {
        let value = Value {
            name: format!("factorial_dim_{}", i),
            ty: Type::I32,
            shape: vec![fact],
        };

        // For a 1D tensor, num_elements should equal the dimension value
        assert_eq!(value.num_elements(), Some(fact));
    }
}

/// Test 4: Nested tensor with prime number dimensions
#[test]
fn test_prime_number_nested_tensor() {
    let primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

    // Create nested tensors with prime dimensions
    let mut current_type = Type::F32;
    for &prime in primes.iter().take(5) {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![prime],
        };
    }

    // Should be valid with all prime dimensions
    assert!(current_type.is_valid_type());

    // Verify we have a nested tensor structure
    let mut depth = 0;
    let mut current = &current_type;
    while let Type::Tensor { element_type, shape } = current {
        // Verify each dimension is a prime number
        assert!(primes.contains(&shape[0]), "Dimension {} is not a prime", shape[0]);
        depth += 1;
        current = element_type;
    }

    // Should have 5 levels of nesting (plus the base F32)
    assert_eq!(depth, 5);
}

/// Test 5: Operation with geometric progression in attribute values
#[test]
fn test_geometric_progression_attributes() {
    let mut op = Operation::new("geometric_progression");
    let mut attrs = HashMap::new();
    
    // Create geometric progression: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    let progression: Vec<i64> = (0..10).map(|i| 2_i64.pow(i as u32)).collect();
    
    for (i, &val) in progression.iter().enumerate() {
        attrs.insert(format!("geo_{}", i), Attribute::Int(val));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10);
    // Verify progression
    for i in 0..10 {
        let expected = 2_i64.pow(i as u32);
        match op.attributes.get(&format!("geo_{}", i)) {
            Some(Attribute::Int(val)) => assert_eq!(*val, expected),
            _ => panic!("Expected Int({})", expected),
        }
    }
}

/// Test 6: Module with palindrome-shaped tensors
#[test]
fn test_palindrome_shape_tensors() {
    let mut module = Module::new("palindrome_shapes");

    let palindrome_shapes = vec![
        vec![1],
        vec![1, 1],
        vec![1, 2, 1],
        vec![1, 2, 3, 2, 1],
        vec![1, 2, 3, 4, 3, 2, 1],
    ];

    for shape in palindrome_shapes {
        module.inputs.push(Value {
            name: format!("palindrome_{}", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("_")),
            ty: Type::F32,
            shape: shape.clone(),
        });
    }

    assert_eq!(module.inputs.len(), 5);
    // Verify palindrome property (shape equals its reverse)
    for input in &module.inputs {
        assert_eq!(input.shape, input.shape.iter().cloned().rev().collect::<Vec<_>>());
    }
}

/// Test 7: Attribute array with power-of-two float values
#[test]
fn test_power_of_two_float_attributes() {
    let powers: Vec<Attribute> = (0..10).map(|i| {
        Attribute::Float(2.0_f64.powi(i as i32))
    }).collect();
    
    let power_array = Attribute::Array(powers);
    
    match power_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 10);
            // Verify powers of 2
            for i in 0..10 {
                match &arr[i] {
                    Attribute::Float(val) => {
                        let expected = 2.0_f64.powi(i as i32);
                        assert!((val - expected).abs() < f64::EPSILON);
                    }
                    _ => panic!("Expected Float at index {}", i),
                }
            }
        }
        _ => panic!("Expected Array attribute"),
    }
}

/// Test 8: Value with triangular number dimensions
#[test]
fn test_triangular_number_dimensions() {
    let triangular_numbers: Vec<usize> = (1..=10)
        .scan(0, |acc, n| {
            *acc += n;
            Some(*acc)
        })
        .collect();
    
    for (i, &tri) in triangular_numbers.iter().enumerate() {
        let value = Value {
            name: format!("triangular_{}", i),
            ty: Type::F64,
            shape: vec![tri],
        };
        
        // Verify the element count matches triangular number
        assert_eq!(value.num_elements(), Some(tri));
    }
}

/// Test 9: Module with operations having perfect square attribute keys
#[test]
fn test_perfect_square_attribute_keys() {
    let mut module = Module::new("perfect_square_keys");
    
    let mut op = Operation::new("square_key_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with perfect square keys
    for i in 1..=10 {
        let square = i * i;
        attrs.insert(format!("key_{}", square), Attribute::Int(square));
    }
    
    op.attributes = attrs;
    module.add_operation(op);
    
    assert_eq!(module.operations[0].attributes.len(), 10);
    // Verify all keys are perfect squares
    for i in 1..=10 {
        let key = format!("key_{}", i * i);
        assert!(module.operations[0].attributes.contains_key(&key));
    }
}

/// Test 10: Nested tensor type with alternating tensor/element pattern
#[test]
fn test_alternating_tensor_element_pattern() {
    // Create pattern: Tensor(Tensor(F32)), Tensor(F32), Tensor(Tensor(Tensor(F32))), etc.
    let patterns: Vec<Type> = vec![
        Type::F32,
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2] },
        Type::Tensor { 
            element_type: Box::new(Type::Tensor { 
                element_type: Box::new(Type::F32), 
                shape: vec![3] 
            }), 
            shape: vec![2] 
        },
        Type::Tensor { 
            element_type: Box::new(Type::Tensor { 
                element_type: Box::new(Type::Tensor { 
                    element_type: Box::new(Type::F32), 
                    shape: vec![4] 
                }), 
                shape: vec![3] 
            }), 
            shape: vec![2] 
        },
    ];
    
    for (i, ty) in patterns.iter().enumerate() {
        assert!(ty.is_valid_type(), "Pattern {} should be valid", i);
    }
    
    // Verify all are distinct
    for i in 0..patterns.len() {
        for j in (i + 1)..patterns.len() {
            assert_ne!(patterns[i], patterns[j], "Patterns {} and {} should be different", i, j);
        }
    }
}