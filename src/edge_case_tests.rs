//! Edge case tests for the Impulse compiler
//! Covering boundary conditions and extreme values

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Operations with minimal input/output counts (0 and 1)
#[test]
fn test_operations_minimal_io_counts() {
    // Operation with no inputs and no outputs
    let op_no_io = Operation::new("noop");
    assert_eq!(op_no_io.inputs.len(), 0);
    assert_eq!(op_no_io.outputs.len(), 0);
    assert_eq!(op_no_io.op_type, "noop");
    assert_eq!(op_no_io.attributes.len(), 0);

    // Operation with one input and one output
    let mut op_single_io = Operation::new("identity");
    op_single_io.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    op_single_io.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });

    assert_eq!(op_single_io.inputs.len(), 1);
    assert_eq!(op_single_io.outputs.len(), 1);
    assert_eq!(op_single_io.op_type, "identity");
}

/// Test 2: Values with maximum possible dimension count
#[test]
fn test_values_maximum_dimension_count() {
    // Create a value with a very high number of dimensions
    let max_dims = vec![1; 100]; // 100 dimensions, each with size 1
    let value = Value {
        name: "max_dim_tensor".to_string(),
        ty: Type::I64,
        shape: max_dims,
    };

    assert_eq!(value.shape.len(), 100);
    
    // Calculate total elements (should be 1 for all 1's)
    let total_elements: usize = value.shape.iter().product();
    assert_eq!(total_elements, 1);
}

/// Test 3: Operations with maximum attribute count
#[test]
fn test_operations_maximum_attributes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("max_attr_op");
    let mut attrs = HashMap::new();

    // Add a large number of attributes
    for i in 0..10_000 {
        attrs.insert(
            format!("attr_{}", i),
            Attribute::String(format!("value_{}", i))
        );
    }

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 10_000);
    assert_eq!(op.op_type, "max_attr_op");
}

/// Test 4: Nested tensor types with extreme nesting depth
#[test]
fn test_tensor_extreme_nesting_depth() {
    let mut current_type = Type::F32;

    // Create a deeply nested tensor type
    for _ in 0..50 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type.clone()),
            shape: vec![2],
        };
    }

    // Verify the structure can be handled
    if let Type::Tensor { element_type: _, shape } = &current_type {
        assert_eq!(shape, &vec![2]);
    } else {
        panic!("Expected a tensor type after nesting");
    }

    // Clone and compare
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

/// Test 5: Values with extreme shape values
#[rstest]
#[case(vec![usize::MAX, 1], usize::MAX)]
#[case(vec![1, usize::MAX], usize::MAX)]
#[case(vec![0, usize::MAX], 0)]
#[case(vec![1, 1, 1], 1)]
#[case(vec![2, 3, 4], 24)]
fn test_values_extreme_shapes(#[case] shape: Vec<usize>, #[case] expected_product: usize) {
    let value = Value {
        name: "extreme_shape_tensor".to_string(),
        ty: Type::F32,
        shape,
    };

    let calculated_product: usize = value.shape.iter().product();
    assert_eq!(calculated_product, expected_product);
}

/// Test 6: Attributes with extremely large string values
#[test]
fn test_attributes_extremely_large_strings() {
    let large_string = "x".repeat(1_000_000); // 1MB string
    let attr = Attribute::String(large_string.clone());

    match attr {
        Attribute::String(s) => {
            assert_eq!(s.len(), large_string.len());
            assert_eq!(s, large_string);
        },
        _ => panic!("Expected String attribute"),
    }
}

/// Test 7: Modules with maximum operation count
#[test]
fn test_modules_maximum_operations() {
    let mut module = Module::new("max_ops_module");

    // Add maximum possible operations
    for i in 0..100_000 {
        let op = Operation::new(&format!("op_{}", i));
        module.add_operation(op);

        // Periodic check to ensure progress
        if i % 25_000 == 0 {
            assert!(module.operations.len() <= i + 1);
        }
    }

    assert_eq!(module.operations.len(), 100_000);
    assert_eq!(module.name, "max_ops_module");
}

/// Test 8: Mixed type operations with all type variants
#[test]
fn test_operations_all_type_variants() {
    use std::collections::HashMap;

    let mut op = Operation::new("mixed_types_op");

    // Add values of all available types
    let types = [
        Type::F32, Type::F64, Type::I32, Type::I64, Type::Bool,
        Type::Tensor { element_type: Box::new(Type::F32), shape: vec![2] }
    ];

    for (i, t) in types.iter().enumerate() {
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: t.clone(),
            shape: vec![i + 1],
        });

        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: t.clone(),
            shape: vec![i + 2],
        });
    }

    // Add various attribute types
    let mut attrs = HashMap::new();
    attrs.insert("int_attr".to_string(), Attribute::Int(42));
    attrs.insert("float_attr".to_string(), Attribute::Float(3.14));
    attrs.insert("string_attr".to_string(), Attribute::String("test".to_string()));
    attrs.insert("bool_attr".to_string(), Attribute::Bool(true));
    attrs.insert("array_attr".to_string(), Attribute::Array(vec![
        Attribute::Int(1), Attribute::Int(2), Attribute::Int(3)
    ]));

    op.attributes = attrs;

    assert_eq!(op.inputs.len(), types.len());
    assert_eq!(op.outputs.len(), types.len());
    assert_eq!(op.attributes.len(), 5);
    assert_eq!(op.op_type, "mixed_types_op");
}

/// Test 9: Special character handling in names
#[rstest]
#[case("")]
#[case("!@#$%^&*()")]
#[case("中文字符")]
#[case("🚀🔥🌟")]
#[case("tab\tchar")]
#[case("newline\nchar")]
#[case("null\0byte")]
fn test_special_character_names(#[case] name: &str) {
    // Test value names with special characters
    let value = Value {
        name: name.to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.name, name);

    // Test operation names with special characters
    let op = Operation::new(name);
    assert_eq!(op.op_type, name);

    // Test module names with special characters
    let module = Module::new(name);
    assert_eq!(module.name, name);
}

/// Test 10: Overflow protection in tensor size calculations
#[test]
fn test_tensor_size_overflow_protection() {
    // Test for potential overflow in tensor size calculations
    
    // A tensor that would have a very large number of elements
    // but not cause actual overflow on 64-bit systems
    let safe_large_tensor = Value {
        name: "safe_large_tensor".to_string(),
        ty: Type::F32,
        shape: vec![10_000, 10_000],  // 100M elements
    };

    assert_eq!(safe_large_tensor.shape, vec![10_000, 10_000]);
    let prod: usize = safe_large_tensor.shape.iter().product();
    assert_eq!(prod, 100_000_000);

    // Test with safe checked multiplication
    let checked_prod: Option<usize> = safe_large_tensor.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    assert_eq!(checked_prod, Some(100_000_000));

    // Test tensor that will definitely have 0 elements due to zeros in shape
    let zero_tensor = Value {
        name: "zero_tensor".to_string(),
        ty: Type::I64,
        shape: vec![5, 0, 10],
    };

    let zero_prod: usize = zero_tensor.shape.iter().product();
    assert_eq!(zero_prod, 0);

    // Test with checked multiplication for zero case
    let checked_zero_prod: Option<usize> = zero_tensor.shape.iter()
        .try_fold(1_usize, |acc, &x| acc.checked_mul(x));
    assert_eq!(checked_zero_prod, Some(0));
}