//! Additional edge case tests for the Impulse compiler
//! Focuses on new edge cases not covered in existing tests

use crate::ir::{Module, Operation, Value, Type, Attribute};
use rstest::rstest;

// Test 1: Arithmetic overflow in shape calculations using checked arithmetic
#[test]
fn test_shape_overflow_scenarios() {
    use std::usize;
    
    // Test cases where shape multiplication would overflow
    let value_with_potential_overflow = Value {
        name: "overflow_test".to_string(),
        ty: Type::F32,
        shape: vec![usize::MAX, 2],  // This would overflow when multiplied
    };
    
    // Use the num_elements method which uses checked arithmetic
    let result = value_with_potential_overflow.num_elements();
    assert_eq!(result, None);  // Should return None due to overflow
    
    // Test a safe case
    let safe_value = Value {
        name: "safe_test".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000],
    };
    
    let result = safe_value.num_elements();
    assert_eq!(result, Some(1_000_000));
}

// Test 2: Empty names and whitespace-only names
#[test]
fn test_empty_and_whitespace_names() {
    let empty_name_value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![1, 2, 3],
    };
    assert_eq!(empty_name_value.name, "");
    
    let whitespace_name_value = Value {
        name: "   \t\n  ".to_string(),
        ty: Type::I64,
        shape: vec![],
    };
    assert_eq!(whitespace_name_value.name, "   \t\n  ");
    
    let empty_name_op = Operation::new("");
    assert_eq!(empty_name_op.op_type, "");
    
    let whitespace_name_op = Operation::new("   \t\n  ");
    assert_eq!(whitespace_name_op.op_type, "   \t\n  ");
    
    let module_empty = Module::new("");
    assert_eq!(module_empty.name, "");
    
    let module_whitespace = Module::new("   \t\n  ");
    assert_eq!(module_whitespace.name, "   \t\n  ");
}

// Test 3: Maximum recursion depth in nested tensor types using explicit count
#[test]
fn test_maximum_recursion_tensor_types() {
    const MAX_DEPTH: usize = 1000; // Set a reasonable maximum depth
    
    let mut current_type = Type::F32;
    
    // Build nested type up to maximum depth
    for _ in 0..MAX_DEPTH {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![1],
        };
    }
    
    // Verify the type validity
    assert!(current_type.is_valid_type());
    
    // Clone the deeply nested type to check clone behavior
    let cloned_type = current_type.clone();
    assert_eq!(current_type, cloned_type);
}

// Test 4: Operations with duplicate input/output names
#[test]
fn test_duplicate_input_output_names() {
    let mut op = Operation::new("duplicate_names_test");
    
    // Add inputs with duplicate names
    op.inputs.push(Value {
        name: "input_a".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    op.inputs.push(Value {
        name: "input_a".to_string(),  // Duplicate name
        ty: Type::I32,
        shape: vec![20],
    });
    
    op.inputs.push(Value {
        name: "input_b".to_string(),
        ty: Type::F64,
        shape: vec![30],
    });
    
    op.inputs.push(Value {
        name: "input_b".to_string(),  // Another duplicate
        ty: Type::I64,
        shape: vec![40],
    });
    
    // Add outputs with duplicate names
    op.outputs.push(Value {
        name: "output_x".to_string(),
        ty: Type::F32,
        shape: vec![5],
    });
    
    op.outputs.push(Value {
        name: "output_x".to_string(),  // Duplicate name
        ty: Type::I32,
        shape: vec![10],
    });
    
    assert_eq!(op.inputs.len(), 4);
    assert_eq!(op.outputs.len(), 2);
    
    // Verify all values exist despite duplicate names
    assert_eq!(op.inputs[0].name, "input_a");
    assert_eq!(op.inputs[1].name, "input_a");
    assert_eq!(op.inputs[2].name, "input_b");
    assert_eq!(op.inputs[3].name, "input_b");
    assert_eq!(op.outputs[0].name, "output_x");
    assert_eq!(op.outputs[1].name, "output_x");
}

// Test 5: Very deeply nested attribute arrays
#[test]
fn test_deeply_nested_attribute_arrays() {
    // Create a very deeply nested array attribute
    let mut nested_attr = Attribute::Int(42);
    
    // Nest 50 levels deep
    for _ in 0..50 {
        nested_attr = Attribute::Array(vec![nested_attr]);
    }
    
    // Validate the nesting
    let mut current_attr = &nested_attr;
    for level in 0..50 {
        match current_attr {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 1, "Array at level {} should have 1 element", level);
                current_attr = &arr[0];
            },
            _ => panic!("Expected Array at nesting level {}", level),
        }
    }
    
    // Should reach the innermost Int(42)
    match current_attr {
        Attribute::Int(42) => {},  // Success
        _ => panic!("Innermost value should be Int(42)"),
    }
    
    // Test cloning of deeply nested attribute
    let cloned_attr = nested_attr.clone();
    assert_eq!(nested_attr, cloned_attr);
}

// Test 6: Mixed case sensitivity in type names/operation names
#[rstest]
#[case("ADD", "add", false)]
#[case("Conv2D", "conv2d", false)]
#[case("MatMul", "MatMul", true)]
#[case("", "", true)]
fn test_case_sensitivity(#[case] name1: &str, #[case] name2: &str, #[case] expected_equal: bool) {
    let op1 = Operation::new(name1);
    let op2 = Operation::new(name2);
    
    if expected_equal {
        assert_eq!(op1.op_type, op2.op_type);
    } else {
        assert_ne!(op1.op_type, op2.op_type);
    }
}

// Test 7: Zero-sized types and empty collections in modules
#[test]
fn test_zero_sized_and_empty_collections() {
    // Create a module with no operations, inputs or outputs
    let empty_module = Module::new("empty_module");
    assert_eq!(empty_module.name, "empty_module");
    assert!(empty_module.operations.is_empty());
    assert!(empty_module.inputs.is_empty());
    assert!(empty_module.outputs.is_empty());
    
    // Create a module with operations but no inputs/outputs
    let mut ops_only_module = Module::new("ops_only");
    ops_only_module.add_operation(Operation::new("op1"));
    ops_only_module.add_operation(Operation::new("op2"));
    
    assert_eq!(ops_only_module.operations.len(), 2);
    assert!(ops_only_module.inputs.is_empty());
    assert!(ops_only_module.outputs.is_empty());
    
    // Create a value with zero-sized type representation (scalar)
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],  // Scalar has empty shape
    };
    
    assert!(scalar_value.shape.is_empty());
    assert_eq!(scalar_value.num_elements(), Some(1));  // Scalar has 1 element
}

// Test 8: Special UTF-8 sequences and emoji in names
#[test]
fn test_unicode_and_emoji_names() {
    let emoji_name = "op_🚀_神经网络_αβγδε_Москва";
    let op = Operation::new(emoji_name);
    assert_eq!(op.op_type, emoji_name);
    
    let chinese_name = "函数_测试操作";
    let op_chinese = Operation::new(chinese_name);
    assert_eq!(op_chinese.op_type, chinese_name);
    
    let emoji_value = Value {
        name: "tensor_🔥_⚡_🎉".to_string(),
        ty: Type::F64,
        shape: vec![2, 3],
    };
    assert_eq!(emoji_value.name, "tensor_🔥_⚡_🎉");
    
    let cyrillic_module = Module::new("модуль_тест");
    assert_eq!(cyrillic_module.name, "модуль_тест");
}

// Test 9: Invalid UTF-8 handling (valid Rust strings are always valid UTF-8, 
// so testing proper string handling)
#[test]
fn test_string_validation_and_handling() {
    // Valid UTF-8 strings with various characters
    let valid_strings = [
        "simple_ascii",
        "with_numbers_12345",
        "with_symbols_!@#$%",
        "with_unicode_αβγδε",
        "with_emojis_🚀🔥🎉",
        "with_control_\n\t\r",
        "with_unicode_escape_\u{26A1}",  // Lightning emoji
        "long_string".repeat(1000),  // Long concatenated string
    ];
    
    for (i, test_string) in valid_strings.iter().enumerate() {
        let op = Operation::new(test_string);
        assert_eq!(op.op_type, *test_string, "Test string #{} failed", i);
        
        // Also test with Value names
        let value = Value {
            name: test_string.repeat(2),  // Double length
            ty: Type::F32,
            shape: vec![1, 2],
        };
        assert_eq!(value.name, test_string.repeat(2), "Test string #{} failed for Value", i);
    }
}

// Test 10: Memory allocation edge cases in collections
#[test]
fn test_memory_allocation_edge_cases() {
    // Create operations with no inputs, outputs, or attributes (minimal footprint)
    let minimal_op = Operation::new("minimal");
    assert_eq!(minimal_op.inputs.len(), 0);
    assert_eq!(minimal_op.outputs.len(), 0);
    assert_eq!(minimal_op.attributes.len(), 0);
    
    // Create a value with empty shape (scalar) - smallest tensor representation
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::Bool,
        shape: vec![],  // Empty shape = scalar
    };
    
    assert!(scalar.shape.is_empty());
    assert_eq!(scalar.num_elements(), Some(1));
    
    // Create module with no content but with a name
    let named_module = Module::new("named_module");
    assert_eq!(named_module.name, "named_module");
    assert_eq!(named_module.operations.len(), 0);
    
    // Test creating many small objects to see if there are issues
    let mut many_minimal_ops = Vec::new();
    for i in 0..100 {
        let op = Operation::new(&format!("minimal_{}", i));
        many_minimal_ops.push(op);
    }
    
    assert_eq!(many_minimal_ops.len(), 100);
    
    // Verify all operations are properly initialized
    for (idx, op) in many_minimal_ops.iter().enumerate() {
        assert_eq!(op.op_type, format!("minimal_{}", idx));
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
    }
}