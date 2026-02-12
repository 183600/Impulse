/// Advanced critical boundary tests - Additional edge cases for compiler robustness
use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// Test 1: Attribute array with circular reference pattern simulation
#[test]
fn test_attribute_array_deeply_nested() {
    // Create deeply nested arrays to test stack depth limits
    let deep_array = Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Array(vec![
                    Attribute::Int(42),
                ]),
            ]),
        ]),
    ]);

    match deep_array {
        Attribute::Array(outer) => {
            assert_eq!(outer.len(), 1);
            match &outer[0] {
                Attribute::Array(level1) => {
                    assert_eq!(level1.len(), 1);
                    match &level1[0] {
                        Attribute::Array(level2) => {
                            assert_eq!(level2.len(), 1);
                            match &level2[0] {
                                Attribute::Array(level3) => {
                                    assert_eq!(level3.len(), 1);
                                    match level3[0] {
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
        _ => panic!("Expected Array at outer level"),
    }
}

/// Test 2: Value with alternating dimension pattern
#[test]
fn test_alternating_dimension_pattern() {
    // Test tensor with alternating dimensions (1, 1000, 1, 1000)
    let value = Value {
        name: "alternating_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1, 1000, 1, 1000],
    };

    assert_eq!(value.num_elements(), Some(1_000_000));
    assert_eq!(value.shape, vec![1, 1000, 1, 1000]);
}

/// Test 3: Operation with attribute keys that are very similar
#[test]
fn test_similar_attribute_keys() {
    let mut op = Operation::new("similar_keys");
    let mut attrs = HashMap::new();

    // Add attributes with very similar keys
    attrs.insert("attribute".to_string(), Attribute::Int(1));
    attrs.insert("attribute_".to_string(), Attribute::Int(2));
    attrs.insert("attribute_1".to_string(), Attribute::Int(3));
    attrs.insert("attribute_2".to_string(), Attribute::Int(4));
    attrs.insert("Attribute".to_string(), Attribute::Int(5)); // Different case

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 5);
    assert_eq!(op.attributes.get("attribute"), Some(&Attribute::Int(1)));
    assert_eq!(op.attributes.get("attribute_"), Some(&Attribute::Int(2)));
    assert_eq!(op.attributes.get("attribute_1"), Some(&Attribute::Int(3)));
    assert_eq!(op.attributes.get("attribute_2"), Some(&Attribute::Int(4)));
    assert_eq!(op.attributes.get("Attribute"), Some(&Attribute::Int(5)));
}

/// Test 4: Value with shape that has single large dimension
#[test]
fn test_single_large_dimension() {
    // Test vector with 100 million elements
    let large_vector = Value {
        name: "large_vector".to_string(),
        ty: Type::F32,
        shape: vec![100_000_000],
    };

    assert_eq!(large_vector.num_elements(), Some(100_000_000));
    assert_eq!(large_vector.shape.len(), 1);
}

/// Test 5: Module with operations referencing similar value names
#[test]
fn test_similar_value_names_in_operations() {
    let mut module = Module::new("similar_names");

    // Create values with very similar names
    let input_a = Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };

    let input_a_ = Value {
        name: "input_".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };

    let input_a_1 = Value {
        name: "input_1".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };

    let mut op = Operation::new("test");
    op.inputs.push(input_a);
    op.inputs.push(input_a_);
    op.inputs.push(input_a_1);

    module.add_operation(op);

    assert_eq!(module.operations[0].inputs.len(), 3);
    assert_eq!(module.operations[0].inputs[0].name, "input");
    assert_eq!(module.operations[0].inputs[1].name, "input_");
    assert_eq!(module.operations[0].inputs[2].name, "input_1");
}

/// Test 6: Attribute with string containing all control characters
#[test]
fn test_control_characters_in_string() {
    // String containing common control characters
    let control_string = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0B\x0C\x0E\x0F\x7F";
    let attr = Attribute::String(control_string.to_string());

    match attr {
        Attribute::String(s) => {
            // Verify the string contains the expected control characters
            assert_eq!(s.len(), 14); // 14 control characters
            assert_eq!(s.chars().nth(0).unwrap() as u32, 0x00);
            assert_eq!(s.chars().nth(13).unwrap() as u32, 0x7F);
        }
        _ => panic!("Expected String attribute with control characters"),
    }
}

/// Test 7: Value with shape that is a perfect square
#[test]
fn test_perfect_square_dimensions() {
    // Test tensors with perfect square dimensions
    let square_2d = Value {
        name: "square_2d".to_string(),
        ty: Type::F32,
        shape: vec![1000, 1000],
    };

    let cube_3d = Value {
        name: "cube_3d".to_string(),
        ty: Type::F32,
        shape: vec![100, 100, 100],
    };

    assert_eq!(square_2d.num_elements(), Some(1_000_000));
    assert_eq!(cube_3d.num_elements(), Some(1_000_000));
}

/// Test 8: Operation with boolean attributes in various combinations
#[test]
fn test_boolean_attribute_combinations() {
    let mut op = Operation::new("bool_test");
    let mut attrs = HashMap::new();

    // Add all combinations of boolean attributes
    attrs.insert("flag1".to_string(), Attribute::Bool(true));
    attrs.insert("flag2".to_string(), Attribute::Bool(false));
    attrs.insert("flag3".to_string(), Attribute::Bool(true));
    attrs.insert("flag4".to_string(), Attribute::Bool(false));
    attrs.insert("flag5".to_string(), Attribute::Bool(true));

    op.attributes = attrs;

    assert_eq!(op.attributes.len(), 5);
    assert_eq!(op.attributes.get("flag1"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("flag2"), Some(&Attribute::Bool(false)));
    assert_eq!(op.attributes.get("flag3"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("flag4"), Some(&Attribute::Bool(false)));
    assert_eq!(op.attributes.get("flag5"), Some(&Attribute::Bool(true)));
}

/// Test 9: Value with shape that powers of 2
#[test]
fn test_power_of_two_dimensions() {
    // Test tensors with dimensions that are powers of 2
    let power_of_2 = Value {
        name: "power_of_2".to_string(),
        ty: Type::F32,
        shape: vec![2, 4, 8, 16, 32],
    };

    assert_eq!(power_of_2.num_elements(), Some(2 * 4 * 8 * 16 * 32));
    assert_eq!(power_of_2.num_elements(), Some(32768));
}

/// Test 10: Module with operations that have duplicate inputs
#[test]
fn test_duplicate_inputs_in_operation() {
    let mut module = Module::new("duplicate_inputs");

    // Create a value that will be used as multiple inputs
    let shared_value = Value {
        name: "shared".to_string(),
        ty: Type::F32,
        shape: vec![10],
    };

    let mut op = Operation::new("duplicate_test");
    
    // Add the same value multiple times as inputs
    for _ in 0..5 {
        op.inputs.push(shared_value.clone());
    }

    module.add_operation(op);

    assert_eq!(module.operations[0].inputs.len(), 5);
    // All inputs should have the same name
    for input in &module.operations[0].inputs {
        assert_eq!(input.name, "shared");
    }
}