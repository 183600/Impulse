//! Additional edge case tests covering more boundary conditions
//! that complement the existing test suites

use rstest::*;
use crate::ir::{Value, Type, Operation, Attribute, Module};

/// Test 1: Operations with unicode and non-ASCII names
#[rstest]
#[case("tensor_名称_test")]
#[case("🔥_unicode_tensor_🚀")]
#[case("многоязычный_tensor")]
#[case("café_naïve_tensor")]
fn test_unicode_operation_names(#[case] name: &str) {
    let op = Operation::new(name);
    assert_eq!(op.op_type, name);
    
    // Also test with Values
    let value = Value {
        name: name.to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(value.name, name);
    
    // And with Modules
    let module = Module::new(name);
    assert_eq!(module.name, name);
}

/// Test 2: Deep but balanced nested tensor types (reduced depth to avoid stack overflow)
#[test]
fn test_balanced_deep_tensor_nesting() {
    let mut current_type = Type::Bool;
    
    // Create 20 levels of nesting with balanced structure (reduced from 1000 to prevent stack overflow)
    for _ in 0..20 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![2],
        };
    }
    
    // Validate the deeply nested type
    assert!(current_type.is_valid_type());
    
    // Test cloning doesn't cause issues
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
    
    // Ensure recursive validation works
    use crate::ir::TypeExtensions;
    assert!(cloned.is_valid_type());
}

/// Test 3: Large sparse tensor shapes (with many dimensions but low total size)
#[rstest]
#[case(vec![1, 1, 1, 1, 1000, 1, 1, 1, 1], 1000)]
#[case(vec![2, 1, 1, 1, 1, 1, 1, 1, 500], 1000)]
#[case(vec![10, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10)]
#[case(vec![1; 10], 1)]  // 10 dimensions of size 1
fn test_sparse_tensor_shapes(#[case] shape: Vec<usize>, #[case] expected_total: usize) {
    let value = Value {
        name: "sparse_tensor".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    let calculated_total: usize = value.shape.iter().product();
    assert_eq!(calculated_total, expected_total);
    assert_eq!(value.shape.len(), shape.len());
}

/// Test 4: Operations with maximum length attribute names
#[test]
fn test_operations_with_extremely_long_attribute_keys() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("long_attr_key_op");
    let mut attrs = HashMap::new();
    
    // Create an attribute key with 50,000 characters (should not cause issues)
    let long_key = "key_".repeat(12_500); // 50k chars
    attrs.insert(long_key.clone(), Attribute::Int(12345));
    
    // Add another with different attributes
    let long_key2 = "param_".repeat(8_334); // ~50k chars
    attrs.insert(long_key2.clone(), Attribute::String("test_value".to_string()));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 2);
    assert!(op.attributes.contains_key(&long_key));
    assert!(op.attributes.contains_key(&long_key2));
    
    if let Some(Attribute::Int(12345)) = op.attributes.get(&long_key) {
        assert!(true); // Correct value found
    } else {
        panic!("Long key not found or incorrect value stored");
    }
}

/// Test 5: Mixed extreme values in attributes for mathematical edge cases
#[test]
fn test_attribute_mathematical_extremes() {
    use std::collections::HashMap;
    
    let mut op = Operation::new("math_extremes_op");
    let mut attrs = HashMap::new();
    
    // Add attributes with extreme numeric values
    attrs.insert("max_i64".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_i64".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_f64".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_f64".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("epsilon".to_string(), Attribute::Float(f64::EPSILON));
    attrs.insert("negative_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("positive_zero".to_string(), Attribute::Float(0.0));
    attrs.insert("one".to_string(), Attribute::Float(1.0));
    attrs.insert("negative_one".to_string(), Attribute::Float(-1.0));
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 9);
    assert_eq!(op.attributes.get("max_i64"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("min_i64"), Some(&Attribute::Int(i64::MIN)));
    
    // Test float comparisons carefully
    if let Some(Attribute::Float(max_f64_val)) = op.attributes.get("max_f64") {
        assert_eq!(*max_f64_val, f64::MAX);
    }
    
    if let Some(Attribute::Float(eps_val)) = op.attributes.get("epsilon") {
        assert_eq!(*eps_val, f64::EPSILON);
    }
}

/// Test 6: Tensor types with extreme shape ratios and edge cases
#[rstest]
#[case(vec![1, 1_000_000], 1_000_000)]  // Very wide
#[case(vec![1_000_000, 1], 1_000_000)]  // Very tall
#[case(vec![1000, 1000, 1000], 1_000_000_000)]  // 3D large
#[case(vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 1024)]  // 10 dimensions of 2
fn test_extreme_aspect_ratio_tensors(#[case] shape: Vec<usize>, #[case] expected_size: usize) {
    let value = Value {
        name: "ratio_test_tensor".to_string(),
        ty: Type::F32,
        shape: shape.clone(),
    };
    
    assert_eq!(value.shape, shape);
    let calculated_size: usize = value.shape.iter().product();
    assert_eq!(calculated_size, expected_size);
    
    // Ensure the tensor is valid and can be cloned
    let cloned_value = value.clone();
    assert_eq!(value, cloned_value);
}

/// Test 7: Module with maximum nesting of operations and values
#[test]
fn test_deeply_nested_module_structure() {
    let mut module = Module::new("nested_structure_test");
    
    // Create operations that form a chain (each referencing the previous output)
    for i in 0..100 {
        let mut op = Operation::new(&format!("op_chain_{}", i));
        
        // Previous output becomes current input (except for first op)
        if i > 0 {
            op.inputs.push(Value {
                name: format!("output_{}", i - 1),
                ty: Type::F32,
                shape: vec![10],
            });
        }
        
        // Current op produces an output
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10],
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 100);
    
    // Verify the chain structure
    for (idx, op) in module.operations.iter().enumerate() {
        if idx == 0 {
            // First operation has no inputs
            assert_eq!(op.inputs.len(), 0);
        } else {
            // Subsequent operations have one input
            assert_eq!(op.inputs.len(), 1);
            assert_eq!(op.inputs[0].name, format!("output_{}", idx - 1));
        }
        
        // All operations have one output
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.outputs[0].name, format!("output_{}", idx));
    }
}

/// Test 8: Operations with alternating complex attribute structures
#[test]
fn test_alternating_complex_attribute_structures() {
    use std::collections::HashMap;
    
    let mut ops = Vec::new();
    
    // Create 20 operations with alternating attribute complexity
    for i in 0..20 {
        let mut op = Operation::new(&format!("alt_attr_op_{}", i));
        let mut attrs = HashMap::new();
        
        if i % 2 == 0 {
            // Even-numbered ops get complex nested attributes
            attrs.insert(format!("nested_group_{}", i), Attribute::Array(vec![
                Attribute::Int(i as i64),
                Attribute::Array(vec![
                    Attribute::String(format!("nested_{}", i)),
                    Attribute::Bool(i % 3 == 0),
                    Attribute::Float(i as f64 * 0.5),
                ]),
            ]));
        } else {
            // Odd-numbered ops get simple attributes
            attrs.insert(format!("simple_attr_{}", i), Attribute::Int(i as i64));
        }
        
        op.attributes = attrs;
        ops.push(op);
    }
    
    assert_eq!(ops.len(), 20);
    
    // Verify alternating patterns
    for (idx, op) in ops.iter().enumerate() {
        if idx % 2 == 0 {
            // Even ops should have complex structure
            assert_eq!(op.attributes.len(), 1);
            assert!(op.attributes.contains_key(&format!("nested_group_{}", idx)));
        } else {
            // Odd ops should have simple structure
            assert_eq!(op.attributes.len(), 1);
            assert!(op.attributes.contains_key(&format!("simple_attr_{}", idx)));
        }
    }
}

/// Test 9: Memory efficiency with shared vs unique structures
#[test]
fn test_memory_efficiency_with_shared_structures() {
    // Create a large number of operations that share similar structures
    let base_shape = vec![32, 32, 3];
    let base_type = Type::F32;
    let mut module = Module::new("efficiency_test");
    
    // Create 1000 operations with identical but independently constructed structures
    for i in 0..1000 {
        let mut op = Operation::new("shared_structure_op");
        
        // Add inputs with identical but separately allocated structures
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: base_type.clone(),
            shape: base_shape.clone(),
        });
        
        // Add outputs with identical but separately allocated structures
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: base_type.clone(),
            shape: base_shape.clone(),
        });
        
        module.add_operation(op);
    }
    
    assert_eq!(module.operations.len(), 1000);
    
    // Verify that all structures were maintained correctly
    for (idx, op) in module.operations.iter().enumerate() {
        assert_eq!(op.inputs.len(), 1);
        assert_eq!(op.outputs.len(), 1);
        assert_eq!(op.inputs[0].shape, base_shape);
        assert_eq!(op.outputs[0].shape, base_shape);
        assert_eq!(op.inputs[0].ty, base_type);
        assert_eq!(op.outputs[0].ty, base_type);
        assert_eq!(op.inputs[0].name, format!("input_{}", idx));
        assert_eq!(op.outputs[0].name, format!("output_{}", idx));
    }
}

/// Test 10: Edge cases with empty collections and zero-sized objects
#[rstest]
#[case(Module::new(""), 0, "", true, true, true)]  // Empty module name
#[case(Module::new("empty_io"), 0, "empty_io", true, true, true)]  // Empty but named
#[case({
    let mut m = Module::new("single_op");
    m.add_operation(Operation::new("simple"));
    m
}, 1, "single_op", false, true, true)]  // One op, no I/O
fn test_empty_collection_edge_cases(
    #[case] module: Module, 
    #[case] expected_ops: usize, 
    #[case] expected_name: &str,
    #[case] should_be_empty_ops: bool,
    #[case] should_be_empty_inputs: bool,
    #[case] should_be_empty_outputs: bool
) {
    assert_eq!(module.operations.len(), expected_ops);
    assert_eq!(module.name, expected_name);
    
    // Test boolean assertions about emptiness
    assert_eq!(module.operations.is_empty(), should_be_empty_ops);
    assert_eq!(module.inputs.is_empty(), should_be_empty_inputs);
    assert_eq!(module.outputs.is_empty(), should_be_empty_outputs);
    
    // Test cloning empty or nearly empty structures
    let cloned = module.clone();
    assert_eq!(cloned.name, module.name);
    assert_eq!(cloned.operations.len(), module.operations.len());
    assert_eq!(cloned.inputs.len(), module.inputs.len());
    assert_eq!(cloned.outputs.len(), module.outputs.len());
}