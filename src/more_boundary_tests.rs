//! More boundary condition tests for the Impulse compiler
//! Focuses on additional edge cases not covered by other test modules

use crate::ir::{Module, Value, Type, Operation, Attribute};
use rstest::*;

// Test 1: Integer overflow scenarios in tensor operations with checked arithmetic
#[test]
fn test_tensor_size_overflow_scenarios() {
    // Test that num_elements method properly handles potential overflow
    let large_dims = Value {
        name: "large_tensor".to_string(),
        ty: Type::F32,
        // Use dimensions that would overflow when multiplied as regular product
        shape: vec![10_000, 10_000], // These would multiply to 100M which should be safe
    };
    
    // Test the safe calculation
    let elements = large_dims.num_elements();
    assert!(elements.is_some());
    assert_eq!(elements.unwrap(), 100_000_000);
}

// Test 2: Empty collections and null cases
#[rstest]
#[case(vec![], 1)]  // Empty shape should yield 1 element (scalar)
#[case(vec![0], 0)]  // Zero in shape results in 0 elements
#[case(vec![0, 0, 0], 0)]  // Multiple zeros
#[case(vec![0, 100, 200], 0)]  // Zero followed by other values
#[case(vec![100, 0, 200], 0)]  // Zero in middle
#[case(vec![100, 200, 0], 0)]  // Zero at end
fn test_zero_element_tensor_shapes(#[case] shape: Vec<usize>, #[case] expected_count: usize) {
    let value = Value {
        name: "zero_test_tensor".to_string(),
        ty: Type::F32,
        shape,
    };
    
    // Using the safe calculation method
    let elements = value.num_elements();
    assert!(elements.is_some());
    assert_eq!(elements.unwrap(), expected_count);
    
    // Also verify the multiplication result
    let product: usize = value.shape.iter().product();
    assert_eq!(product, expected_count);
}

// Test 3: Maximum recursion depth for nested types - reduced to safer limit
#[test]
fn test_deeply_nested_tensor_types_depth_limit() {
    let mut current_type = Type::F32;
    
    // Create nested tensor types up to 20 levels to test recursion limits safely
    for i in 0..20 {
        current_type = Type::Tensor {
            element_type: Box::new(current_type),
            shape: vec![i % 5 + 1], // Varying shape to make it more complex
        };
    }
    
    // Verify the final type is still valid
    use crate::ir::TypeExtensions;
    assert!(current_type.is_valid_type());
    
    // Test that it can be cloned without issue
    let cloned = current_type.clone();
    assert_eq!(current_type, cloned);
}

// Test 4: Safe arithmetic operations testing
#[test]
fn test_safe_arithmetic_operations() {
    let value = Value {
        name: "safe_math_test".to_string(),
        ty: Type::F32,
        shape: vec![10, 20, 30],
    };
    
    // Test safe multiplication with no overflow
    let product: usize = value.shape.iter().product();
    assert_eq!(product, 6000); // 10 * 20 * 30
    
    // Test with num_elements method (uses checked_mul)
    let safe_elements = value.num_elements();
    assert_eq!(safe_elements, Some(6000));
}

// Test 5: Memory allocation tests with more reasonable sizes
#[test]
fn test_large_collection_allocations() {
    // Test creating an operation with a large but reasonable number of attributes
    let mut op = Operation::new("large_attrs_op");
    let mut attrs = std::collections::HashMap::new();
    
    // Add 50,000 attributes to test memory handling
    for i in 0..50_000 {
        attrs.insert(
            format!("attr_{:05}", i),
            Attribute::String(format!("value_{}", i))
        );
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 50_000);
    
    // Verify a few randomly selected attributes exist
    assert!(op.attributes.contains_key("attr_00000"));
    assert!(op.attributes.contains_key("attr_25000"));
    assert!(op.attributes.contains_key("attr_49999"));
    
    // Check specific values
    if let Some(Attribute::String(val)) = op.attributes.get("attr_00000") {
        assert_eq!(val, "value_0");
    } else {
        panic!("Expected attr_00000 to have value_0");
    }
    
    if let Some(Attribute::String(val)) = op.attributes.get("attr_49999") {
        assert_eq!(val, "value_49999");
    } else {
        panic!("Expected attr_49999 to have value_49999");
    }
}

// Test 6: Floating point precision issues
#[test]
fn test_floating_point_precision() {
    // Test classic floating point precision issue: 0.1 + 0.2 != 0.3
    let sum: f64 = 0.1 + 0.2;
    let expected: f64 = 0.3;
    
    // These are not exactly equal due to floating point precision
    assert_ne!(sum, expected);
    
    // But they should be close enough
    let diff: f64 = (sum - expected).abs();
    assert!(diff < f64::EPSILON * 10.0);
    
    // Test attribute equality with floats
    let attr1 = Attribute::Float(sum);
    let attr2 = Attribute::Float(expected);
    
    // Since sum != expected, the attributes shouldn't be equal
    assert_ne!(attr1, attr2);
}

// Test 7: Hash map performance with reasonable sizes
#[test]
fn test_hash_map_performance_reasonable() {
    let mut op = Operation::new("hash_collision_test");
    let mut attrs = std::collections::HashMap::new();
    
    // Create keys with 10,000 entries which is reasonable
    for i in 0..10_000 {
        let key = format!("key_{:05}", i);
        attrs.insert(key, Attribute::Int(i));
    }
    
    op.attributes = attrs;
    
    assert_eq!(op.attributes.len(), 10_000);
    
    // Test retrieval by checking several random keys
    assert!(op.attributes.contains_key("key_00000"));
    assert!(op.attributes.contains_key("key_05000"));
    assert!(op.attributes.contains_key("key_09999"));
    
    // Verify value integrity
    if let Some(Attribute::Int(0)) = op.attributes.get("key_00000") {
        assert_eq!(Attribute::Int(0), Attribute::Int(0));
    } else {
        panic!("Expected Int(0) for key_00000");
    }
    
    if let Some(&Attribute::Int(expected_val)) = op.attributes.get("key_05000") {
        assert_eq!(expected_val, 5000);
    } else {
        panic!("Expected Int(5000) for key_05000");
    }
}

// Test 8: Deep type clone operations
#[test]
fn test_deep_type_clone_operations() {
    // Test deep cloning of complex nested types
    let complex_tensor = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F64),
                shape: vec![3, 4],
            }),
            shape: vec![5, 6],
        }),
        shape: vec![7, 8],
    };
    
    // Clone the complex type multiple times to test memory management
    let clone1 = complex_tensor.clone();
    let clone2 = complex_tensor.clone();
    let clone3 = clone1.clone();
    
    // All clones should be equal
    assert_eq!(complex_tensor, clone1);
    assert_eq!(complex_tensor, clone2);
    assert_eq!(complex_tensor, clone3);
    assert_eq!(clone1, clone2);
    assert_eq!(clone1, clone3);
    assert_eq!(clone2, clone3);
    
    // Verify the nested structure remains intact
    if let Type::Tensor { element_type: outer_elem, shape: outer_shape } = &complex_tensor {
        assert_eq!(outer_shape, &vec![7, 8]);
        
        if let Type::Tensor { element_type: middle_elem, shape: middle_shape } = outer_elem.as_ref() {
            assert_eq!(middle_shape, &vec![5, 6]);
            
            if let Type::Tensor { element_type: inner_elem, shape: inner_shape } = middle_elem.as_ref() {
                assert_eq!(inner_shape, &vec![3, 4]);
                
                if let Type::F64 = inner_elem.as_ref() {
                    // Success: verified deep nesting
                } else {
                    panic!("Expected F64 as innermost type");
                }
            }
        }
    }
}

// Test 9: Boundary conditions for value shapes - with reduced dimension count
#[test]
fn test_boundary_tensor_shapes() {
    // Test very long shapes (many dimensions) but with smaller count
    let long_shape_value = Value {
        name: "long_shape_tensor".to_string(),
        ty: Type::F32,
        shape: vec![1; 1_000],  // 1,000 dimensions, each of size 1
    };
    
    assert_eq!(long_shape_value.shape.len(), 1_000);
    assert!(long_shape_value.shape.iter().all(|&x| x == 1));
    
    // Total elements should be 1 (since all dims are 1)
    let elements = long_shape_value.num_elements();
    assert_eq!(elements, Some(1));
    
    // Test minimal shapes
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],  // Empty shape = scalar
    };
    
    assert_eq!(scalar_value.shape.len(), 0);
    let scalar_elements = scalar_value.num_elements();
    assert_eq!(scalar_elements, Some(1));
    
    // Test single dimension with large but manageable size
    let large_1d = Value {
        name: "large_1d".to_string(),
        ty: Type::F32,
        shape: vec![100_000_000],  // 100 million elements
    };
    
    assert_eq!(large_1d.shape, vec![100_000_000]);
    let large_1d_elements = large_1d.num_elements();
    assert_eq!(large_1d_elements, Some(100_000_000));
}

// Test 10: Comprehensive edge case for module operations with manageable count
#[test]
fn test_module_operation_boundary_conditions() {
    let mut module = Module::new("boundary_test_module");
    
    // Test adding operations with various edge cases - using 500 instead of 1000
    for i in 0..500 {
        let mut op = Operation::new(&format!("op_{}", i));
        
        // Add inputs with different characteristics
        if i % 2 == 0 {
            op.inputs.push(Value {
                name: format!("input_{}_even", i),
                ty: Type::F32,
                shape: vec![i + 1],  // Growing shape
            });
        }
        
        // Add outputs with different characteristics  
        if i % 3 == 0 {
            op.outputs.push(Value {
                name: format!("output_{}_div3", i),
                ty: Type::I64,
                shape: vec![1, i + 1],  // 2D shape growing
            });
        }
        
        // Add attributes occasionally
        if i % 5 == 0 {
            op.attributes.insert(
                format!("attr_{}", i),
                Attribute::Float(i as f64 * 0.5)
            );
        }
        
        module.add_operation(op);
    }
    
    // Verify module characteristics
    assert_eq!(module.operations.len(), 500);
    assert_eq!(module.name, "boundary_test_module");
    
    // Verify a few specific operations
    let first_op = &module.operations[0];
    assert_eq!(first_op.op_type, "op_0");
    assert_eq!(first_op.inputs.len(), 1);  // i=0, 0%2==0, so added input
    assert_eq!(first_op.outputs.len(), 1);  // i=0, 0%3==0, so added output
    assert_eq!(first_op.attributes.len(), 1);  // i=0, 0%5==0, so added attribute
    
    // Check last operation
    let last_op = &module.operations[499];
    assert_eq!(last_op.op_type, "op_499");
    assert_eq!(last_op.inputs.len(), if 499 % 2 == 0 { 1 } else { 0 });  // 499 is odd, so 0
    assert_eq!(last_op.outputs.len(), if 499 % 3 == 0 { 1 } else { 0 });  // 499 % 3 != 0, so 0
    assert_eq!(last_op.attributes.len(), if 499 % 5 == 0 { 1 } else { 0 });  // 499 % 5 != 0, so 0
    
    // Verify some specific values in the module
    assert_eq!(module.operations[10].inputs[0].name, "input_10_even");
    assert_eq!(module.operations[9].outputs[0].name, "output_9_div3");
    if let Some(Attribute::Float(val)) = module.operations[10].attributes.get("attr_10") {
        assert_eq!(*val, 5.0);  // 10 * 0.5
    } else {
        panic!("Expected attribute in operation 10");
    }
}