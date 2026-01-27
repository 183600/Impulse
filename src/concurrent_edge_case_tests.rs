//! Additional concurrent access edge case tests for the Impulse compiler
//! Testing thread-safety and race condition prevention

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::sync::{Arc, Mutex};
use std::thread;

// Test 1: Test concurrent read access to shared IR objects
#[test]
fn test_concurrent_read_access_to_ir_objects() {
    let shared_value = Arc::new(Value {
        name: "shared_value".to_string(),
        ty: Type::F32,
        shape: vec![10, 20, 30],
    });

    let mut handles = vec![];
    
    // Spawn multiple threads that read the same value
    for _ in 0..10 {
        let value_clone = shared_value.clone();
        let handle = thread::spawn(move || {
            // Perform multiple reads to ensure consistency
            let name = &value_clone.name;
            let ty = &value_clone.ty;
            let shape = &value_clone.shape;
            
            assert_eq!(name, "shared_value");
            assert_eq!(ty, &Type::F32);
            assert_eq!(shape, &vec![10, 20, 30]);
            
            let elements = value_clone.num_elements();
            assert_eq!(elements, Some(6000)); // 10 * 20 * 30
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

// Test 2: Test concurrent operations on separate objects
#[test]
fn test_concurrent_operations_on_separate_objects() {
    let mut handles = vec![];
    
    // Each thread creates and manipulates its own objects independently
    for i in 0..5 {
        let handle = thread::spawn(move || {
            let value = Value {
                name: format!("thread_local_{}", i),
                ty: match i % 4 {
                    0 => Type::F32,
                    1 => Type::F64,
                    2 => Type::I32,
                    _ => Type::I64,
                },
                shape: vec![i + 1, i + 2],
            };
            
            // Perform some operations on the local object
            let elements = value.num_elements();
            assert_eq!(elements, Some((i + 1) * (i + 2)));
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

// Test 3: Test mutex-protected shared state access
#[test]
fn test_mutex_protected_shared_state() {
    let shared_module = Arc::new(Mutex::new(Module::new("shared_module")));
    
    let mut handles = vec![];
    
    // Multiple threads trying to add operations to the same module
    for i in 0..5 {
        let module_clone = shared_module.clone();
        let handle = thread::spawn(move || {
            let op = Operation::new(&format!("op_from_thread_{}", i));
            
            // Lock the mutex to safely modify the shared module
            let mut module = module_clone.lock().unwrap();
            module.add_operation(op);
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify that all operations were added (though possibly in different order)
    let final_module = shared_module.lock().unwrap();
    assert_eq!(final_module.operations.len(), 5);
    
    // Verify the module name is unchanged
    assert_eq!(final_module.name, "shared_module");
}

// Test 4: Test attribute manipulation in concurrent context
#[test]
fn test_concurrent_attribute_manipulation() {
    // Create a reference-counted collection of attributes
    let shared_attrs = Arc::new(Mutex::new(std::collections::HashMap::<String, Attribute>::new()));
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let attrs_clone = shared_attrs.clone();
        let handle = thread::spawn(move || {
            let attr = match i % 3 {
                0 => Attribute::Int(i as i64),
                1 => Attribute::Float(i as f64),
                _ => Attribute::String(format!("thread_{}", i)),
            };
            
            let mut attrs = attrs_clone.lock().unwrap();
            attrs.insert(format!("attr_{}", i), attr);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Check that all attributes were inserted
    let final_attrs = shared_attrs.lock().unwrap();
    assert_eq!(final_attrs.len(), 10);
    
    // Verify a few specific values
    if let Some(Attribute::Int(0)) = final_attrs.get("attr_0") {
        assert!(true);
    } else {
        panic!("Expected Int(0) for attr_0");
    }
    
    if let Some(Attribute::Float(1.0)) = final_attrs.get("attr_1") {
        assert!(true);
    } else {
        panic!("Expected Float(1.0) for attr_1");
    }
}

// Test 5: Race condition test for type comparisons
#[test]
fn test_race_condition_type_comparisons() {
    let type_a = Arc::new(Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![10, 10],
    });
    
    let type_b = Arc::new(Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![10, 10],
    });
    
    let mut handles = vec![];
    
    // Multiple threads comparing the same types
    for _ in 0..20 {
        let ta = type_a.clone();
        let tb = type_b.clone();
        let handle = thread::spawn(move || {
            // Both types should be equal
            assert_eq!(ta, tb);
            
            // Cloning and comparing should work consistently
            let clone_a = (*ta).clone();
            let clone_b = (*tb).clone();
            assert_eq!(clone_a, clone_b);
            assert_eq!((*ta), clone_b);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

// Test 6: Concurrent tensor shape calculations
#[test]
fn test_concurrent_tensor_shape_calculations() {
    let shapes = vec![
        vec![10, 20, 30],
        vec![5, 5, 5, 5],
        vec![100, 100],
        vec![0, 10, 20],
        vec![1, 1, 1, 1, 1],
    ];
    
    let mut handles = vec![];
    
    for (idx, shape) in shapes.iter().enumerate() {
        let shape_clone = shape.clone();
        let handle = thread::spawn(move || {
            let value = Value {
                name: format!("value_{}", idx),
                ty: Type::F32,
                shape: shape_clone,
            };
            
            let elements = value.num_elements();
            let expected_product: usize = value.shape.iter().product();
            assert_eq!(elements, Some(expected_product));
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

// Test 7: Thread-safe module creation and modification
#[test]
fn test_thread_safe_module_operations() {
    let shared_counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for i in 0..10 {
        let counter = shared_counter.clone();
        let handle = thread::spawn(move || {
            // Each thread creates its own module
            let mut module = Module::new(&format!("thread_module_{}", i));
            
            // Add some operations
            for j in 0..i {
                let mut op = Operation::new(&format!("op_{}_{}", i, j));
                op.inputs.push(Value {
                    name: format!("input_{}_{}", i, j),
                    ty: Type::F32,
                    shape: vec![j + 1],
                });
                module.add_operation(op);
            }
            
            // Update shared counter
            let mut counter_guard = counter.lock().unwrap();
            *counter_guard += module.operations.len();
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let final_count = *shared_counter.lock().unwrap();
    // Sum of 0+1+2+...+9 = 45 operations total
    assert_eq!(final_count, 45);
}

// Test 8: Shared nested type structures in multithreaded context
#[test]
fn test_shared_nested_type_structures_multithreaded() {
    let shared_nested_type = Arc::new(Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![3, 4],
        }),
        shape: vec![5, 6],
    });
    
    let mut handles = vec![];
    
    for _ in 0..15 {
        let nested_type = shared_nested_type.clone();
        let handle = thread::spawn(move || {
            // Test that the nested structure is accessible and consistent
            if let Type::Tensor { element_type: outer_elem, shape: outer_shape } = &*nested_type {
                assert_eq!(outer_shape, &vec![5, 6]);
                
                if let Type::Tensor { element_type: inner_elem, shape: inner_shape } = outer_elem.as_ref() {
                    assert_eq!(inner_shape, &vec![3, 4]);
                    
                    if let Type::F64 = inner_elem.as_ref() {
                        // Success: verified deep nesting
                    } else {
                        panic!("Expected F64 as innermost type");
                    }
                }
            }
            
            // Test cloning in this thread
            let thread_clone = (*nested_type).clone();
            assert_eq!(*nested_type, thread_clone);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

// Test 9: Concurrent access to operation collections
#[test]
fn test_concurrent_access_to_operation_collections() {
    let shared_ops = Arc::new(Mutex::new(Vec::<Operation>::new()));
    
    let mut handles = vec![];
    
    for i in 0..8 {
        let ops_clone = shared_ops.clone();
        let handle = thread::spawn(move || {
            let mut op = Operation::new(&format!("concurrent_op_{}", i));
            
            // Add inputs and attributes based on thread id
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::I32,
                shape: vec![i + 1],
            });
            
            op.attributes.insert(
                format!("thread_id_{}", i),
                Attribute::Int(i as i64)
            );
            
            // Add to shared collection
            let mut ops = ops_clone.lock().unwrap();
            ops.push(op);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let final_ops = shared_ops.lock().unwrap();
    assert_eq!(final_ops.len(), 8);
    
    // Instead of relying on order, verify all expected operations exist
    let mut found_ops = 0;
    for op in final_ops.iter() {
        for i in 0..8 {
            if op.op_type == format!("concurrent_op_{}", i) {
                found_ops += 1;
                break;
            }
        }
    }
    assert_eq!(found_ops, 8);
}

// Test 10: Edge case with rapid thread creation/destruction
#[test]
fn test_rapid_thread_creation_destruction_with_ir_objects() {
    let start = std::time::Instant::now();
    
    // Create many short-lived threads that manipulate IR objects
    let mut handles = vec![];
    
    for i in 0..50 {
        let handle = thread::spawn(move || {
            // Create and immediately process an IR object
            let value = Value {
                name: format!("rapid_thread_{}", i),
                ty: Type::Bool,
                shape: vec![i % 10 + 1],
            };
            
            // Perform quick operations
            assert_eq!(value.shape.len(), 1);
            assert!(value.shape[0] > 0 && value.shape[0] <= 10);
            
            // Return computation result
            value.num_elements()
        });
        handles.push(handle);
    }
    
    // Collect results
    let results: Result<Vec<_>, _> = handles.into_iter().map(|h| h.join()).collect();
    let results = results.expect("Thread join failed");
    
    // Verify we got 50 results
    assert_eq!(results.len(), 50);
    
    // Verify timing is reasonable (should complete within seconds)
    assert!(start.elapsed().as_secs() < 10);
}