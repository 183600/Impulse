//! Comprehensive integration boundary tests
//! Tests covering integration across modules with boundary conditions

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::passes::{PassManager, Pass};
use crate::backends::{BackendManager, Backend};
use crate::utils::{calculate_tensor_size_safe, gcd, lcm, round_up_to_multiple, next_power_of_2};
use crate::utils::{validate_module, validate_operation};
use anyhow::Result;
use std::collections::HashMap;

/// Test 1: Backend manager with unknown target
#[test]
fn test_backend_manager_unknown_target() {
    let manager = BackendManager::new();
    let module = Module::new("test");
    
    let result = manager.compile(&module, "nonexistent_backend");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Unknown target"));
}

/// Test 2: Backend compilation with large module
#[test]
fn test_backend_large_module_compilation() {
    let manager = BackendManager::new();
    let mut module = Module::new("large_test");
    
    // Create a module with many operations
    for i in 0..1000 {
        let mut op = Operation::new(&format!("op_{}", i));
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        module.add_operation(op);
    }
    
    let result = manager.compile(&module, "cpu");
    assert!(result.is_ok());
    assert!(result.unwrap().len() > 0);
}

/// Test 3: Pass manager with failing pass mid-sequence
#[test]
fn test_pass_manager_failing_pass_mid_sequence() {
    struct FirstPass;
    impl Pass for FirstPass {
        fn run(&self, module: &mut Module) -> Result<()> {
            module.name.push_str("_first");
            Ok(())
        }
        fn name(&self) -> &'static str { "FirstPass" }
    }
    
    struct FailingPass;
    impl Pass for FailingPass {
        fn run(&self, _module: &mut Module) -> Result<()> {
            anyhow::bail!("Intentional failure")
        }
        fn name(&self) -> &'static str { "FailingPass" }
    }
    
    struct LastPass;
    impl Pass for LastPass {
        fn run(&self, module: &mut Module) -> Result<()> {
            module.name.push_str("_last");
            Ok(())
        }
        fn name(&self) -> &'static str { "LastPass" }
    }
    
    let mut pm = PassManager::new();
    pm.add_pass(Box::new(FirstPass));
    pm.add_pass(Box::new(FailingPass));
    pm.add_pass(Box::new(LastPass));
    
    let mut module = Module::new("test");
    let result = pm.run_passes(&mut module);
    
    assert!(result.is_err());
    // First pass should have run, but not last pass
    assert!(module.name.contains("_first"));
    assert!(!module.name.contains("_last"));
}

/// Test 4: Validate module with conflicting input/output names
#[test]
fn test_validate_module_name_conflicts() {
    let mut module = Module::new("conflict_test");
    
    // Add input
    module.inputs.push(Value {
        name: "shared_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    // Add output with same name
    module.outputs.push(Value {
        name: "shared_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    
    let result = validate_module(&module);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("share the same name"));
}

/// Test 5: GCD with large prime numbers
#[test]
fn test_gcd_large_primes() {
    let large_prime1 = 999983;
    let large_prime2 = 999979;
    
    let result = gcd(large_prime1, large_prime2);
    assert_eq!(result, 1); // Primes are coprime
}

/// Test 6: LCM overflow prevention
#[test]
fn test_lcm_overflow_prevention() {
    // Test with values that could cause overflow in naive implementation
    let a = 1000000;
    let b = 1000000;
    
    let result = lcm(a, b);
    assert_eq!(result, 1000000); // LCM of equal numbers is the number itself
}

/// Test 7: Round up to multiple with zero multiple
#[test]
fn test_round_up_zero_multiple() {
    let result = round_up_to_multiple(100, 0);
    assert_eq!(result, 100); // Should return original value when multiple is 0
}

/// Test 8: Next power of 2 for usize::MAX / 2 + 1
#[test]
fn test_next_power_of_2_large() {
    let large_value = usize::MAX / 2 + 1;
    let result = next_power_of_2(large_value);
    assert!(result.is_power_of_two());
    assert!(result >= large_value);
}

/// Test 9: Calculate tensor size with potential overflow dimensions
#[test]
fn test_calculate_tensor_size_overflow() {
    // Test dimensions that would overflow
    let overflow_dims = [usize::MAX, 2];
    let result = calculate_tensor_size_safe(&overflow_dims);
    assert!(result.is_none());
}

/// Test 10: Operation with extremely deep attribute nesting
#[test]
fn test_operation_deep_attribute_nesting() {
    let mut op = Operation::new("deep_nested");
    let mut attrs = HashMap::new();
    
    // Create deeply nested attribute structure
    let mut current = Attribute::Int(42);
    for _ in 0..10 {
        current = Attribute::Array(vec![current]);
    }
    
    attrs.insert("deep_key".to_string(), current);
    op.attributes = attrs;
    
    assert!(op.attributes.contains_key("deep_key"));
    
    // Verify the structure
    match op.attributes.get("deep_key") {
        Some(Attribute::Array(arr)) => {
            assert!(!arr.is_empty());
        }
        _ => panic!("Expected Array attribute"),
    }
}