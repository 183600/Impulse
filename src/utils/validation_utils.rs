//! Validation utilities for the Impulse compiler

use crate::ir::{Value, Type, Operation};

/// Validates that a value's shape is valid (no invalid dimensions)
pub fn validate_value_shape(value: &Value) -> Result<(), String> {
    // Check for any obviously invalid dimensions
    for &dim in &value.shape {
        // While 0-dimension tensors are valid in some contexts,
        // we might want to warn or validate them specially
        if dim > 1_000_000 && dim != 0 {  // Large but not impossibly large
            return Err(format!("Dimension {} in shape {:?} seems unusually large", dim, value.shape));
        }
    }
    
    Ok(())
}

/// Validates that a type is internally consistent
pub fn validate_type(ty: &Type) -> Result<(), String> {
    match ty {
        Type::Tensor { element_type, shape } => {
            // Validate the nested type
            validate_type(element_type)?;
            
            // Validate shape dimensions
            for &dim in shape {
                if dim > 1_000_000 && dim != 0 {
                    return Err(format!("Dimension {} in tensor shape seems unusually large", dim));
                }
            }
        },
        _ => {
            // Basic types are considered valid
        }
    }
    
    Ok(())
}

/// Validates that an operation meets basic structural requirements
pub fn validate_operation(op: &Operation) -> Result<(), String> {
    // Validate input values
    for input in &op.inputs {
        validate_value_shape(input)?;
        validate_type(&input.ty)?;
    }
    
    // Validate output values
    for output in &op.outputs {
        validate_value_shape(output)?;
        validate_type(&output.ty)?;
    }
    
    // Validate operation name
    if op.op_type.chars().count() > 1_000_000 {
        return Err("Operation type name is unusually long".to_string());
    }
    
    // Check for duplicate names within inputs
    let mut seen_names = std::collections::HashSet::new();
    for input in &op.inputs {
        if seen_names.contains(&input.name) {
            return Err(format!("Duplicate input name detected: {}", input.name));
        }
        seen_names.insert(&input.name);
    }
    
    // Check for duplicate names within outputs
    let mut output_names = std::collections::HashSet::new();
    for output in &op.outputs {
        if output_names.contains(&output.name) {
            return Err(format!("Duplicate output name detected: {}", output.name));
        }
        output_names.insert(&output.name);
    }
    
    // Check for conflict between input names and output names
    for input in &op.inputs {
        for output in &op.outputs {
            if input.name == output.name {
                return Err(format!("Input and output share the same name: {}", input.name));
            }
        }
    }
    
    Ok(())
}

/// Validates that all shapes in a module are valid
pub fn validate_module_shapes(module: &crate::ir::Module) -> Result<(), String> {
    // Validate all inputs
    for input in &module.inputs {
        validate_value_shape(input)?;
        validate_type(&input.ty)?;
    }
    
    // Validate all outputs
    for output in &module.outputs {
        validate_value_shape(output)?;
        validate_type(&output.ty)?;
    }
    
    // Validate all operations in the module
    for op in &module.operations {
        validate_operation(op)?;
    }
    
    Ok(())
}

/// Validates that a module has unique names for inputs and outputs
pub fn validate_module_uniqueness(module: &crate::ir::Module) -> Result<(), String> {
    // Check for duplicate input names
    let mut input_names = std::collections::HashSet::new();
    for input in &module.inputs {
        if input_names.contains(&input.name) {
            return Err(format!("Duplicate input name detected: {}", input.name));
        }
        input_names.insert(&input.name);
    }
    
    // Check for duplicate output names
    let mut output_names = std::collections::HashSet::new();
    for output in &module.outputs {
        if output_names.contains(&output.name) {
            return Err(format!("Duplicate output name detected: {}", output.name));
        }
        output_names.insert(&output.name);
    }
    
    // Check for conflict between input names and output names at module level
    for input in &module.inputs {
        for output in &module.outputs {
            if input.name == output.name {
                return Err(format!("Module input and output share the same name: {}", input.name));
            }
        }
    }
    
    // Check for conflicts between module inputs/outputs and operation inputs/outputs
    for op in &module.operations {
        // Check operation inputs against module inputs
        for op_input in &op.inputs {
            for module_input in &module.inputs {
                if op_input.name == module_input.name {
                    return Err(format!("Operation input '{}' conflicts with module input", op_input.name));
                }
            }
            // Check operation inputs against module outputs
            for module_output in &module.outputs {
                if op_input.name == module_output.name {
                    return Err(format!("Operation input '{}' conflicts with module output", op_input.name));
                }
            }
        }
        
        // Check operation outputs against module inputs
        for op_output in &op.outputs {
            for module_input in &module.inputs {
                if op_output.name == module_input.name {
                    return Err(format!("Operation output '{}' conflicts with module input", op_output.name));
                }
            }
            // Check operation outputs against module outputs
            for module_output in &module.outputs {
                if op_output.name == module_output.name {
                    return Err(format!("Operation output '{}' conflicts with module output", op_output.name));
                }
            }
        }
    }

    Ok(())
}

/// Validates an entire module for consistency and correctness
pub fn validate_module(module: &crate::ir::Module) -> Result<(), String> {
    // First validate the basic shapes
    validate_module_shapes(module)?;
    
    // Validate uniqueness of names
    validate_module_uniqueness(module)?;
    
    // Additional module-level validations
    if module.name.is_empty() {
        return Err("Module name cannot be empty".to_string());
    }
    
    // Validate the lengths are reasonable
    if module.name.len() > 10_000_000 {
        return Err("Module name is unusually long".to_string());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Value, Type, Operation};
    

    #[test]
    fn test_validate_value_shape_valid() {
        let valid_value = Value {
            name: "valid".to_string(),
            ty: Type::F32,
            shape: vec![10, 20],
        };
        
        assert!(validate_value_shape(&valid_value).is_ok());
    }

    #[test]
    fn test_validate_value_shape_empty() {
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        
        assert!(validate_value_shape(&scalar).is_ok());
    }

    #[test]
    fn test_validate_value_shape_with_zero() {
        let zero_tensor = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5],
        };
        
        assert!(validate_value_shape(&zero_tensor).is_ok());
    }

    #[test]
    fn test_validate_type_basic() {
        assert!(validate_type(&Type::F32).is_ok());
        assert!(validate_type(&Type::I32).is_ok());
        assert!(validate_type(&Type::Bool).is_ok());
    }

    #[test]
    fn test_validate_type_nested() {
        let nested = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        
        assert!(validate_type(&nested).is_ok());
    }

    #[test]
    fn test_validate_operation_valid() {
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        op.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        assert!(validate_operation(&op).is_ok());
    }

    #[test]
    fn test_validate_operation_empty() {
        let op = Operation::new("noop");
        assert!(validate_operation(&op).is_ok());
    }
}