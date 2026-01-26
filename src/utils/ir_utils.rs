//! IR utility functions for the Impulse compiler

use crate::ir::{Value, Type, Operation};

/// Helper function to determine if a value represents a scalar (has no dimensions)
pub fn is_scalar(value: &Value) -> bool {
    value.shape.is_empty()
}

/// Helper function to determine if a value represents a vector (has exactly one dimension)
pub fn is_vector(value: &Value) -> bool {
    value.shape.len() == 1
}

/// Helper function to determine if a value represents a matrix (has exactly two dimensions)
pub fn is_matrix(value: &Value) -> bool {
    value.shape.len() == 2
}

/// Helper function to get the rank (number of dimensions) of a tensor
pub fn get_rank(value: &Value) -> usize {
    value.shape.len()
}

/// Helper function to get the total number of elements in a tensor
pub fn get_num_elements(value: &Value) -> Option<usize> {
    if value.shape.is_empty() {
        // Scalar
        Some(1)
    } else {
        value.shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }
}

/// Helper function to get the element type of a tensor (extracting from nested types if needed)
pub fn get_element_type(ty: &Type) -> &Type {
    match ty {
        Type::Tensor { element_type, .. } => get_element_type(element_type),
        _ => ty,
    }
}

/// Convert Type to string representation
pub fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::F32 => "f32".to_string(),
        Type::F64 => "f64".to_string(),
        Type::I32 => "i32".to_string(),
        Type::I64 => "i64".to_string(),
        Type::Bool => "bool".to_string(),
        Type::Tensor { element_type, shape } => {
            let element_str = type_to_string(element_type);
            let shape_str = shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
            format!("tensor<{}, [{}]>", element_str, shape_str)
        }
    }
}

/// Count operations by type in a module
pub fn count_operations_by_type(module: &crate::ir::Module) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    
    for op in &module.operations {
        *counts.entry(op.op_type.clone()).or_insert(0) += 1;
    }
    
    counts
}

/// Find operations by type in a module
pub fn find_operations_by_type<'a>(module: &'a crate::ir::Module, op_type: &str) -> Vec<&'a Operation> {
    module.operations.iter().filter(|op| op.op_type == op_type).collect()
}

/// Helper function to calculate tensor size in bytes based on type and shape
pub fn calculate_tensor_size(data_type: &Type, shape: &[usize]) -> Result<usize, String> {
    match data_type {
        Type::F32 => {
            let num_elements = if shape.is_empty() {
                // Scalar has 1 element
                1
            } else {
                shape.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| "Overflow in shape calculation".to_string())
                })?
            };
            num_elements.checked_mul(4).ok_or_else(|| "Overflow in final size calculation".to_string())
        },
        Type::F64 => {
            let num_elements = if shape.is_empty() {
                // Scalar has 1 element
                1
            } else {
                shape.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| "Overflow in shape calculation".to_string())
                })?
            };
            num_elements.checked_mul(8).ok_or_else(|| "Overflow in final size calculation".to_string())
        },
        Type::I32 => {
            let num_elements = if shape.is_empty() {
                // Scalar has 1 element
                1
            } else {
                shape.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| "Overflow in shape calculation".to_string())
                })?
            };
            num_elements.checked_mul(4).ok_or_else(|| "Overflow in final size calculation".to_string())
        },
        Type::I64 => {
            let num_elements = if shape.is_empty() {
                // Scalar has 1 element
                1
            } else {
                shape.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| "Overflow in shape calculation".to_string())
                })?
            };
            num_elements.checked_mul(8).ok_or_else(|| "Overflow in final size calculation".to_string())
        },
        Type::Bool => {
            let num_elements = if shape.is_empty() {
                // Scalar has 1 element
                1
            } else {
                shape.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| "Overflow in shape calculation".to_string())
                })?
            };
            num_elements.checked_mul(1).ok_or_else(|| "Overflow in final size calculation".to_string())
        },
        Type::Tensor { element_type, shape: inner_shape } => {
            // For nested tensors, we need to combine the outer shape with the inner shape
            // First calculate the number of elements in the outer shape
            let outer_elements = if shape.is_empty() {
                // If outer shape is empty, just consider the tensor as a single entity
                1
            } else {
                shape.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim).ok_or_else(|| "Overflow in outer shape calculation".to_string())
                })?
            };

            // Then calculate the size of the inner tensor
            let inner_tensor_size = calculate_tensor_size(element_type, inner_shape)?;

            // Total size is outer elements * inner tensor size
            outer_elements.checked_mul(inner_tensor_size)
                .ok_or_else(|| "Overflow in nested tensor size calculation".to_string())
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Value, Type};

    #[test]
    fn test_is_scalar() {
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(is_scalar(&scalar));

        let vector = Value {
            name: "vector".to_string(),
            ty: Type::F32,
            shape: vec![5],
        };
        assert!(!is_scalar(&vector));

        let matrix = Value {
            name: "matrix".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        assert!(!is_scalar(&matrix));
    }

    #[test]
    fn test_is_vector() {
        let vector = Value {
            name: "vector".to_string(),
            ty: Type::F32,
            shape: vec![5],
        };
        assert!(is_vector(&vector));

        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(!is_vector(&scalar));

        let matrix = Value {
            name: "matrix".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        assert!(!is_vector(&matrix));
    }

    #[test]
    fn test_is_matrix() {
        let matrix = Value {
            name: "matrix".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        assert!(is_matrix(&matrix));

        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert!(!is_matrix(&scalar));

        let three_d = Value {
            name: "3d".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        assert!(!is_matrix(&three_d));
    }

    #[test]
    fn test_get_rank() {
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(get_rank(&scalar), 0);

        let vector = Value {
            name: "vector".to_string(),
            ty: Type::F32,
            shape: vec![5],
        };
        assert_eq!(get_rank(&vector), 1);

        let matrix = Value {
            name: "matrix".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        assert_eq!(get_rank(&matrix), 2);

        let three_d = Value {
            name: "3d".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        assert_eq!(get_rank(&three_d), 3);
    }

    #[test]
    fn test_get_num_elements() {
        let scalar = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],
        };
        assert_eq!(get_num_elements(&scalar), Some(1));

        let vector = Value {
            name: "vector".to_string(),
            ty: Type::F32,
            shape: vec![5],
        };
        assert_eq!(get_num_elements(&vector), Some(5));

        let matrix = Value {
            name: "matrix".to_string(),
            ty: Type::F32,
            shape: vec![2, 3],
        };
        assert_eq!(get_num_elements(&matrix), Some(6));

        let zero_tensor = Value {
            name: "zero_tensor".to_string(),
            ty: Type::F32,
            shape: vec![2, 0, 4],
        };
        assert_eq!(get_num_elements(&zero_tensor), Some(0));
    }

    #[test]
    fn test_get_element_type() {
        // Direct types
        let f32_type = Type::F32;
        assert_eq!(get_element_type(&f32_type), &Type::F32);

        let i32_type = Type::I32;
        assert_eq!(get_element_type(&i32_type), &Type::I32);

        // Nested tensor
        let nested_tensor = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            }),
            shape: vec![3],
        };

        assert_eq!(get_element_type(&nested_tensor), &Type::F32);
    }
}