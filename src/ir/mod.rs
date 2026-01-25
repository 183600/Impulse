//! Intermediate Representation (IR) module for the Impulse compiler
//! Defines the core data structures for representing computation graphs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A module represents a complete computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub name: String,
    pub operations: Vec<Operation>,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
}

/// An operation in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub op_type: String,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub attributes: HashMap<String, Attribute>,
}

/// A value in the computation graph (tensor/variable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    pub name: String,
    pub ty: Type,
    pub shape: Vec<usize>,
}

/// Type represents data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Type {
    F32,
    F64,
    I32,
    I64,
    Bool,
    Tensor { element_type: Box<Type>, shape: Vec<usize> },
}

/// Attributes for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Attribute {
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<Attribute>),
    Bool(bool),
}


impl Module {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_operation(&mut self, op: Operation) {
        self.operations.push(op);
    }
}

impl Operation {
    pub fn new(op_type: &str) -> Self {
        Self {
            op_type: op_type.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let module = Module::new("test");
        assert_eq!(module.name, "test");
        assert!(module.operations.is_empty());
        assert!(module.inputs.is_empty());
        assert!(module.outputs.is_empty());
    }

    #[test]
    fn test_operation_creation() {
        let op = Operation::new("add");
        assert_eq!(op.op_type, "add");
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(op.attributes.is_empty());
    }

    #[test]
    fn test_value_creation() {
        let value = Value {
            name: "test_val".to_string(),
            ty: Type::F32,
            shape: vec![2, 3, 4],
        };
        assert_eq!(value.name, "test_val");
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_nested_tensor() {
        let nested = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        };
        match nested {
            Type::Tensor { element_type, shape } => {
                match element_type.as_ref() {
                    Type::F32 => {}, // Success
                    _ => panic!("Expected F32 as element type"),
                }
                assert_eq!(shape, vec![2, 2]);
            },
            _ => panic!("Expected Tensor type"),
        }
    }

    #[test]
    fn test_attribute_types() {
        let attrs = [
            Attribute::Int(42),
            Attribute::Float(3.14),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
        ];

        match attrs[0] {
            Attribute::Int(42) => {}, // Success
            _ => panic!("Expected Int(42)"),
        }

        match attrs[1] {
            Attribute::Float(val) if (val - 3.14).abs() < f64::EPSILON => {}, // Success
            _ => panic!("Expected Float(3.14)"),
        }

        match &attrs[2] {
            Attribute::String(s) if s == "test" => {}, // Success
            _ => panic!("Expected String(\"test\")"),
        }

        match attrs[3] {
            Attribute::Bool(true) => {}, // Success
            _ => panic!("Expected Bool(true)"),
        }
    }

    #[test]
    fn test_attribute_array() {
        let inner_attrs = vec![Attribute::Int(1), Attribute::Int(2)];
        let array_attr = Attribute::Array(inner_attrs);
        
        match array_attr {
            Attribute::Array(vec) => {
                assert_eq!(vec.len(), 2);
                match vec[0] {
                    Attribute::Int(1) => {}, // Success
                    _ => panic!("Expected first element to be Int(1)"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    #[test]
    fn test_module_add_operation() {
        let mut module = Module::new("test");
        let op = Operation::new("add");
        module.add_operation(op);
        
        assert_eq!(module.operations.len(), 1);
        assert_eq!(module.operations[0].op_type, "add");
    }

    #[test]
    fn test_empty_shape() {
        let value = Value {
            name: "scalar".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape represents a scalar
        };
        assert!(value.shape.is_empty());
    }

    #[test]
    fn test_zero_in_shape() {
        let value = Value {
            name: "zero_dim".to_string(),
            ty: Type::F32,
            shape: vec![10, 0, 5],  // Contains zero, making total size 0
        };
        assert_eq!(value.shape, vec![10, 0, 5]);
        // This represents a tensor with zero elements
        let total_size: usize = value.shape.iter().product();
        assert_eq!(total_size, 0);
    }

    #[test]
    fn test_deeply_nested_tensor() {
        // Create a deeply nested tensor type: tensor<tensor<f32, [2]>, [3]>
        let inner_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        };
        let outer_type = Type::Tensor {
            element_type: Box::new(inner_type),
            shape: vec![3],
        };

        match outer_type {
            Type::Tensor { element_type: outer_element, shape: outer_shape } => {
                assert_eq!(outer_shape, vec![3]);
                
                // Check the nested type
                match outer_element.as_ref() {
                    Type::Tensor { element_type: inner_element, shape: inner_shape } => {
                        assert_eq!(inner_shape, &vec![2]);
                        
                        // Check the innermost type
                        match inner_element.as_ref() {
                            Type::F32 => {}, // Success
                            _ => panic!("Expected F32 as innermost type"),
                        }
                    },
                    _ => panic!("Expected Tensor as inner type"),
                }
            },
            _ => panic!("Expected Tensor as outer type"),
        }
    }

    #[test]
    fn test_recursive_tensor_equivalence() {
        // Two equivalent nested tensor types should be equal
        let type1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        
        let type2 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3],
        };
        
        assert_eq!(type1, type2);
        
        // Two different nested tensor types should not be equal
        let type3 = Type::Tensor {
            element_type: Box::new(Type::I32),  // Different element type
            shape: vec![2, 3],
        };
        
        assert_ne!(type1, type3);
        
        // Different shapes
        let type4 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 4],  // Different shape
        };
        
        assert_ne!(type1, type4);
    }

    #[test]
    fn test_copy_cloned_nested_types() {
        let original = Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![10, 20],
        };
        
        // Test cloning
        let cloned = original.clone();
        assert_eq!(original, cloned);
        
        // Verify both are tensor types with same properties
        match (&original, &cloned) {
            (Type::Tensor { shape: orig_shape, .. }, Type::Tensor { shape: cloned_shape, .. }) => {
                assert_eq!(orig_shape, cloned_shape);
            },
            _ => panic!("Both should be tensor types"),
        }
    }
}