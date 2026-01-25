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

    #[test]
    fn test_deeply_nested_tensor_operations() {
        // Test creating and manipulating deeply nested tensor types
        let level1 = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2],
        };
        let level2 = Type::Tensor {
            element_type: Box::new(level1),
            shape: vec![3],
        };
        let level3 = Type::Tensor {
            element_type: Box::new(level2),
            shape: vec![4],
        };

        // Verify nested structure
        match &level3 {
            Type::Tensor { element_type: boxed_level2, shape: outer_shape } => {
                assert_eq!(outer_shape, &vec![4]);
                
                // Access nested type
                match boxed_level2.as_ref() {
                    Type::Tensor { element_type: boxed_level1, shape: middle_shape } => {
                        assert_eq!(middle_shape, &vec![3]);
                        
                        // Access innermost type
                        match boxed_level1.as_ref() {
                            Type::Tensor { element_type: inner_type, shape: inner_shape } => {
                                assert_eq!(inner_shape, &vec![2]);
                                
                                match inner_type.as_ref() {
                                    Type::F32 => {}, // Success
                                    _ => panic!("Innermost type should be F32"),
                                }
                            },
                            _ => panic!("Middle type should be Tensor<F32, [2]>"),
                        }
                    },
                    _ => panic!("Outer type should be Tensor<Tensor<F32, [2]>, [3]>"),
                }
            },
            _ => panic!("Level3 type should be Tensor<...>"),
        }
    }

    #[test]
    fn test_tensor_comparison_edge_cases() {
        // Different element types
        let f32_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        };
        let f64_tensor = Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![2, 2],
        };
        assert_ne!(f32_tensor, f64_tensor);

        // Different shapes
        let shape1_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        };
        let shape2_tensor = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![4],
        };
        assert_ne!(shape1_tensor, shape2_tensor);

        // Same everything should be equal
        let tensor_a = Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![5, 5],
        };
        let tensor_b = Type::Tensor {
            element_type: Box::new(Type::I32),
            shape: vec![5, 5],
        };
        assert_eq!(tensor_a, tensor_b);
    }

    #[test]
    fn test_complex_value_shapes() {
        let complex_value = Value {
            name: "complex_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224],  // Common in image processing
        };
        
        assert_eq!(complex_value.name, "complex_tensor");
        assert_eq!(complex_value.ty, Type::F32);
        assert_eq!(complex_value.shape, vec![1, 3, 224, 224]);
        
        // Calculate total elements
        let total_elements: usize = complex_value.shape.iter().product();
        assert_eq!(total_elements, 1 * 3 * 224 * 224); // 150,528 elements
    }

    #[test]
    fn test_empty_and_zero_dimension_tensors() {
        // Test scalar value (0-dimensional tensor)
        let scalar = Value {
            name: "scalar_value".to_string(),
            ty: Type::F32,
            shape: vec![],  // Empty shape means scalar
        };
        
        assert_eq!(scalar.name, "scalar_value");
        assert_eq!(scalar.ty, Type::F32);
        assert_eq!(scalar.shape.len(), 0);
        assert!(scalar.shape.is_empty());
        
        // Scalar has 1 element
        let scalar_elements: usize = scalar.shape.iter().product();
        assert_eq!(scalar_elements, 1);

        // Test tensor with zero in dimensions (represents empty tensor)
        let zero_dim_tensor = Value {
            name: "zero_dim_tensor".to_string(),
            ty: Type::I32,
            shape: vec![5, 0, 10],  // Contains 0, so total elements = 0
        };
        
        assert_eq!(zero_dim_tensor.shape, vec![5, 0, 10]);
        
        // Zero-dimensional tensor has 0 elements
        let zero_elements: usize = zero_dim_tensor.shape.iter().product();
        assert_eq!(zero_elements, 0);

        // Test another zero-dimensional case
        let zero_tensor = Value {
            name: "zero_tensor".to_string(),
            ty: Type::Bool,
            shape: vec![0],
        };
        
        assert_eq!(zero_tensor.shape, vec![0]);
        let zero_tensor_elements: usize = zero_tensor.shape.iter().product();
        assert_eq!(zero_tensor_elements, 0);

        // Test 1-dimensional zero-length tensor
        let empty_1d = Value {
            name: "empty_1d".to_string(),
            ty: Type::F64,
            shape: vec![0],
        };
        
        assert_eq!(empty_1d.shape, vec![0]);
        assert_eq!(empty_1d.shape[0], 0);
    }

    #[test]
    fn test_operation_with_complex_structure() {
        use std::collections::HashMap;
        
        let mut complex_op = Operation::new("conv2d_complex");
        complex_op.inputs.push(Value {
            name: "input_image".to_string(),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224],
        });
        complex_op.inputs.push(Value {
            name: "weights".to_string(),
            ty: Type::F32,
            shape: vec![64, 3, 7, 7],
        });
        complex_op.outputs.push(Value {
            name: "feature_map".to_string(),
            ty: Type::F32,
            shape: vec![1, 64, 112, 112],
        });
        
        let mut attrs = HashMap::new();
        attrs.insert("padding".to_string(), Attribute::Int(3));
        attrs.insert("stride".to_string(), Attribute::Int(2));
        attrs.insert("activation".to_string(), Attribute::String("relu".to_string()));
        complex_op.attributes = attrs;
        
        assert_eq!(complex_op.op_type, "conv2d_complex");
        assert_eq!(complex_op.inputs.len(), 2);
        assert_eq!(complex_op.outputs.len(), 1);
        assert_eq!(complex_op.attributes.len(), 3);
        assert!(complex_op.attributes.contains_key("padding"));
        assert!(complex_op.attributes.contains_key("stride"));
        assert!(complex_op.attributes.contains_key("activation"));
    }
}