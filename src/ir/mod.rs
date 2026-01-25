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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Extension trait for Type operations
pub trait TypeExtensions {
    fn is_valid_type(&self) -> bool;
}

impl TypeExtensions for Type {
    fn is_valid_type(&self) -> bool {
        match self {
            Type::F32 | Type::F64 | Type::I32 | Type::I64 | Type::Bool => true,
            Type::Tensor { element_type, .. } => {
                // Recursively validate the nested type
                element_type.is_valid_type()
            }
        }
    }
}

/// Attributes for operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    
    #[test]
    fn test_large_tensor_dimensions() {
        // Test very large tensor dimensions to check for potential overflow in calculations
        // While this doesn't cause overflow in the current implementation due to usize, 
        // it tests the limits of the system
        let huge_shape = vec![1000, 1000, 100];  // 100M elements
        let value = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: huge_shape,
        };
        
        assert_eq!(value.shape.len(), 3);
        // Ensure the shape values are preserved
        assert_eq!(value.shape[0], 1000);
        assert_eq!(value.shape[1], 1000);
        assert_eq!(value.shape[2], 100);
    }
    
    #[test]
    fn test_deep_recursion_in_tensor_types() {
        // Test creating a deeply nested tensor type to make sure we don't hit stack limits
        let mut current_type = Type::F32;
        for _ in 0..100 {  // Create 100 levels of nesting
            current_type = Type::Tensor {
                element_type: Box::new(current_type),
                shape: vec![2],
            };
        }
        
        // Ensure the final type is valid
        if let Type::Tensor { shape, .. } = &current_type {
            assert_eq!(shape, &vec![2]);
        } else {
            panic!("Expected a tensor type after nesting");
        }
        
        // Test cloning of this deeply nested type
        let cloned_type = current_type.clone();
        assert_eq!(current_type, cloned_type);
    }
    
    #[test]
    fn test_module_with_many_operations() {
        // Test creating a module with many operations to test memory management
        let mut module = Module::new("large_module");
        
        // Add 10,000 operations to the module
        for i in 0..10000 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
            
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 10000);
        assert_eq!(module.name, "large_module");
        
        // Verify some operations still have correct data
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[9999].op_type, "op_9999");
        assert_eq!(module.operations[5000].op_type, "op_5000");
    }
    
    #[test]
    fn test_very_large_tensor_sizes() {
        // Test tensor with very large size to test bounds checking
        let large_value = Value {
            name: "large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1000, 1000],  // 1 million elements
        };
        
        assert_eq!(large_value.shape, vec![1000, 1000]);
        let product: usize = large_value.shape.iter().product();
        assert_eq!(product, 1_000_000);
        
        // Test with even larger sizes
        let very_large_value = Value {
            name: "very_large_tensor".to_string(),
            ty: Type::F32,
            shape: vec![10_000, 10_000],  // 100 million elements
        };
        
        assert_eq!(very_large_value.shape, vec![10_000, 10_000]);
        let large_product: usize = very_large_value.shape.iter().product();
        assert_eq!(large_product, 100_000_000);
    }
    
    #[test]
    fn test_extreme_shaped_tensors() {
        // Test tensors with extreme aspect ratios (very wide or very tall)
        let thin_tensor = Value {
            name: "thin_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, 1_000_000],  // 1 row, 1M columns
        };
        
        assert_eq!(thin_tensor.shape, vec![1, 1_000_000]);
        let thin_product: usize = thin_tensor.shape.iter().product();
        assert_eq!(thin_product, 1_000_000);
        
        let tall_tensor = Value {
            name: "tall_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1_000_000, 1],  // 1M rows, 1 column
        };
        
        assert_eq!(tall_tensor.shape, vec![1_000_000, 1]);
        let tall_product: usize = tall_tensor.shape.iter().product();
        assert_eq!(tall_product, 1_000_000);
        
        // Test single dimensional tensors with extreme sizes
        let long_vector = Value {
            name: "long_vector".to_string(),
            ty: Type::F32,
            shape: vec![10_000_000],  // 10 million element vector
        };
        
        assert_eq!(long_vector.shape, vec![10_000_000]);
        let long_product: usize = long_vector.shape.iter().product();
        assert_eq!(long_product, 10_000_000);
    }

    #[test]
    fn test_operation_with_extremely_long_names() {
        // Test creating an operation with an extremely long name
        let extremely_long_name = "a".repeat(10_000); // 10k character name
        let op = Operation::new(&extremely_long_name);
        
        assert_eq!(op.op_type, extremely_long_name);
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
    }

    #[test]
    fn test_value_with_extremely_long_name() {
        // Test creating a value with an extremely long name
        let extremely_long_name = "x".repeat(10_000); // 10k character name
        let value = Value {
            name: extremely_long_name.clone(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name, extremely_long_name);
        assert_eq!(value.ty, Type::F32);
        assert_eq!(value.shape, vec![1, 2, 3]);
    }

    #[test]
    fn test_integer_overflow_in_shape_products() {
        // Test cases that might cause integer overflow when computing shape products
        // Use values that would cause overflow if multiplied naively
        // Instead, we'll test with large but safe values that still stress the system
        
        // Test with shapes that have large dimensions but still fit in usize
        let huge_but_safe_shape = vec![100_000, 100_000];  // Would be 10 billion elements
        let value = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: huge_but_safe_shape,
        };
        
        // Calculate the product safely
        let product: usize = value.shape.iter().copied().product();
        assert_eq!(product, 10_000_000_000); // 10 billion
        
        // Test with a shape that would definitely result in 0 due to containing 0
        let zero_shape = vec![1000, 0, 5000];
        let zero_value = Value {
            name: "zero_tensor".to_string(),
            ty: Type::I64,
            shape: zero_shape,
        };
        
        let zero_product: usize = zero_value.shape.iter().copied().product();
        assert_eq!(zero_product, 0);
    }

    #[test]
    fn test_operation_with_max_possible_inputs_outputs() {
        use std::collections::HashMap;
        
        // Create an operation with a very large number of inputs and outputs to test limits
        let mut op = Operation::new("max_io_op");
        
        // Add a large number of inputs
        for i in 0..50_000 {
            op.inputs.push(Value {
                name: format!("input_{:08}", i), // Zero-padded to make unique
                ty: Type::F32,
                shape: vec![1], // Minimal shape
            });
        }
        
        // Add a large number of outputs  
        for i in 0..25_000 {
            op.outputs.push(Value {
                name: format!("output_{:08}", i), // Zero-padded to make unique
                ty: Type::F32,
                shape: vec![1], // Minimal shape
            });
        }
        
        // Add many attributes too
        let mut attrs = HashMap::new();
        for i in 0..10_000 {
            attrs.insert(
                format!("attribute_{:06}", i), 
                Attribute::String(format!("value_{}", i))
            );
        }
        op.attributes = attrs;
        
        assert_eq!(op.inputs.len(), 50_000);
        assert_eq!(op.outputs.len(), 25_000);
        assert_eq!(op.attributes.len(), 10_000);
    }

    #[test]
    fn test_module_with_extremely_long_name() {
        // Test creating a module with an extremely long name
        let extremely_long_name = "module_".repeat(1000) + "end"; // Around 7k characters
        let module = Module::new(extremely_long_name.clone());
        
        assert_eq!(module.name, extremely_long_name);
        assert_eq!(module.operations.len(), 0);
        assert_eq!(module.inputs.len(), 0);
        assert_eq!(module.outputs.len(), 0);
    }

    #[test]
    fn test_integer_overflow_with_large_dimensions() {
        // Test potential integer overflow in shape products
        // Use values that when multiplied together would exceed usize::MAX for most systems
        // This tests for potential arithmetic overflow issues
        
        // Since actual overflow is hard to achieve on 64-bit systems, 
        // we focus on testing large but realistic multiplications
        
        // Test with a shape that would result in a very large product (but not necessarily overflowing)
        let huge_tensor = Value {
            name: "huge_tensor".to_string(),
            ty: Type::F32,
            shape: vec![46340, 46340],  // 46340^2 ≈ 2.1 billion, close to u32::MAX
        };
        
        assert_eq!(huge_tensor.shape, vec![46340, 46340]);
        let product: usize = huge_tensor.shape.iter().product();
        assert_eq!(product, 46340 * 46340);  // ~2.1 billion
    }

    #[test]
    fn test_operation_with_max_inputs_outputs() {
        // Create an operation with many inputs and outputs to test limits
        let mut op = Operation::new("multimodal_op");
        
        // Add many inputs (testing potential memory allocation issues)
        for i in 0..1000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
        }
        
        // Add many outputs
        for i in 0..1000 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![10, 10],
            });
        }
        
        assert_eq!(op.inputs.len(), 1000);
        assert_eq!(op.outputs.len(), 1000);
        assert_eq!(op.op_type, "multimodal_op");
    }
    
    #[test]
    fn test_attribute_equality_edge_cases() {
        // Test equality behavior of attributes in various edge cases
        let attr_int1 = Attribute::Int(42);
        let attr_int2 = Attribute::Int(42);
        let attr_int3 = Attribute::Int(43);
        
        assert_eq!(attr_int1, attr_int2);
        assert_ne!(attr_int1, attr_int3);
        
        let attr_float1 = Attribute::Float(3.14);
        let attr_float2 = Attribute::Float(3.14);
        let attr_float3 = Attribute::Float(3.15);
        
        // Compare floats carefully due to floating point precision
        assert_eq!(attr_float1, attr_float2);
        assert_ne!(attr_float1, attr_float3);
        
        let attr_str1 = Attribute::String("hello".to_string());
        let attr_str2 = Attribute::String("hello".to_string());
        let attr_str3 = Attribute::String("world".to_string());
        
        assert_eq!(attr_str1, attr_str2);
        assert_ne!(attr_str1, attr_str3);
        assert_ne!(attr_str2, attr_str3);
    }

    #[test]
    fn test_string_attribute_with_special_characters() {
        // Test string attributes containing special characters
        let special_string = "Special chars: \n\t\r\u{0000}!@#$%^&*()_+{}[]|\\:\";'<>?,./";
        let attr = Attribute::String(special_string.to_string());
        
        match attr {
            Attribute::String(s) => assert_eq!(s, special_string),
            _ => panic!("Expected String attribute"),
        }
    }

    #[test]
    fn test_array_attribute_with_mixed_types() {
        // Test array attributes containing mixed attribute types
        let mixed_array = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(2.5),
            Attribute::String("three".to_string()),
            Attribute::Bool(true),
        ]);
        
        match mixed_array {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 4);
                
                match arr[0] {
                    Attribute::Int(1) => (),
                    _ => panic!("First element should be Int(1)"),
                }
                
                match arr[1] {
                    Attribute::Float(val) if (val - 2.5).abs() < f64::EPSILON => (),
                    _ => panic!("Second element should be Float(2.5)"),
                }
                
                match &arr[2] {
                    Attribute::String(s) if s == "three" => (),
                    _ => panic!("Third element should be String(\"three\")"),
                }
                
                match arr[3] {
                    Attribute::Bool(true) => (),
                    _ => panic!("Fourth element should be Bool(true)"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    #[test]
    fn test_nested_array_attributes() {
        // Test arrays inside arrays (nested arrays)
        let nested_array = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
            ]),
            Attribute::Array(vec![
                Attribute::Int(3),
                Attribute::Int(4),
            ]),
        ]);
        
        match nested_array {
            Attribute::Array(outer_arr) => {
                assert_eq!(outer_arr.len(), 2);
                
                match &outer_arr[0] {
                    Attribute::Array(inner_arr) => {
                        assert_eq!(inner_arr.len(), 2);
                        if let Attribute::Int(1) = inner_arr[0] {}
                        else { panic!("Expected Int(1)"); }
                        
                        if let Attribute::Int(2) = inner_arr[1] {}
                        else { panic!("Expected Int(2)"); }
                    },
                    _ => panic!("Expected Array as first element"),
                }
                
                match &outer_arr[1] {
                    Attribute::Array(inner_arr) => {
                        assert_eq!(inner_arr.len(), 2);
                        if let Attribute::Int(3) = inner_arr[0] {}
                        else { panic!("Expected Int(3)"); }
                        
                        if let Attribute::Int(4) = inner_arr[1] {}
                        else { panic!("Expected Int(4)"); }
                    },
                    _ => panic!("Expected Array as second element"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }
}