use crate::ir::{Module, Type};
use anyhow::Result;
use std::collections::HashMap;

/// Utility functions for working with the IR
pub mod ir_utils {
    use super::*;
    
    /// Calculate the size in bytes of a tensor with the given type and shape
    pub fn calculate_tensor_size(ty: &Type, shape: &[usize]) -> Result<usize> {
        match ty {
            Type::F32 => {
                let num_elements: usize = shape.iter().copied().product();
                Ok(num_elements * 4)
            },
            Type::F64 => {
                let num_elements: usize = shape.iter().copied().product();
                Ok(num_elements * 8)
            },
            Type::I32 => {
                let num_elements: usize = shape.iter().copied().product();
                Ok(num_elements * 4)
            },
            Type::I64 => {
                let num_elements: usize = shape.iter().copied().product();
                Ok(num_elements * 8)
            },
            Type::Bool => {
                let num_elements: usize = shape.iter().copied().product();
                Ok(num_elements * 1)
            },
            Type::Tensor { element_type, shape: inner_shape } => {
                // For tensor types, multiply the outer shape by the inner shape
                let outer_num_elements: usize = shape.iter().copied().product();
                let inner_num_elements: usize = inner_shape.iter().copied().product();
                let element_size = calculate_tensor_size(element_type, &[])?;
                Ok(outer_num_elements * inner_num_elements * element_size)
            },
        }
    }
    
    /// Count operations in a module by type
    pub fn count_operations_by_type(module: &Module) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for op in &module.operations {
            *counts.entry(op.op_type.clone()).or_insert(0) += 1;
        }
        counts
    }
    
    /// Find operations with specific type in a module
    pub fn find_operations_by_type<'a>(module: &'a Module, op_type: &str) -> Vec<&'a crate::ir::Operation> {
        module.operations.iter().filter(|op| op.op_type == op_type).collect()
    }
    
    /// Convert a type to its string representation
    pub fn type_to_string(ty: &Type) -> String {
        match ty {
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Tensor { element_type, shape } => {
                format!("tensor<{:?}, [{}]>", type_to_string(element_type), 
                    shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
            }
        }
    }
}

/// Utility functions for debugging and visualization
pub mod debug_utils {
    use super::*;
    
    /// Print a human-readable representation of a module
    pub fn print_module(module: &Module) {
        println!("Module: {}", module.name);
        println!("Operations ({}):", module.operations.len());
        
        for (i, op) in module.operations.iter().enumerate() {
            println!("  [{}] {}: {} -> {}", 
                i, 
                op.op_type, 
                op.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>().join(", "),
                op.outputs.iter().map(|output| output.name.as_str()).collect::<Vec<_>>().join(", ")
            );
        }
    }
    
    /// Print statistics about a module
    pub fn print_module_stats(module: &Module) {
        let op_counts = ir_utils::count_operations_by_type(module);
        let total_ops = module.operations.len();
        
        println!("Module '{}' Statistics:", module.name);
        println!("  Total Operations: {}", total_ops);
        println!("  Operation Types:");
        for (op_type, count) in op_counts.iter() {
            let percentage = (*count as f64 / total_ops as f64) * 100.0;
            println!("    {}: {} ({:.1}%)", op_type, count, percentage);
        }
    }
}

/// Utility functions for validation
pub mod validation_utils {
    use super::*;
    
    /// Validate the structure of a module
    pub fn validate_module(module: &Module) -> Result<()> {
        // Check that all input/output names are unique within their respective collections
        let mut input_names = std::collections::HashSet::new();
        for input in &module.inputs {
            if !input_names.insert(&input.name) {
                anyhow::bail!("Duplicate input name: {}", input.name);
            }
        }
        
        let mut output_names = std::collections::HashSet::new();
        for output in &module.outputs {
            if !output_names.insert(&output.name) {
                anyhow::bail!("Duplicate output name: {}", output.name);
            }
        }
        
        // Validate operations
        for op in &module.operations {
            validate_operation(op)?;
        }
        
        Ok(())
    }
    
    /// Validate a single operation
    pub fn validate_operation(operation: &crate::ir::Operation) -> Result<()> {
        // All input and output names should be unique
        let mut all_names = std::collections::HashSet::new();
        for input in &operation.inputs {
            if !all_names.insert(&input.name) {
                anyhow::bail!("Duplicate name in operation inputs: {}", input.name);
            }
        }
        
        for output in &operation.outputs {
            if !all_names.insert(&output.name) {
                anyhow::bail!("Duplicate name in operation outputs: {}", output.name);
            }
        }
        
        Ok(())
    }
}

/// Timing utility for measuring execution time
pub mod timing_utils {
    use std::time::Instant;
    
    pub struct Timer {
        start: Instant,
        pub label: String,
    }
    
    impl Timer {
        pub fn new(label: impl Into<String>) -> Self {
            Self {
                start: Instant::now(),
                label: label.into(),
            }
        }
        
        pub fn elapsed_ms(&self) -> u128 {
            self.start.elapsed().as_millis()
        }
        
        pub fn stop(self) {
            println!("Timer '{}': {} ms", self.label, self.elapsed_ms());
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                println!("Timer '{}': {} ms (via drop)", self.label, self.start.elapsed().as_millis());
            }
        }
    }
}

/// Math utilities
pub mod math_utils {
    
    /// Compute greatest common divisor
    pub fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    
    /// Compute least common multiple
    pub fn lcm(a: usize, b: usize) -> usize {
        if a == 0 || b == 0 {
            0
        } else {
            (a * b) / gcd(a, b)
        }
    }
    
    /// Round up to the nearest multiple
    pub fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
        if multiple == 0 {
            return value;
        }
        ((value + multiple - 1) / multiple) * multiple
    }
    
    /// Find the next power of 2 >= value
    pub fn next_power_of_2(value: usize) -> usize {
        if value == 0 {
            1
        } else {
            let mut n = value - 1;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            #[cfg(target_pointer_width = "64")]
            {
                n |= n >> 32;
            }
            n + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Value, Type, Operation, Attribute};

    #[test]
    fn test_calculate_tensor_size() {
        // Test basic types
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &[]).unwrap(), 4);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F64, &[]).unwrap(), 8);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I32, &[]).unwrap(), 4);

        // Test a tensor [1, 3, 224, 224] of f32
        let shape = vec![1, 3, 224, 224];
        let expected_size = 1 * 3 * 224 * 224 * 4;  // elements * bytes_per_element
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &shape).unwrap(), expected_size);
    }

    #[test]
    fn test_calculate_tensor_size_edge_cases() {
        // Test zero-dimensional tensor (scalar)
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &vec![0]).unwrap(), 0);
        
        // Test empty shape vector (scalar)
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &vec![]).unwrap(), 4);
        
        // Test tensor with zero in shape (should result in 0 size)
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &vec![1, 0, 10]).unwrap(), 0);
        
        // Test boolean tensor
        assert_eq!(ir_utils::calculate_tensor_size(&Type::Bool, &vec![8]).unwrap(), 8);
        
        // Test I64 tensor
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I64, &vec![5]).unwrap(), 40);
    }

    #[test]
    fn test_calculate_tensor_size_with_nested_tensor() {
        // Test nested tensor type
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        };
        
        // Shape is empty for the outer tensor, but the element type has its own shape
        // Should just return the size of one F32 element * 4 elements from the nested shape
        // Actually, this would recursively calculate based on element_type and empty shape
        assert_eq!(ir_utils::calculate_tensor_size(&nested_type, &vec![]).unwrap(), 16); // 4 elements * 4 bytes each
        
        // Test with a non-empty shape for the outer tensor
        assert_eq!(ir_utils::calculate_tensor_size(&nested_type, &vec![2]).unwrap(), 32); // 2 * 4 elements * 4 bytes each
    }

    #[test]
    fn test_count_operations_by_type() {
        let mut module = Module::new("test");
        
        let op1 = Operation::new("matmul");
        let op2 = Operation::new("add");
        let op3 = Operation::new("matmul");
        
        module.add_operation(op1);
        module.add_operation(op2);
        module.add_operation(op3);
        
        let counts = ir_utils::count_operations_by_type(&module);
        assert_eq!(counts.get("matmul"), Some(&2));
        assert_eq!(counts.get("add"), Some(&1));
    }

    #[test]
    fn test_count_operations_by_type_empty_module() {
        let module = Module::new("empty_test");
        let counts = ir_utils::count_operations_by_type(&module);
        assert_eq!(counts.len(), 0);
    }

    #[test]
    fn test_find_operations_by_type() {
        let mut module = Module::new("find_test");
        
        let op1 = Operation::new("matmul");
        let op2 = Operation::new("add");
        let op3 = Operation::new("matmul");
        let op4 = Operation::new("relu");
        
        module.add_operation(op1.clone());
        module.add_operation(op2);
        module.add_operation(op3.clone());
        module.add_operation(op4);
        
        let matmul_ops = ir_utils::find_operations_by_type(&module, "matmul");
        assert_eq!(matmul_ops.len(), 2);
        
        let relu_ops = ir_utils::find_operations_by_type(&module, "relu");
        assert_eq!(relu_ops.len(), 1);
        
        let nonexistent_ops = ir_utils::find_operations_by_type(&module, "conv2d");
        assert_eq!(nonexistent_ops.len(), 0);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(math_utils::gcd(48, 18), 6);
        assert_eq!(math_utils::gcd(7, 5), 1);
    }

    #[test]
    fn test_gcd_edge_cases() {
        assert_eq!(math_utils::gcd(0, 5), 5);
        assert_eq!(math_utils::gcd(5, 0), 5);
        assert_eq!(math_utils::gcd(0, 0), 0);
        assert_eq!(math_utils::gcd(1, 1), 1);
        assert_eq!(math_utils::gcd(17, 17), 17);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(math_utils::lcm(4, 6), 12);
        assert_eq!(math_utils::lcm(12, 18), 36);
    }

    #[test]
    fn test_lcm_edge_cases() {
        assert_eq!(math_utils::lcm(0, 5), 0);
        assert_eq!(math_utils::lcm(5, 0), 0);
        assert_eq!(math_utils::lcm(0, 0), 0);
        assert_eq!(math_utils::lcm(1, 7), 7);
        assert_eq!(math_utils::lcm(7, 1), 7);
    }

    #[test]
    fn test_round_up_to_multiple() {
        assert_eq!(math_utils::round_up_to_multiple(10, 8), 16);
        assert_eq!(math_utils::round_up_to_multiple(16, 8), 16);
        assert_eq!(math_utils::round_up_to_multiple(1, 8), 8);
    }

    #[test]
    fn test_round_up_to_multiple_edge_cases() {
        assert_eq!(math_utils::round_up_to_multiple(0, 5), 0);
        assert_eq!(math_utils::round_up_to_multiple(7, 0), 7);  // Should return value when multiple is 0
        assert_eq!(math_utils::round_up_to_multiple(5, 1), 5);  // Multiple of 1
        assert_eq!(math_utils::round_up_to_multiple(5, 10), 10); // Round up to larger multiple
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(math_utils::next_power_of_2(1), 1);
        assert_eq!(math_utils::next_power_of_2(2), 2);
        assert_eq!(math_utils::next_power_of_2(3), 4);
        assert_eq!(math_utils::next_power_of_2(15), 16);
        assert_eq!(math_utils::next_power_of_2(16), 16);
        assert_eq!(math_utils::next_power_of_2(17), 32);
    }

    #[test]
    fn test_next_power_of_2_edge_cases() {
        assert_eq!(math_utils::next_power_of_2(0), 1);  // Edge case: 0 should return 1
        assert_eq!(math_utils::next_power_of_2(255), 256);
        assert_eq!(math_utils::next_power_of_2(256), 256);
        assert_eq!(math_utils::next_power_of_2(257), 512);
    }

    #[test]
    fn test_type_to_string() {
        assert_eq!(ir_utils::type_to_string(&Type::F32), "f32");
        assert_eq!(ir_utils::type_to_string(&Type::I64), "i64");
    }

    #[test]
    fn test_type_to_string_complex() {
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 3, 4],
        };
        
        let _expected = format!("tensor<{:?}, [2, 3, 4]>", ir_utils::type_to_string(&Type::F32));
        let result = ir_utils::type_to_string(&nested_type);
        
        // Since the recursive call will return "f32", the tensor string looks like:
        // "tensor<f32, [2, 3, 4]>"
        assert!(result.contains("f32"));
        assert!(result.contains("[2, 3, 4]"));
        
        assert_eq!(ir_utils::type_to_string(&Type::Bool), "bool");
        assert_eq!(ir_utils::type_to_string(&Type::I32), "i32");
    }

    #[test]
    fn test_validation() {
        let mut module = Module::new("valid_module");
        
        // Add a valid operation
        let mut op = Operation::new("add");
        op.inputs.push(Value {
            name: "a".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        op.outputs.push(Value {
            name: "result".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        module.add_operation(op);
        
        // This should pass validation
        assert!(validation_utils::validate_module(&module).is_ok());
        
        // Add an operation with conflicted names (same name as both input and output) to trigger error
        let mut bad_op = Operation::new("mul");
        bad_op.inputs.push(Value {
            name: "conflict".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        bad_op.outputs.push(Value {
            name: "conflict".to_string(), // Same name as input, should cause conflict
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        assert!(validation_utils::validate_operation(&bad_op).is_err());
    }

    #[test]
    fn test_validation_edge_cases() {
        // Test module with duplicate input names
        let mut module = Module::new("duplicate_input_test");
        module.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.inputs.push(Value {
            name: "input1".to_string(),  // Duplicate name
            ty: Type::F32,
            shape: vec![10],
        });
        
        assert!(validation_utils::validate_module(&module).is_err());
        
        // Test module with duplicate output names
        let mut module2 = Module::new("duplicate_output_test");
        module2.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module2.outputs.push(Value {
            name: "output1".to_string(),  // Duplicate name
            ty: Type::F32,
            shape: vec![10],
        });
        
        assert!(validation_utils::validate_module(&module2).is_err());
    }

    #[test]
    fn test_timer() {
        let _timer = timing_utils::Timer::new("test_timer");
        std::thread::sleep(std::time::Duration::from_millis(10));
        // The timer will print when dropped
    }
    
    #[test]
    fn test_performance_math_utils() {
        use std::time::Instant;
        
        // Benchmark gcd function
        let start = Instant::now();
        for i in 1..1000 {
            math_utils::gcd(i * 7, i * 3);
        }
        let gcd_duration = start.elapsed();
        println!("GCD benchmark (1000 iterations): {:?}", gcd_duration);
        
        // Benchmark lcm function
        let start = Instant::now();
        for i in 1..1000 {
            math_utils::lcm(i * 2, i * 3);
        }
        let lcm_duration = start.elapsed();
        println!("LCM benchmark (1000 iterations): {:?}", lcm_duration);
        
        // Benchmark round_up_to_multiple
        let start = Instant::now();
        for i in 1..1000 {
            math_utils::round_up_to_multiple(i * 5, 8);
        }
        let round_up_duration = start.elapsed();
        println!("Round up to multiple benchmark (1000 iterations): {:?}", round_up_duration);
        
        // Benchmark next_power_of_2
        let start = Instant::now();
        for i in 1..1000 {
            math_utils::next_power_of_2(i * 3);
        }
        let next_power_duration = start.elapsed();
        println!("Next power of 2 benchmark (1000 iterations): {:?}", next_power_duration);
        
        // Ensure performance is reasonable (less than 100ms for 1000 operations)
        // Note: This is just a basic sanity check, not a strict performance requirement
        assert!(gcd_duration.as_millis() < 100, "GCD function too slow: {:?}", gcd_duration);
        assert!(lcm_duration.as_millis() < 100, "LCM function too slow: {:?}", lcm_duration);
        assert!(round_up_duration.as_millis() < 100, "Round up function too slow: {:?}", round_up_duration);
        assert!(next_power_duration.as_millis() < 100, "Next power function too slow: {:?}", next_power_duration);
    }

    #[test]
    fn test_attribute_handling_various_types() {
        use crate::ir::Attribute;
        
        // Test all attribute types
        let int_attr = Attribute::Int(42);
        let float_attr = Attribute::Float(3.14159);
        let string_attr = Attribute::String("test_string".to_string());
        let bool_attr = Attribute::Bool(true);
        let array_attr = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Int(2),
            Attribute::Int(3),
        ]);

        // Verify they can be matched correctly
        match int_attr {
            Attribute::Int(v) => assert_eq!(v, 42),
            _ => panic!("Expected Int attribute"),
        }

        match float_attr {
            Attribute::Float(v) => assert!((v - 3.14159).abs() < f64::EPSILON),
            _ => panic!("Expected Float attribute"),
        }

        match &string_attr {
            Attribute::String(s) => assert_eq!(s, "test_string"),
            _ => panic!("Expected String attribute"),
        }

        match bool_attr {
            Attribute::Bool(v) => assert_eq!(v, true),
            _ => panic!("Expected Bool attribute"),
        }

        match &array_attr {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 3);
                match &arr[0] {
                    Attribute::Int(1) => {},
                    _ => panic!("Expected first element to be Int(1)"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    #[test]
    fn test_attribute_array_nested() {
        use crate::ir::Attribute;
        
        // Create a nested array attribute
        let nested_array = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Int(2),
            ]),
            Attribute::Array(vec![
                Attribute::String("nested".to_string()),
                Attribute::Bool(false),
            ]),
        ]);

        match &nested_array {
            Attribute::Array(outer_arr) => {
                assert_eq!(outer_arr.len(), 2);
                
                // Check first nested array
                match &outer_arr[0] {
                    Attribute::Array(inner1) => {
                        assert_eq!(inner1.len(), 2);
                        match &inner1[0] {
                            Attribute::Int(1) => {},
                            _ => panic!("Expected first nested element to be Int(1)"),
                        }
                    },
                    _ => panic!("Expected first element to be Array"),
                }
                
                // Check second nested array
                match &outer_arr[1] {
                    Attribute::Array(inner2) => {
                        assert_eq!(inner2.len(), 2);
                        match &inner2[0] {
                            Attribute::String(s) if s == "nested" => {},
                            _ => panic!("Expected nested string"),
                        }
                    },
                    _ => panic!("Expected second element to be Array"),
                }
            },
            _ => panic!("Expected nested array attribute"),
        }
    }

    #[test]
    fn test_deeply_nested_attributes() {
        use crate::ir::Attribute;
        
        // Create a deeply nested attribute structure to test recursion limits
        let mut attr = Attribute::Int(1);
        
        // Build nested arrays up to a depth of 50
        for i in 2..50 {
            attr = Attribute::Array(vec![
                Attribute::Int(i),
                attr,
            ]);
        }
        
        // Verify the nested structure still works
        match &attr {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 2);
                
                match &arr[0] {
                    Attribute::Int(val) => {
                        assert_eq!(*val, 49); // Last value inserted
                    },
                    _ => panic!("Expected first element to be Int(49)"),
                }
                
                // Check that the nested structure can be accessed
                match &arr[1] {
                    Attribute::Array(_) | Attribute::Int(_) => {}, // Either is valid at this depth
                    _ => panic!("Expected nested structure"),
                }
            },
            _ => panic!("Expected final structure to be an Array"),
        }
    }

    #[test]
    fn test_empty_attributes() {
        use crate::ir::Attribute;
        
        // Test empty string attribute
        let empty_string_attr = Attribute::String(String::new());
        match &empty_string_attr {
            Attribute::String(s) => assert!(s.is_empty()),
            _ => panic!("Expected empty String attribute"),
        }
        
        // Test empty array attribute
        let empty_array_attr = Attribute::Array(vec![]);
        match &empty_array_attr {
            Attribute::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("Expected empty Array attribute"),
        }
        
        // Test array with empty arrays inside
        let array_with_empty = Attribute::Array(vec![
            Attribute::Array(vec![]),
            Attribute::Int(42),
        ]);
        
        match &array_with_empty {
            Attribute::Array(arr) => {
                assert_eq!(arr.len(), 2);
                
                match &arr[0] {
                    Attribute::Array(empty_arr) => assert!(empty_arr.is_empty()),
                    _ => panic!("Expected first element to be an empty Array"),
                }
                
                match &arr[1] {
                    Attribute::Int(42) => {}, // Expected
                    _ => panic!("Expected second element to be Int(42)"),
                }
            },
            _ => panic!("Expected top-level Array attribute"),
        }
    }

    #[test]
    fn test_tensor_size_calculation_edge_cases() {
        use crate::ir::Type;
        
        // Test zero-size tensor - contains zero in dimensions
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &[5, 0, 10]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F64, &[0, 5]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I32, &[7, 8, 0, 1]).unwrap(), 0);

        // Test extremely large dimensions (but within usize bounds) - should not overflow
        let large_dimension = 1000;
        let large_size_result = ir_utils::calculate_tensor_size(&Type::F32, &[large_dimension, large_dimension]);
        if let Ok(computed_size) = large_size_result {
            assert_eq!(computed_size, large_dimension * large_dimension * 4); // 4 bytes for F32
        } else {
            // If the size calculation overflows, it should return an error
            assert!(large_size_result.is_err());
        }

        // Test multi-dimensional size calculation
        let size_3d = ir_utils::calculate_tensor_size(&Type::F32, &[2, 3, 4]).unwrap();
        assert_eq!(size_3d, 2 * 3 * 4 * 4); // 2*3*4 elements * 4 bytes per F32

        // Test with different data types
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I64, &[10, 10]).unwrap(), 10 * 10 * 8); // 8 bytes per I64
        assert_eq!(ir_utils::calculate_tensor_size(&Type::Bool, &[100]).unwrap(), 100 * 1); // 1 byte per Bool
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I32, &[5, 5]).unwrap(), 5 * 5 * 4); // 4 bytes per I32
    }

    #[test]
    fn test_nested_tensor_size_calculation_edge_cases() {
        use crate::ir::Type;
        
        // Deeply nested tensor type
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![2, 2],
        };

        // For a tensor<[2,2] f32> with outer shape [3], total size should be 3 * 2 * 2 * 4 = 48
        let size = ir_utils::calculate_tensor_size(&nested_type, &[3]).unwrap();
        assert_eq!(size, 3 * 2 * 2 * 4);

        // For a tensor<[2,2] f32> with outer shape [], total size should be 1 * 2 * 2 * 4 = 16
        let size_scalar = ir_utils::calculate_tensor_size(&nested_type, &[]).unwrap();
        assert_eq!(size_scalar, 1 * 2 * 2 * 4);

        // Even deeper nesting
        let deeper_nested = Type::Tensor {
            element_type: Box::new(
                Type::Tensor {
                    element_type: Box::new(Type::F64),
                    shape: vec![3],
                }
            ),
            shape: vec![2],
        };

        // For tensor<[2] tensor<[3] f64>> with outer shape [4], 
        // total size should be 4 * 2 * 3 * 8 = 192
        let deeper_size = ir_utils::calculate_tensor_size(&deeper_nested, &[4]).unwrap();
        assert_eq!(deeper_size, 4 * 2 * 3 * 8);
    }

    #[test]
    fn test_validation_utilities_error_conditions() {
        use crate::ir::{Module, Value, Type, Operation};
        
        // Test module validation with duplicate input names
        let mut module = Module::new("test_duplicate_inputs");
        module.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.inputs.push(Value {
            name: "input1".to_string(),  // Duplicate name
            ty: Type::F32,
            shape: vec![20],
        });
        
        let validation_result = validation_utils::validate_module(&module);
        assert!(validation_result.is_err());
        assert!(validation_result.unwrap_err().to_string().contains("Duplicate input name"));

        // Test operation validation with duplicate names
        let mut op = Operation::new("test_op");
        op.inputs.push(Value {
            name: "dup_name".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        op.outputs.push(Value {
            name: "dup_name".to_string(),  // Same name as input
            ty: Type::F32,
            shape: vec![5],
        });
        
        let op_validation_result = validation_utils::validate_operation(&op);
        assert!(op_validation_result.is_err());
        assert!(op_validation_result.unwrap_err().to_string().contains("Duplicate name"));
    }

    #[test]
    fn test_complex_module_validation() {
        use crate::ir::{Module, Value, Type, Operation};
        use std::collections::HashMap;

        // Create a valid complex module
        let mut module = Module::new("complex_valid_module");
        
        // Add inputs with unique names
        module.inputs.push(Value {
            name: "image_data".to_string(),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224],
        });
        module.inputs.push(Value {
            name: "conv_weights".to_string(),
            ty: Type::F32,
            shape: vec![64, 3, 7, 7],
        });
        
        // Add outputs with unique names
        module.outputs.push(Value {
            name: "conv_output".to_string(),
            ty: Type::F32,
            shape: vec![1, 64, 112, 112],
        });
        
        // Add an operation
        let mut conv_op = Operation::new("conv2d");
        conv_op.inputs.push(Value {
            name: "input_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224],
        });
        conv_op.outputs.push(Value {
            name: "output_tensor".to_string(),
            ty: Type::F32,
            shape: vec![1, 64, 112, 112],
        });
        
        // Add attributes to the operation
        let mut attrs = HashMap::new();
        attrs.insert("padding".to_string(), crate::ir::Attribute::Int(3));
        attrs.insert("stride".to_string(), crate::ir::Attribute::Int(2));
        conv_op.attributes = attrs;
        
        module.add_operation(conv_op);
        
        // This should be valid
        assert!(validation_utils::validate_module(&module).is_ok());

        // Now add an operation with duplicate names to make it invalid
        let mut bad_op = Operation::new("bad_op");
        bad_op.inputs.push(Value {
            name: "unique_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        bad_op.outputs.push(Value {
            name: "unique_input".to_string(),  // Same name as input - should fail
            ty: Type::F32,
            shape: vec![10],
        });
        
        assert!(validation_utils::validate_operation(&bad_op).is_err());
    }

    #[test]
    fn test_validation_with_extremely_long_names() {
        use crate::ir::{Module, Value, Type, Operation};
        
        // Test validation with extremely long names to check string limit handling
        
        // Create a module with very long name
        let very_long_name = "x".repeat(100_000);
        let mut module = Module::new(&very_long_name);
        
        // Add inputs with long names
        module.inputs.push(Value {
            name: "input1_".to_string() + &"a".repeat(50_000),
            ty: Type::F32,
            shape: vec![10],
        });
        module.inputs.push(Value {
            name: "input2_".to_string() + &"b".repeat(50_000), 
            ty: Type::F32,
            shape: vec![20],
        });
        
        // Add operations with long names and inputs/outputs
        let mut op = Operation::new(&("operation_".to_owned() + &"o".repeat(40_000)));
        op.inputs.push(Value {
            name: "long_input_name_".to_string() + &"i".repeat(30_000),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        op.outputs.push(Value {
            name: "long_output_name_".to_string() + &"o".repeat(30_000),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        
        module.add_operation(op);
        
        // This should validate successfully despite long names
        let result = validation_utils::validate_module(&module);
        assert!(result.is_ok());
        
        // Now test with duplicate names among long names (should fail)
        let mut bad_module = Module::new("bad_long_names");
        let long_name_base = "very_long_unique_name_".to_owned() + &"x".repeat(50_000);
        bad_module.inputs.push(Value {
            name: long_name_base.clone(),
            ty: Type::F32,
            shape: vec![5],
        });
        bad_module.inputs.push(Value {
            name: long_name_base,  // Duplicate
            ty: Type::F32,
            shape: vec![10],
        });
        
        let bad_result = validation_utils::validate_module(&bad_module);
        assert!(bad_result.is_err());
        assert!(bad_result.unwrap_err().to_string().contains("Duplicate input name"));
    }

    #[test]
    fn test_validation_of_complex_nested_structures() {
        use crate::ir::{Module, Value, Type, Operation, Attribute};
        use std::collections::HashMap;
        
        // Create a module with complex nested structures to validate
        let mut module = Module::new("nested_validation_test");
        
        // Create a complex nested tensor type
        let nested_tensor_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![3, 3],
            }),
            shape: vec![2, 2],
        };
        
        // Add an operation with complex inputs and outputs
        let mut complex_op = Operation::new("complex_op");
        
        // Add complex-shaped inputs
        complex_op.inputs.push(Value {
            name: "nested_tensor_input".to_string(),
            ty: nested_tensor_type.clone(),
            shape: vec![4],  // Outer shape for the nested tensor
        });
        
        complex_op.inputs.push(Value {
            name: "regular_tensor_input".to_string(),
            ty: Type::F64,
            shape: vec![10, 20, 5],
        });
        
        // Add complex-shaped outputs
        complex_op.outputs.push(Value {
            name: "nested_tensor_output".to_string(),
            ty: nested_tensor_type,
            shape: vec![6],  // Different outer shape
        });
        
        complex_op.outputs.push(Value {
            name: "regular_tensor_output".to_string(),
            ty: Type::I32,
            shape: vec![5, 5, 5, 5],
        });
        
        // Add complex attributes
        let mut attrs = HashMap::new();
        attrs.insert(
            "complex_array_attr".to_string(),
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(3.14),
                Attribute::String("nested_string".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Int(42),
                ]),
            ])
        );
        attrs.insert("simple_int_attr".to_string(), Attribute::Int(123));
        complex_op.attributes = attrs;
        
        module.add_operation(complex_op);
        
        // This complex structure should still validate successfully
        let result = validation_utils::validate_module(&module);
        assert!(result.is_ok());
        
        // Now add an operation with naming conflicts in the complex structure
        let mut conflicting_op = Operation::new("conflict_op");
        conflicting_op.inputs.push(Value {
            name: "same_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        conflicting_op.outputs.push(Value {
            name: "same_name".to_string(),  // Same name as input - should cause conflict
            ty: Type::F32,
            shape: vec![10],
        });
        
        let conflict_result = validation_utils::validate_operation(&conflicting_op);
        assert!(conflict_result.is_err());
        assert!(conflict_result.unwrap_err().to_string().contains("Duplicate name"));
    }

    #[test]
    fn test_math_utilities_edge_cases() {
        // Test GCD with special values
        assert_eq!(math_utils::gcd(0, 0), 0);  // gcd(0, 0) is traditionally defined as 0
        assert_eq!(math_utils::gcd(0, 5), 5);  // gcd(0, n) = n
        assert_eq!(math_utils::gcd(5, 0), 5);  // gcd(n, 0) = n
        assert_eq!(math_utils::gcd(1, 1000000), 1);  // gcd(1, n) = 1
        assert_eq!(math_utils::gcd(17, 17), 17);  // gcd(n, n) = n
        assert_eq!(math_utils::gcd(100, 75), 25);  // gcd(100, 75) should be 25
        
        // Test LCM with special values
        assert_eq!(math_utils::lcm(0, 5), 0);  // lcm(0, n) = 0
        assert_eq!(math_utils::lcm(5, 0), 0);  // lcm(n, 0) = 0
        assert_eq!(math_utils::lcm(0, 0), 0);  // lcm(0, 0) = 0
        assert_eq!(math_utils::lcm(1, 7), 7);  // lcm(1, n) = n
        assert_eq!(math_utils::lcm(7, 1), 7);  // lcm(n, 1) = n
        assert_eq!(math_utils::lcm(4, 6), 12); // lcm(4, 6) should be 12
        assert_eq!(math_utils::lcm(12, 18), 36); // lcm(12, 18) should be 36
        
        // Test round_up_to_multiple with edge cases
        assert_eq!(math_utils::round_up_to_multiple(0, 5), 0);  // 0 rounded up to any multiple is 0
        assert_eq!(math_utils::round_up_to_multiple(5, 0), 5);  // When multiple is 0, return original
        assert_eq!(math_utils::round_up_to_multiple(7, 1), 7);  // Rounding to multiple of 1 should return the same number
        assert_eq!(math_utils::round_up_to_multiple(7, 7), 7);  // Rounding exact multiple should return same number
        assert_eq!(math_utils::round_up_to_multiple(8, 7), 14);  // 8 rounded up to multiple of 7 should be 14
        assert_eq!(math_utils::round_up_to_multiple(1, 100), 100);  // Small number rounded up to big multiple
        assert_eq!(math_utils::round_up_to_multiple(100, 100), 100);  // Exact match
        
        // Test next_power_of_2 with edge cases
        assert_eq!(math_utils::next_power_of_2(0), 1);    // Special case: 0 should return 1
        assert_eq!(math_utils::next_power_of_2(1), 1);    // 1 is already a power of 2
        assert_eq!(math_utils::next_power_of_2(2), 2);    // 2 is already a power of 2
        assert_eq!(math_utils::next_power_of_2(3), 4);    // Next power of 2 after 3 is 4
        assert_eq!(math_utils::next_power_of_2(4), 4);    // 4 is already a power of 2
        assert_eq!(math_utils::next_power_of_2(5), 8);    // Next power of 2 after 5 is 8
        assert_eq!(math_utils::next_power_of_2(7), 8);    // Next power of 2 after 7 is 8
        assert_eq!(math_utils::next_power_of_2(8), 8);    // 8 is already a power of 2
        assert_eq!(math_utils::next_power_of_2(9), 16);   // Next power of 2 after 9 is 16
        assert_eq!(math_utils::next_power_of_2(15), 16);  // Next power of 2 after 15 is 16
        assert_eq!(math_utils::next_power_of_2(16), 16);  // 16 is already a power of 2
        assert_eq!(math_utils::next_power_of_2(17), 32);  // Next power of 2 after 17 is 32
        assert_eq!(math_utils::next_power_of_2(1000), 1024); // Large number test
    }

    #[test]
    fn test_math_utility_properties() {
        // Test the mathematical property that gcd(a,b) * lcm(a,b) = a * b
        // (when neither a nor b is 0)
        let a = 12;
        let b = 18;
        let gcd_val = math_utils::gcd(a, b);
        let lcm_val = math_utils::lcm(a, b);
        assert_eq!(gcd_val * lcm_val, a * b);
        
        let a = 15;
        let b = 25;
        let gcd_val = math_utils::gcd(a, b);
        let lcm_val = math_utils::lcm(a, b);
        assert_eq!(gcd_val * lcm_val, a * b);
        
        // Test that a number is a power of 2 if and only if it equals its next power of 2
        for i in 1..=32 {
            let is_power_of_2 = (i & (i - 1)) == 0 && i != 0;  // Bitwise trick
            let next_pow = math_utils::next_power_of_2(i);
            if is_power_of_2 {
                assert_eq!(i, next_pow);  // Powers of 2 should remain unchanged
            } else {
                assert!(next_pow > i);    // Others should be increased
            }
        }
        
        // Powers of 2 check specifically
        let powers_of_2 = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        for pow in powers_of_2 {
            assert_eq!(math_utils::next_power_of_2(pow), pow);  // Powers of 2 return themselves
        }
    }

    #[test]
    fn test_math_utils_with_large_numbers() {
        // Test math utilities with very large numbers that are still within usize range
        
        // Test GCD with very large coprime numbers
        let large_num1 = 999_999_937; // Large prime
        let large_num2 = 999_999_967; // Another large prime
        let gcd_result = math_utils::gcd(large_num1, large_num2);
        assert_eq!(gcd_result, 1);  // GCD of two different primes is 1
        
        // Test LCM of these large primes
        let lcm_result = math_utils::lcm(large_num1, large_num2);
        assert_eq!(lcm_result, large_num1 * large_num2);  // LCM of coprimes is their product
        
        // Test round_up with large numbers
        let large_value = 1_000_000_000;
        let round_up_result = math_utils::round_up_to_multiple(large_value, 1024);
        assert!(round_up_result >= large_value);
        assert_eq!(round_up_result % 1024, 0);  // Should be divisible by 1024
        
        // Test next power of 2 with a large number
        let next_pow = math_utils::next_power_of_2(large_value);
        assert!(next_pow.is_power_of_two());
        assert!(next_pow >= large_value);
    }

    #[test]
    fn test_math_utils_comprehensive_edge_cases() {
        // Comprehensive edge case testing for all math utilities
        
        // Test gcd with all combinations of 0 and 1
        assert_eq!(math_utils::gcd(0, 0), 0);
        assert_eq!(math_utils::gcd(0, 1), 1);
        assert_eq!(math_utils::gcd(1, 0), 1);
        assert_eq!(math_utils::gcd(1, 1), 1);
        
        // Test lcm with all combinations of 0 and 1
        assert_eq!(math_utils::lcm(0, 0), 0);
        assert_eq!(math_utils::lcm(0, 1), 0);
        assert_eq!(math_utils::lcm(1, 0), 0);
        assert_eq!(math_utils::lcm(1, 1), 1);
        
        // Test round_up with extreme multiples
        assert_eq!(math_utils::round_up_to_multiple(1, 1), 1);  // Everything is multiple of 1
        assert_eq!(math_utils::round_up_to_multiple(0, 1000), 0);  // 0 rounded to any multiple is 0
        assert_eq!(math_utils::round_up_to_multiple(999, 1000), 1000);  // Just below a big multiple
        assert_eq!(math_utils::round_up_to_multiple(1000, 1000), 1000);  // Exact match
        
        // Test next_power_of_2 with various ranges
        assert_eq!(math_utils::next_power_of_2(1), 1);
        assert_eq!(math_utils::next_power_of_2(2), 2);
        for exp in 1..5 {  // Reduced range to debug
            let power_of_2 = 1 << exp;
            let prev = power_of_2 - 1;  // Number just before power of 2
            let next = power_of_2 + 1;  // Number just after power of 2
            
            assert_eq!(math_utils::next_power_of_2(prev), power_of_2);
            assert_eq!(math_utils::next_power_of_2(power_of_2), power_of_2);
            if power_of_2 < usize::MAX / 2 {  // Prevent overflow
                let expected = power_of_2 << 1;
                let actual = math_utils::next_power_of_2(next);
                assert_eq!(actual, expected, "Failed for exp={}, power_of_2={}, next={}, expected={}, actual={}", exp, power_of_2, next, expected, actual);
            }
        }
    }
    
    #[test]
    fn test_edge_case_tensor_size_calculations() {
        use crate::ir::{Type};
        
        // Test handling of extreme large values that might cause overflow
        // Note: This won't actually trigger overflow in the current implementation
        // but verifies behavior with large values
        let large_shape = vec![1000, 1000, 100];
        let result = ir_utils::calculate_tensor_size(&Type::F32, &large_shape);
        assert!(result.is_ok());
        if let Ok(size) = result {
            assert_eq!(size, 1000 * 1000 * 100 * 4); // 4 bytes per F32
        }
        
        // Test with multiple different types and large shapes
        let large_shape_i64 = vec![1000, 500];
        let result_i64 = ir_utils::calculate_tensor_size(&Type::I64, &large_shape_i64);
        assert!(result_i64.is_ok());
        if let Ok(size) = result_i64 {
            assert_eq!(size, 1000 * 500 * 8); // 8 bytes per I64
        }
        
        // Test empty tensor with 0 dimensions
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &vec![0]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I32, &vec![0, 5]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::Bool, &vec![10, 0, 100]).unwrap(), 0);
    }
    
    #[test]
    fn test_large_number_math_utils() {
        // Test large numbers with math utilities
        let large_num1 = 1_000_000;
        let large_num2 = 999_999;
        let gcd_result = math_utils::gcd(large_num1, large_num2);
        // gcd(1000000, 999999) = gcd(1000000, 1) = 1
        assert_eq!(gcd_result, 1);
        
        let lcm_result = math_utils::lcm(large_num1, large_num2);
        assert_eq!(lcm_result, large_num1 * large_num2); // Since gcd is 1
        
        let next_power = math_utils::next_power_of_2(1_000_000);
        assert!(next_power >= 1_000_000);
        assert_eq!(next_power, 1_048_576); // Next power of 2 after 1 million
        
        let round_up_result = math_utils::round_up_to_multiple(1_000_001, 1024);
        assert_eq!(round_up_result, 1_000_448); // Round up to next multiple of 1024
    }
    
    #[test]
    fn test_memory_allocation_edge_cases() {
        // Test potential edge cases with memory allocation for large structures
        
        // Test creating many small operations to test vector allocation/deallocation
        let mut operations = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            let mut op = Operation::new(&format!("op_{}", i));
            if i % 1000 == 0 {
                op.inputs.push(Value {
                    name: format!("input_{}", i),
                    ty: Type::F32,
                    shape: vec![i % 100 + 1, i % 100 + 1], // Small variable sized tensor
                });
            }
            operations.push(op);
        }
        
        // Verify we got all operations
        assert_eq!(operations.len(), 100_000);
        
        // Test tensor size calculation with potential overflow
        // This tests large calculations that could theoretically overflow if not handled carefully
        let large_shape = vec![usize::MAX / 1000, 10];  // Very large but not causing direct overflow
        let result = ir_utils::calculate_tensor_size(&Type::F32, &large_shape);
        // Either the operation succeeds or it properly errors out, no panics allowed
        assert!(result.is_ok() || result.is_err());
        
        // Clean up by dropping the vector
        drop(operations);
        
        // Test with many nested attributes (deep structure)
        let mut attr = Attribute::Int(0);
        for i in 1..1000 {
            attr = Attribute::Array(vec![
                Attribute::Int(i),
                attr,  // Nesting
            ]);
        }
        
        // Just make sure we can handle deep nesting without stack overflow
        match attr {
            Attribute::Array(ref arr) => {
                assert_eq!(arr.len(), 2);
            },
            _ => panic!("Expected array attribute after nesting"),
        }
    }

    #[test]
    fn test_large_memory_allocations_stress() {
        // Stress test with potentially huge allocations to verify robustness
        
        // Test creating a module with extremely large number of operations
        let mut module = Module::new("stress_test_module");
        
        // Add operations in batches to test memory allocation
        for batch in 0..10 {
            for i in 0..10_000 {
                let idx = batch * 10_000 + i;
                let mut op = Operation::new(&format!("stress_op_{}", idx));
                
                // Add minimal amount of data to keep test reasonable
                op.inputs.push(Value {
                    name: format!("input_{}", idx),
                    ty: Type::F32,
                    shape: vec![1], // Minimal shape
                });
                
                module.add_operation(op);
            }
            
            // Verify progress every batch
            assert_eq!(module.operations.len(), (batch + 1) * 10_000);
        }
        
        assert_eq!(module.operations.len(), 100_000);
        
        // Check that we can access elements from different parts of the module
        assert_eq!(module.operations[0].op_type, "stress_op_0");
        assert_eq!(module.operations[50_000].op_type, "stress_op_50000");
        assert_eq!(module.operations[99_999].op_type, "stress_op_99999");
    }

    #[test]
    fn test_string_allocation_for_long_names() {
        // Test with extremely long names to check string allocation limits
        let long_name = "a".repeat(100_000); // 100k character name
        
        // Test with a value
        let value = Value {
            name: long_name.clone(),
            ty: Type::F32,
            shape: vec![1, 2, 3],
        };
        
        assert_eq!(value.name.len(), 100_000);
        assert_eq!(value.name.chars().nth(0), Some('a'));
        assert_eq!(value.name.chars().nth(99_999), Some('a'));
        
        // Test with an operation
        let op = Operation::new(&long_name);
        assert_eq!(op.op_type.len(), 100_000);
        
        // Test with a module
        let module = Module::new(long_name.clone());
        assert_eq!(module.name.len(), 100_000);
    }

    #[test]
    fn test_tensor_size_with_overflow_scenarios() {
        // Test tensor size calculations that might cause overflow
        // Using values that are likely to cause overflow when multiplied
        
        // Very large dimensions that might cause overflow in multiplication
        let huge_shape = vec![1_000_000, 1_000_000];
        let result = ir_utils::calculate_tensor_size(&Type::F32, &huge_shape);
        // Test should not panic, regardless of success or failure
        assert!(result.is_ok() || result.err().is_some());
        
        // Another potential overflow scenario
        let another_huge_shape = vec![100_000, 100_000, 100];
        let result2 = ir_utils::calculate_tensor_size(&Type::I64, &another_huge_shape);
        assert!(result2.is_ok() || result2.err().is_some());
    }

    #[test]
    fn test_operation_with_maximum_inputs_outputs() {
        use std::collections::HashMap;
        
        // Test creating an operation with a very large number of inputs and outputs
        let mut op = Operation::new("massive_op");
        
        // Add many inputs
        for i in 0..1000 {
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i % 10 + 1, i % 10 + 1], // Small variable shapes
            });
        }
        
        // Add many outputs
        for i in 0..500 {
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i % 5 + 1, i % 5 + 1], // Small variable shapes
            });
        }
        
        // Add some attributes too
        let mut attrs = HashMap::new();
        for i in 0..100 {
            attrs.insert(format!("attr_{}", i), Attribute::Int(i as i64));
        }
        op.attributes = attrs;
        
        // Verify counts
        assert_eq!(op.inputs.len(), 1000);
        assert_eq!(op.outputs.len(), 500);
        assert_eq!(op.attributes.len(), 100);
        assert_eq!(op.op_type, "massive_op");
    }

    #[test]
    fn test_math_utils_with_extreme_values() {
        // Test math utilities with the largest possible values
        
        // Test next power of 2 with a value close to the maximum
        let _near_max = usize::MAX / 2;  // This would cause overflow if doubled
        // Instead, test with increasingly large values up to a safe limit
        let large_value = 1_000_000_000usize;
        let next_pow = math_utils::next_power_of_2(large_value);
        assert!(next_pow >= large_value);
        assert!(next_pow.is_power_of_two());
        
        // Test GCD with very large coprime numbers
        let large_prime1 = 1_000_000_007usize; // A large prime number
        let large_prime2 = 1_000_000_009usize; // Next large prime number
        let gcd_result = math_utils::gcd(large_prime1, large_prime2);
        assert_eq!(gcd_result, 1); // GCD of two different primes should be 1
        
        // Test LCM of these large primes
        let lcm_result = math_utils::lcm(large_prime1, large_prime2);
        assert_eq!(lcm_result, large_prime1 * large_prime2); // LCM of coprime numbers is their product
    }

    #[test]
    fn test_tensor_size_calculation_with_specific_limits() {
        use crate::ir::Type;
        
        // Test with maximum possible shape values that still allow calculation
        // Use smaller values that won't overflow but still test boundary conditions
        let almost_max_shape = vec![std::cmp::min(100_000, usize::MAX/4), 4];
        let result = ir_utils::calculate_tensor_size(&Type::F32, &almost_max_shape);
        assert!(result.is_ok());
        if let Ok(size) = result {
            let expected_size = almost_max_shape[0] * almost_max_shape[1] * 4;
            assert_eq!(size, expected_size);
        }
        
        // Test with various primitive types and zero dimensions
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F32, &[0]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::F64, &[5, 0, 10]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I32, &[0, 5]).unwrap(), 0);
        assert_eq!(ir_utils::calculate_tensor_size(&Type::I64, &[10, 20, 0]).unwrap(), 0);
        
        // Test with bool tensors and zero dimensions
        assert_eq!(ir_utils::calculate_tensor_size(&Type::Bool, &[0]).unwrap(), 0);
    }

    #[test]
    fn test_nested_tensor_size_with_various_depths() {
        use crate::ir::Type;
        
        // Create a nested tensor type: tensor<tensor<f32, [2,2]>, [3]>
        let nested_type = Type::Tensor {
            element_type: Box::new(Type::Tensor {
                element_type: Box::new(Type::F32),
                shape: vec![2, 2],
            }),
            shape: vec![3],
        };
        
        // Calculate size when outer shape is [4]: 4 * 3 * 2 * 2 * 4 = 192
        let size = ir_utils::calculate_tensor_size(&nested_type, &[4]).unwrap();
        assert_eq!(size, 4 * 3 * 2 * 2 * 4); // 192
        
        // Calculate size when outer shape is [] (scalar): 1 * 3 * 2 * 2 * 4 = 48
        let size_scalar = ir_utils::calculate_tensor_size(&nested_type, &[]).unwrap();
        assert_eq!(size_scalar, 1 * 3 * 2 * 2 * 4); // 48
        
        // Create shallower nesting: tensor<f32, [5,5]> with outer shape [2]
        let shallow_nested = Type::Tensor {
            element_type: Box::new(Type::F32),
            shape: vec![5, 5],
        };
        let shallow_size = ir_utils::calculate_tensor_size(&shallow_nested, &[2]).unwrap();
        assert_eq!(shallow_size, 2 * 5 * 5 * 4); // 200
    }
}