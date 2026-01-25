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
    use crate::ir::{Module, Value, Type, Operation};

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
        let timer = timing_utils::Timer::new("test_timer");
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
}