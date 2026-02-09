//! Additional edge case tests for operation validation and module functionality in Impulse compiler
//! This file contains tests covering validation boundary conditions and module operations

use crate::{
    ir::{Module, Value, Type, Operation, Attribute},
    utils::validation_utils,
};
use std::collections::HashMap;

#[cfg(test)]
mod validation_and_module_edge_case_tests {
    use super::*;

    #[test]
    fn test_operation_validation_edge_cases() {
        // Test operation with no inputs or outputs but with attributes
        let mut op_no_io_with_attrs = Operation::new("no_io_op");
        op_no_io_with_attrs.attributes.insert("flag".to_string(), Attribute::Bool(true));
        
        let result = validation_utils::validate_operation(&op_no_io_with_attrs);
        assert!(result.is_ok()); // Should be valid

        // Test operation with inputs but no outputs
        let mut op_inputs_only = Operation::new("inputs_only");
        op_inputs_only.inputs.push(Value {
            name: "inp1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op_inputs_only.inputs.push(Value {
            name: "inp2".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_operation(&op_inputs_only);
        assert!(result.is_ok()); // Should be valid

        // Test operation with outputs but no inputs
        let mut op_outputs_only = Operation::new("outputs_only");
        op_outputs_only.outputs.push(Value {
            name: "out1".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        
        let result = validation_utils::validate_operation(&op_outputs_only);
        assert!(result.is_ok()); // Should be valid

        // Test operation with duplicate input names (should fail)
        let mut op_dup_inputs = Operation::new("dup_inputs_op");
        op_dup_inputs.inputs.push(Value {
            name: "same_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op_dup_inputs.inputs.push(Value {
            name: "same_name".to_string(),  // Duplicate
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_operation(&op_dup_inputs);
        assert!(result.is_err()); // Should fail

        // Test operation with duplicate output names (should fail)
        let mut op_dup_outputs = Operation::new("dup_outputs_op");
        op_dup_outputs.outputs.push(Value {
            name: "out1".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op_dup_outputs.outputs.push(Value {
            name: "out1".to_string(),  // Duplicate
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_operation(&op_dup_outputs);
        assert!(result.is_err()); // Should fail

        // Test operation with same name in inputs and outputs (should fail)
        let mut op_io_conflict = Operation::new("io_conflict_op");
        op_io_conflict.inputs.push(Value {
            name: "shared_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op_io_conflict.outputs.push(Value {
            name: "shared_name".to_string(),  // Same name as input
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_operation(&op_io_conflict);
        assert!(result.is_err()); // Should fail

        // Test operation with many inputs and outputs with unique names
        let mut op_many_unique = Operation::new("many_unique_op");
        for i in 0..100 {
            op_many_unique.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
            op_many_unique.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![i + 1],
            });
        }
        
        let result = validation_utils::validate_operation(&op_many_unique);
        assert!(result.is_ok()); // Should be valid
        assert_eq!(op_many_unique.inputs.len(), 100);
        assert_eq!(op_many_unique.outputs.len(), 100);
    }

    #[test]
    fn test_module_validation_detailed() {
        // Test completely valid module
        let mut valid_module = Module::new("valid_module");
        
        // Add unique input names
        valid_module.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        valid_module.inputs.push(Value {
            name: "input2".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // Add unique output names
        valid_module.outputs.push(Value {
            name: "output1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // Add operations
        for i in 0..10 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("op_input_{}", i),
                ty: Type::F32,
                shape: vec![5, 5],
            });
            op.outputs.push(Value {
                name: format!("op_output_{}", i),
                ty: Type::F32,
                shape: vec![5, 5],
            });
            valid_module.add_operation(op);
        }
        
        assert!(validation_utils::validate_module(&valid_module).is_ok());

        // Test module with duplicate input names (should fail)
        let mut dup_input_module = Module::new("dup_input_mod");
        dup_input_module.inputs.push(Value {
            name: "dupe".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        dup_input_module.inputs.push(Value {
            name: "dupe".to_string(),  // Duplicate
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_module(&dup_input_module);
        assert!(result.is_err());

        // Test module with duplicate output names (should fail)
        let mut dup_output_module = Module::new("dup_output_mod");
        dup_output_module.outputs.push(Value {
            name: "same_output".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        dup_output_module.outputs.push(Value {
            name: "same_output".to_string(),  // Duplicate
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_module(&dup_output_module);
        assert!(result.is_err());

        // Test empty module (should be valid)
        let empty_module = Module::new("empty");
        assert!(validation_utils::validate_module(&empty_module).is_ok());
    }

    #[test]
    fn test_module_operation_manipulation() {
        let mut module = Module::new("manipulation_test");
        
        // Add operations one by one
        for i in 0..5 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![i + 1, i + 1],
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 5);
        
        // Test accessing operations
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[4].op_type, "op_4");
        
        // Test module with many operations to test memory handling
        for i in 5..1000 {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![2, 2], // Small shape for performance
            });
            module.add_operation(op);
        }
        
        assert_eq!(module.operations.len(), 1000);
        
        // Check random accesses to ensure integrity
        assert_eq!(module.operations[0].op_type, "op_0");
        assert_eq!(module.operations[500].op_type, "op_500");
        assert_eq!(module.operations[999].op_type, "op_999");
        
        // Check that all operations have the expected input
        assert_eq!(module.operations[0].inputs[0].name, "input_0");
        assert_eq!(module.operations[500].inputs[0].name, "input_500");
        assert_eq!(module.operations[999].inputs[0].name, "input_999");
    }

    #[test]
    fn test_module_uniqueness_validation() {
        // Create a module validation test with more complex scenarios
        
        // Test validation with unicode names
        let mut unicode_module = Module::new("unicode_模块_モジュール_модуль_.Module");
        
        unicode_module.inputs.push(Value {
            name: "input_名称_名_имя".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        unicode_module.outputs.push(Value {
            name: "output_出力_вывод".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        let mut unicode_op = Operation::new("op_操作_операция");
        unicode_op.inputs.push(Value {
            name: "op_input_入力_ввод".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        unicode_op.outputs.push(Value {
            name: "output_出力_вывод".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        
        unicode_module.add_operation(unicode_op);
        
        // Unicode names should be valid, but there's a duplicate "output_出力_вывод" in input and output names
        let result = validation_utils::validate_module(&unicode_module);
        assert!(result.is_err()); // Should fail because input and output share the same name
        
        // Fix the issue by making names unique
        unicode_module.outputs.clear();
        unicode_module.outputs.push(Value {
            name: "unique_output_出力_вывод".to_string(), // Made unique
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_module(&unicode_module);
        assert!(result.is_ok()); // Should pass now
    }

    #[test]
    fn test_complex_module_with_interconnected_operations() {
        // Create a more realistic module with interconnected operations
        let mut module = Module::new("connected_module");
        
        // Add global inputs and outputs
        module.inputs.push(Value {
            name: "global_input".to_string(),
            ty: Type::F32,
            shape: vec![1, 3, 224, 224], // Typical image input shape
        });
        
        module.outputs.push(Value {
            name: "global_output".to_string(),
            ty: Type::F32,
            shape: vec![1, 1000], // Classification output
        });
        
        // Create a chain of operations
        let mut prev_output_name = "global_input".to_string();
        
        for i in 0..10 {
            let mut op = Operation::new(&format!("layer_{}", i));
            
            // Connect to previous output
            op.inputs.push(Value {
                name: if i == 0 { "op_input_from_global".to_string() } else { prev_output_name },
                ty: Type::F32,
                shape: if i == 0 { vec![1, 3, 224, 224] } else { vec![1, 64] }, // Adjust shape as needed
            });
            
            let output_name = format!("layer_{}_output", i);
            op.outputs.push(Value {
                name: output_name.clone(),
                ty: Type::F32,
                shape: vec![1, 64], // Common intermediate shape
            });
            
            // Add some attributes to make it realistic
            let mut attrs = HashMap::new();
            attrs.insert("bias".to_string(), Attribute::Bool(i % 2 == 0));
            attrs.insert("activation".to_string(), 
                         Attribute::String(if i % 3 == 0 { "relu".to_string() } else { "linear".to_string() }));
            op.attributes = attrs;
            
            module.add_operation(op);
            prev_output_name = format!("layer_{}_output", i);
        }
        
        // Validation should pass
        assert!(validation_utils::validate_module(&module).is_ok());
        
        // Check module structure
        assert_eq!(module.operations.len(), 10);
        assert_eq!(module.inputs.len(), 1);
        assert_eq!(module.outputs.len(), 1);
        
        // Check that all operations have the right attributes
        for (i, op) in module.operations.iter().enumerate() {
            assert_eq!(op.op_type, format!("layer_{}", i));
            assert_eq!(op.attributes.len(), 2);
            
            // Verify attributes
            assert!(op.attributes.contains_key("bias"));
            assert!(op.attributes.contains_key("activation"));
            
            if let Some(&Attribute::Bool(has_bias)) = op.attributes.get("bias") {
                assert_eq!(has_bias, i % 2 == 0);
            } else {
                panic!("Expected bias to be Bool");
            }
        }
    }

    #[test]
    fn test_operation_validation_with_complex_attributes() {
        // Test operation validation doesn't interfere with complex attributes
        
        let mut op_with_complex_attrs = Operation::new("complex_attr_op");
        
        // Add complex nested attributes
        let complex_attr = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::String("nested".to_string()),
            ]),
            Attribute::Array(vec![
                Attribute::Bool(true),
                Attribute::Float(3.14),
            ]),
        ]);
        
        op_with_complex_attrs.attributes.insert("complex_structure".to_string(), complex_attr);
        
        // Add inputs and outputs
        op_with_complex_attrs.inputs.push(Value {
            name: "input_1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        op_with_complex_attrs.outputs.push(Value {
            name: "output_1".to_string(),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // Validation should succeed even with complex attributes
        assert!(validation_utils::validate_operation(&op_with_complex_attrs).is_ok());
        
        // Test validation with duplicate names AND complex attributes
        let mut op_conflict_attrs = Operation::new("conflict_op");
        op_conflict_attrs.attributes.insert("some_attr".to_string(), Attribute::Int(42));
        
        // Add duplicate names
        op_conflict_attrs.inputs.push(Value {
            name: "shared_name".to_string(),
            ty: Type::F32,
            shape: vec![5],
        });
        op_conflict_attrs.outputs.push(Value {
            name: "shared_name".to_string(),  // Same as input
            ty: Type::F32,
            shape: vec![5],
        });
        
        // This should still fail even with attributes present
        assert!(validation_utils::validate_operation(&op_conflict_attrs).is_err());
    }

    #[test]
    fn test_module_with_extreme_name_lengths() {
        // Test validation with extremely long names
        
        let mut long_name_module = Module::new(&"a".repeat(1_000_000)); // 1 million character module name
        
        // Add inputs with long names
        long_name_module.inputs.push(Value {
            name: "input_".to_string() + &"b".repeat(500_000), // 500k + 6 characters
            ty: Type::F32,
            shape: vec![10],
        });
        
        long_name_module.inputs.push(Value {
            name: "input_".to_string() + &"c".repeat(500_000), // Different name for uniqueness
            ty: Type::F32,
            shape: vec![20],
        });
        
        long_name_module.outputs.push(Value {
            name: "output_".to_string() + &"d".repeat(400_000), // 400k + 7 characters
            ty: Type::F32,
            shape: vec![15],
        });
        
        // Add an operation with long names
        let mut long_op = Operation::new(&("op_".to_string() + &"e".repeat(300_000)));
        long_op.inputs.push(Value {
            name: "long_input_".to_string() + &"f".repeat(200_000),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        long_op.outputs.push(Value {
            name: "long_output_".to_string() + &"g".repeat(200_000),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        
        long_name_module.add_operation(long_op);
        
        // This should validate successfully despite long names
        let result = validation_utils::validate_module(&long_name_module);
        assert!(result.is_ok());
        
        // Now add a duplicate to test that validation still catches errors with long names
        let mut duplicate_module = Module::new("duplicate_test");
        let duplicate_name = "long_name_".to_string() + &"x".repeat(100_000);
        
        duplicate_module.inputs.push(Value {
            name: duplicate_name.clone(),
            ty: Type::F32,
            shape: vec![5],
        });
        duplicate_module.outputs.push(Value {
            name: duplicate_name,  // Same name as input
            ty: Type::F32,
            shape: vec![10],
        });
        
        let result = validation_utils::validate_module(&duplicate_module);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_module_operations() {
        // Test modules with sparse operation patterns (e.g., lots of gaps in numbering)
        
        let mut sparse_module = Module::new("sparse_test");
        
        // Add operations with sparse numbering
        for &i in &[0, 1, 5, 10, 50, 100, 500, 999] {
            let mut op = Operation::new(&format!("op_{}", i));
            op.inputs.push(Value {
                name: format!("input_{}", i),
                ty: Type::F32,
                shape: vec![2, 2],
            });
            op.outputs.push(Value {
                name: format!("output_{}", i),
                ty: Type::F32,
                shape: vec![2, 2],
            });
            sparse_module.add_operation(op);
        }
        
        assert_eq!(sparse_module.operations.len(), 8);
        
        // Validation should work fine with sparse operations
        let result = validation_utils::validate_module(&sparse_module);
        assert!(result.is_ok());
        
        // Verify the operations are in the order they were added
        let expected_order = vec!["op_0", "op_1", "op_5", "op_10", "op_50", "op_100", "op_500", "op_999"];
        for (i, expected_name) in expected_order.iter().enumerate() {
            assert_eq!(sparse_module.operations[i].op_type, *expected_name);
        }
    }

    #[test]
    fn test_module_with_special_character_names() {
        // Test module validation with special characters in names
        
        let mut special_module = Module::new("module_with_special_chars!@#$%^&*()");
        
        // Add inputs with special characters
        special_module.inputs.push(Value {
            name: "input-with.dots_and_underscores spaces".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        
        special_module.inputs.push(Value {
            name: "input-with.dots_and_underscores spaces".to_string(), // Duplicate to test validation
            ty: Type::F32,
            shape: vec![20],
        });
        
        // This should fail due to duplicate names
        assert!(validation_utils::validate_module(&special_module).is_err());
        
        // Fix the duplicate and try again
        special_module.inputs.pop(); // Remove the duplicate
        
        special_module.inputs.push(Value {
            name: "different-input with!special@chars#and%spaces".to_string(),
            ty: Type::F32,
            shape: vec![20],
        });
        
        special_module.outputs.push(Value {
            name: "output with\ttabs\nand\rnewlines".to_string(),
            ty: Type::F32,
            shape: vec![15],
        });
        
        // Add an operation with special character names
        let mut special_op = Operation::new("operation-with.special!@#$%^&*()chars");
        special_op.inputs.push(Value {
            name: "op-input with\"quotes'and<special>chars".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        special_op.outputs.push(Value {
            name: "op-output with{curly}brackets[brackets]|pipes".to_string(),
            ty: Type::F32,
            shape: vec![5, 5],
        });
        
        special_module.add_operation(special_op);
        
        // Validation should succeed with special character names
        let result = validation_utils::validate_module(&special_module);
        assert!(result.is_ok());
        
        // Verify the names are preserved correctly
        assert_eq!(special_module.name, "module_with_special_chars!@#$%^&*()");
        assert_eq!(special_module.inputs[0].name, "input-with.dots_and_underscores spaces");
        assert_eq!(special_module.inputs[1].name, "different-input with!special@chars#and%spaces");
        assert_eq!(special_module.outputs[0].name, "output with\ttabs\nand\rnewlines");
    }

    #[test]
    fn test_validation_performance_with_large_modules() {
        // Test that validation performs reasonably with large modules
        
        let mut large_module = Module::new("performance_test_module");
        
        // Add many inputs (this is unusual but valid)
        for i in 0..100 {
            large_module.inputs.push(Value {
                name: format!("global_input_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
        }
        
        // Add many outputs (this is unusual but valid)
        for i in 0..50 {
            large_module.outputs.push(Value {
                name: format!("global_output_{}", i),
                ty: Type::F32,
                shape: vec![10],
            });
        }
        
        // Add many operations
        for op_idx in 0..500 {
            let mut op = Operation::new(&format!("op_{}", op_idx));
            
            // Add a few inputs and outputs to each operation
            for inp_idx in 0..5 {
                op.inputs.push(Value {
                    name: format!("op{}_input_{}", op_idx, inp_idx),
                    ty: Type::F32,
                    shape: vec![inp_idx + 1],
                });
            }
            
            for out_idx in 0..3 {
                op.outputs.push(Value {
                    name: format!("op{}_output_{}", op_idx, out_idx),
                    ty: Type::F32,
                    shape: vec![out_idx + 2],
                });
            }
            
            large_module.add_operation(op);
        }
        
        // Measure validation time (this is more of a performance test)
        let start_time = std::time::Instant::now();
        let result = validation_utils::validate_module(&large_module);
        let duration = start_time.elapsed();
        
        // Validation should complete and succeed (the module is valid)
        assert!(result.is_ok());
        
        // The validation should happen in a reasonable time
        assert!(duration.as_millis() < 1000); // Should complete in under 1 second
        
        // Verify the module structure
        assert_eq!(large_module.inputs.len(), 100);
        assert_eq!(large_module.outputs.len(), 50);
        assert_eq!(large_module.operations.len(), 500);
        
        // Check that the first and last operations are as expected
        assert_eq!(large_module.operations[0].op_type, "op_0");
        assert_eq!(large_module.operations[499].op_type, "op_499");
        
        // Check inputs/outputs of first and last operations
        assert_eq!(large_module.operations[0].inputs.len(), 5);
        assert_eq!(large_module.operations[0].outputs.len(), 3);
        assert_eq!(large_module.operations[499].inputs.len(), 5);
        assert_eq!(large_module.operations[499].outputs.len(), 3);
    }
}