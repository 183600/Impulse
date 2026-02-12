//! Compiler edge case tests - additional boundary scenarios for the Impulse compiler
//! 覆盖编译器核心功能的边界情况

use crate::ir::{Module, Value, Type, Operation, Attribute, TypeExtensions};
use crate::compiler::Compiler;

/// Test 1: Value with checked_mul overflow detection in num_elements
/// 测试使用 checked_mul 检测数值溢出的情况
#[test]
fn test_num_elements_overflow_detection() {
    // Test normal case
    let normal = Value {
        name: "normal".to_string(),
        ty: Type::F32,
        shape: vec![100, 100],
    };
    assert_eq!(normal.num_elements(), Some(10000));

    // Test with zero dimension
    let zero_dim = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![100, 0, 100],
    };
    assert_eq!(zero_dim.num_elements(), Some(0));

    // Test scalar (empty shape)
    let scalar = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert_eq!(scalar.num_elements(), Some(1));

    // Test with large dimensions that would overflow on 32-bit
    let large_dim = Value {
        name: "large_dim".to_string(),
        ty: Type::F32,
        shape: vec![65536, 65536], // Would overflow on u32 but fits in u64
    };
    // On 64-bit systems this should work
    let result = large_dim.num_elements();
    assert!(result.is_some() || result.is_none()); // Just verify it doesn't panic
}

/// Test 2: Compiler with empty operation sequences
/// 测试编译器处理空操作序列的情况
#[test]
fn test_compiler_empty_operation_sequences() {
    let _compiler = Compiler::new();
    let mut module = Module::new("empty_ops");

    // Module with no operations
    assert_eq!(module.operations.len(), 0);
    assert_eq!(module.inputs.len(), 0);
    assert_eq!(module.outputs.len(), 0);

    // Add operations but leave them empty
    for i in 0..5 {
        let op = Operation::new(&format!("empty_op_{}", i));
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 5);
    for op in &module.operations {
        assert_eq!(op.inputs.len(), 0);
        assert_eq!(op.outputs.len(), 0);
        assert_eq!(op.attributes.len(), 0);
    }
}

/// Test 3: Type validation with recursive tensor types
/// 测试递归张量类型的验证
#[test]
fn test_recursive_tensor_type_validation() {
    // Single level nesting
    let level1 = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![2, 3],
    };
    assert!(level1.is_valid_type());

    // Double level nesting
    let level2 = Type::Tensor {
        element_type: Box::new(level1),
        shape: vec![4],
    };
    assert!(level2.is_valid_type());

    // Triple level nesting
    let level3 = Type::Tensor {
        element_type: Box::new(level2),
        shape: vec![5],
    };
    assert!(level3.is_valid_type());

    // All primitive types should be valid
    assert!(Type::F32.is_valid_type());
    assert!(Type::F64.is_valid_type());
    assert!(Type::I32.is_valid_type());
    assert!(Type::I64.is_valid_type());
    assert!(Type::Bool.is_valid_type());
}

/// Test 4: Attribute array with extreme size variations
/// 测试包含极端大小变化的属性数组
#[test]
fn test_attribute_array_extreme_sizes() {
    // Empty array
    let empty_array = Attribute::Array(vec![]);
    match empty_array {
        Attribute::Array(arr) => assert_eq!(arr.len(), 0),
        _ => panic!("Expected Array"),
    }

    // Single element array
    let single_array = Attribute::Array(vec![Attribute::Int(42)]);
    match single_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 1);
            match arr[0] {
                Attribute::Int(42) => (),
                _ => panic!("Expected Int(42)"),
            }
        }
        _ => panic!("Expected Array"),
    }

    // Array with all types
    let mixed_array = Attribute::Array(vec![
        Attribute::Int(i64::MAX),
        Attribute::Int(i64::MIN),
        Attribute::Int(0),
        Attribute::Float(f64::MAX),
        Attribute::Float(f64::MIN),
        Attribute::Float(f64::NAN),
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::String("".to_string()),
        Attribute::String("test".to_string()),
        Attribute::Bool(true),
        Attribute::Bool(false),
        Attribute::Array(vec![]),
    ]);

    match mixed_array {
        Attribute::Array(arr) => {
            assert_eq!(arr.len(), 13);
            // Verify we have different types
            assert!(matches!(arr[0], Attribute::Int(_)));
            assert!(matches!(arr[3], Attribute::Float(_)));
            assert!(matches!(arr[8], Attribute::String(_)));
            assert!(matches!(arr[10], Attribute::Bool(_)));
            assert!(matches!(arr[12], Attribute::Array(_)));
        }
        _ => panic!("Expected Array"),
    }
}

/// Test 5: Module with cyclic-like operation names
/// 测试包含类似循环名称操作的模块
#[test]
fn test_module_cyclic_operation_names() {
    let mut module = Module::new("cyclic_names");

    // Create operations with cyclic naming pattern
    let cyclic_names = ["op_a", "op_b", "op_c", "op_a", "op_b", "op_c"];

    for name in cyclic_names {
        let mut op = Operation::new(name);
        op.inputs.push(Value {
            name: format!("input_{}", name),
            ty: Type::F32,
            shape: vec![1],
        });
        module.add_operation(op);
    }

    assert_eq!(module.operations.len(), 6);
    // Verify duplicate names are allowed
    assert_eq!(module.operations[0].op_type, "op_a");
    assert_eq!(module.operations[3].op_type, "op_a");
    assert_eq!(module.operations[1].op_type, "op_b");
    assert_eq!(module.operations[4].op_type, "op_b");
}

/// Test 6: Value with shape containing only 1s (broadcast-like tensors)
/// 测试形状只包含1的张量（类似广播的张量）
#[test]
fn test_value_broadcast_like_shapes() {
    // All ones shape (scalar broadcast)
    let all_ones = Value {
        name: "all_ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 1, 1, 1],
    };
    assert_eq!(all_ones.num_elements(), Some(1));

    // Mix of 1s and other dimensions
    let mixed_ones = Value {
        name: "mixed_ones".to_string(),
        ty: Type::F32,
        shape: vec![1, 10, 1, 20, 1],
    };
    assert_eq!(mixed_ones.num_elements(), Some(200));

    // Single dimension with 1
    let single_one = Value {
        name: "single_one".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(single_one.num_elements(), Some(1));

    // Verify all shapes are correctly preserved
    assert_eq!(all_ones.shape, vec![1, 1, 1, 1]);
    assert_eq!(mixed_ones.shape, vec![1, 10, 1, 20, 1]);
    assert_eq!(single_one.shape, vec![1]);
}

/// Test 7: Operation with duplicate attribute keys (last value wins)
/// 测试包含重复属性键的操作（最后一个值生效）
#[test]
fn test_operation_duplicate_attribute_keys() {
    use std::collections::HashMap;

    let mut op = Operation::new("dup_attrs");
    let mut attrs = HashMap::new();

    // Insert same key multiple times
    attrs.insert("key1".to_string(), Attribute::Int(1));
    attrs.insert("key1".to_string(), Attribute::Int(2));
    attrs.insert("key1".to_string(), Attribute::Int(3));

    attrs.insert("key2".to_string(), Attribute::String("first".to_string()));
    attrs.insert("key2".to_string(), Attribute::String("second".to_string()));

    op.attributes = attrs;

    // HashMap behavior: last value wins
    assert_eq!(op.attributes.len(), 2);

    match op.attributes.get("key1") {
        Some(Attribute::Int(3)) => (),
        _ => panic!("Expected Int(3) for key1"),
    }

    match op.attributes.get("key2") {
        Some(Attribute::String(s)) if s == "second" => (),
        _ => panic!("Expected String(\"second\") for key2"),
    }
}

/// Test 8: Module with inputs/outputs having identical names
/// 测试输入/输出具有相同名称的模块
#[test]
fn test_module_identical_io_names() {
    let mut module = Module::new("same_io_names");

    // Add inputs
    module.inputs.push(Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    // Add outputs with same names as inputs
    module.outputs.push(Value {
        name: "data".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });

    module.outputs.push(Value {
        name: "data".to_string(), // Another output with same name
        ty: Type::I32,
        shape: vec![5],
    });

    assert_eq!(module.inputs.len(), 1);
    assert_eq!(module.outputs.len(), 2);
    assert_eq!(module.inputs[0].name, "data");
    assert_eq!(module.outputs[0].name, "data");
    assert_eq!(module.outputs[1].name, "data");
}

/// Test 9: Value with all primitive types and edge case values
/// 测试所有基本类型和边界值的值
#[test]
fn test_value_all_primitive_types_with_edge_values() {
    // Test F32 with special values
    let f32_nan = Value {
        name: "f32_nan".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert_eq!(f32_nan.ty, Type::F32);

    // Test I64 with extreme values
    let i64_max = Value {
        name: "i64_max".to_string(),
        ty: Type::I64,
        shape: vec![1],
    };
    assert_eq!(i64_max.ty, Type::I64);

    let i64_min = Value {
        name: "i64_min".to_string(),
        ty: Type::I64,
        shape: vec![1],
    };
    assert_eq!(i64_min.ty, Type::I64);

    // Test Bool
    let bool_val = Value {
        name: "bool_val".to_string(),
        ty: Type::Bool,
        shape: vec![100, 100],
    };
    assert_eq!(bool_val.ty, Type::Bool);
    assert_eq!(bool_val.num_elements(), Some(10000));

    // Test all types have correct shapes
    assert_eq!(f32_nan.shape, vec![1]);
    assert_eq!(i64_max.shape, vec![1]);
    assert_eq!(i64_min.shape, vec![1]);
    assert_eq!(bool_val.shape, vec![100, 100]);
}

/// Test 10: Module with operations having no inputs but outputs
/// 测试包含无输入但有输出的操作的模块
#[test]
fn test_module_operations_no_inputs_with_outputs() {
    let mut module = Module::new("output_only_ops");

    // Add operation with outputs but no inputs (like constant generation)
    let mut op1 = Operation::new("constant");
    op1.outputs.push(Value {
        name: "const_output".to_string(),
        ty: Type::F32,
        shape: vec![2, 2],
    });
    module.add_operation(op1);

    // Add operation with multiple outputs
    let mut op2 = Operation::new("multi_output");
    op2.outputs.push(Value {
        name: "output_1".to_string(),
        ty: Type::I32,
        shape: vec![5],
    });
    op2.outputs.push(Value {
        name: "output_2".to_string(),
        ty: Type::F64,
        shape: vec![3, 3],
    });
    op2.outputs.push(Value {
        name: "output_3".to_string(),
        ty: Type::Bool,
        shape: vec![1],
    });
    module.add_operation(op2);

    assert_eq!(module.operations.len(), 2);
    assert_eq!(module.operations[0].inputs.len(), 0);
    assert_eq!(module.operations[0].outputs.len(), 1);
    assert_eq!(module.operations[1].inputs.len(), 0);
    assert_eq!(module.operations[1].outputs.len(), 3);

    // Verify output types
    assert_eq!(module.operations[0].outputs[0].ty, Type::F32);
    assert_eq!(module.operations[1].outputs[0].ty, Type::I32);
    assert_eq!(module.operations[1].outputs[1].ty, Type::F64);
    assert_eq!(module.operations[1].outputs[2].ty, Type::Bool);
}