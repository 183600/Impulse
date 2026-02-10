//! Advanced boundary tests for the Impulse compiler
//! 覆盖更多边界情况的测试用例

use crate::ir::{Module, Value, Type, Operation, Attribute};
use crate::utils::{calculate_tensor_size, type_to_string, count_operations_by_type, find_operations_by_type};
use crate::utils::math_utils::{next_power_of_2, round_up_to_multiple};
use crate::utils::validation_utils::{validate_operation, validate_module, validate_value_shape};
use crate::runtime::{Device, MemoryHandle, MemoryInfo};
use crate::passes::{PassManager, ConstantFoldPass, DeadCodeEliminationPass};
use crate::autotuning::{AutoTuner, TuneParams, SearchSpace};
use crate::backends::{BackendManager, CpuBackend, CudaBackend, Backend};
use crate::frontend::Frontend;
use crate::transforms::{TransformPipeline, create_transformer_optimization_pipeline};
use std::collections::HashMap;

/// 测试1: 计算张量大小时的边界情况 - 测试不同类型和形状的组合
#[test]
fn test_calculate_tensor_size_boundary_cases() {
    // 测试标量（空形状）
    let scalar_size = calculate_tensor_size(&Type::F32, &[]).unwrap();
    assert_eq!(scalar_size, 4); // F32 = 4 bytes, scalar = 1 element
    
    // 测试零维张量
    let zero_size = calculate_tensor_size(&Type::I32, &[0]).unwrap();
    assert_eq!(zero_size, 0);
    
    // 测试包含零的多维张量
    let zero_multi_size = calculate_tensor_size(&Type::F64, &[10, 0, 5]).unwrap();
    assert_eq!(zero_multi_size, 0);
    
    // 测试F64类型（8字节）
    let f64_size = calculate_tensor_size(&Type::F64, &[2, 3]).unwrap();
    assert_eq!(f64_size, 2 * 3 * 8);
    
    // 测试Bool类型（1字节）
    let bool_size = calculate_tensor_size(&Type::Bool, &[100, 100]).unwrap();
    assert_eq!(bool_size, 100 * 100 * 1);
    
    // 测试I64类型（8字节）
    let i64_size = calculate_tensor_size(&Type::I64, &[5, 10]).unwrap();
    assert_eq!(i64_size, 5 * 10 * 8);
}

/// 测试2: type_to_string 的边界情况 - 测试嵌套张量类型的字符串表示
#[test]
fn test_type_to_string_edge_cases() {
    // 测试基本类型
    assert_eq!(type_to_string(&Type::F32), "f32");
    assert_eq!(type_to_string(&Type::I64), "i64");
    assert_eq!(type_to_string(&Type::Bool), "bool");
    
    // 测试单维张量
    let tensor1d = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![10],
    };
    assert_eq!(type_to_string(&tensor1d), "tensor<f32, [10]>");
    
    // 测试多维张量
    let tensor3d = Type::Tensor {
        element_type: Box::new(Type::I32),
        shape: vec![2, 3, 4],
    };
    assert_eq!(type_to_string(&tensor3d), "tensor<i32, [2, 3, 4]>");
    
    // 测试嵌套张量
    let nested = Type::Tensor {
        element_type: Box::new(Type::Tensor {
            element_type: Box::new(Type::F64),
            shape: vec![2, 2],
        }),
        shape: vec![3],
    };
    let nested_str = type_to_string(&nested);
    assert!(nested_str.contains("tensor<f64, [2, 2]>"));
    assert!(nested_str.contains("[3]"));
    
    // 测试空形状张量（标量张量）
    let scalar_tensor = Type::Tensor {
        element_type: Box::new(Type::F32),
        shape: vec![],
    };
    assert_eq!(type_to_string(&scalar_tensor), "tensor<f32, []>");
}

/// 测试3: count_operations_by_type 的边界情况 - 测试空模块和重复操作类型
#[test]
fn test_count_operations_by_type_boundary() {
    // 测试空模块
    let empty_module = Module::new("empty");
    let empty_counts = count_operations_by_type(&empty_module);
    assert_eq!(empty_counts.len(), 0);
    
    // 测试单一操作类型
    let mut single_type_module = Module::new("single_type");
    for _ in 0..5 {
        single_type_module.add_operation(Operation::new("add"));
    }
    let single_counts = count_operations_by_type(&single_type_module);
    assert_eq!(single_counts.len(), 1);
    assert_eq!(single_counts.get("add"), Some(&5));
    
    // 测试多种操作类型
    let mut multi_type_module = Module::new("multi_type");
    multi_type_module.add_operation(Operation::new("add"));
    multi_type_module.add_operation(Operation::new("add"));
    multi_type_module.add_operation(Operation::new("multiply"));
    multi_type_module.add_operation(Operation::new("conv2d"));
    multi_type_module.add_operation(Operation::new("add"));
    let multi_counts = count_operations_by_type(&multi_type_module);
    assert_eq!(multi_counts.len(), 3);
    assert_eq!(multi_counts.get("add"), Some(&3));
    assert_eq!(multi_counts.get("multiply"), Some(&1));
    assert_eq!(multi_counts.get("conv2d"), Some(&1));
}

/// 测试4: find_operations_by_type 的边界情况 - 测试查找不存在类型和空结果
#[test]
fn test_find_operations_by_type_boundary() {
    let mut module = Module::new("test");
    
    // 添加一些操作
    module.add_operation(Operation::new("add"));
    module.add_operation(Operation::new("multiply"));
    module.add_operation(Operation::new("add"));
    
    // 查找存在的类型
    let add_ops = find_operations_by_type(&module, "add");
    assert_eq!(add_ops.len(), 2);
    
    // 查找存在的单一操作
    let mul_ops = find_operations_by_type(&module, "multiply");
    assert_eq!(mul_ops.len(), 1);
    
    // 查找不存在的类型
    let nonexistent_ops = find_operations_by_type(&module, "nonexistent");
    assert_eq!(nonexistent_ops.len(), 0);
    
    // 测试空模块
    let empty_module = Module::new("empty");
    let empty_ops = find_operations_by_type(&empty_module, "add");
    assert_eq!(empty_ops.len(), 0);
}

/// 测试5: next_power_of_2 和 round_up_to_multiple 的边界情况
#[test]
fn test_math_utilities_boundary_cases() {
    // 测试 next_power_of_2
    assert_eq!(next_power_of_2(0), 1);
    assert_eq!(next_power_of_2(1), 1);
    assert_eq!(next_power_of_2(2), 2);
    assert_eq!(next_power_of_2(3), 4);
    assert_eq!(next_power_of_2(1024), 1024);
    assert_eq!(next_power_of_2(1025), 2048);
    
    // 测试大数
    assert_eq!(next_power_of_2(1_000_000), 1_048_576); // 2^20
    assert_eq!(next_power_of_2(10_000_000), 16_777_216); // 2^24
    
    // 测试 round_up_to_multiple
    assert_eq!(round_up_to_multiple(5, 1), 5);
    assert_eq!(round_up_to_multiple(5, 0), 5); // 特殊情况：multiple=0
    assert_eq!(round_up_to_multiple(15, 16), 16);
    assert_eq!(round_up_to_multiple(32, 16), 32);
    assert_eq!(round_up_to_multiple(1, 1024), 1024);
    assert_eq!(round_up_to_multiple(1025, 1024), 2048);
    
    // 测试边界值
    assert_eq!(round_up_to_multiple(100, 100), 100);
    assert_eq!(round_up_to_multiple(101, 100), 200);
}

/// 测试6: Device 和 MemoryInfo 的边界情况
#[test]
fn test_device_and_memory_info_boundary() {
    // 测试不同设备类型的内存信息
    let cpu = Device::Cpu;
    let cpu_info = cpu.memory_info();
    assert!(cpu_info.total > 0);
    assert!(cpu_info.free > 0);
    assert!(cpu_info.free <= cpu_info.total);
    
    let cuda = Device::Cuda { device_id: 0 };
    let cuda_info = cuda.memory_info();
    assert!(cuda_info.total > 0);
    assert!(cuda_info.free > 0);
    
    let npu = Device::Npu { vendor: "test".to_string(), device_id: 0 };
    let npu_info = npu.memory_info();
    assert!(npu_info.total > 0);
    assert!(npu_info.free > 0);
    
    // 测试设备名称生成
    assert_eq!(cpu.name(), "CPU");
    assert_eq!(cuda.name(), "CUDA:0");
    assert_eq!(npu.name(), "test:0");
    
    // 测试设备相等性
    let cpu1 = Device::Cpu;
    let cpu2 = Device::Cpu;
    assert_eq!(cpu1, cpu2);
    
    let cuda0 = Device::Cuda { device_id: 0 };
    let cuda1 = Device::Cuda { device_id: 1 };
    assert_ne!(cuda0, cuda1);
}

/// 测试7: PassManager 的边界情况 - 测试重复添加、空传递失败等
#[test]
fn test_pass_manager_edge_cases() {
    // 测试空 PassManager
    let mut pm = PassManager::new();
    let mut empty_module = Module::new("empty");
    let result = pm.run_passes(&mut empty_module);
    assert!(result.is_ok());
    
    // 测试添加相同类型的多个 pass
    pm.add_pass(Box::new(ConstantFoldPass));
    pm.add_pass(Box::new(ConstantFoldPass));
    pm.add_pass(Box::new(ConstantFoldPass));
    assert_eq!(pm.passes.len(), 3);
    
    // 测试运行多个相同类型的 pass
    let mut test_module = Module::new("test");
    let result = pm.run_passes(&mut test_module);
    assert!(result.is_ok());
}

/// 测试8: AutoTuner 和 SearchSpace 的边界情况
#[test]
fn test_autotuner_boundary_cases() {
    // 测试空的 AutoTuner
    let tuner = AutoTuner::new();
    assert_eq!(tuner.cache.len(), 0);
    
    // 测试默认 SearchSpace
    let search_space = SearchSpace::default();
    assert!(!search_space.gemm_tile_sizes.is_empty());
    assert!(!search_space.conv_tile_sizes.is_empty());
    assert!(!search_space.attention_block_sizes.is_empty());
    assert!(!search_space.vector_widths.is_empty());
    
    // 测试生成候选参数的有效性
    let op = Operation::new("gemm");
    let candidates = tuner.generate_candidates(&op).unwrap();
    assert!(!candidates.is_empty());
    
    // 验证所有参数都是正值
    for candidate in candidates {
        if let TuneParams::Gemm { tile_m, tile_n, tile_k, vector_width } = candidate {
            assert!(tile_m > 0);
            assert!(tile_n > 0);
            assert!(tile_k > 0);
            assert!(vector_width > 0);
        }
    }
}

/// 测试9: Backend 的边界情况 - 测试未知目标和编译失败处理
#[test]
fn test_backend_edge_cases() {
    let mut manager = BackendManager::new();
    
    // 测试列出后端
    let backends = manager.list_backends();
    assert!(!backends.is_empty());
    assert!(backends.contains(&"cpu".to_string()));
    
    // 测试已知目标编译
    let module = Module::new("test");
    let cpu_result = manager.compile(&module, "cpu");
    assert!(cpu_result.is_ok());
    
    // 测试未知目标
    let unknown_result = manager.compile(&module, "unknown_target");
    assert!(unknown_result.is_err());
    
    // 测试 CPU 后端的属性
    let cpu_backend = CpuBackend::new();
    assert!(!cpu_backend.target_triple().is_empty());
    assert!(!cpu_backend.data_layout().is_empty());
    
    // 测试 CUDA 后端的属性
    let cuda_backend = CudaBackend::new();
    assert!(!cuda_backend.target_triple().is_empty());
    assert!(!cuda_backend.data_layout().is_empty());
}

/// 测试10: validate_operation 和 validate_module 的复杂边界情况
#[test]
fn test_validation_complex_boundary_cases() {
    // 测试重复输入名称
    let mut duplicate_input_op = Operation::new("test");
    duplicate_input_op.inputs.push(Value {
        name: "same_name".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    duplicate_input_op.inputs.push(Value {
        name: "same_name".to_string(),
        ty: Type::F32,
        shape: vec![20],
    });
    let result = validate_operation(&duplicate_input_op);
    assert!(result.is_err());
    
    // 测试输入和输出名称冲突
    let mut conflict_op = Operation::new("test");
    conflict_op.inputs.push(Value {
        name: "conflict".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    conflict_op.outputs.push(Value {
        name: "conflict".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    let result = validate_operation(&conflict_op);
    assert!(result.is_err());
    
    // 测试模块级名称冲突
    let mut module = Module::new("test");
    module.inputs.push(Value {
        name: "module_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    module.outputs.push(Value {
        name: "module_input".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    let result = validate_module(&module);
    assert!(result.is_err());
    
    // 测试空模块名称
    let empty_name_module = Module::new("");
    let result = validate_module(&empty_name_module);
    assert!(result.is_err());
}