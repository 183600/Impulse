//! 边界测试套件 - 覆盖更多边界情况的测试用例
//! Boundary test suite covering more edge cases

use crate::ir::{Module, Value, Type, Operation};
use crate::utils::calculate_tensor_size;
use crate::utils::math_utils::{gcd, lcm, next_power_of_2, round_up_to_multiple};
use crate::utils::validation_utils::{validate_operation, validate_module, validate_value_shape};
use crate::runtime::{Device, ExecutionContext, Runtime};
use crate::passes::{PassManager, ConstantFoldPass, Pass};

/// 测试1: 计算张量大小时，极端形状和类型的组合
#[test]
fn test_calculate_tensor_size_extreme_combinations() {
    // 测试空形状（标量）
    let scalar_f32 = calculate_tensor_size(&Type::F32, &[]).unwrap();
    assert_eq!(scalar_f32, 4);
    
    let scalar_i64 = calculate_tensor_size(&Type::I64, &[]).unwrap();
    assert_eq!(scalar_i64, 8);
    
    // 测试包含零的形状
    let zero_size = calculate_tensor_size(&Type::F32, &[0, 100, 100]).unwrap();
    assert_eq!(zero_size, 0);
    
    // 测试单元素张量
    let single_element = calculate_tensor_size(&Type::F32, &[1]).unwrap();
    assert_eq!(single_element, 4);
    
    // 测试Bool类型（1字节）
    let bool_large = calculate_tensor_size(&Type::Bool, &[1000, 1000]).unwrap();
    assert_eq!(bool_large, 1_000_000);
    
    // 测试大尺寸张量
    let large_f32 = calculate_tensor_size(&Type::F32, &[1024, 1024]).unwrap();
    assert_eq!(large_f32, 1024 * 1024 * 4);
    
    // 测试多维度形状
    let multidim = calculate_tensor_size(&Type::I32, &[2, 3, 4, 5]).unwrap();
    assert_eq!(multidim, 2 * 3 * 4 * 5 * 4);
}

/// 测试2: GCD和LCM函数的边界情况
#[test]
fn test_gcd_lcm_boundary_cases() {
    // 测试GCD边界情况
    assert_eq!(gcd(0, 0), 0);
    assert_eq!(gcd(0, 5), 5);
    assert_eq!(gcd(5, 0), 5);
    assert_eq!(gcd(1, 1), 1);
    assert_eq!(gcd(1, 1000), 1);
    assert_eq!(gcd(1000, 1), 1);
    
    // 测试互质数
    assert_eq!(gcd(17, 13), 1);
    assert_eq!(gcd(999983, 999979), 1); // 两个大质数
    
    // 测试LCM边界情况
    assert_eq!(lcm(0, 0), 0);
    assert_eq!(lcm(0, 5), 0);
    assert_eq!(lcm(5, 0), 0);
    assert_eq!(lcm(1, 1), 1);
    assert_eq!(lcm(1, 1000), 1000);
    
    // 测试大数
    assert_eq!(lcm(100000, 100000), 100000);
    assert_eq!(lcm(12, 18), 36);
}

/// 测试3: next_power_of_2函数的极端边界情况
#[test]
fn test_next_power_of_2_extreme_edge_cases() {
    // 测试基本值
    assert_eq!(next_power_of_2(0), 1);
    assert_eq!(next_power_of_2(1), 1);
    assert_eq!(next_power_of_2(2), 2);
    
    // 测试2的幂次方
    assert_eq!(next_power_of_2(256), 256);
    assert_eq!(next_power_of_2(65536), 65536);
    
    // 测试2的幂次方减1
    assert_eq!(next_power_of_2(255), 256);
    assert_eq!(next_power_of_2(65535), 65536);
    
    // 测试大数
    assert_eq!(next_power_of_2(1000000), 1048576); // 2^20
    assert_eq!(next_power_of_2(50000000), 67108864); // 2^26
}

/// 测试4: round_up_to_multiple函数的边界情况
#[test]
fn test_round_up_to_multiple_boundary() {
    // 测试multiple=0的特殊情况
    assert_eq!(round_up_to_multiple(100, 0), 100);
    assert_eq!(round_up_to_multiple(0, 0), 0);
    
    // 测试value=0
    assert_eq!(round_up_to_multiple(0, 10), 0);
    assert_eq!(round_up_to_multiple(0, 1), 0);
    
    // 测试value等于multiple
    assert_eq!(round_up_to_multiple(100, 100), 100);
    assert_eq!(round_up_to_multiple(1024, 1024), 1024);
    
    // 测试value等于multiple减1
    assert_eq!(round_up_to_multiple(99, 100), 100);
    assert_eq!(round_up_to_multiple(1023, 1024), 1024);
    
    // 测试value等于multiple加1
    assert_eq!(round_up_to_multiple(101, 100), 200);
    assert_eq!(round_up_to_multiple(1025, 1024), 2048);
    
    // 测试multiple=1（应该返回原值）
    assert_eq!(round_up_to_multiple(100, 1), 100);
    assert_eq!(round_up_to_multiple(0, 1), 0);
}

/// 测试5: Device和MemoryInfo的边界情况
#[test]
fn test_device_memory_info_edge_cases() {
    // 测试CPU设备
    let cpu = Device::Cpu;
    let cpu_info = cpu.memory_info();
    assert!(cpu_info.total > 0);
    assert!(cpu_info.free > 0);
    assert!(cpu_info.free <= cpu_info.total);
    
    // 测试CUDA设备（不同device_id）
    let cuda0 = Device::Cuda { device_id: 0 };
    let cuda1 = Device::Cuda { device_id: 1 };
    assert_ne!(cuda0, cuda1);
    
    let cuda0_info = cuda0.memory_info();
    let cuda1_info = cuda1.memory_info();
    assert!(cuda0_info.total > 0);
    assert!(cuda1_info.total > 0);
    
    // 测试NPU设备（不同vendor）
    let npu1 = Device::Npu { vendor: "vendor1".to_string(), device_id: 0 };
    let npu2 = Device::Npu { vendor: "vendor2".to_string(), device_id: 0 };
    assert_ne!(npu1, npu2);
    
    let npu1_info = npu1.memory_info();
    assert!(npu1_info.total > 0);
    assert!(npu1_info.free > 0);
    
    // 测试设备名称
    assert_eq!(cpu.name(), "CPU");
    assert_eq!(cuda0.name(), "CUDA:0");
    assert_eq!(cuda1.name(), "CUDA:1");
    assert_eq!(npu1.name(), "vendor1:0");
}

/// 测试6: PassManager的边界情况 - 测试空Pass和重复Pass
#[test]
fn test_pass_manager_edge_cases() {
    // 测试空PassManager
    let empty_pm = PassManager::new();
    let mut empty_module = Module::new("empty_test");
    let result = empty_pm.run_passes(&mut empty_module);
    assert!(result.is_ok());
    assert_eq!(empty_module.operations.len(), 0);
    
    // 测试添加相同类型的多个Pass
    let mut pm = PassManager::new();
    pm.add_pass(Box::new(ConstantFoldPass));
    pm.add_pass(Box::new(ConstantFoldPass));
    pm.add_pass(Box::new(ConstantFoldPass));
    assert_eq!(pm.passes.len(), 3);
    
    // 测试运行相同类型的多个Pass
    let mut test_module = Module::new("test");
    let result = pm.run_passes(&mut test_module);
    assert!(result.is_ok());
    
    // 测试自定义Pass
    struct CustomTestPass;
    
    impl Pass for CustomTestPass {
        fn run(&self, module: &mut Module) -> anyhow::Result<()> {
            module.name.push_str("_modified");
            Ok(())
        }
        
        fn name(&self) -> &'static str {
            "CustomTestPass"
        }
    }
    
    let mut pm2 = PassManager::new();
    pm2.add_pass(Box::new(CustomTestPass));
    
    let mut custom_module = Module::new("custom");
    let result = pm2.run_passes(&mut custom_module);
    assert!(result.is_ok());
    assert_eq!(custom_module.name, "custom_modified");
}

/// 测试7: validate_value_shape的边界情况
#[test]
fn test_validate_value_shape_boundary() {
    // 测试正常形状
    let normal_value = Value {
        name: "normal".to_string(),
        ty: Type::F32,
        shape: vec![10, 20, 30],
    };
    assert!(validate_value_shape(&normal_value).is_ok());
    
    // 测试空形状（标量）
    let scalar_value = Value {
        name: "scalar".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    assert!(validate_value_shape(&scalar_value).is_ok());
    
    // 测试包含零的形状
    let zero_dim_value = Value {
        name: "zero_dim".to_string(),
        ty: Type::F32,
        shape: vec![10, 0, 5],
    };
    assert!(validate_value_shape(&zero_dim_value).is_ok());
    
    // 测试单个元素的形状
    let single_element = Value {
        name: "single".to_string(),
        ty: Type::F32,
        shape: vec![1],
    };
    assert!(validate_value_shape(&single_element).is_ok());
    
    // 测试大维度（但合理）
    let large_dim_value = Value {
        name: "large_dim".to_string(),
        ty: Type::F32,
        shape: vec![10000, 100],
    };
    assert!(validate_value_shape(&large_dim_value).is_ok());
    
    // 测试极端维度（应该失败）
    let extreme_dim_value = Value {
        name: "extreme_dim".to_string(),
        ty: Type::F32,
        shape: vec![2_000_000], // 超过1,000,000的限制
    };
    assert!(validate_value_shape(&extreme_dim_value).is_err());
}

/// 测试8: validate_operation的边界情况 - 测试重复名称和冲突
#[test]
fn test_validate_operation_boundary() {
    // 测试正常的操作
    let mut normal_op = Operation::new("add");
    normal_op.inputs.push(Value {
        name: "input1".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    normal_op.inputs.push(Value {
        name: "input2".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    normal_op.outputs.push(Value {
        name: "output".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    assert!(validate_operation(&normal_op).is_ok());
    
    // 测试空操作
    let empty_op = Operation::new("noop");
    assert!(validate_operation(&empty_op).is_ok());
    
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
    assert!(validate_operation(&duplicate_input_op).is_err());
    
    // 测试输入输出名称冲突
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
    assert!(validate_operation(&conflict_op).is_err());
    
    // 测试超长操作类型名称
    let mut long_name_op = Operation::new(&"a".repeat(1_000_001));
    long_name_op.inputs.push(Value {
        name: "input".to_string(),
        ty: Type::F32,
        shape: vec![1],
    });
    assert!(validate_operation(&long_name_op).is_err());
}

/// 测试9: validate_module的边界情况
#[test]
fn test_validate_module_boundary() {
    // 测试正常模块
    let mut normal_module = Module::new("normal_module");
    normal_module.inputs.push(Value {
        name: "module_input".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    normal_module.outputs.push(Value {
        name: "module_output".to_string(),
        ty: Type::F32,
        shape: vec![10, 10],
    });
    assert!(validate_module(&normal_module).is_ok());
    
    // 测试空模块名称
    let empty_name_module = Module::new("");
    assert!(validate_module(&empty_name_module).is_err());
    
    // 测试模块输入输出名称冲突
    let mut conflict_module = Module::new("conflict_module");
    conflict_module.inputs.push(Value {
        name: "conflict".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    conflict_module.outputs.push(Value {
        name: "conflict".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    assert!(validate_module(&conflict_module).is_err());
    
    // 测试超长模块名称
    let long_name_module = Module::new(&"m".repeat(10_000_001));
    assert!(validate_module(&long_name_module).is_err());
    
    // 测试包含无效操作的模块
    let mut invalid_module = Module::new("invalid_module");
    let mut invalid_op = Operation::new("test");
    invalid_op.inputs.push(Value {
        name: "same".to_string(),
        ty: Type::F32,
        shape: vec![10],
    });
    invalid_op.inputs.push(Value {
        name: "same".to_string(),
        ty: Type::F32,
        shape: vec![20],
    });
    invalid_module.add_operation(invalid_op);
    assert!(validate_module(&invalid_module).is_err());
}

/// 测试10: Runtime和ExecutionContext的边界情况
#[test]
fn test_runtime_execution_context_boundary() {
    // 测试Runtime创建
    let mut runtime = Runtime::new();
    assert!(!runtime.devices.is_empty());
    assert!(runtime.devices.contains(&Device::Cpu));
    
    // 测试获取不存在的缓存模块
    assert!(runtime.get_cached_module("nonexistent").is_none());
    
    // 测试缓存和获取模块
    let test_module_data = vec![1u8, 2u8, 3u8, 4u8];
    runtime.cache_module("test_key".to_string(), test_module_data.clone());
    let cached = runtime.get_cached_module("test_key").unwrap();
    assert_eq!(cached, &test_module_data);
    
    // 测试创建ExecutionContext
    let ctx = runtime.create_context(None).unwrap();
    assert_eq!(ctx.device, runtime.default_device);
    
    // 测试使用CPU设备创建ExecutionContext
    let cpu_ctx = runtime.create_context(Some(Device::Cpu)).unwrap();
    assert_eq!(cpu_ctx.device, Device::Cpu);
    
    // 测试内存分配
    let mut alloc_ctx = ExecutionContext::new(Device::Cpu).unwrap();
    let handle1 = alloc_ctx.allocate_tensor(1024).unwrap();
    assert!(handle1.size >= 1024);
    
    let handle2 = alloc_ctx.allocate_tensor(2048).unwrap();
    assert!(handle2.size >= 2048);
    assert_ne!(handle1.id, handle2.id);
    
    // 测试最小内存分配
    let min_handle = alloc_ctx.allocate_tensor(1).unwrap();
    assert!(min_handle.size >= 1);
    
    // 测试大内存分配
    let large_handle = alloc_ctx.allocate_tensor(1024 * 1024).unwrap();
    assert!(large_handle.size >= 1024 * 1024);
}