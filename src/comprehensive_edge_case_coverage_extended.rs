//! 扩展边界情况测试 - 覆盖尚未充分测试的边界场景
//! 使用标准库 assert! 和 assert_eq! 进行验证

#[cfg(test)]
mod comprehensive_edge_case_coverage_extended {
    use crate::ir::{Module, Value, Type, Operation};
    use crate::utils::{math_utils, validation_utils};

    /// 测试 1: GCD 边界情况 - 相同值、1 和大质数
    #[test]
    fn test_gcd_boundary_cases() {
        // 相同值的 GCD 是自身
        assert_eq!(math_utils::gcd(42, 42), 42);
        assert_eq!(math_utils::gcd(1, 1), 1);
        assert_eq!(math_utils::gcd(0, 0), 0);

        // 与 1 的 GCD 始终为 1
        assert_eq!(math_utils::gcd(1, 1000000), 1);
        assert_eq!(math_utils::gcd(1000000, 1), 1);

        // 连续互质数
        assert_eq!(math_utils::gcd(17, 16), 1);
        assert_eq!(math_utils::gcd(1000003, 1000002), 1);

        // 大数 GCD
        assert_eq!(math_utils::gcd(1000000, 500000), 500000);
        assert_eq!(math_utils::gcd(999999, 333333), 333333);
    }

    /// 测试 2: LCM 边界情况 - 溢出预防和大质数
    #[test]
    fn test_lcm_boundary_cases() {
        // 相同值的 LCM 是自身
        assert_eq!(math_utils::lcm(7, 7), 7);
        assert_eq!(math_utils::lcm(100, 100), 100);

        // 1 与任何数的 LCM 是该数本身
        assert_eq!(math_utils::lcm(1, 1000), 1000);
        assert_eq!(math_utils::lcm(1000, 1), 1000);

        // 互质数的 LCM 是它们的乘积
        assert_eq!(math_utils::lcm(7, 9), 63);
        assert_eq!(math_utils::lcm(13, 17), 221);

        // 一个数是另一个数的倍数
        assert_eq!(math_utils::lcm(10, 20), 20);
        assert_eq!(math_utils::lcm(20, 10), 20);
    }

    /// 测试 3: next_power_of_2 边界情况 - 大数和边界值
    #[test]
    fn test_next_power_of_2_extended() {
        // 2 的幂次方应该返回自身
        assert_eq!(math_utils::next_power_of_2(1), 1);
        assert_eq!(math_utils::next_power_of_2(2), 2);
        assert_eq!(math_utils::next_power_of_2(4), 4);
        assert_eq!(math_utils::next_power_of_2(8), 8);
        assert_eq!(math_utils::next_power_of_2(16), 16);
        assert_eq!(math_utils::next_power_of_2(32), 32);
        assert_eq!(math_utils::next_power_of_2(64), 64);
        assert_eq!(math_utils::next_power_of_2(128), 128);
        assert_eq!(math_utils::next_power_of_2(256), 256);
        assert_eq!(math_utils::next_power_of_2(512), 512);
        assert_eq!(math_utils::next_power_of_2(1024), 1024);
        assert_eq!(math_utils::next_power_of_2(2048), 2048);
        assert_eq!(math_utils::next_power_of_2(4096), 4096);
        assert_eq!(math_utils::next_power_of_2(8192), 8192);
        assert_eq!(math_utils::next_power_of_2(16384), 16384);
        assert_eq!(math_utils::next_power_of_2(32768), 32768);
        assert_eq!(math_utils::next_power_of_2(65536), 65536);

        // 接近 2 的幂次方的值
        assert_eq!(math_utils::next_power_of_2(3), 4);
        assert_eq!(math_utils::next_power_of_2(7), 8);
        assert_eq!(math_utils::next_power_of_2(15), 16);
        assert_eq!(math_utils::next_power_of_2(31), 32);
        assert_eq!(math_utils::next_power_of_2(63), 64);
        assert_eq!(math_utils::next_power_of_2(127), 128);
        assert_eq!(math_utils::next_power_of_2(255), 256);
        assert_eq!(math_utils::next_power_of_2(511), 512);
        assert_eq!(math_utils::next_power_of_2(1023), 1024);
        assert_eq!(math_utils::next_power_of_2(2047), 2048);
    }

    /// 测试 4: round_up_to_multiple 边界情况 - 特殊值和大数
    #[test]
    fn test_round_up_to_multiple_boundary() {
        // 值是 multiple 的倍数
        assert_eq!(math_utils::round_up_to_multiple(0, 8), 0);
        assert_eq!(math_utils::round_up_to_multiple(8, 8), 8);
        assert_eq!(math_utils::round_up_to_multiple(16, 8), 16);
        assert_eq!(math_utils::round_up_to_multiple(1024, 1024), 1024);

        // multiple 为 1 时返回原值
        assert_eq!(math_utils::round_up_to_multiple(0, 1), 0);
        assert_eq!(math_utils::round_up_to_multiple(100, 1), 100);
        assert_eq!(math_utils::round_up_to_multiple(999999, 1), 999999);

        // 值小于 multiple
        assert_eq!(math_utils::round_up_to_multiple(1, 8), 8);
        assert_eq!(math_utils::round_up_to_multiple(7, 8), 8);
        assert_eq!(math_utils::round_up_to_multiple(5, 16), 16);

        // 大数情况
        assert_eq!(math_utils::round_up_to_multiple(1000000, 8), 1000000);
        assert_eq!(math_utils::round_up_to_multiple(1000001, 8), 1000008);
        assert_eq!(math_utils::round_up_to_multiple(1000007, 8), 1000008);
    }

    /// 测试 5: 验证操作中重复输入名称的检测
    #[test]
    fn test_validate_operation_duplicate_input_names() {
        let mut op = Operation::new("test_op");

        // 添加重复名称的输入
        op.inputs.push(Value {
            name: "duplicate_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op.inputs.push(Value {
            name: "duplicate_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        let result = validation_utils::validate_operation(&op);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Duplicate input name"));
    }

    /// 测试 6: 验证操作中输入和输出名称冲突
    #[test]
    fn test_validate_operation_input_output_name_conflict() {
        let mut op = Operation::new("conflict_op");

        op.inputs.push(Value {
            name: "conflicting_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        op.outputs.push(Value {
            name: "conflicting_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        let result = validation_utils::validate_operation(&op);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Input and output share the same name"));
    }

    /// 测试 7: 模块名称边界情况 - 空名称和超长名称
    #[test]
    fn test_validate_module_name_boundary() {
        // 测试空名称
        let empty_module = Module::new("");
        let result = validation_utils::validate_module(&empty_module);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Module name cannot be empty"));

        // 测试正常名称
        let normal_module = Module::new("normal_module");
        assert!(validation_utils::validate_module(&normal_module).is_ok());

        // 测试特殊字符名称
        let special_module = Module::new("module_with-special.chars_123");
        assert!(validation_utils::validate_module(&special_module).is_ok());
    }

    /// 测试 8: 深度嵌套的张量类型验证
    #[test]
    fn test_validate_deeply_nested_tensor_type() {
        // 创建多层嵌套的张量类型
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

        // 验证深度嵌套的类型
        let result = validation_utils::validate_type(&level3);
        assert!(result.is_ok());
    }

    /// 测试 9: 模块级输入输出名称冲突验证
    #[test]
    fn test_validate_module_level_name_conflicts() {
        let mut module = Module::new("conflict_module");

        // 添加同名输入和输出
        module.inputs.push(Value {
            name: "conflicting_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });
        module.outputs.push(Value {
            name: "conflicting_name".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        let result = validation_utils::validate_module_uniqueness(&module);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Module input and output share the same name"));
    }

    /// 测试 10: 操作输入与模块输入名称冲突验证
    #[test]
    fn test_validate_operation_module_input_name_conflict() {
        let mut module = Module::new("name_conflict_module");

        // 添加模块输入
        module.inputs.push(Value {
            name: "module_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        // 添加操作，其输入与模块输入同名
        let mut op = Operation::new("test_op");
        op.inputs.push(Value {
            name: "module_input".to_string(),
            ty: Type::F32,
            shape: vec![10],
        });

        module.add_operation(op);

        let result = validation_utils::validate_module_uniqueness(&module);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Operation input"));
    }
}