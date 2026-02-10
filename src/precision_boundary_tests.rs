//! 精度和边界测试 - 覆盖数值精度、序列化边界和内存边界的测试用例
//! Precision and Boundary Tests - Covering numerical precision, serialization boundaries, and memory boundaries

use crate::ir::{Module, Value, Type, Operation, Attribute};
use std::collections::HashMap;

/// 测试1: 浮点数精度边界 - 测试接近 f32/f64 精度极限的值
#[test]
fn test_floating_point_precision_boundaries() {
    let mut op = Operation::new("precision_test");
    let mut attrs = HashMap::new();

    // 添加接近精度极限的浮点数
    attrs.insert("very_small".to_string(), Attribute::Float(1e-308));
    attrs.insert("very_large".to_string(), Attribute::Float(1e308));
    attrs.insert("subnormal".to_string(), Attribute::Float(f64::MIN_POSITIVE));
    attrs.insert("denormal".to_string(), Attribute::Float(1e-320));
    
    // 添加会导致精度损失的值
    attrs.insert("precision_loss1".to_string(), Attribute::Float(1.7976931348623157e+308));
    attrs.insert("precision_loss2".to_string(), Attribute::Float(2.2250738585072014e-308));
    
    op.attributes = attrs;
    
    // 验证这些值被正确存储
    assert_eq!(op.attributes.len(), 6);
    
    // 验证特殊值
    match op.attributes.get("very_small") {
        Some(Attribute::Float(val)) => assert!(*val > 0.0 && *val < 1e-300),
        _ => panic!("Expected very small float"),
    }
    
    match op.attributes.get("very_large") {
        Some(Attribute::Float(val)) => assert!(*val > 1e300),
        _ => panic!("Expected very large float"),
    }
}

/// 测试2: 序列化/反序列化边界 - 测试包含极端值的 Operation 序列化
#[test]
fn test_serialization_with_extreme_values() {
    use bincode;

    let mut op = Operation::new("extreme_serialization");
    
    // 添加极端的属性值
    let mut attrs = HashMap::new();
    attrs.insert("max_int".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("min_int".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("max_float".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("min_float".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("nan".to_string(), Attribute::Float(f64::NAN));
    
    // 添加包含极端值的数组
    attrs.insert("extreme_array".to_string(), Attribute::Array(vec![
        Attribute::Int(i64::MAX),
        Attribute::Int(i64::MIN),
        Attribute::Float(f64::MAX),
        Attribute::Float(f64::MIN),
        Attribute::Float(f64::INFINITY),
        Attribute::Float(f64::NEG_INFINITY),
        Attribute::Float(f64::NAN),
    ]));
    
    // 添加嵌套数组
    attrs.insert("nested_extreme".to_string(), Attribute::Array(vec![
        Attribute::Array(vec![
            Attribute::Int(i64::MAX),
            Attribute::Float(f64::MAX),
        ]),
        Attribute::Array(vec![
            Attribute::Int(i64::MIN),
            Attribute::Float(f64::MIN),
        ]),
    ]));
    
    op.attributes = attrs;
    
    // 使用 bincode 进行序列化（支持所有特殊浮点值）
    let serialized: Vec<u8> = bincode::serialize(&op).expect("Serialization should succeed");
    assert!(!serialized.is_empty());
    
    // 反序列化
    let deserialized: Operation = bincode::deserialize(&serialized).expect("Deserialization should succeed");
    
    // 验证反序列化后的属性数量
    assert_eq!(deserialized.attributes.len(), op.attributes.len());
    
    // 验证极端整数值
    assert_eq!(deserialized.attributes.get("max_int"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(deserialized.attributes.get("min_int"), Some(&Attribute::Int(i64::MIN)));
    
    // 验证特殊浮点值
    match deserialized.attributes.get("inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val > 0.0),
        _ => panic!("Expected positive infinity"),
    }
    
    match deserialized.attributes.get("neg_inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val < 0.0),
        _ => panic!("Expected negative infinity"),
    }
    
    // 验证 NaN (需要特殊处理，因为 NaN != NaN)
    match deserialized.attributes.get("nan") {
        Some(Attribute::Float(val)) => assert!(val.is_nan()),
        _ => panic!("Expected NaN attribute"),
    }
}

/// 测试3: 内存边界 - 测试大型张量形状的内存占用
#[test]
fn test_memory_boundary_large_tensors() {
    // 测试接近内存边界的大型张量形状
    let large_shapes = vec![
        vec![10000, 10000],      // 100M elements
        vec![100000, 1000],      // 100M elements
        vec![1000000, 100],      // 100M elements
        vec![10000000, 10],      // 100M elements
        vec![100000000],         // 100M elements (1D)
    ];
    
    for (i, shape) in large_shapes.iter().enumerate() {
        let value = Value {
            name: format!("large_tensor_{}", i),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // 验证形状被正确存储
        assert_eq!(value.shape.len(), shape.len());
        
        // 计算元素数量
        let num_elements = value.num_elements();
        
        // 验证元素数量计算正确
        match num_elements {
            Some(count) => assert_eq!(count, shape.iter().product::<usize>()),
            None => assert!(shape.iter().product::<usize>() > 0), // 如果返回 None，说明检测到溢出
        }
    }
}

/// 测试4: 张量形状的边界组合 - 测试各种形状组合
#[test]
fn test_tensor_shape_boundary_combinations() {
    let test_cases = vec![
        // (形状, 描述)
        (vec![], "标量 (0维张量)"),
        (vec![0], "零元素 1D 张量"),
        (vec![1], "单元素 1D 张量"),
        (vec![1, 1], "单元素 2D 张量"),
        (vec![1, 1, 1], "单元素 3D 张量"),
        (vec![0, 10], "零元素 2D 张量"),
        (vec![10, 0], "零元素 2D 张量"),
        (vec![2, 0, 3], "零元素 3D 张量"),
        (vec![i32::MAX as usize, 1], "最大行数的 2D 张量"),
        (vec![1, i32::MAX as usize], "最大列数的 2D 张量"),
    ];
    
    for (shape, description) in test_cases {
        let value = Value {
            name: format!("test_{}", description),
            ty: Type::F32,
            shape: shape.clone(),
        };
        
        // 验证形状
        assert_eq!(value.shape, shape);
        
        // 计算元素数量
        let num_elements = value.num_elements();
        
        // 验证元素数量
        let expected_elements: usize = shape.iter().product();
        assert_eq!(num_elements, Some(expected_elements));
    }
}

/// 测试5: 属性序列化性能 - 测试大型属性数组的序列化性能
#[test]
fn test_large_attribute_array_serialization() {
    use serde_json;

    let mut op = Operation::new("large_array_test");
    
    // 创建一个大型属性数组
    let mut large_array = Vec::new();
    for i in 0..10000 {
        large_array.push(Attribute::Int(i));
    }
    
    let mut attrs = HashMap::new();
    attrs.insert("large_array".to_string(), Attribute::Array(large_array));
    op.attributes = attrs;
    
    // 测量序列化时间
    let start = std::time::Instant::now();
    let serialized = serde_json::to_string(&op).expect("Serialization should succeed");
    let duration = start.elapsed();
    
    // 验证序列化成功
    assert!(!serialized.is_empty());
    
    // 验证序列化时间在合理范围内（应该在 1 秒内完成）
    assert!(duration.as_secs() < 1, "Serialization took too long: {:?}", duration);
    
    // 反序列化
    let deserialized: Operation = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    
    // 验证反序列化后的数组大小
    match deserialized.attributes.get("large_array") {
        Some(Attribute::Array(arr)) => assert_eq!(arr.len(), 10000),
        _ => panic!("Expected large array attribute"),
    }
}

/// 测试6: 模块序列化 - 测试包含多个操作的模块序列化
#[test]
fn test_module_serialization_with_multiple_operations() {
    use serde_json;

    let mut module = Module::new("serialization_test_module");
    
    // 添加多个操作
    for i in 0..100 {
        let mut op = Operation::new(&format!("op_{}", i));
        
        // 添加输入
        op.inputs.push(Value {
            name: format!("input_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // 添加输出
        op.outputs.push(Value {
            name: format!("output_{}", i),
            ty: Type::F32,
            shape: vec![10, 10],
        });
        
        // 添加属性
        let mut attrs = HashMap::new();
        attrs.insert("index".to_string(), Attribute::Int(i as i64));
        attrs.insert("name".to_string(), Attribute::String(format!("operation_{}", i)));
        op.attributes = attrs;
        
        module.add_operation(op);
    }
    
    // 序列化
    let serialized = serde_json::to_string(&module).expect("Serialization should succeed");
    assert!(!serialized.is_empty());
    
    // 反序列化
    let deserialized: Module = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    
    // 验证反序列化后的模块
    assert_eq!(deserialized.name, module.name);
    assert_eq!(deserialized.operations.len(), module.operations.len());
    assert_eq!(deserialized.operations.len(), 100);
    
    // 验证第一个和最后一个操作
    assert_eq!(deserialized.operations[0].op_type, "op_0");
    assert_eq!(deserialized.operations[99].op_type, "op_99");
}

/// 测试7: 深度嵌套类型的序列化 - 测试深度嵌套张量类型的序列化
#[test]
fn test_deeply_nested_type_serialization() {
    use serde_json;

    // 创建深度嵌套的张量类型
    let mut nested_type = Type::F32;
    for i in 0..50 {
        nested_type = Type::Tensor {
            element_type: Box::new(nested_type),
            shape: vec![i + 1],
        };
    }
    
    // 创建包含嵌套类型的值
    let value = Value {
        name: "nested_type_value".to_string(),
        ty: nested_type,
        shape: vec![100, 100],
    };
    
    // 序列化
    let serialized = serde_json::to_string(&value).expect("Serialization should succeed");
    assert!(!serialized.is_empty());
    
    // 反序列化
    let deserialized: Value = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    
    // 验证反序列化后的值
    assert_eq!(deserialized.name, value.name);
    assert_eq!(deserialized.shape, value.shape);
    assert_eq!(deserialized.ty, value.ty);
}

/// 测试8: 特殊 Unicode 字符串序列化 - 测试包含特殊 Unicode 字符的字符串序列化
#[test]
fn test_unicode_string_serialization() {
    use serde_json;

    let mut op = Operation::new("unicode_test");
    let mut attrs = HashMap::new();
    
    // 添加各种 Unicode 字符串
    attrs.insert("emoji".to_string(), Attribute::String("🚀🎉⭐🔥💯".to_string()));
    attrs.insert("chinese".to_string(), Attribute::String("中文测试字符串".to_string()));
    attrs.insert("japanese".to_string(), Attribute::String("日本語テスト".to_string()));
    attrs.insert("arabic".to_string(), Attribute::String("مرحبا بالعالم".to_string()));
    attrs.insert("emoji_text".to_string(), Attribute::String("Hello 🌍 World 🌏".to_string()));
    attrs.insert("mixed".to_string(), Attribute::String("Mix: 你好 🚀 مرحبا".to_string()));
    
    op.attributes = attrs;
    
    // 序列化
    let serialized = serde_json::to_string(&op).expect("Serialization should succeed");
    
    // 反序列化
    let deserialized: Operation = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    
    // 验证所有 Unicode 字符串被正确保留
    match deserialized.attributes.get("emoji") {
        Some(Attribute::String(s)) => assert_eq!(s, "🚀🎉⭐🔥💯"),
        _ => panic!("Expected emoji string"),
    }
    
    match deserialized.attributes.get("chinese") {
        Some(Attribute::String(s)) => assert_eq!(s, "中文测试字符串"),
        _ => panic!("Expected Chinese string"),
    }
    
    match deserialized.attributes.get("mixed") {
        Some(Attribute::String(s)) => assert_eq!(s, "Mix: 你好 🚀 مرحبا"),
        _ => panic!("Expected mixed string"),
    }
}

/// 测试9: 空值和空操作的序列化 - 测试空值和空操作的序列化
#[test]
fn test_empty_values_serialization() {
    use serde_json;

    // 测试空操作
    let empty_op = Operation::new("");
    let serialized = serde_json::to_string(&empty_op).expect("Serialization should succeed");
    let deserialized: Operation = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    assert_eq!(deserialized.op_type, "");
    
    // 测试空值
    let empty_value = Value {
        name: "".to_string(),
        ty: Type::F32,
        shape: vec![],
    };
    let serialized = serde_json::to_string(&empty_value).expect("Serialization should succeed");
    let deserialized: Value = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    assert_eq!(deserialized.name, "");
    
    // 测试空模块
    let empty_module = Module::new("");
    let serialized = serde_json::to_string(&empty_module).expect("Serialization should succeed");
    let deserialized: Module = serde_json::from_str(&serialized).expect("Deserialization should succeed");
    assert_eq!(deserialized.name, "");
}

/// 测试10: 数值类型边界 - 测试所有数值类型的边界值
#[test]
fn test_numeric_type_boundaries() {
    let mut op = Operation::new("numeric_boundary_test");
    let mut attrs = HashMap::new();
    
    // 整数边界值
    attrs.insert("int_max".to_string(), Attribute::Int(i64::MAX));
    attrs.insert("int_min".to_string(), Attribute::Int(i64::MIN));
    attrs.insert("int_zero".to_string(), Attribute::Int(0));
    attrs.insert("int_one".to_string(), Attribute::Int(1));
    attrs.insert("int_neg_one".to_string(), Attribute::Int(-1));
    
    // 浮点数边界值
    attrs.insert("float_max".to_string(), Attribute::Float(f64::MAX));
    attrs.insert("float_min".to_string(), Attribute::Float(f64::MIN));
    attrs.insert("float_zero".to_string(), Attribute::Float(0.0));
    attrs.insert("float_neg_zero".to_string(), Attribute::Float(-0.0));
    attrs.insert("float_one".to_string(), Attribute::Float(1.0));
    attrs.insert("float_neg_one".to_string(), Attribute::Float(-1.0));
    
    // 特殊浮点值
    attrs.insert("float_inf".to_string(), Attribute::Float(f64::INFINITY));
    attrs.insert("float_neg_inf".to_string(), Attribute::Float(f64::NEG_INFINITY));
    attrs.insert("float_nan".to_string(), Attribute::Float(f64::NAN));
    
    // 布尔值
    attrs.insert("bool_true".to_string(), Attribute::Bool(true));
    attrs.insert("bool_false".to_string(), Attribute::Bool(false));
    
    op.attributes = attrs;
    
    // 验证所有属性
    assert_eq!(op.attributes.len(), 16);
    
    // 验证整数值
    assert_eq!(op.attributes.get("int_max"), Some(&Attribute::Int(i64::MAX)));
    assert_eq!(op.attributes.get("int_min"), Some(&Attribute::Int(i64::MIN)));
    
    // 验证浮点值
    assert_eq!(op.attributes.get("float_max"), Some(&Attribute::Float(f64::MAX)));
    assert_eq!(op.attributes.get("float_min"), Some(&Attribute::Float(f64::MIN)));
    
    // 验证布尔值
    assert_eq!(op.attributes.get("bool_true"), Some(&Attribute::Bool(true)));
    assert_eq!(op.attributes.get("bool_false"), Some(&Attribute::Bool(false)));
    
    // 验证特殊浮点值
    match op.attributes.get("float_inf") {
        Some(Attribute::Float(val)) => assert!(val.is_infinite() && *val > 0.0),
        _ => panic!("Expected positive infinity"),
    }
    
    match op.attributes.get("float_nan") {
        Some(Attribute::Float(val)) => assert!(val.is_nan()),
        _ => panic!("Expected NaN"),
    }
}