//! Additional edge case tests for attribute handling in the Impulse compiler
//! Focused specifically on edge cases with different attribute types

use crate::ir::{Module, Operation, Attribute};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Test 1: Deeply nested array attributes
    #[test]
    fn test_deeply_nested_array_attributes() {
        // Create a nested array that goes 10 levels deep
        let mut nested_attribute = Attribute::Int(42);
        
        for _ in 0..10 {
            nested_attribute = Attribute::Array(vec![nested_attribute]);
        }
        
        // Verify the structure was created properly
        match &nested_attribute {
            Attribute::Array(inner) => {
                assert_eq!(inner.len(), 1);
                // Continue checking recursively if needed
            },
            _ => panic!("Expected nested Array attribute"),
        }
        
        // Test equality and cloning
        let cloned = nested_attribute.clone();
        assert_eq!(nested_attribute, cloned);
    }

    /// Test 2: Complex heterogeneous arrays
    #[test]
    fn test_complex_heterogeneous_array_attributes() {
        let complex_array = Attribute::Array(vec![
            Attribute::Int(1),
            Attribute::Float(1.5),
            Attribute::String("hello".to_string()),
            Attribute::Bool(true),
            Attribute::Array(vec![
                Attribute::Int(2),
                Attribute::String("nested".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(false),
                    Attribute::Float(3.14),
                ])
            ]),
        ]);

        match &complex_array {
            Attribute::Array(items) => {
                assert_eq!(items.len(), 5);
                
                // Check each item individually
                match &items[0] {
                    Attribute::Int(1) => {},
                    _ => panic!("First item should be Int(1)"),
                }
                
                match &items[1] {
                    Attribute::Float(val) if (val - 1.5).abs() < f64::EPSILON => {},
                    _ => panic!("Second item should be Float(1.5)"),
                }
                
                match &items[2] {
                    Attribute::String(s) if s == "hello" => {},
                    _ => panic!("Third item should be String(\"hello\")"),
                }
                
                match &items[3] {
                    Attribute::Bool(true) => {},
                    _ => panic!("Fourth item should be Bool(true)"),
                }
            },
            _ => panic!("Expected Array attribute"),
        }
    }

    /// Test 3: Parameterized tests for different attribute types
    #[rstest]
    #[case(Attribute::Int(42))]
    #[case(Attribute::Float(3.14159))]
    #[case(Attribute::String("test_string".to_string()))]
    #[case(Attribute::Bool(true))]
    #[case(Attribute::Array(vec![Attribute::Int(1), Attribute::String("nested".to_string())]))]
    fn test_different_attribute_types(#[case] attr: Attribute) {
        // Test that each attribute can be cloned safely
        let cloned = attr.clone();
        assert_eq!(attr, cloned);
    }

    /// Test 4: Very large string attributes
    #[test]
    fn test_large_string_attributes() {
        use std::collections::HashMap;
        
        let mut op = Operation::new("large_string_attr_test");
        
        let mut attrs = HashMap::new();
        
        // Add increasingly large strings
        attrs.insert("small_string".to_string(), Attribute::String("small".to_string()));
        attrs.insert("medium_string".to_string(), Attribute::String("m".repeat(1_000)));
        attrs.insert("large_string".to_string(), Attribute::String("l".repeat(100_000)));
        attrs.insert("huge_string".to_string(), Attribute::String("h".repeat(1_000_000)));  // 1MB string
        
        op.attributes = attrs;
        
        assert_eq!(op.attributes.len(), 4);
        
        // Verify each string was stored correctly
        if let Some(Attribute::String(s)) = op.attributes.get("small_string") {
            assert_eq!(s, "small");
        }
        
        if let Some(Attribute::String(s)) = op.attributes.get("large_string") {
            assert_eq!(s.len(), 100_000);
        }
        
        if let Some(Attribute::String(s)) = op.attributes.get("huge_string") {
            assert_eq!(s.len(), 1_000_000);
        }
    }

    /// Test 5: Arrays of maximum length
    #[test]
    fn test_maximum_length_array_attributes() {
        // Create an array with many elements
        let mut large_array_items = Vec::new();
        
        // Add 100,000 integers to the array
        for i in 0..100_000 {
            large_array_items.push(Attribute::Int(i as i64));
        }
        
        let large_array = Attribute::Array(large_array_items);
        
        match &large_array {
            Attribute::Array(items) => {
                assert_eq!(items.len(), 100_000);
                
                // Check a few specific items
                if let Attribute::Int(0) = items[0] {}
                else { panic!("First item should be Int(0)"); }
                
                if let Attribute::Int(99_999) = items[99_999] {}
                else { panic!("Last item should be Int(99999)"); }
                
                if let Attribute::Int(50_000) = items[50_000] {}
                else { panic!("Middle item should be Int(50000)"); }
            },
            _ => panic!("Expected Array attribute"),
        }
        
        // Ensure cloning works with large arrays
        let cloned = large_array.clone();
        assert_eq!(large_array, cloned);
    }

    /// Test 6: Equality tests for complex attribute structures
    #[test]
    fn test_attribute_equality_complex_structures() {
        // Create two identical complex attributes
        let complex_attr1 = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
            ]),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
        ]);
        
        let complex_attr2 = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                Attribute::Float(2.5),
            ]),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
        ]);
        
        let complex_attr3 = Attribute::Array(vec![
            Attribute::Array(vec![
                Attribute::Int(1),
                // Different float value
                Attribute::Float(2.6),
            ]),
            Attribute::String("test".to_string()),
            Attribute::Bool(true),
        ]);
        
        // The first two should be equal
        assert_eq!(complex_attr1, complex_attr2);
        
        // The third should be different
        assert_ne!(complex_attr1, complex_attr3);
    }

    /// Test 7: Attribute type conversions and comparisons
    #[test]
    fn test_attribute_type_variations() {
        // Test that similar values with different types are not equal
        let int_attr = Attribute::Int(42);
        let float_attr = Attribute::Float(42.0);
        let string_attr = Attribute::String("42".to_string());
        
        // These should all be different despite "similar" values
        assert_ne!(int_attr, float_attr);
        assert_ne!(int_attr, string_attr);
        assert_ne!(float_attr, string_attr);
        
        // Test some border cases with floats
        let float_pos_zero = Attribute::Float(0.0);
        let float_neg_zero = Attribute::Float(-0.0);
        let int_zero = Attribute::Int(0);
        
        // Positive and negative zero floats are typically considered equal mathematically,
        // but we'll check the implementation here
        assert_ne!(float_pos_zero, int_zero);
        assert_ne!(float_neg_zero, int_zero);
    }

    /// Test 8: Mixed nested structures with all types
    #[test]
    fn test_mixed_nested_structures() {
        let mixed_structure = Attribute::Array(vec![
            Attribute::Int(100),
            Attribute::Array(vec![
                Attribute::String("nested_string".to_string()),
                Attribute::Array(vec![
                    Attribute::Bool(true),
                    Attribute::Float(42.42),
                    Attribute::Array(vec![
                        Attribute::Int(999),
                        Attribute::String("deeply_nested".to_string()),
                    ])
                ])
            ]),
            Attribute::Bool(false),
        ]);
        
        // Verify the top-level structure
        match &mixed_structure {
            Attribute::Array(top_level) => {
                assert_eq!(top_level.len(), 3);
                
                // Check first element: Int(100)
                match &top_level[0] {
                    Attribute::Int(100) => {},
                    _ => panic!("First element should be Int(100)"),
                }
                
                // Check second element: Array with nested structures
                match &top_level[1] {
                    Attribute::Array(nested1) => {
                        assert_eq!(nested1.len(), 2);
                        
                        match &nested1[0] {
                            Attribute::String(s) if s == "nested_string" => {},
                            _ => panic!("Nested element should be String(\"nested_string\")"),
                        }
                        
                        match &nested1[1] {
                            Attribute::Array(deeply_nested) => {
                                assert_eq!(deeply_nested.len(), 3);
                                
                                match &deeply_nested[0] {
                                    Attribute::Bool(true) => {},
                                    _ => panic!("Deeply nested element should be Bool(true)"),
                                }
                            },
                            _ => panic!("Deeply nested element should be Array"),
                        }
                    },
                    _ => panic!("Second element should be Array"),
                }
                
                // Check third element: Bool(false)
                match &top_level[2] {
                    Attribute::Bool(false) => {},
                    _ => panic!("Third element should be Bool(false)"),
                }
            },
            _ => panic!("Expected top-level Array"),
        }
    }

    /// Test 9: Empty and minimal attribute structures
    #[test]
    fn test_empty_minimal_attribute_structures() {
        // Empty array
        let empty_array = Attribute::Array(vec![]);
        match &empty_array {
            Attribute::Array(items) => assert_eq!(items.len(), 0),
            _ => panic!("Expected empty Array"),
        }
        
        // Array with only empty arrays
        let nested_empty = Attribute::Array(vec![
            Attribute::Array(vec![]),
            Attribute::Array(vec![]),
        ]);
        match &nested_empty {
            Attribute::Array(outer) => {
                assert_eq!(outer.len(), 2);
                for item in outer {
                    match item {
                        Attribute::Array(inner) => assert_eq!(inner.len(), 0),
                        _ => panic!("Expected empty Array"),
                    }
                }
            },
            _ => panic!("Expected Array"),
        }
        
        // Empty string
        let empty_string = Attribute::String("".to_string());
        match &empty_string {
            Attribute::String(s) => assert_eq!(s, ""),
            _ => panic!("Expected empty String"),
        }
    }

    /// Test 10: Attribute operations in complex module setup
    #[test]
    fn test_attributes_in_complex_module_operations() {
        let mut module = Module::new("attribute_complexity_test");
        
        // Add multiple operations, each with complex attributes
        for i in 0..1000 {
            let mut op = Operation::new(&format!("op_with_attrs_{}", i));
            
            use std::collections::HashMap;
            let mut attrs = HashMap::new();
            
            // Add various types of attributes
            attrs.insert(
                format!("int_attr_{}", i), 
                Attribute::Int(i as i64)
            );
            attrs.insert(
                format!("float_attr_{}", i), 
                Attribute::Float(i as f64 * 1.5)
            );
            attrs.insert(
                format!("string_attr_{}", i), 
                Attribute::String(format!("value_{}", i))
            );
            attrs.insert(
                format!("bool_attr_{}", i), 
                Attribute::Bool(i % 2 == 0)
            );
            
            // Add a small array attribute
            attrs.insert(
                format!("array_attr_{}", i), 
                Attribute::Array(vec![
                    Attribute::Int(i as i64),
                    Attribute::String(format!("item_{}", i)),
                    Attribute::Bool(i % 3 == 0),
                ])
            );
            
            op.attributes = attrs;
            module.add_operation(op);
        }
        
        // Verify the module was built correctly
        assert_eq!(module.operations.len(), 1000);
        assert_eq!(module.name, "attribute_complexity_test");
        
        // Verify a few operations
        for idx in [0, 499, 999] {
            let op = &module.operations[idx];
            assert_eq!(op.attributes.len(), 5); // int, float, string, bool, array
            
            let expected_int_key = format!("int_attr_{}", idx);
            assert!(op.attributes.contains_key(&expected_int_key));
            
            let expected_array_key = format!("array_attr_{}", idx);
            if let Some(Attribute::Array(items)) = op.attributes.get(&expected_array_key) {
                assert_eq!(items.len(), 3);
            } else {
                panic!("Expected array attribute for operation {}", idx);
            }
        }
    }
}