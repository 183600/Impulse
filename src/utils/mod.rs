//! Utility functions for the Impulse compiler

pub mod ir_utils;
pub mod validation_utils;
pub mod math_utils;

pub use ir_utils::*;
pub use validation_utils::*;
pub use math_utils::*;


/// Safely calculate tensor size to prevent overflow
pub fn calculate_tensor_size_safe(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        // Scalar has size 1
        return Some(1);
    }

    shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tensor_size_safe() {
        // Test normal cases
        assert_eq!(calculate_tensor_size_safe(&[2, 3, 4]), Some(24));
        assert_eq!(calculate_tensor_size_safe(&[10, 10]), Some(100));
        assert_eq!(calculate_tensor_size_safe(&[5]), Some(5));
        
        // Test scalar
        assert_eq!(calculate_tensor_size_safe(&[]), Some(1));
        
        // Test zero dimension
        assert_eq!(calculate_tensor_size_safe(&[10, 0, 5]), Some(0));
        assert_eq!(calculate_tensor_size_safe(&[0]), Some(0));
        
        // Test potential overflow (would depend on platform)
        // Using smaller values to avoid actual overflow on most platforms
        let large_dims = &[10_000, 10_000];
        let result = calculate_tensor_size_safe(large_dims);
        assert!(result.is_some()); // Should not overflow for these values
    }

    #[test]
    fn test_calculate_tensor_size_safe_overflow() {
        // Test for potential overflow by using values that would likely overflow
        // when multiplied on most systems
        let dims = &[usize::MAX, 2];
        let result = calculate_tensor_size_safe(dims);
        assert!(result.is_none());
    }
}