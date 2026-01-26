//! Math utilities for the Impulse compiler

/// Calculate greatest common divisor of two numbers
pub fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Calculate least common multiple of two numbers
pub fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        a * b / gcd(a, b)
    }
}

/// Round a number up to the next multiple of another number
pub fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    
    let remainder = value % multiple;
    if remainder == 0 {
        value
    } else {
        value + (multiple - remainder)
    }
}

/// Find the next power of 2 greater than or equal to the given number
pub fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    
    // Find the position of the highest bit
    let mut result = 1;
    while result < n {
        result <<= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(1071, 462), 21);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(12, 8), 24);
        assert_eq!(lcm(17, 13), 221);
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(lcm(5, 0), 0);
    }

    #[test]
    fn test_round_up_to_multiple() {
        assert_eq!(round_up_to_multiple(10, 16), 16);
        assert_eq!(round_up_to_multiple(16, 16), 16);
        assert_eq!(round_up_to_multiple(17, 16), 32);
        assert_eq!(round_up_to_multiple(1, 1024), 1024);
        assert_eq!(round_up_to_multiple(5, 1), 5);
        assert_eq!(round_up_to_multiple(5, 0), 5); // Special case
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(4), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
        assert_eq!(next_power_of_2(1000), 1024);
        assert_eq!(next_power_of_2(1024), 1024);
        assert_eq!(next_power_of_2(1025), 2048);
    }
}