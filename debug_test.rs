#[test]
fn debug_math_utils_issue() {
    let value = 3;
    let result = math_utils::next_power_of_2(value);
    println!("next_power_of_2({}) = {}", value, result);
    assert_eq!(result, 4, "Expected 4 but got {}", result);
}
