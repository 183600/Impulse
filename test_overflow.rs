fn main() {
    let max_val = usize::MAX;
    let half_max = max_val / 2;
    let result = half_max * 2;
    println!("max: {}", max_val);
    println!("half max: {}", half_max);
    println!("half max * 2: {}", result);
    println!("Does it overflow?: {}", result > max_val);
    println!("Are they equal?: {}", result == max_val);
    
    // Try with checked_mul to see what happens
    let checked_result = 1usize.checked_mul(half_max).and_then(|x| x.checked_mul(2));
    println!("Checked multiplication result: {:?}", checked_result);
}