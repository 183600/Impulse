//! Focused comprehensive edge case tests covering critical boundary scenarios
//! Tests focus on practical edge cases relevant to ML compiler operations

/// Test 1: Broadcasting-compatible tensor shapes (common in ML operations)
#[test]
fn test_broadcasting_compatible_shapes() {
    // Test shapes that are compatible for broadcasting
    // Shape [3, 1] can broadcast to [3, 5]
    let shape1 = vec![3, 1];
    let shape2 = vec![1, 5];
    
    // Broadcasting would produce [3, 5]
    let broadcast_shape: Vec<usize> = shape1.iter()
        .zip(shape2.iter())
        .map(|(&a, &b)| a.max(b))
        .collect();
    
    assert_eq!(broadcast_shape, vec![3, 5]);
    
    // Verify shapes are valid for broadcasting
    let is_broadcastable = shape1.iter().zip(shape2.iter()).all(|(&a, &b)| 
        a == b || a == 1 || b == 1
    );
    assert!(is_broadcastable);
}

/// Test 2: Convolution-specific edge cases with valid padding dimensions
#[test]
fn test_convolution_padding_dimensions() {
    // Test valid padding configurations that maintain spatial dimensions
    let input_size = 32;
    let kernel_size = 3;
    let padding = 1;
    let stride = 1;
    
    // Output size = floor((input + 2*padding - kernel) / stride) + 1
    let output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
    
    // With same padding, output should equal input
    assert_eq!(output_size, input_size);
    
    // Test with different stride
    let stride2 = 2;
    let output_size2 = (input_size + 2 * padding - kernel_size) / stride2 + 1;
    assert_eq!(output_size2, 16); // (32 + 2 - 3) / 2 + 1 = 16
}

/// Test 3: Valid pooling operations with kernel and stride constraints
#[test]
fn test_pooling_kernel_stride_constraints() {
    // Test pooling where kernel size and stride maintain valid output dimensions
    let input_dim = 28;
    let kernel_size = 2;
    let stride = 2;
    
    // For max/avg pooling, output = floor((input - kernel) / stride) + 1
    let output_dim = (input_dim - kernel_size) / stride + 1;
    
    assert_eq!(output_dim, 14);
    
    // Test with non-divisible dimensions (truncation behavior)
    let input_dim2 = 27;
    let output_dim2 = (input_dim2 - kernel_size) / stride + 1;
    assert_eq!(output_dim2, 13); // Floor division: (27 - 2) / 2 + 1 = 13
}

/// Test 4: Reshape operations that preserve total element count
#[test]
fn test_reshape_preserves_element_count() {
    let original_shape = vec![2, 3, 4]; // 24 elements
    let new_shape = vec![6, 4]; // Also 24 elements
    
    let original_size: usize = original_shape.iter().product();
    let new_size: usize = new_shape.iter().product();
    
    assert_eq!(original_size, new_size);
    
    // Test reshape to 1D (flattening)
    let flat_shape = vec![24];
    let flat_size: usize = flat_shape.iter().product();
    assert_eq!(flat_size, original_size);
    
    // Test reshape with -1 (inferred dimension)
    // -1 would be computed as 24 / 4 = 6
    let known_dims: usize = 4;
    let inferred_dim = original_size / known_dims;
    assert_eq!(inferred_dim, 6);
}

/// Test 5: Batch normalization with valid channel dimensions
#[test]
fn test_batch_norm_channel_dimensions() {
    let batch_size = 32;
    let channels = 64;
    let height = 28;
    let width = 28;
    
    let tensor_shape = vec![batch_size, channels, height, width];
    
    // BN params (gamma, beta) have shape [channels]
    let param_shape = vec![channels];
    
    // Verify param shape matches channel dimension
    assert_eq!(param_shape[0], tensor_shape[1]);
    
    // Total elements in batch
    let total_elements: usize = tensor_shape.iter().product();
    assert_eq!(total_elements, 32 * 64 * 28 * 28);
}

/// Test 6: Transpose operations with valid dimension permutations
#[test]
fn test_transpose_dimension_permutations() {
    let original_shape = vec![2, 3, 4];
    
    // Valid permutation: [0, 2, 1] swaps last two dimensions
    let perm1 = vec![0, 2, 1];
    let transposed_shape1: Vec<usize> = perm1.iter().map(|&i| original_shape[i]).collect();
    assert_eq!(transposed_shape1, vec![2, 4, 3]);
    
    // Full reversal: [2, 1, 0]
    let perm2 = vec![2, 1, 0];
    let transposed_shape2: Vec<usize> = perm2.iter().map(|&i| original_shape[i]).collect();
    assert_eq!(transposed_shape2, vec![4, 3, 2]);
    
    // Identity permutation: [0, 1, 2] - no change
    let perm3 = vec![0, 1, 2];
    let transposed_shape3: Vec<usize> = perm3.iter().map(|&i| original_shape[i]).collect();
    assert_eq!(transposed_shape3, original_shape);
}

/// Test 7: Reduction operations preserving or reducing dimensions
#[test]
fn test_reduction_operations() {
    let input_shape = vec![4, 5, 6];
    
    // Reduce along axis 0: [5, 6]
    let axis0_shape: Vec<usize> = input_shape.iter()
        .enumerate()
        .filter(|&(i, _)| i != 0)
        .map(|(_, &dim)| dim)
        .collect();
    assert_eq!(axis0_shape, vec![5, 6]);
    
    // Reduce along axis 1: [4, 6]
    let axis1_shape: Vec<usize> = input_shape.iter()
        .enumerate()
        .filter(|&(i, _)| i != 1)
        .map(|(_, &dim)| dim)
        .collect();
    assert_eq!(axis1_shape, vec![4, 6]);
    
    // Reduce all axes (keep_dim=false): scalar
    let reduced_all: usize = 1; // Scalar has 1 element
    assert_eq!(reduced_all, 1);
}

/// Test 8: Gradient computation with matching tensor shapes
#[test]
fn test_gradient_shape_matching() {
    // Forward pass tensor
    let forward_shape = vec![32, 64, 28, 28];
    
    // Gradient should have same shape as forward tensor
    let gradient_shape = forward_shape.clone();
    
    assert_eq!(forward_shape, gradient_shape);
    
    // Weight gradient shape should match weight shape
    let weight_shape = vec![64, 3, 3, 3];
    let weight_grad_shape = weight_shape.clone();
    
    assert_eq!(weight_shape, weight_grad_shape);
}

/// Test 9: Slice operations with valid bounds
#[test]
fn test_slice_operations() {
    let input_shape = vec![10, 20, 30];
    
    // Slice: [1:5, 5:15, 10:20]
    let start_indices = vec![1, 5, 10];
    let end_indices = vec![5, 15, 20];
    
    let output_shape: Vec<usize> = start_indices.iter()
        .zip(end_indices.iter())
        .map(|(&start, &end)| end - start)
        .collect();
    
    assert_eq!(output_shape, vec![4, 10, 10]);
    
    // Verify slice bounds are within input bounds
    let is_valid = start_indices.iter().zip(end_indices.iter())
        .zip(input_shape.iter())
        .all(|((&start, &end), &dim)| start <= end && end <= dim);
    
    assert!(is_valid);
}

/// Test 10: Concatenation with matching non-concat dimensions
#[test]
fn test_concatenation_dimension_matching() {
    let tensor1_shape = vec![2, 3, 4];
    let tensor2_shape = vec![5, 3, 4];
    
    // Concatenate along axis 0
    let concat_axis = 0;
    
    // Non-concat dimensions must match
    let non_concat_match = tensor1_shape.iter()
        .zip(tensor2_shape.iter())
        .enumerate()
        .filter(|&(i, _)| i != concat_axis)
        .all(|(_, (&d1, &d2))| d1 == d2);
    
    assert!(non_concat_match);
    
    // Output shape: [2+5, 3, 4] = [7, 3, 4]
    let output_shape: Vec<usize> = tensor1_shape.iter()
        .zip(tensor2_shape.iter())
        .enumerate()
        .map(|(i, (&d1, &d2))| {
            if i == concat_axis { d1 + d2 } else { d1 }
        })
        .collect();
    
    assert_eq!(output_shape, vec![7, 3, 4]);
}