//! Autotuning system for automatically optimizing kernels
//! Finds optimal parameters for operations like GEMM, convolution, attention, etc.

use crate::ir::{Module, Operation};
use anyhow::Result;
use std::collections::HashMap;

/// Tuning parameters for different operation types
#[derive(Debug, Clone)]
pub enum TuneParams {
    Gemm {
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        vector_width: usize,
    },
    Conv {
        tile_h: usize,
        tile_w: usize,
        tile_c: usize,
        vector_width: usize,
    },
    Attention {
        block_m: usize,
        block_n: usize,
        stages: usize,
        num_warps: usize,
    },
    LayerNorm {
        vector_width: usize,
        block_size: usize,
    },
}

/// Auto-tuner for finding optimal kernel configurations
pub struct AutoTuner {
    pub cache: HashMap<String, TuneParams>,
    pub search_space: SearchSpace,
}

impl AutoTuner {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            search_space: SearchSpace::default(),
        }
    }

    /// Tune an operation and return the best parameters
    pub fn tune_operation(&mut self, operation: &Operation, module: &Module) -> Result<TuneParams> {
        println!("Tuning operation: {}", operation.op_type);
        
        // Generate candidates based on the operation type
        let candidates = self.generate_candidates(operation)?;
        
        // For now, just return the first candidate as a placeholder
        // In practice, we would benchmark each candidate and return the best one
        match candidates.first() {
            Some(params) => {
                // Cache the result
                let key = self.generate_cache_key(operation, module);
                self.cache.insert(key, params.clone());
                
                Ok(params.clone())
            },
            None => Ok(self.get_default_params(&operation.op_type)),
        }
    }

    /// Generate tuning candidates for an operation
    fn generate_candidates(&self, operation: &Operation) -> Result<Vec<TuneParams>> {
        match operation.op_type.as_str() {
            "gemm" | "matmul" => self.generate_gemm_candidates(),
            "conv" | "conv2d" => self.generate_conv_candidates(),
            "attention" => self.generate_attention_candidates(),
            "layer_norm" | "layernorm" => self.generate_layernorm_candidates(),
            _ => Ok(vec![self.get_default_params(&operation.op_type)]),
        }
    }

    /// Generate candidates for GEMM operations
    fn generate_gemm_candidates(&self) -> Result<Vec<TuneParams>> {
        let mut candidates = Vec::new();
        
        for &tile_m in &[64, 128, 256] {
            for &tile_n in &[64, 128, 256] {
                for &tile_k in &[8, 16, 32] {
                    for &vector_width in &[1, 2, 4, 8] {
                        candidates.push(TuneParams::Gemm {
                            tile_m,
                            tile_n,
                            tile_k,
                            vector_width,
                        });
                    }
                }
            }
        }
        
        Ok(candidates)
    }

    /// Generate candidates for convolution operations
    fn generate_conv_candidates(&self) -> Result<Vec<TuneParams>> {
        let mut candidates = Vec::new();
        
        for &tile_h in &[14, 28] {
            for &tile_w in &[14, 28] {
                for &tile_c in &[8, 16, 32] {
                    for &vector_width in &[1, 2, 4] {
                        candidates.push(TuneParams::Conv {
                            tile_h,
                            tile_w,
                            tile_c,
                            vector_width,
                        });
                    }
                }
            }
        }
        
        Ok(candidates)
    }

    /// Generate candidates for attention operations
    fn generate_attention_candidates(&self) -> Result<Vec<TuneParams>> {
        let mut candidates = Vec::new();
        
        for &block_m in &[64, 128, 256] {
            for &block_n in &[32, 64, 128] {
                for &stages in &[2, 3, 4] {
                    for &num_warps in &[4, 8, 16] {
                        candidates.push(TuneParams::Attention {
                            block_m,
                            block_n,
                            stages,
                            num_warps,
                        });
                    }
                }
            }
        }
        
        Ok(candidates)
    }

    /// Generate candidates for layer norm operations
    fn generate_layernorm_candidates(&self) -> Result<Vec<TuneParams>> {
        let mut candidates = Vec::new();
        
        for &vector_width in &[4, 8, 16, 32] {
            for &block_size in &[128, 256, 512] {
                candidates.push(TuneParams::LayerNorm {
                    vector_width,
                    block_size,
                });
            }
        }
        
        Ok(candidates)
    }

    /// Get default parameters for an unknown operation type
    fn get_default_params(&self, op_type: &str) -> TuneParams {
        match op_type {
            "gemm" | "matmul" => TuneParams::Gemm {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                vector_width: 4,
            },
            "conv" | "conv2d" => TuneParams::Conv {
                tile_h: 14,
                tile_w: 14,
                tile_c: 16,
                vector_width: 2,
            },
            "attention" => TuneParams::Attention {
                block_m: 128,
                block_n: 64,
                stages: 3,
                num_warps: 8,
            },
            "layer_norm" | "layernorm" => TuneParams::LayerNorm {
                vector_width: 8,
                block_size: 256,
            },
            _ => TuneParams::Gemm {
                tile_m: 64,
                tile_n: 64,
                tile_k: 16,
                vector_width: 2,
            },
        }
    }

    /// Generate a cache key for an operation
    fn generate_cache_key(&self, operation: &Operation, module: &Module) -> String {
        format!(
            "{}_{}_{}",
            operation.op_type,
            module.name,
            // For simplicity, using a hash of input/output shapes
            self.hash_shapes(&operation.inputs, &operation.outputs)
        )
    }

    /// Simple hash of shapes for cache key
    fn hash_shapes(&self, inputs: &[crate::ir::Value], outputs: &[crate::ir::Value]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for val in inputs.iter().chain(outputs.iter()) {
            val.shape.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Benchmark a specific parameter set
    pub fn benchmark(&self, operation: &Operation, params: &TuneParams) -> Result<f64> {
        // In a real implementation, this would execute the kernel with the given parameters
        // and return the execution time or performance metric
        
        println!("Benchmarking operation: {} with params: {:?}", operation.op_type, params);
        
        // For now, return a dummy performance metric
        // In reality, this would measure execution time, throughput, etc.
        Ok(100.0) // GFLOPS or some other metric
    }
}

/// Search space for hyperparameters
pub struct SearchSpace {
    pub gemm_tile_sizes: Vec<usize>,
    pub conv_tile_sizes: Vec<usize>,
    pub attention_block_sizes: Vec<usize>,
    pub vector_widths: Vec<usize>,
    pub num_warps_options: Vec<usize>,
    pub stages_options: Vec<usize>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            gemm_tile_sizes: vec![32, 64, 128, 256],
            conv_tile_sizes: vec![7, 14, 28, 56],
            attention_block_sizes: vec![16, 32, 64, 128, 256],
            vector_widths: vec![1, 2, 4, 8, 16, 32],
            num_warps_options: vec![1, 2, 4, 8, 16, 32],
            stages_options: vec![2, 3, 4, 5],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Operation, Value, Type};

    #[test]
    fn test_autotuner_creation() {
        let tuner = AutoTuner::new();
        assert_eq!(tuner.cache.len(), 0);
    }

    #[test]
    fn test_default_gemm_params() {
        let tuner = AutoTuner::new();
        let params = tuner.get_default_params("gemm");
        
        match params {
            TuneParams::Gemm { tile_m, tile_n, tile_k, vector_width } => {
                assert_eq!(tile_m, 128);
                assert_eq!(tile_n, 128);
                assert_eq!(tile_k, 32);
                assert_eq!(vector_width, 4);
            },
            _ => panic!("Expected GEMM params"),
        }
    }

    #[test]
    fn test_generate_gemm_candidates_size() {
        let tuner = AutoTuner::new();
        let candidates = tuner.generate_gemm_candidates().unwrap();
        
        // 3 tile_m options * 3 tile_n options * 3 tile_k options * 4 vector_width options = 108 combinations
        assert_eq!(candidates.len(), 108);
    }
    
    #[test]
    fn test_hash_shapes_for_cache_key() {
        let tuner = AutoTuner::new();
        
        // Create two sets of inputs/outputs with identical shapes
        let inputs1 = vec![
            Value { name: "input1".to_string(), ty: Type::F32, shape: vec![10, 20] },
            Value { name: "input2".to_string(), ty: Type::F32, shape: vec![20, 30] },
        ];
        let outputs1 = vec![
            Value { name: "output1".to_string(), ty: Type::F32, shape: vec![10, 30] },
        ];
        
        let inputs2 = vec![
            Value { name: "input_a".to_string(), ty: Type::F32, shape: vec![10, 20] }, // Same shape as inputs1
            Value { name: "input_b".to_string(), ty: Type::F32, shape: vec![20, 30] }, // Same shape as inputs1
        ];
        let outputs2 = vec![
            Value { name: "output_a".to_string(), ty: Type::F32, shape: vec![10, 30] }, // Same shape as outputs1
        ];
        
        // The hashes should be the same because the shapes are identical
        let hash1 = tuner.hash_shapes(&inputs1, &outputs1);
        let hash2 = tuner.hash_shapes(&inputs2, &outputs2);
        
        assert_eq!(hash1, hash2);
    }
    
    #[test]
    fn test_autotuner_with_unknown_operation() {
        let mut tuner = AutoTuner::new();
        let operation = Operation::new("unknown_op_type");
        let module = Module::new("test_module");
        
        // This should return default params instead of crashing
        let result = tuner.tune_operation(&operation, &module);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_search_space_defaults() {
        let search_space = SearchSpace::default();
        
        // Validate that the default configuration has the expected values
        assert_eq!(search_space.gemm_tile_sizes, vec![32, 64, 128, 256]);
        assert_eq!(search_space.conv_tile_sizes, vec![7, 14, 28, 56]);
        assert_eq!(search_space.attention_block_sizes, vec![16, 32, 64, 128, 256]);
        assert_eq!(search_space.vector_widths, vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(search_space.num_warps_options, vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(search_space.stages_options, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_generate_candidates_for_unsupported_operation() {
        let tuner = AutoTuner::new();
        
        // Create an operation with an unsupported type
        let op = Operation {
            op_type: "unsupported_op".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: std::collections::HashMap::new(),
        };
        
        let candidates = tuner.generate_candidates(&op).unwrap();
        // Should return default parameters for unsupported ops
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_tuning_cache_behavior() {
        let mut tuner = AutoTuner::new();
        
        let module = Module::new("test_cache_module");
        let op = Operation::new("gemm");
        
        // Initially cache should be empty
        assert_eq!(tuner.cache.len(), 0);
        
        // Tune the operation - this should populate the cache
        let result = tuner.tune_operation(&op, &module);
        assert!(result.is_ok());
        
        // Now cache should have one entry
        assert_eq!(tuner.cache.len(), 1);
        
        // Generate the same key to ensure we can retrieve from cache
        let key = tuner.generate_cache_key(&op, &module);
        assert!(tuner.cache.contains_key(&key));
    }

    #[test]
    fn test_extreme_parameter_generation() {
        let tuner = AutoTuner::new();
        
        // Generate candidates for each operation type
        let gemm_candidates = tuner.generate_gemm_candidates().unwrap();
        let conv_candidates = tuner.generate_conv_candidates().unwrap();
        let attention_candidates = tuner.generate_attention_candidates().unwrap();
        let layernorm_candidates = tuner.generate_layernorm_candidates().unwrap();
        
        // Ensure all parameter spaces are populated
        assert!(!gemm_candidates.is_empty());
        assert!(!conv_candidates.is_empty());
        assert!(!attention_candidates.is_empty());
        assert!(!layernorm_candidates.is_empty());
        
        // Check that all candidates have valid positive parameters
        for candidate in gemm_candidates {
            if let TuneParams::Gemm { tile_m, tile_n, tile_k, vector_width } = candidate {
                assert!(tile_m > 0);
                assert!(tile_n > 0);
                assert!(tile_k > 0);
                assert!(vector_width > 0);
            }
        }
        
        for candidate in conv_candidates {
            if let TuneParams::Conv { tile_h, tile_w, tile_c, vector_width } = candidate {
                assert!(tile_h > 0);
                assert!(tile_w > 0);
                assert!(tile_c > 0);
                assert!(vector_width > 0);
            }
        }
    }
}
