//! Additional tests focusing on memory allocation edge cases

#[cfg(test)]
mod memory_allocation_edge_case_tests {
    use crate::runtime::{Device, ExecutionContext, MemoryHandle, Runtime};

    #[test]
    fn test_memory_allocation_with_extreme_sizes() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        
        // Test allocation with extremely small size
        let tiny_handle = ctx.allocate_tensor(1);
        assert!(tiny_handle.is_ok());
        
        // Test allocation with larger sizes
        let small_handle = ctx.allocate_tensor(1024); // 1KB
        assert!(small_handle.is_ok());
        
        let medium_handle = ctx.allocate_tensor(1024 * 1024); // 1MB
        assert!(medium_handle.is_ok());
        
        let large_handle = ctx.allocate_tensor(10 * 1024 * 1024); // 10MB
        assert!(large_handle.is_ok());
    }

    #[test]
    fn test_memory_allocation_failure_scenarios() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        
        // Note: We can't actually force allocation failure without low-level manipulation
        // but we can test the allocation and deallocation cycle
        
        let handle = ctx.allocate_tensor(1000).unwrap();
        assert!(handle.size >= 1000);
        
        // Test deallocation
        let dealloc_result = ctx.memory_pool.deallocate(handle);
        assert!(dealloc_result.is_ok());
    }

    #[test]
    fn test_memory_pool_with_many_allocations() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        let mut handles = Vec::new();
        
        // Perform many allocations
        for i in 0..1000 {
            let handle_result = ctx.allocate_tensor(i * 100 + 1);
            assert!(handle_result.is_ok());
            handles.push(handle_result.unwrap());
        }
        
        assert_eq!(handles.len(), 1000);
        
        // Verify all handles are unique
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                assert_ne!(handles[i].id, handles[j].id, "Duplicate handle IDs detected");
            }
        }
    }

    #[test]
    fn test_memory_pool_allocation_deallocation_cycles() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        let mut handles = Vec::new();
        
        // Perform allocations and deallocations in cycles
        for cycle in 0..10 {
            // Allocate in this cycle
            for i in 0..100 {
                let handle = ctx.allocate_tensor((cycle * 100 + i) * 10 + 1).unwrap();
                handles.push(handle);
            }
            
            // Deallocate half of them
            for i in (0..handles.len()).step_by(2) {
                if i < handles.len() {
                    let handle = handles.remove(i);
                    let _ = ctx.memory_pool.deallocate(handle);
                }
            }
        }
        
        // Check that remaining allocations are still tracked
        assert!(!handles.is_empty());
    }

    #[test]
    fn test_memory_pool_concurrent_access_simulation() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        
        // Simulate a scenario of multiple allocations
        let mut active_handles = Vec::new();
        
        // Phase 1: Allocate many handles
        for i in 0..500 {
            let handle = ctx.allocate_tensor(100 + i).unwrap();
            active_handles.push(handle);
        }
        
        // Phase 2: Deallocate every third handle
        let mut indices_to_remove = Vec::new();
        for i in (0..active_handles.len()).step_by(3) {
            indices_to_remove.push(i);
        }
        
        // Actually remove them in reverse order to maintain indices
        for &idx in indices_to_remove.iter().rev() {
            if idx < active_handles.len() {
                let handle = active_handles.remove(idx);
                let _ = ctx.memory_pool.deallocate(handle);
            }
        }
        
        // Phase 3: Allocate more handles to ensure memory pool is functioning
        for i in 0..200 {
            let handle = ctx.allocate_tensor(1000 + i).unwrap();
            active_handles.push(handle);
        }
        
        assert!(active_handles.len() > 200); // Should have more than 200 handles
    }

    #[test]
    fn test_memory_handle_properties() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        
        // Test creating handles with various properties
        let handle1 = ctx.allocate_tensor(1).unwrap();
        let handle2 = ctx.allocate_tensor(1000).unwrap();
        let handle3 = ctx.allocate_tensor(1_000_000).unwrap(); // 1MB
        
        // Each handle should have different properties
        assert!(handle2.size >= handle1.size);
        assert!(handle3.size >= handle2.size);
        
        // Each handle should be associated with the correct device
        assert_eq!(handle1.device, Device::Cpu);
        assert_eq!(handle2.device, Device::Cpu);
        assert_eq!(handle3.device, Device::Cpu);
    }

    #[test]
    fn test_runtime_memory_management() {
        let mut runtime = Runtime::new();
        
        // Test execution context creation and cleanup simulation
        for _ in 0..10 {
            let mut ctx = runtime.create_context(Some(Device::Cpu)).unwrap();
            
            // Do some allocations in the context
            for i in 0..10 {
                let _handle = ctx.allocate_tensor(i * 100 + 1).unwrap();
            }
        }
    }

    #[test]
    fn test_execution_context_lifecycle() {
        // Test creating and dropping execution contexts
        for i in 0..5 {
            let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
            
            // Do some work in the context
            let handle = ctx.allocate_tensor(i * 1000 + 1).unwrap();
            
            // The context will be dropped here, which should clean up properly
        }
    }

    #[test]
    fn test_memory_pool_fragmentation() {
        let mut ctx = ExecutionContext::new(Device::Cpu).unwrap();
        let mut handles = Vec::new();
        
        // Create many allocations of different sizes
        for i in 0..100 {
            let size = 100 + (i * 50);
            let handle = ctx.allocate_tensor(size).unwrap();
            handles.push(Some(handle));
        }
        
        // Free some of them randomly
        for i in (0..handles.len()).step_by(3) {
            if let Some(handle) = handles[i].take() {
                let _ = ctx.memory_pool.deallocate(handle);
            }
        }
        
        // Allocate more to test reuse of freed space
        for i in 100..150 {
            let size = 200 + (i * 25);
            let handle = ctx.allocate_tensor(size).unwrap();
            handles.push(Some(handle));
        }
        
        // Count remaining active handles
        let active_count = handles.iter().filter(|h| h.is_some()).count();
        assert!(active_count > 50); // Should have more than 50 active handles
    }

    #[test]
    fn test_large_number_of_contexts() {
        // Test creating a large number of contexts to test resource management
        for i in 0..50 {
            let ctx_result = ExecutionContext::new(Device::Cpu);
            assert!(ctx_result.is_ok());
            
            let mut ctx = ctx_result.unwrap();
            
            // Do minimal work to verify context is functional
            let handle_result = ctx.allocate_tensor(i + 1);
            assert!(handle_result.is_ok());
            
            // Context gets dropped here
        }
    }
}