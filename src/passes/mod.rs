//! Optimization passes for the Impulse compiler
//! Contains various transformation and optimization passes

use crate::ir::Module;
use anyhow::Result;

/// Manager for running optimization passes
pub struct PassManager {
    pub passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the manager
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run all registered passes on the module
    pub fn run_passes(&self, module: &mut Module) -> Result<()> {
        for pass in &self.passes {
            pass.run(module)?;
        }
        Ok(())
    }
}

/// Trait that all passes must implement
pub trait Pass: Send + Sync {
    /// Run the pass on the given module
    fn run(&self, module: &mut Module) -> Result<()>;
    
    /// Name of the pass for debugging
    fn name(&self) -> &'static str;
}

/// Constant folding pass - folds constant computations at compile time
pub struct ConstantFoldPass;

impl Pass for ConstantFoldPass {
    fn run(&self, _module: &mut Module) -> Result<()> {
        // TODO: Implement constant folding logic
        println!("Running constant folding pass...");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ConstantFoldPass"
    }
}

/// Dead code elimination pass - removes unused operations
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn run(&self, _module: &mut Module) -> Result<()> {
        // TODO: Implement dead code elimination logic
        println!("Running dead code elimination pass...");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DeadCodeEliminationPass"
    }
}

/// Operator fusion pass - fuses compatible operations
pub struct OperatorFusionPass;

impl Pass for OperatorFusionPass {
    fn run(&self, _module: &mut Module) -> Result<()> {
        // TODO: Implement operator fusion logic
        println!("Running operator fusion pass...");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "OperatorFusionPass"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Value, Type};

    #[test]
    fn test_pass_manager_creation() {
        let pm = PassManager::new();
        assert_eq!(pm.passes.len(), 0);
    }

    #[test]
    fn test_pass_manager_add_pass() {
        let mut pm = PassManager::new();
        pm.add_pass(Box::new(ConstantFoldPass));
        assert_eq!(pm.passes.len(), 1);
    }

    #[test]
    fn test_pass_manager_run_passes() {
        let mut pm = PassManager::new();
        pm.add_pass(Box::new(ConstantFoldPass));
        pm.add_pass(Box::new(DeadCodeEliminationPass));
        
        let mut module = Module::new("test_module");
        // Add a simple test operation to the module
        let mut test_op = crate::ir::Operation::new("test_op");
        test_op.inputs.push(Value {
            name: "input1".to_string(),
            ty: Type::F32,
            shape: vec![1, 2],
        });
        module.add_operation(test_op);
        
        let result = pm.run_passes(&mut module);
        assert!(result.is_ok());
        assert_eq!(module.operations.len(), 1); // Should still have the operation
    }

    #[test]
    fn test_pass_manager_empty() {
        let pm = PassManager::new();
        let mut module = Module::new("empty_test_module");
        
        let result = pm.run_passes(&mut module);
        assert!(result.is_ok());
        assert_eq!(module.operations.len(), 0);
    }

    #[test]
    fn test_constant_fold_pass() {
        let pass = ConstantFoldPass;
        assert_eq!(pass.name(), "ConstantFoldPass");
        
        let mut module = Module::new("const_fold_test");
        let result = pass.run(&mut module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dead_code_elimination_pass() {
        let pass = DeadCodeEliminationPass;
        assert_eq!(pass.name(), "DeadCodeEliminationPass");
        
        let mut module = Module::new("dce_test");
        let result = pass.run(&mut module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_operator_fusion_pass() {
        let pass = OperatorFusionPass;
        assert_eq!(pass.name(), "OperatorFusionPass");
        
        let mut module = Module::new("fusion_test");
        let result = pass.run(&mut module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pass_execution_order() {
        // Create a custom pass for testing order
        struct OrderTestPass {
            id: u32,
        }
        
        impl Pass for OrderTestPass {
            fn run(&self, module: &mut Module) -> Result<()> {
                // Add the pass ID to the module name to track execution order
                module.name.push_str(&format!("_{}", self.id));
                Ok(())
            }

            fn name(&self) -> &'static str {
                "OrderTestPass"
            }
        }

        let mut pm = PassManager::new();
        pm.add_pass(Box::new(OrderTestPass { id: 1 }));
        pm.add_pass(Box::new(OrderTestPass { id: 2 }));
        pm.add_pass(Box::new(OrderTestPass { id: 3 }));

        let mut module = Module::new("order_test");
        pm.run_passes(&mut module).unwrap();

        // The order should be preserved: 1, 2, 3
        assert_eq!(module.name, "order_test_1_2_3");
    }

    #[test]
    fn test_pass_with_failure_simulation() {
        // Create a pass that fails to test error handling
        struct FailingPass;
        
        impl Pass for FailingPass {
            fn run(&self, _module: &mut Module) -> Result<()> {
                anyhow::bail!("Simulated pass failure")
            }

            fn name(&self) -> &'static str {
                "FailingPass"
            }
        }

        let mut pm = PassManager::new();
        pm.add_pass(Box::new(ConstantFoldPass));
        pm.add_pass(Box::new(FailingPass));
        pm.add_pass(Box::new(DeadCodeEliminationPass));

        let mut module = Module::new("failure_test");
        let result = pm.run_passes(&mut module);

        // The failure should propagate
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Simulated pass failure"));
    }
}