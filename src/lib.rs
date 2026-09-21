pub mod recipe;
pub mod sovereign_service;

pub use recipe::{ContextRecipe, RecipeStore, SharedRecipeStore};
pub use sovereign_service::{BranchState, SovereignInferenceService};

use aegis::inference::ProtocolInferenceBackend;
use std::sync::Arc;

pub struct LocalStack {
    pub inference_service: Arc<SovereignInferenceService>,
    pub protocol_backend: Arc<ProtocolInferenceBackend>,
    pub recipe_store: Arc<RecipeStore>,
}

impl LocalStack {
    pub fn new() -> Result<Self, String> {
        let sovereign = Arc::new(SovereignInferenceService::new_default()?);
        let backend = Arc::new(ProtocolInferenceBackend::new(
            sovereign.clone(),
            "sovereign-embedded-model".to_string(),
        ));
        let recipe_store = Arc::new(RecipeStore::new());

        Ok(Self {
            inference_service: sovereign,
            protocol_backend: backend,
            recipe_store,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_inference_protocol::*;

    #[tokio::test]
    async fn test_local_stack_in_process_composition() {
        let stack = LocalStack::new().expect("Failed to initialize LocalStack");
        let caps = stack.inference_service.get_capabilities().await.unwrap();
        assert!(caps.context_branching);
        assert!(caps.physical_cow);

        let req = InferenceRequest {
            request_id: uuid::Uuid::new_v4(),
            model: "sovereign-embedded-model".to_string(),
            messages: vec![InferenceMessage {
                role: "user".to_string(),
                content: "ping".to_string(),
            }],
            context: None,
            max_tokens: 4,
            temperature: 0.0,
            stop_sequences: vec![],
        };

        let res = stack.inference_service.infer(req).await.unwrap();
        assert_eq!(res.finish_reason, "stop");
        assert!(res.completion_tokens > 0);
    }

    #[tokio::test]
    async fn test_local_stack_branch_and_cow() {
        let stack = LocalStack::new().expect("Failed to initialize LocalStack");

        let prompt: Vec<u32> = (1..=32).collect();
        let recipe = ContextRecipe::new(prompt.clone());
        stack.recipe_store.insert(recipe.clone()).unwrap();

        let (_root_h, parent_ctx) = stack
            .inference_service
            .create_root_context(&prompt, &recipe)
            .expect("create root context");

        let branch_req = BranchContextRequest {
            operation_id: uuid::Uuid::new_v4(),
            parent_context: parent_ctx.clone(),
            child_branch_id: BranchId::new_v4(),
            isolation: parent_ctx.isolation,
        };

        let receipt = stack.inference_service.branch_context(branch_req).await.unwrap();
        assert!(receipt.shared_pages > 0);
        assert_eq!(receipt.copied_pages, 0);

        let infer_req = InferenceRequest {
            request_id: uuid::Uuid::new_v4(),
            model: "sovereign-embedded-model".to_string(),
            messages: vec![InferenceMessage {
                role: "user".to_string(),
                content: "step".to_string(),
            }],
            context: Some(receipt.child_context),
            max_tokens: 2,
            temperature: 0.0,
            stop_sequences: vec![],
        };

        let res = stack.inference_service.infer(infer_req).await.unwrap();
        assert_eq!(res.completion_tokens, 2);
        assert!(res.context.is_some());
        let ctx = res.context.unwrap();
        assert_eq!(ctx.generation, Generation(3)); // 1 root -> 2 fork -> 3 after step
        assert!(ctx.token_count > 32);
    }
}
