pub mod sovereign_service;

pub use sovereign_service::SovereignInferenceService;

use aegis::inference::ProtocolInferenceBackend;
use std::sync::Arc;

pub struct LocalStack {
    pub inference_service: Arc<SovereignInferenceService>,
    pub protocol_backend: Arc<ProtocolInferenceBackend>,
}

impl LocalStack {
    pub fn new() -> Result<Self, String> {
        let sovereign = Arc::new(SovereignInferenceService::new_default()?);
        let backend = Arc::new(ProtocolInferenceBackend::new(
            sovereign.clone(),
            "sovereign-embedded-model".to_string(),
        ));

        Ok(Self {
            inference_service: sovereign,
            protocol_backend: backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_inference_protocol::*;
    use aien_protocol_types::Digest32;

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

        let parent_ctx = InferenceContextRef {
            abi_version: 1,
            context_id: ContextId::new_v4(),
            branch_id: BranchId::new_v4(),
            generation: Generation(1),
            model: ModelFingerprint {
                weights_digest: Digest32([0u8; 32]),
                model_config_digest: Digest32([0u8; 32]),
            },
            tokenizer: TokenizerFingerprint {
                tokenizer_digest: Digest32([0u8; 32]),
            },
            kv_format: KvFormatFingerprint {
                format_version: 1,
                dtype: "BF16".to_string(),
            },
            logical_state_digest: Digest32([0u8; 32]),
            token_count: 32,
            lineage: ContextLineage {
                parent_context: None,
                parent_branch: None,
                parent_digest: None,
                fork_token_index: None,
            },
            isolation: CacheIsolationKey {
                domain_id: uuid::Uuid::new_v4(),
                domain_digest: Digest32([1u8; 32]),
            },
            binding: None,
            recovery: ContextRecipeRef {
                recipe_id: uuid::Uuid::new_v4(),
                recipe_digest: Digest32([2u8; 32]),
            },
        };

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
    }
}
