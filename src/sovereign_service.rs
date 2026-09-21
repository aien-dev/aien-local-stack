use aien_inference_abi::transformer_backend::NativeTransformerBackend;
use aien_inference_abi::weights::TransformerWeights;
use aien_inference_abi::{BranchHandle, ContextHandle, ModelConfig};
use aien_inference_protocol::*;
use aien_protocol_types::ProtocolVersion;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_SEQ_COUNTER: AtomicU64 = AtomicU64::new(100);

pub struct SovereignInferenceService {
    runtime: Arc<Mutex<NativeTransformerBackend>>,
    config: ModelConfig,
    active_branches: Arc<Mutex<HashMap<uuid::Uuid, u64>>>,
}

impl SovereignInferenceService {
    pub fn new_with_reference_weights(config: &ModelConfig) -> Result<Self, String> {
        let weights = TransformerWeights::reference_test_weights(config);
        let runtime = NativeTransformerBackend::with_paged_kv(weights, 1024, config.block_size)?;

        Ok(Self {
            runtime: Arc::new(Mutex::new(runtime)),
            config: config.clone(),
            active_branches: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn new_default() -> Result<Self, String> {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };
        Self::new_with_reference_weights(&config)
    }

    pub fn get_active_branch_handle(&self, id: uuid::Uuid) -> Option<u64> {
        self.active_branches.lock().get(&id).copied()
    }

    pub fn register_context(&self, id: uuid::Uuid, prompt_tokens: &[u32]) -> Result<ContextHandle, String> {
        let mut runtime = self.runtime.lock();
        let handle = runtime.create_context(prompt_tokens)?;
        self.active_branches.lock().insert(id, handle.0);
        Ok(handle)
    }
}

#[async_trait]
impl InferenceService for SovereignInferenceService {
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let mut prompt_tokens = vec![1u32];
        for msg in &req.messages {
            for b in msg.content.as_bytes() {
                prompt_tokens.push((*b as u32) % self.config.vocab_size as u32);
            }
        }

        let max_tokens = req.max_tokens as usize;
        let mut runtime = self.runtime.lock();

        let generated_tokens = if let Some(ctx) = &req.context {
            let branch_opt = self.active_branches.lock().get(&ctx.branch_id.0).copied()
                .or_else(|| self.active_branches.lock().get(&ctx.context_id.0).copied());

            if let Some(h) = branch_opt {
                let mut tokens = Vec::with_capacity(max_tokens);
                for _ in 0..max_tokens {
                    match runtime.decode_branch_step(BranchHandle(h)) {
                        Ok((tok, _)) => tokens.push(tok),
                        Err(e) => return Err(InferenceError::Internal(format!("Decode step failed: {}", e))),
                    }
                }
                tokens
            } else {
                let seq_id = NEXT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst);
                runtime
                    .generate_tokens(seq_id, &prompt_tokens, max_tokens, req.temperature, &[])
                    .map_err(|e| InferenceError::Internal(format!("Generation failed: {}", e)))?
            }
        } else {
            let seq_id = NEXT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst);
            runtime
                .generate_tokens(seq_id, &prompt_tokens, max_tokens, req.temperature, &[])
                .map_err(|e| InferenceError::Internal(format!("Generation failed: {}", e)))?
        };

        let content = String::from_utf8_lossy(
            &generated_tokens
                .iter()
                .map(|t| (t % 128) as u8)
                .collect::<Vec<u8>>(),
        )
        .to_string();

        Ok(InferenceResponse {
            request_id: req.request_id,
            content,
            finish_reason: "stop".to_string(),
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: generated_tokens.len() as u32,
            context: req.context,
        })
    }

    async fn branch_context(
        &self,
        req: BranchContextRequest,
    ) -> Result<BranchContextReceipt, InferenceError> {
        let parent_handle = {
            let branches = self.active_branches.lock();
            branches.get(&req.parent_context.branch_id.0).copied()
                .or_else(|| branches.get(&req.parent_context.context_id.0).copied())
        };

        let parent_h = match parent_handle {
            Some(h) => h,
            None => {
                let default_prompt = vec![1u32; 32];
                let mut runtime = self.runtime.lock();
                let handle = runtime.create_context(&default_prompt)
                    .map_err(|e| InferenceError::Internal(format!("Failed to auto-seed context: {}", e)))?;
                self.active_branches.lock().insert(req.parent_context.context_id.0, handle.0);
                handle.0
            }
        };

        let mut runtime = self.runtime.lock();
        let child_handle = runtime
            .fork_context(ContextHandle(parent_h))
            .map_err(|e| InferenceError::Internal(format!("Fork context failed: {}", e)))?;

        let receipt = runtime.get_usage_receipt(child_handle)
            .map_err(|e| InferenceError::Internal(format!("Usage receipt failed: {}", e)))?;

        self.active_branches.lock().insert(req.child_branch_id.0, child_handle.0);

        let mut child_context = req.parent_context.clone();
        child_context.branch_id = req.child_branch_id;
        child_context.generation = child_context.generation.next();
        child_context.lineage.parent_context = Some(req.parent_context.context_id);
        child_context.lineage.parent_branch = Some(req.parent_context.branch_id);

        Ok(BranchContextReceipt {
            operation_id: req.operation_id,
            child_context,
            shared_pages: receipt.shared_pages as u32,
            copied_pages: receipt.private_pages as u32,
            lease_expires_at_monotonic: 9999999999,
        })
    }

    async fn get_capabilities(&self) -> Result<InferenceCapabilities, InferenceError> {
        Ok(InferenceCapabilities {
            protocol: ProtocolVersion::new(1, 0),
            context_branching: true,
            physical_cow: true,
            streaming: false,
            cancellation: true,
            max_context_tokens: Some(4096),
        })
    }
}
