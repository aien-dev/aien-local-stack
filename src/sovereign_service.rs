use crate::recipe::ContextRecipe;
use aien_inference_abi::transformer_backend::NativeTransformerBackend;
use aien_inference_abi::weights::TransformerWeights;
use aien_inference_abi::{BranchHandle, ContextHandle, ModelConfig};
use aien_inference_protocol::*;
use aien_kv_cache::KvMetrics;
use aien_platform_linux::{LinuxComputeDevice, LinuxMemoryKind};
use aien_protocol_types::{Digest32, ProtocolVersion};
use aien_provenance::{compute_canonical_json_digest, compute_sha256};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

static NEXT_SEQ_COUNTER: AtomicU64 = AtomicU64::new(100);

#[derive(Clone, Debug)]
pub struct BranchState {
    pub handle: u64,
    pub branch_id: Uuid,
    pub context_id: Uuid,
    pub generation: Generation,
    pub token_count: u64,
    pub tokens: Vec<u32>,
    pub logical_state_digest: Digest32,
    pub recipe_ref: ContextRecipeRef,
    pub isolation: CacheIsolationKey,
    pub lineage: ContextLineage,
}

impl BranchState {
    pub fn compute_digest(branch_id: &Uuid, tokens: &[u32]) -> Digest32 {
        let mut bytes = Vec::with_capacity(16 + tokens.len() * 4);
        bytes.extend_from_slice(branch_id.as_bytes());
        for &tok in tokens {
            bytes.extend_from_slice(&tok.to_le_bytes());
        }
        compute_sha256(&bytes)
    }

    pub fn to_context_ref(&self, service: &SovereignInferenceService) -> InferenceContextRef {
        InferenceContextRef {
            abi_version: 1,
            context_id: ContextId(self.context_id),
            branch_id: BranchId(self.branch_id),
            generation: self.generation,
            model: service.model_fingerprint(),
            tokenizer: service.tokenizer_fingerprint(),
            kv_format: service.kv_format_fingerprint(),
            logical_state_digest: self.logical_state_digest,
            token_count: self.token_count,
            lineage: self.lineage.clone(),
            isolation: self.isolation,
            binding: Some(OpaqueBindingRef {
                engine_id: service.engine_id.clone(),
                runtime_epoch: service.runtime_epoch(),
                binding_id: self.branch_id,
                lease_generation: 1,
            }),
            recovery: self.recipe_ref.clone(),
        }
    }
}

pub struct SovereignInferenceService {
    runtime: Arc<Mutex<NativeTransformerBackend>>,
    config: ModelConfig,
    active_branches: Arc<Mutex<HashMap<Uuid, BranchState>>>,
    epoch: Arc<AtomicU64>,
    pub engine_id: String,
    pub backend_name: String,
    pub is_accelerated: bool,
    pub memory_kind: String,
}

impl SovereignInferenceService {
    pub fn new_with_reference_weights(config: &ModelConfig, epoch: u64) -> Result<Self, String> {
        let weights = TransformerWeights::reference_test_weights(config);
        let runtime = NativeTransformerBackend::with_paged_kv(weights, 1024, config.block_size)?;

        let device = LinuxComputeDevice::new();
        let mem_kind = match device.preferred_memory_kind() {
            LinuxMemoryKind::AtsSystem => "AtsSystem (Hardware Coherent Grace-Blackwell)",
            LinuxMemoryKind::HmmSystem => "HmmSystem (Heterogeneous Memory Management)",
            LinuxMemoryKind::CudaManagedFallback => "CudaManagedFallback",
        };

        Ok(Self {
            runtime: Arc::new(Mutex::new(runtime)),
            config: config.clone(),
            active_branches: Arc::new(Mutex::new(HashMap::new())),
            epoch: Arc::new(AtomicU64::new(epoch)),
            engine_id: "aien-sovereign-gb10".to_string(),
            backend_name: "ReferenceCpuBackend (CPU Golden Oracle)".to_string(),
            is_accelerated: false,
            memory_kind: mem_kind.to_string(),
        })
    }

    pub fn new_default() -> Result<Self, String> {
        Self::new_with_epoch(1)
    }

    pub fn new_with_epoch(epoch: u64) -> Result<Self, String> {
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
        Self::new_with_reference_weights(&config, epoch)
    }

    pub fn model_fingerprint(&self) -> ModelFingerprint {
        let cfg_digest = compute_canonical_json_digest(&self.config).unwrap_or(Digest32::ZERO);
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&(self.config.hidden_dim as u64).to_le_bytes());
        w_bytes.extend_from_slice(&(self.config.num_layers as u64).to_le_bytes());
        w_bytes.extend_from_slice(&(self.config.vocab_size as u64).to_le_bytes());
        let w_digest = compute_sha256(&w_bytes);

        ModelFingerprint {
            weights_digest: w_digest,
            model_config_digest: cfg_digest,
        }
    }

    pub fn tokenizer_fingerprint(&self) -> TokenizerFingerprint {
        let mut t_bytes = Vec::new();
        t_bytes.extend_from_slice(&(self.config.vocab_size as u64).to_le_bytes());
        t_bytes.extend_from_slice(b"sovereign-reference-tokenizer-v1");
        TokenizerFingerprint {
            tokenizer_digest: compute_sha256(&t_bytes),
        }
    }

    pub fn kv_format_fingerprint(&self) -> KvFormatFingerprint {
        KvFormatFingerprint {
            format_version: 1,
            dtype: "FP32".to_string(),
        }
    }

    pub fn runtime_epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    pub fn bump_runtime_epoch(&self) -> u64 {
        let new_epoch = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        self.active_branches.lock().clear();
        new_epoch
    }

    pub fn create_root_context(
        &self,
        prompt_tokens: &[u32],
        recipe: &ContextRecipe,
    ) -> Result<(ContextHandle, InferenceContextRef), String> {
        let mut runtime = self.runtime.lock();
        let handle = runtime.create_context(prompt_tokens)?;

        let root_ctx_id = Uuid::new_v4();
        let root_branch_id = Uuid::new_v4();
        let state_digest = BranchState::compute_digest(&root_branch_id, prompt_tokens);

        let branch_state = BranchState {
            handle: handle.0,
            branch_id: root_branch_id,
            context_id: root_ctx_id,
            generation: Generation(1),
            token_count: prompt_tokens.len() as u64,
            tokens: prompt_tokens.to_vec(),
            logical_state_digest: state_digest,
            recipe_ref: recipe.to_ref(),
            isolation: CacheIsolationKey {
                domain_id: Uuid::new_v4(),
                domain_digest: compute_sha256(b"sovereign-tenant-root"),
            },
            lineage: ContextLineage {
                parent_context: None,
                parent_branch: None,
                parent_digest: None,
                fork_token_index: None,
            },
        };

        let ctx_ref = branch_state.to_context_ref(self);
        self.active_branches
            .lock()
            .insert(root_branch_id, branch_state);

        Ok((handle, ctx_ref))
    }

    pub fn reconstruct_from_recipe(
        &self,
        branch_id: Uuid,
        recipe: &ContextRecipe,
    ) -> Result<(ContextHandle, InferenceContextRef), String> {
        let tokens = recipe.full_tokens_for_branch(&branch_id);
        let mut runtime = self.runtime.lock();
        let handle = runtime.create_context(&tokens)?;

        let ctx_id = Uuid::new_v4();
        let state_digest = BranchState::compute_digest(&branch_id, &tokens);

        let branch_state = BranchState {
            handle: handle.0,
            branch_id,
            context_id: ctx_id,
            generation: Generation(1),
            token_count: tokens.len() as u64,
            tokens,
            logical_state_digest: state_digest,
            recipe_ref: recipe.to_ref(),
            isolation: CacheIsolationKey {
                domain_id: Uuid::new_v4(),
                domain_digest: compute_sha256(b"sovereign-reconstruction-domain"),
            },
            lineage: ContextLineage {
                parent_context: None,
                parent_branch: None,
                parent_digest: None,
                fork_token_index: None,
            },
        };

        let ctx_ref = branch_state.to_context_ref(self);
        self.active_branches.lock().insert(branch_id, branch_state);

        Ok((handle, ctx_ref))
    }

    pub fn get_block_table(&self, id: Uuid) -> Option<Vec<usize>> {
        let handle = self.active_branches.lock().get(&id).map(|b| b.handle)?;
        let runtime = self.runtime.lock();
        let kv_mgr = runtime.kv_manager.as_ref()?;
        let mgr = kv_mgr.read();
        let tbl = mgr.get_block_table(handle)?;
        Some(tbl.block_ids.clone())
    }

    pub fn get_block_refcount(&self, block_id: usize) -> Option<usize> {
        let runtime = self.runtime.lock();
        let kv_mgr = runtime.kv_manager.as_ref()?;
        let mgr = kv_mgr.read();
        mgr.get_block(block_id).map(|b| b.ref_count)
    }

    pub fn get_cow_faults(&self) -> usize {
        let runtime = self.runtime.lock();
        runtime
            .kv_manager
            .as_ref()
            .map(|m| m.read().cow_faults())
            .unwrap_or(0)
    }

    pub fn get_kv_metrics(&self) -> Option<KvMetrics> {
        let runtime = self.runtime.lock();
        runtime.kv_manager.as_ref().map(|m| m.read().metrics())
    }

    pub fn get_branch_tokens(&self, branch_id: Uuid) -> Option<Vec<u32>> {
        let branches = self.active_branches.lock();
        branches.get(&branch_id).map(|s| s.tokens.clone())
    }

    pub fn snapshot_branch_recipe(
        &self,
        branch_id: Uuid,
        recipe: &ContextRecipe,
    ) -> Option<ContextRecipe> {
        let branches = self.active_branches.lock();
        let state = branches.get(&branch_id)?;
        let root_len = recipe.root_prompt_tokens.len();
        if state.tokens.len() >= root_len
            && state.tokens[..root_len] == recipe.root_prompt_tokens[..]
        {
            let delta = state.tokens[root_len..].to_vec();
            Some(recipe.clone().with_branch_delta(branch_id, delta))
        } else {
            None
        }
    }

    pub fn select_branch(
        &self,
        selected_branch: Uuid,
        sibling_branches: &[Uuid],
    ) -> Result<(), String> {
        let mut runtime = self.runtime.lock();
        let mut branches = self.active_branches.lock();

        if !branches.contains_key(&selected_branch) {
            return Err(format!("Selected branch {} not found", selected_branch));
        }

        for &sibling in sibling_branches {
            if sibling == selected_branch {
                continue;
            }
            if let Some(st) = branches.remove(&sibling) {
                let _ = runtime.release_branch(BranchHandle(st.handle));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl InferenceService for SovereignInferenceService {
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        if let Some(ctx) = &req.context {
            if let Some(binding) = &ctx.binding {
                let current_epoch = self.runtime_epoch();
                if binding.runtime_epoch != current_epoch {
                    return Err(InferenceError::StaleRuntimeEpoch);
                }
            }

            if ctx.kv_format.dtype != "FP32" {
                return Err(InferenceError::Internal(format!(
                    "KV format mismatch: expected FP32, got {}",
                    ctx.kv_format.dtype
                )));
            }
        }

        let max_tokens = req.max_tokens as usize;
        let mut runtime = self.runtime.lock();

        let (generated_tokens, updated_ctx) = if let Some(ctx) = &req.context {
            let mut branches = self.active_branches.lock();
            let state = branches
                .get_mut(&ctx.branch_id.0)
                .ok_or(InferenceError::ContextNotFound(ctx.context_id))?;

            if ctx.generation != state.generation {
                return Err(InferenceError::StaleGeneration {
                    expected: state.generation,
                    actual: ctx.generation,
                });
            }

            // Extract branch delta tokens from request messages
            let mut delta_tokens = Vec::new();
            for msg in &req.messages {
                for b in msg.content.as_bytes() {
                    delta_tokens.push((*b as u32) % self.config.vocab_size as u32);
                }
            }

            // Ingest branch delta into physical KV cache (triggers CoW on shared block!)
            if !delta_tokens.is_empty() {
                runtime
                    .append_branch_tokens(BranchHandle(state.handle), &delta_tokens)
                    .map_err(|e| {
                        InferenceError::Internal(format!("Failed to append branch tokens: {}", e))
                    })?;
                state.tokens.extend_from_slice(&delta_tokens);
            }

            // Autoregressively decode tokens
            let mut tokens = Vec::with_capacity(max_tokens);
            for _ in 0..max_tokens {
                match runtime.decode_branch_step(BranchHandle(state.handle)) {
                    Ok((tok, _)) => tokens.push(tok),
                    Err(e) => {
                        return Err(InferenceError::Internal(format!(
                            "Decode step failed: {}",
                            e
                        )))
                    }
                }
            }

            // Ingest generated tokens into logical branch state so state.tokens,
            // token_count, and logical_state_digest accurately reflect post-generation state
            state.tokens.extend_from_slice(&tokens);

            // Advance branch state
            state.generation = state.generation.next();
            state.token_count = state.tokens.len() as u64;
            state.logical_state_digest =
                BranchState::compute_digest(&state.branch_id, &state.tokens);

            let new_ctx_ref = state.to_context_ref(self);
            (tokens, Some(new_ctx_ref))
        } else {
            let mut prompt_tokens = vec![1u32];
            for msg in &req.messages {
                for b in msg.content.as_bytes() {
                    prompt_tokens.push((*b as u32) % self.config.vocab_size as u32);
                }
            }
            let seq_id = NEXT_SEQ_COUNTER.fetch_add(1, Ordering::SeqCst);
            let toks = runtime
                .generate_tokens(seq_id, &prompt_tokens, max_tokens, req.temperature, &[])
                .map_err(|e| InferenceError::Internal(format!("Generation failed: {}", e)))?;
            (toks, None)
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
            prompt_tokens: req.messages.iter().map(|m| m.content.len() as u32).sum(),
            completion_tokens: generated_tokens.len() as u32,
            context: updated_ctx,
        })
    }

    async fn branch_context(
        &self,
        req: BranchContextRequest,
    ) -> Result<BranchContextReceipt, InferenceError> {
        let parent_state = {
            let branches = self.active_branches.lock();
            branches
                .get(&req.parent_context.branch_id.0)
                .cloned()
                .ok_or(InferenceError::ContextNotFound(
                    req.parent_context.context_id,
                ))?
        };

        let mut runtime = self.runtime.lock();
        let child_handle = runtime
            .fork_context(ContextHandle(parent_state.handle))
            .map_err(|e| InferenceError::Internal(format!("Fork context failed: {}", e)))?;

        let receipt = runtime
            .get_usage_receipt(child_handle)
            .map_err(|e| InferenceError::Internal(format!("Usage receipt failed: {}", e)))?;

        let child_state = BranchState {
            handle: child_handle.0,
            branch_id: req.child_branch_id.0,
            context_id: parent_state.context_id,
            generation: parent_state.generation.next(),
            token_count: parent_state.token_count,
            tokens: parent_state.tokens.clone(),
            logical_state_digest: parent_state.logical_state_digest,
            recipe_ref: parent_state.recipe_ref.clone(),
            isolation: req.isolation,
            lineage: ContextLineage {
                parent_context: Some(req.parent_context.context_id),
                parent_branch: Some(req.parent_context.branch_id),
                parent_digest: Some(parent_state.logical_state_digest),
                fork_token_index: Some(parent_state.token_count),
            },
        };

        let child_context_ref = child_state.to_context_ref(self);
        self.active_branches
            .lock()
            .insert(req.child_branch_id.0, child_state);

        Ok(BranchContextReceipt {
            operation_id: req.operation_id,
            child_context: child_context_ref,
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
