use aien_inference_protocol::*;
use aien_local_stack::SovereignInferenceService;
use aien_protocol_types::Digest32;
use uuid::Uuid;

#[tokio::test]
async fn test_flagship_cow_multi_branch_gb10_lifecycle() {
    let service = SovereignInferenceService::new_default()
        .expect("Failed to create SovereignInferenceService");

    // Stage 1: Prefill root sequence context (40 tokens = 2 full blocks + 1 partial block)
    let root_ctx_id = Uuid::new_v4();
    let root_branch_id = Uuid::new_v4();
    let prompt_tokens: Vec<u32> = (1..=40).map(|x| (x * 7) % 256).collect();

    let _root_handle = service
        .register_context(root_branch_id, &prompt_tokens)
        .expect("Root context prefill failed");

    let root_blocks = service.get_block_table(root_branch_id).unwrap();
    assert_eq!(root_blocks.len(), 3, "40 tokens / 16 block_size must allocate 3 blocks");
    for &b in &root_blocks {
        assert_eq!(service.get_block_refcount(b), Some(1));
    }
    assert_eq!(service.get_cow_faults(), 0);

    let root_context_ref = InferenceContextRef {
        abi_version: 1,
        context_id: ContextId(root_ctx_id),
        branch_id: BranchId(root_branch_id),
        generation: Generation(1),
        model: ModelFingerprint {
            weights_digest: Digest32([1u8; 32]),
            model_config_digest: Digest32([2u8; 32]),
        },
        tokenizer: TokenizerFingerprint {
            tokenizer_digest: Digest32([3u8; 32]),
        },
        kv_format: KvFormatFingerprint {
            format_version: 1,
            dtype: "BF16".to_string(),
        },
        logical_state_digest: Digest32([4u8; 32]),
        token_count: 40,
        lineage: ContextLineage {
            parent_context: None,
            parent_branch: None,
            parent_digest: None,
            fork_token_index: None,
        },
        isolation: CacheIsolationKey {
            domain_id: Uuid::new_v4(),
            domain_digest: Digest32([5u8; 32]),
        },
        binding: Some(OpaqueBindingRef {
            engine_id: "aien-sovereign-gb10".to_string(),
            runtime_epoch: service.runtime_epoch(),
            binding_id: root_branch_id,
            lease_generation: 1,
        }),
        recovery: ContextRecipeRef {
            recipe_id: Uuid::new_v4(),
            recipe_digest: Digest32([6u8; 32]),
        },
    };

    // Stage 2: Fork into 3 agent branches
    let mut branch_contexts = Vec::new();
    for _ in 0..3 {
        let child_branch_id = BranchId::new_v4();
        let branch_req = BranchContextRequest {
            operation_id: Uuid::new_v4(),
            parent_context: root_context_ref.clone(),
            child_branch_id,
            isolation: root_context_ref.isolation,
        };
        let receipt = service.branch_context(branch_req).await.unwrap();
        assert_eq!(receipt.shared_pages, 3);
        assert_eq!(receipt.copied_pages, 0);
        branch_contexts.push(receipt.child_context);
    }

    let m2 = service.get_kv_metrics().unwrap();
    assert_eq!(m2.used_blocks, 3, "Flat zero-copy invariant across forks");
    for &b in &root_blocks {
        assert_eq!(service.get_block_refcount(b), Some(4), "1 root + 3 branches = ref_count 4");
    }

    // Stage 3: Divergent token emissions & CoW page fault isolation
    // Step Branch 0
    let req0 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-gb10".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "code-gen".to_string(),
        }],
        context: Some(branch_contexts[0].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let res0 = service.infer(req0).await.unwrap();
    assert_eq!(res0.completion_tokens, 1);
    assert_eq!(service.get_cow_faults(), 1, "Branch 0 triggers exactly 1 CoW fault");

    let tail_block = *root_blocks.last().unwrap();
    assert_eq!(service.get_block_refcount(tail_block), Some(3));

    // Step Branch 1
    let req1 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-gb10".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "security-audit".to_string(),
        }],
        context: Some(branch_contexts[1].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let res1 = service.infer(req1).await.unwrap();
    assert_eq!(res1.completion_tokens, 1);
    assert_eq!(service.get_cow_faults(), 2, "Branch 1 triggers second CoW fault");
    assert_eq!(service.get_block_refcount(tail_block), Some(2));

    // Branch 2 remains unmodified
    let b2_blocks = service.get_block_table(branch_contexts[2].branch_id.0).unwrap();
    assert_eq!(b2_blocks, root_blocks);

    // Stage 4: Logical Merge (SelectBranch)
    let winner_id = branch_contexts[0].branch_id.0;
    let sibling_ids = [branch_contexts[1].branch_id.0, branch_contexts[2].branch_id.0];
    service.select_branch(winner_id, &sibling_ids).unwrap();

    // Stage 5: Crash & Epoch Recovery from ContextRecipeRef
    let old_epoch = service.bump_runtime_epoch();
    let stale_req = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-gb10".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "resume".to_string(),
        }],
        context: Some(branch_contexts[0].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    match service.infer(stale_req).await {
        Err(InferenceError::StaleRuntimeEpoch) => {}
        other => panic!("Expected StaleRuntimeEpoch, got {:?}", other),
    }

    // Reconstruct from recipe
    let mut recipe_tokens = prompt_tokens.clone();
    recipe_tokens.push(42);
    let recon_id = Uuid::new_v4();
    let _recon_handle = service.reconstruct_from_recipe(recon_id, &recipe_tokens).unwrap();

    let mut recon_ctx = branch_contexts[0].clone();
    recon_ctx.branch_id = BranchId(recon_id);
    recon_ctx.binding = Some(OpaqueBindingRef {
        engine_id: "aien-sovereign-gb10".to_string(),
        runtime_epoch: service.runtime_epoch(),
        binding_id: recon_id,
        lease_generation: 1,
    });

    let recon_req = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-gb10".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "continue".to_string(),
        }],
        context: Some(recon_ctx),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let recon_res = service.infer(recon_req).await.unwrap();
    assert_eq!(recon_res.completion_tokens, 1);
    assert_eq!(service.runtime_epoch(), old_epoch);
}
