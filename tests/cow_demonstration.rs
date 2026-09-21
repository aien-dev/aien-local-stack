use aien_inference_protocol::*;
use aien_local_stack::{ContextRecipe, RecipeStore, SovereignInferenceService};
use uuid::Uuid;

#[tokio::test]
async fn test_flagship_cow_multi_branch_hardened_lifecycle() {
    let temp_dir = std::env::temp_dir().join(format!("aien-test-recipe-store-{}", Uuid::new_v4()));
    let recipe_store = RecipeStore::with_storage_dir(&temp_dir)
        .expect("Failed to initialize file-backed RecipeStore");

    let service = SovereignInferenceService::new_with_epoch(1).expect("Service creation failed");

    // Stage 1: Prefill root context with durable ContextRecipe
    let prompt: Vec<u32> = (1..=40).map(|x| (x * 7) % 256).collect();
    let mut recipe = ContextRecipe::new(prompt.clone());
    recipe_store.insert(recipe.clone()).unwrap();

    let (_root_handle, root_context) = service
        .create_root_context(&prompt, &recipe)
        .expect("create root context failed");

    assert_eq!(root_context.generation, Generation(1));
    assert_eq!(root_context.token_count, 40);
    assert_eq!(root_context.kv_format.dtype, "FP32");
    assert_eq!(service.get_cow_faults(), 0);

    let root_blocks = service.get_block_table(root_context.branch_id.0).unwrap();
    assert_eq!(root_blocks.len(), 3);
    for &b in &root_blocks {
        assert_eq!(service.get_block_refcount(b), Some(1));
    }

    // Stage 2: Fork into 3 branches
    let mut branch_contexts = Vec::new();
    for _ in 0..3 {
        let child_branch_id = BranchId::new_v4();
        let branch_req = BranchContextRequest {
            operation_id: Uuid::new_v4(),
            parent_context: root_context.clone(),
            child_branch_id,
            isolation: root_context.isolation,
        };
        let receipt = service.branch_context(branch_req).await.unwrap();
        assert_eq!(receipt.shared_pages, 3);
        assert_eq!(receipt.copied_pages, 0);
        assert_eq!(receipt.child_context.generation, Generation(2));
        branch_contexts.push(receipt.child_context);
    }

    let m2 = service.get_kv_metrics().unwrap();
    assert_eq!(m2.used_blocks, 3, "Flat zero-copy invariant");
    for &b in &root_blocks {
        assert_eq!(
            service.get_block_refcount(b),
            Some(4),
            "Root + 3 branches = 4"
        );
    }

    // Stage 3: True Branch Divergence
    // Branch 0 receives "gen" delta
    let req0 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "gen".to_string(),
        }],
        context: Some(branch_contexts[0].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let res0 = service.infer(req0).await.unwrap();
    let ctx0 = res0.context.unwrap();
    assert_eq!(service.get_cow_faults(), 1);
    assert_eq!(ctx0.generation, Generation(3));
    // 40 root tokens + 3 "gen" delta tokens + 1 generated token = 44 tokens
    assert_eq!(ctx0.token_count, 44);

    // Branch 1 receives "sec" delta
    let req1 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "sec".to_string(),
        }],
        context: Some(branch_contexts[1].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let res1 = service.infer(req1).await.unwrap();
    let ctx1 = res1.context.unwrap();
    assert_eq!(service.get_cow_faults(), 2);
    assert_eq!(ctx1.generation, Generation(3));
    assert_eq!(ctx1.token_count, 44);

    // Verify true logical divergence
    assert_ne!(ctx0.logical_state_digest, ctx1.logical_state_digest);

    // Branch 2 remains unmodified and fully shared
    let b2_blocks = service
        .get_block_table(branch_contexts[2].branch_id.0)
        .unwrap();
    assert_eq!(b2_blocks, root_blocks);

    // Snapshot Branch 0 state (including generated tokens) into durable recipe
    recipe = service
        .snapshot_branch_recipe(ctx0.branch_id.0, &recipe)
        .expect("Snapshot recipe failed");
    recipe_store.insert(recipe.clone()).unwrap();

    // Stage 4: Zero-copy logical merge
    let winner_id = ctx0.branch_id.0;
    let sibling_ids = [
        branch_contexts[1].branch_id.0,
        branch_contexts[2].branch_id.0,
    ];
    service.select_branch(winner_id, &sibling_ids).unwrap();

    // Stage 5: Authentic Engine Crash & RecipeStore Recovery
    let pre_crash_digest = ctx0.logical_state_digest;
    let pre_crash_recipe_ref = recipe.to_ref();

    // Complete destruction of engine instance and recipe store memory
    drop(service);
    drop(recipe_store);

    let new_service = SovereignInferenceService::new_with_epoch(2).expect("Restart failed");
    assert_eq!(new_service.runtime_epoch(), 2);

    let recovered_recipe_store =
        RecipeStore::with_storage_dir(&temp_dir).expect("Failed to reload RecipeStore from disk");

    // Stale epoch rejected
    let stale_req = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "continue".to_string(),
        }],
        context: Some(ctx0.clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    match new_service.infer(stale_req).await {
        Err(InferenceError::StaleRuntimeEpoch) => {}
        other => panic!("Expected StaleRuntimeEpoch, got {:?}", other),
    }

    // Recipe store recovery from disk
    let verified_recipe = recovered_recipe_store
        .verify_and_get(&pre_crash_recipe_ref)
        .unwrap();
    let (_recon_handle, recon_ctx) = new_service
        .reconstruct_from_recipe(winner_id, &verified_recipe)
        .unwrap();

    assert_eq!(recon_ctx.logical_state_digest, pre_crash_digest);
    assert_eq!(recon_ctx.token_count, 44);

    let _ = std::fs::remove_dir_all(&temp_dir);

    // Continued inference on reconstructed branch
    let cont_req = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "next".to_string(),
        }],
        context: Some(recon_ctx),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let cont_res = new_service.infer(cont_req).await.unwrap();
    assert_eq!(cont_res.completion_tokens, 1);
}
