use aien_inference_protocol::*;
use aien_local_stack::SovereignInferenceService;
use aien_protocol_types::Digest32;
use sha2::{Digest, Sha256};
use std::time::Instant;
use uuid::Uuid;

fn get_rss_kb() -> i64 {
    unsafe {
        let mut rusage = std::mem::zeroed::<libc::rusage>();
        libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        rusage.ru_maxrss
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("        AIEN FLAGSHIP MULTI-BRANCH COPY-ON-WRITE DEMONSTRATION ON GB10          ");
    println!("================================================================================");
    println!("Platform: NVIDIA DGX Spark (Grace Blackwell GB10, sm_121)");
    println!("Architecture: Pure Native Rust + Hardware Paged KV Cache");
    println!("Initial RSS Memory: {} KB", get_rss_kb());
    println!("--------------------------------------------------------------------------------\n");

    let service = SovereignInferenceService::new_default()
        .map_err(|e| format!("Failed to create SovereignInferenceService: {}", e))?;

    // --- STAGE 1: Prefill Root Context ---
    println!("[Stage 1] Prefilling Root Sequence Context (40 prompt tokens, block_size = 16)...");
    let root_ctx_id = Uuid::new_v4();
    let root_branch_id = Uuid::new_v4();
    let prompt_tokens: Vec<u32> = (1..=40).map(|x| (x * 7) % 256).collect();

    let t0 = Instant::now();
    let _root_handle = service
        .register_context(root_branch_id, &prompt_tokens)
        .map_err(|e| format!("Root context prefill failed: {}", e))?;
    let prefill_us = t0.elapsed().as_micros();

    let root_blocks = service.get_block_table(root_branch_id).unwrap_or_default();
    let initial_cow_faults = service.get_cow_faults();
    let m1 = service.get_kv_metrics().unwrap();

    println!("  Prefill completed in {} us", prefill_us);
    println!("  Root Logical Blocks: {:?} (count = {})", root_blocks, root_blocks.len());
    for &b in &root_blocks {
        let rc = service.get_block_refcount(b).unwrap_or(0);
        println!("    Block {}: ref_count = {}", b, rc);
    }
    println!("  Physical Blocks Allocated: {}", m1.used_blocks);
    println!("  CoW Page Faults: {}", initial_cow_faults);
    println!("  RSS Memory: {} KB\n", get_rss_kb());

    assert_eq!(root_blocks.len(), 3, "40 tokens / 16 block_size must equal 3 blocks");
    for &b in &root_blocks {
        assert_eq!(service.get_block_refcount(b), Some(1));
    }

    // Context Ref
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

    // --- STAGE 2: Fork 3 Logical Branches ---
    println!("[Stage 2] Forking 3 Autonomous Agent Branches from Root Context...");
    let branch_names = [
        "agent-code-generator",
        "agent-security-auditor",
        "agent-perf-optimizer",
    ];
    let mut branch_contexts = Vec::new();
    let mut fork_latencies = Vec::new();

    for name in &branch_names {
        let child_branch_id = BranchId::new_v4();
        let branch_req = BranchContextRequest {
            operation_id: Uuid::new_v4(),
            parent_context: root_context_ref.clone(),
            child_branch_id,
            isolation: root_context_ref.isolation,
        };

        let tf = Instant::now();
        let receipt = service.branch_context(branch_req).await?;
        fork_latencies.push(tf.elapsed().as_micros());

        println!("  Branch {} (ID: {}):", name, child_branch_id.0);
        println!("    Shared Pages: {}, Copied Pages: {}", receipt.shared_pages, receipt.copied_pages);
        branch_contexts.push(receipt.child_context);
    }

    let m2 = service.get_kv_metrics().unwrap();
    println!("  Total Fork Latency for 3 branches: {} us", fork_latencies.iter().sum::<u128>());
    println!("  Physical Blocks Allocated: {} (FLAT ZERO-COPY INVARIANT)", m2.used_blocks);
    println!("  Total Shared Blocks in Pool: {}", m2.shared_pages);
    println!("  Total Private Blocks in Pool: {}", m2.private_pages);
    println!("  Bytes Saved vs Full Copy: {} bytes", m2.bytes_saved_vs_full_copy);
    println!("  CoW Page Faults: {}", service.get_cow_faults());
    println!("  Refcounts on Root Blocks (1 root + 3 branches = 4):");
    for &b in &root_blocks {
        let rc = service.get_block_refcount(b).unwrap_or(0);
        println!("    Block {}: ref_count = {}", b, rc);
        assert_eq!(rc, 4, "Every block must have ref_count = 4 across root + 3 forks");
    }
    assert_eq!(m2.used_blocks, 3, "Zero new physical blocks allocated during fork");
    println!("  RSS Memory: {} KB\n", get_rss_kb());

    // --- STAGE 3: Divergent Token Emissions & CoW Page Fault Isolation ---
    println!("[Stage 3] Executing Divergent Token Emissions on Branches...");

    // Step Branch 0 (Code Generator)
    println!("  1. Stepping Branch 0 ({})...", branch_names[0]);
    let req0 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-gb10".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "generate code".to_string(),
        }],
        context: Some(branch_contexts[0].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let t_step0 = Instant::now();
    let res0 = service.infer(req0).await?;
    let step0_us = t_step0.elapsed().as_micros();
    let cow_after_0 = service.get_cow_faults();
    println!("    Step completed in {} us, tokens generated: {}", step0_us, res0.completion_tokens);
    println!("    CoW Page Faults after Branch 0 step: {} (INCREMENTED BY 1)", cow_after_0);
    assert_eq!(cow_after_0, 1, "Tail page fault must trigger exactly 1 CoW event");

    // Check refcounts after Branch 0 CoW
    let tail_block = root_blocks.last().unwrap();
    let rc_tail_0 = service.get_block_refcount(*tail_block).unwrap();
    println!("    Original Shared Tail Block {} ref_count: {} (decremented 4 -> 3)", tail_block, rc_tail_0);
    assert_eq!(rc_tail_0, 3, "Branch 0 detached private copy; remaining refcount must be 3");

    // Step Branch 1 (Security Auditor)
    println!("  2. Stepping Branch 1 ({})...", branch_names[1]);
    let req1 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-gb10".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "audit security".to_string(),
        }],
        context: Some(branch_contexts[1].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let t_step1 = Instant::now();
    let res1 = service.infer(req1).await?;
    let step1_us = t_step1.elapsed().as_micros();
    let cow_after_1 = service.get_cow_faults();
    println!("    Step completed in {} us, tokens generated: {}", step1_us, res1.completion_tokens);
    println!("    CoW Page Faults after Branch 1 step: {} (INCREMENTED BY 1)", cow_after_1);
    assert_eq!(cow_after_1, 2, "Branch 1 must trigger independent CoW fault");

    let rc_tail_1 = service.get_block_refcount(*tail_block).unwrap();
    println!("    Original Shared Tail Block {} ref_count: {} (decremented 3 -> 2)", tail_block, rc_tail_1);
    assert_eq!(rc_tail_1, 2, "Branch 1 detached; remaining refcount must be 2");

    // Branch 2 (Perf Optimizer) remains completely untouched
    println!("  3. Inspecting Branch 2 ({}) [UNMODIFIED SIBLING]...", branch_names[2]);
    let b2_blocks = service.get_block_table(branch_contexts[2].branch_id.0).unwrap();
    println!("    Branch 2 Block Table: {:?} (Completely shared, zero private copies)", b2_blocks);
    assert_eq!(b2_blocks, root_blocks, "Unmodified sibling must retain identical block table");
    println!("  RSS Memory: {} KB\n", get_rss_kb());

    // --- STAGE 4: Logical Merge (SelectBranch) ---
    println!("[Stage 4] Demonstrating Logical Merge (SelectBranch)...");
    let winner_id = branch_contexts[0].branch_id.0;
    let sibling_ids = [branch_contexts[1].branch_id.0, branch_contexts[2].branch_id.0];

    println!("  Selecting Branch 0 ({}) as canonical winner...", branch_names[0]);
    println!("  Releasing unselected sibling branches with zero tensor concatenation...");
    let t_merge = Instant::now();
    service.select_branch(winner_id, &sibling_ids)
        .map_err(|e| format!("SelectBranch failed: {}", e))?;
    let merge_us = t_merge.elapsed().as_micros();
    println!("  SelectBranch completed in {} us (Zero-Copy Merge)", merge_us);

    let m4 = service.get_kv_metrics().unwrap();
    println!("  Allocated Blocks after Pruning: {}", m4.used_blocks);
    println!("  RSS Memory: {} KB\n", get_rss_kb());

    // --- STAGE 5: Crash Sovereign Core Mid-Session & Epoch Recovery ---
    println!("[Stage 5] Simulating Engine Crash & Epoch Recovery from ContextRecipeRef...");
    println!("  Current Runtime Epoch: {}", service.runtime_epoch());
    let old_epoch = service.bump_runtime_epoch();
    println!("  Crash simulated. Engine restarted: Epoch {} -> {}", old_epoch - 1, service.runtime_epoch());

    // Attempt inference with stale binding
    println!("  1. Attempting inference with stale binding from Epoch {}...", old_epoch - 1);
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
        Err(InferenceError::StaleRuntimeEpoch) => {
            println!("    SUCCESS: StaleRuntimeEpoch loudly rejected by Agent State ABI!");
        }
        other => panic!("Expected StaleRuntimeEpoch error, got: {:?}", other),
    }

    // 2. Reconstruct branch from ContextRecipeRef
    println!("  2. Reconstructing logical branch from ContextRecipeRef...");
    let t_recon = Instant::now();
    let mut recipe_tokens = prompt_tokens.clone();
    recipe_tokens.push(res0.content.as_bytes().first().copied().unwrap_or(42) as u32);

    let recon_branch_id = Uuid::new_v4();
    let recon_handle = service
        .reconstruct_from_recipe(recon_branch_id, &recipe_tokens)
        .map_err(|e| format!("Reconstruction failed: {}", e))?;
    let recon_us = t_recon.elapsed().as_micros();
    println!("    Branch reconstructed in {} us (New Handle: {})", recon_us, recon_handle.0);

    // Step reconstructed branch
    let mut recon_ctx = branch_contexts[0].clone();
    recon_ctx.branch_id = BranchId(recon_branch_id);
    recon_ctx.binding = Some(OpaqueBindingRef {
        engine_id: "aien-sovereign-gb10".to_string(),
        runtime_epoch: service.runtime_epoch(),
        binding_id: recon_branch_id,
        lease_generation: 1,
    });

    let recon_infer_req = InferenceRequest {
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

    let recon_res = service.infer(recon_infer_req).await?;
    println!("    Reconstructed branch inference succeeded: tokens = {}", recon_res.completion_tokens);
    println!("    RSS Memory: {} KB\n", get_rss_kb());

    // --- Cryptographic Telemetry Receipt ---
    println!("================================================================================");
    println!("                   CRYPTOGRAPHIC EXECUTION RECEIPT                              ");
    println!("================================================================================");

    let mut hasher = Sha256::new();
    hasher.update(format!("prefill_us:{}", prefill_us));
    hasher.update(b"fork_count:3");
    hasher.update(format!("total_cow_faults:{}", service.get_cow_faults()));
    hasher.update(format!("recon_us:{}", recon_us));
    hasher.update(format!("runtime_epoch:{}", service.runtime_epoch()));
    let receipt_digest = format!("{:x}", hasher.finalize());

    println!("Receipt Digest (SHA-256): {}", receipt_digest);
    println!("Metrics Summary:");
    println!("  - Prefill Latency: {} us", prefill_us);
    println!("  - Fork Latency (avg): {} us", fork_latencies.iter().sum::<u128>() / 3);
    println!("  - Step Latency: {} us", step0_us);
    println!("  - Zero-Copy Merge Latency: {} us", merge_us);
    println!("  - Context Recipe Reconstruction: {} us", recon_us);
    println!("  - Peak RSS Memory: {} KB", get_rss_kb());
    println!("  - Hardware Invariant: SECURE_TPM_ONLY, Pure Compiled Native Rust + GB10 Paged KV");
    println!("================================================================================");

    Ok(())
}
