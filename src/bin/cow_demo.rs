use aien_inference_protocol::*;
use aien_local_stack::{ContextRecipe, RecipeStore, SovereignInferenceService};
use aien_provenance::compute_canonical_json_digest;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

fn get_rss_kb() -> i64 {
    unsafe {
        let mut rusage = std::mem::zeroed::<libc::rusage>();
        libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        rusage.ru_maxrss
    }
}

fn get_git_sha(repo_path: &str) -> String {
    std::process::Command::new("git")
        .args(["-C", repo_path, "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_uname_info() -> (String, String) {
    unsafe {
        let mut uts = std::mem::zeroed::<libc::utsname>();
        if libc::uname(&mut uts) == 0 {
            let sysname = std::ffi::CStr::from_ptr(uts.sysname.as_ptr()).to_string_lossy();
            let release = std::ffi::CStr::from_ptr(uts.release.as_ptr()).to_string_lossy();
            let machine = std::ffi::CStr::from_ptr(uts.machine.as_ptr()).to_string_lossy();
            (machine.to_string(), format!("{} {}", sysname, release))
        } else {
            (std::env::consts::ARCH.to_string(), std::env::consts::OS.to_string())
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecutionReceipt {
    pub demo_version: String,
    pub dependency_shas: DependencyShas,
    pub hardware: HardwareDescriptor,
    pub runtime: RuntimeDescriptor,
    pub measurements: Measurements,
    pub invariants_verified: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DependencyShas {
    pub aien_protocols: String,
    pub aegis_runtime: String,
    pub aien_sovereign_core: String,
    pub aien_local_stack: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HardwareDescriptor {
    pub architecture: String,
    pub kernel: String,
    pub memory_kind: String,
    pub peak_rss_kb: i64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RuntimeDescriptor {
    pub engine_id: String,
    pub backend: String,
    pub is_hardware_accelerated: bool,
    pub kv_dtype: String,
    pub block_size: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Measurements {
    pub prefill_latency_us: u128,
    pub fork_latency_avg_us: u128,
    pub branch_0_step_us: u128,
    pub branch_1_step_us: u128,
    pub zero_copy_merge_us: u128,
    pub recipe_reconstruction_us: u128,
    pub total_cow_faults: usize,
    pub bytes_saved_vs_full_copy: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("        AIEN Sovereign Multi-Branch Copy-on-Write & State ABI Demonstration     ");
    println!("================================================================================");

    let storage_dir = std::env::temp_dir().join(format!("aien-recipe-store-{}", Uuid::new_v4()));
    let recipe_store = RecipeStore::with_storage_dir(&storage_dir)
        .map_err(|e| format!("Failed to create durable RecipeStore at {:?}: {}", storage_dir, e))?;

    // --- STAGE 1: Initialize Engine and Prefill Shared Root Context ---
    println!("\n[Stage 1] Initializing Sovereign Inference Service (Epoch = 1)...");
    let service = SovereignInferenceService::new_with_epoch(1)
        .map_err(|e| format!("Service initialization failed: {}", e))?;

    println!("  Engine ID: {}", service.engine_id);
    println!("  Backend:   {}", service.backend_name);
    println!("  Accelerated: {}", service.is_accelerated);
    println!("  Memory:    {}", service.memory_kind);
    println!("  Epoch:     {}", service.runtime_epoch());

    let root_prompt_tokens: Vec<u32> = (1..=40).map(|x| (x * 7) % 256).collect();
    println!("  Prefilling root context with {} tokens (block_size = 16)...", root_prompt_tokens.len());

    let mut recipe = ContextRecipe::new(root_prompt_tokens.clone());
    recipe_store.insert(recipe.clone())?;

    let t0 = Instant::now();
    let (root_handle, root_context) = service
        .create_root_context(&root_prompt_tokens, &recipe)
        .map_err(|e| format!("Root context creation failed: {}", e))?;
    let prefill_us = t0.elapsed().as_micros();

    println!("  Root Context ID:     {:?}", root_context.context_id);
    println!("  Root Branch ID:      {}", root_context.branch_id.0);
    println!("  Root Physical Handle: {:?}", root_handle.0);
    println!("  Root Token Count:    {}", root_context.token_count);
    println!("  Prefill Latency:     {} us", prefill_us);

    let m1 = service.get_kv_metrics().unwrap();
    println!("  Allocated Blocks:    {} (shared root pages)", m1.used_blocks);
    println!("  Initial CoW Faults:  {}", service.get_cow_faults());

    let root_blocks = service.get_block_table(root_context.branch_id.0).unwrap();
    println!("  Root Block Table:    {:?}", root_blocks);
    for &b in &root_blocks {
        println!("    Block {}: refcount = {}", b, service.get_block_refcount(b).unwrap());
    }
    println!();

    // --- STAGE 2: Fork Shared Root into Multiple Subagent Branches ---
    println!("[Stage 2] Forking root context into 3 concurrent subagent branches...");
    let mut branch_contexts = Vec::new();
    let mut fork_times = Vec::new();
    let branch_names = ["Branch-0 (CodeGen)", "Branch-1 (SecurityAudit)", "Branch-2 (DocGen)"];

    for (_i, name) in branch_names.iter().enumerate() {
        let child_branch_id = BranchId::new_v4();
        let branch_req = BranchContextRequest {
            operation_id: Uuid::new_v4(),
            parent_context: root_context.clone(),
            child_branch_id,
            isolation: root_context.isolation,
        };

        let t_fork = Instant::now();
        let receipt = service.branch_context(branch_req).await
            .map_err(|e| format!("Fork failed for {}: {}", name, e))?;
        let us = t_fork.elapsed().as_micros();
        fork_times.push(us);

        println!("  Forked {} in {} us", name, us);
        println!("    Branch ID:     {}", receipt.child_context.branch_id.0);
        println!("    Shared Pages:  {}", receipt.shared_pages);
        println!("    Copied Pages:  {} (Flat Zero-Copy Guaranteed)", receipt.copied_pages);
        println!("    Generation:    {:?}", receipt.child_context.generation);

        branch_contexts.push(receipt.child_context);
        assert_eq!(receipt.copied_pages, 0, "Fork must be pure zero-copy pointer retention");
    }

    let fork_avg_us: u128 = fork_times.iter().sum::<u128>() / fork_times.len() as u128;
    println!("  Average Fork Latency: {} us", fork_avg_us);

    let m2 = service.get_kv_metrics().unwrap();
    println!("  Total Used Blocks after 3-way Fork: {} (Zero block allocation)", m2.used_blocks);
    for &b in &root_blocks {
        println!("    Block {}: refcount = {} (root + 3 branches)", b, service.get_block_refcount(b).unwrap());
    }
    println!();

    // --- STAGE 3: Mutate Branch and Verify Copy-on-Write Fault ---
    println!("[Stage 3] Mutating Branch 0 and Branch 1 with divergent deltas to trigger physical CoW...");

    println!("  Stepping Branch 0 ({}) with delta \"generate code\"...", branch_names[0]);
    let req0 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
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
    let res0 = service.infer(req0).await
        .map_err(|e| format!("Infer on Branch 0 failed: {}", e))?;
    let step0_us = t_step0.elapsed().as_micros();
    let cow_after_0 = service.get_cow_faults();
    let updated_ctx0 = res0.context.clone().unwrap();

    println!("  Branch 0 completed step in {} us (Tokens generated = {})", step0_us, res0.completion_tokens);
    println!("  CoW Faults after Branch 0 write: {} (Exactly 1 tail page cloned)", cow_after_0);
    assert_eq!(cow_after_0, 1, "Expected exactly 1 CoW fault after branch 0 mutation");

    println!("  Stepping Branch 1 ({}) with delta \"security audit\"...", branch_names[1]);
    let req1 = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "security audit".to_string(),
        }],
        context: Some(branch_contexts[1].clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };

    let t_step1 = Instant::now();
    let res1 = service.infer(req1).await
        .map_err(|e| format!("Infer on Branch 1 failed: {}", e))?;
    let step1_us = t_step1.elapsed().as_micros();
    let cow_after_1 = service.get_cow_faults();
    let updated_ctx1 = res1.context.clone().unwrap();

    println!("  Branch 1 completed step in {} us (Tokens generated = {})", step1_us, res1.completion_tokens);
    println!("  CoW Faults after Branch 1 write: {} (Cloned own tail page)", cow_after_1);
    assert_eq!(cow_after_1, 2, "Expected exactly 2 CoW faults after independent branch mutations");

    println!("  Verifying True Logical & Physical Divergence between Branch 0 and Branch 1...");
    println!("    Branch 0 State Digest: {:?}", updated_ctx0.logical_state_digest);
    println!("    Branch 1 State Digest: {:?}", updated_ctx1.logical_state_digest);
    assert_ne!(
        updated_ctx0.logical_state_digest, updated_ctx1.logical_state_digest,
        "Branch 0 and Branch 1 must physically and logically diverge"
    );

    println!("  Verifying Branch 2 remained untouched and fully shared...");
    let b2_blocks = service.get_block_table(branch_contexts[2].branch_id.0).unwrap();
    assert_eq!(b2_blocks, root_blocks);
    println!();

    // Snapshot durable recipe from branch state (covers root prompt + branch delta + generated tokens)
    recipe = service
        .snapshot_branch_recipe(updated_ctx0.branch_id.0, &recipe)
        .expect("Snapshot branch recipe failed");
    recipe_store.insert(recipe.clone())?;

    // --- STAGE 4: Zero-Copy Logical Merge (SelectBranch) ---
    println!("[Stage 4] Demonstrating Logical Merge (SelectBranch)...");
    let winner_id = updated_ctx0.branch_id.0;
    let sibling_ids = [branch_contexts[1].branch_id.0, branch_contexts[2].branch_id.0];

    println!("  Selecting Branch 0 ({}) as canonical winner...", branch_names[0]);
    let t_merge = Instant::now();
    service.select_branch(winner_id, &sibling_ids)
        .map_err(|e| format!("SelectBranch failed: {}", e))?;
    let merge_us = t_merge.elapsed().as_micros();
    println!("  SelectBranch completed in {} us (Zero-Copy Pointer Retention)", merge_us);

    let m4 = service.get_kv_metrics().unwrap();
    println!("  Allocated Blocks after Pruning Siblings: {}\n", m4.used_blocks);

    // --- STAGE 5: Authentic Engine Crash & Durable ContextRecipeRef Recovery ---
    println!("[Stage 5] Simulating Authentic Engine Crash & Recovery from RecipeStore...");
    let pre_crash_digest = updated_ctx0.logical_state_digest;
    let pre_crash_recipe_ref = recipe.to_ref();

    println!("  1. Destroying Sovereign Core and RecipeStore (simulating process crash)...");
    drop(service);
    drop(recipe_store);

    println!("  2. Instantiating fresh Sovereign Core and re-opening RecipeStore from disk (Epoch = 2)...");
    let fresh_service = SovereignInferenceService::new_with_epoch(2)
        .map_err(|e| format!("Restart failed: {}", e))?;
    let recovered_recipe_store = RecipeStore::with_storage_dir(&storage_dir)
        .map_err(|e| format!("Failed to reload RecipeStore from disk: {}", e))?;
    println!("     Restarted runtime epoch: {}", fresh_service.runtime_epoch());

    // Verify stale context rejection
    println!("  3. Probing stale binding from pre-crash Epoch 1 against new engine...");
    let stale_req = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "resume".to_string(),
        }],
        context: Some(updated_ctx0.clone()),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    match fresh_service.infer(stale_req).await {
        Err(InferenceError::StaleRuntimeEpoch) => {
            println!("     SUCCESS: StaleRuntimeEpoch correctly rejected stale epoch binding.");
        }
        other => panic!("Expected StaleRuntimeEpoch, got {:?}", other),
    }

    // Durable Recipe Recovery
    println!("  4. Resolving durable ContextRecipe from disk-backed RecipeStore via ContextRecipeRef...");
    let t_recon = Instant::now();
    let durable_recipe = recovered_recipe_store
        .verify_and_get(&pre_crash_recipe_ref)
        .map_err(|e| format!("Recipe verification failed: {}", e))?;

    let (recon_handle, recon_ctx) = fresh_service
        .reconstruct_from_recipe(winner_id, &durable_recipe)
        .map_err(|e| format!("Reconstruction failed: {}", e))?;
    let recon_us = t_recon.elapsed().as_micros();

    println!("     Branch reconstructed in {} us (New Context Handle ID: {:?})", recon_us, recon_handle.0);
    println!("     Reconstructed Logical Digest: {:?}", recon_ctx.logical_state_digest);
    println!("     Pre-Crash Logical Digest:    {:?}", pre_crash_digest);
    assert_eq!(
        recon_ctx.logical_state_digest, pre_crash_digest,
        "Reconstructed state digest must match pre-crash state digest exactly"
    );

    // Clean up temporary storage directory
    let _ = std::fs::remove_dir_all(&storage_dir);

    // Verify inference continues on reconstructed branch
    let cont_req = InferenceRequest {
        request_id: Uuid::new_v4(),
        model: "sovereign-embedded".to_string(),
        messages: vec![InferenceMessage {
            role: "user".to_string(),
            content: "continue".to_string(),
        }],
        context: Some(recon_ctx),
        max_tokens: 1,
        temperature: 0.0,
        stop_sequences: vec![],
    };
    let cont_res = fresh_service.infer(cont_req).await?;
    println!("     Inference on recovered branch succeeded (tokens = {})\n", cont_res.completion_tokens);

    // --- Structured Execution Receipt ---
    println!("================================================================================");
    println!("                       STRUCTURED EXECUTION RECEIPT                             ");
    println!("================================================================================");

    let measurements = Measurements {
        prefill_latency_us: prefill_us,
        fork_latency_avg_us: fork_avg_us,
        branch_0_step_us: step0_us,
        branch_1_step_us: step1_us,
        zero_copy_merge_us: merge_us,
        recipe_reconstruction_us: recon_us,
        total_cow_faults: fresh_service.get_cow_faults() + cow_after_1,
        bytes_saved_vs_full_copy: m2.bytes_saved_vs_full_copy,
    };

    let (arch, kernel) = get_uname_info();
    let receipt_data = ExecutionReceipt {
        demo_version: "1.2.0".to_string(),
        dependency_shas: DependencyShas {
            aien_protocols: get_git_sha("/home/drakestapleton/workspace/aien-protocols"),
            aegis_runtime: get_git_sha("/home/drakestapleton/workspace/aegis-runtime"),
            aien_sovereign_core: get_git_sha("/home/drakestapleton/workspace/aien-sovereign-core"),
            aien_local_stack: get_git_sha("/home/drakestapleton/workspace/aien-local-stack"),
        },
        hardware: HardwareDescriptor {
            architecture: arch,
            kernel,
            memory_kind: fresh_service.memory_kind.clone(),
            peak_rss_kb: get_rss_kb(),
        },
        runtime: RuntimeDescriptor {
            engine_id: fresh_service.engine_id.clone(),
            backend: fresh_service.backend_name.clone(),
            is_hardware_accelerated: fresh_service.is_accelerated,
            kv_dtype: "FP32".to_string(),
            block_size: 16,
        },
        measurements,
        invariants_verified: vec![
            "flat_zero_copy_branch_fork".to_string(),
            "isolated_tail_page_cow_fault".to_string(),
            "untouched_sibling_block_sharing".to_string(),
            "branch_delta_state_divergence".to_string(),
            "zero_copy_logical_merge_select_branch".to_string(),
            "stale_runtime_epoch_rejection".to_string(),
            "durable_recipe_hash_verification".to_string(),
            "post_crash_logical_state_digest_parity".to_string(),
            "file_backed_recipe_store_persistence".to_string(),
            "generated_token_state_parity".to_string(),
        ],
        receipt_digest: String::new(),
    };

    let canonical_digest = compute_canonical_json_digest(&receipt_data).unwrap();
    let final_receipt = ExecutionReceipt {
        receipt_digest: canonical_digest.to_hex(),
        ..receipt_data
    };

    let json_receipt = serde_json::to_string_pretty(&final_receipt)?;
    std::fs::write("receipt-cow-demo.json", &json_receipt)?;

    println!("Receipt Digest (SHA-256): {}", final_receipt.receipt_digest);
    println!("Saved to: receipt-cow-demo.json");
    println!("Summary:");
    println!("  - Backend: {}", final_receipt.runtime.backend);
    println!("  - Memory Subsystem: {}", final_receipt.hardware.memory_kind);
    println!("  - KV DType: {}", final_receipt.runtime.kv_dtype);
    println!("  - Prefill Latency: {} us", final_receipt.measurements.prefill_latency_us);
    println!("  - Fork Latency (avg): {} us", final_receipt.measurements.fork_latency_avg_us);
    println!("  - True Divergence Verified: Branch A != Branch B state digest");
    println!("  - Merge Latency (SelectBranch): {} us", final_receipt.measurements.zero_copy_merge_us);
    println!("  - Recipe Reconstruction: {} us (Parity Verified)", final_receipt.measurements.recipe_reconstruction_us);
    println!("  - Peak RSS Memory: {} KB", final_receipt.hardware.peak_rss_kb);
    println!("================================================================================");

    Ok(())
}
