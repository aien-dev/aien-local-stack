use aien_inference_protocol::ContextRecipeRef;
use aien_protocol_types::Digest32;
use aien_provenance::compute_canonical_json_digest;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextRecipe {
    pub recipe_id: Uuid,
    pub root_prompt_tokens: Vec<u32>,
    pub branch_deltas: BTreeMap<Uuid, Vec<u32>>,
}

impl ContextRecipe {
    pub fn new(root_prompt_tokens: Vec<u32>) -> Self {
        Self {
            recipe_id: Uuid::new_v4(),
            root_prompt_tokens,
            branch_deltas: BTreeMap::new(),
        }
    }

    pub fn with_branch_delta(mut self, branch_id: Uuid, delta: Vec<u32>) -> Self {
        self.branch_deltas.insert(branch_id, delta);
        self
    }

    pub fn compute_digest(&self) -> Digest32 {
        compute_canonical_json_digest(self).unwrap_or(Digest32::ZERO)
    }

    pub fn to_ref(&self) -> ContextRecipeRef {
        ContextRecipeRef {
            recipe_id: self.recipe_id,
            recipe_digest: self.compute_digest(),
        }
    }

    pub fn full_tokens_for_branch(&self, branch_id: &Uuid) -> Vec<u32> {
        let mut tokens = self.root_prompt_tokens.clone();
        if let Some(delta) = self.branch_deltas.get(branch_id) {
            tokens.extend_from_slice(delta);
        }
        tokens
    }
}

pub struct RecipeStore {
    storage_dir: Option<PathBuf>,
    recipes: RwLock<BTreeMap<Uuid, ContextRecipe>>,
}

impl RecipeStore {
    pub fn new() -> Self {
        Self {
            storage_dir: None,
            recipes: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn with_storage_dir(dir: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path = dir.into();
        fs::create_dir_all(&path)?;

        let store = Self {
            storage_dir: Some(path.clone()),
            recipes: RwLock::new(BTreeMap::new()),
        };

        if let Ok(entries) = fs::read_dir(&path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(content) = fs::read_to_string(&p) {
                        if let Ok(recipe) = serde_json::from_str::<ContextRecipe>(&content) {
                            store.recipes.write().insert(recipe.recipe_id, recipe);
                        }
                    }
                }
            }
        }

        Ok(store)
    }

    pub fn insert(&self, recipe: ContextRecipe) -> Result<(), std::io::Error> {
        if let Some(dir) = &self.storage_dir {
            let filename = format!("{}.json", recipe.recipe_id);
            let file_path = dir.join(filename);
            let temp_path = dir.join(format!("{}.tmp", recipe.recipe_id));
            let content = serde_json::to_string_pretty(&recipe)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            fs::write(&temp_path, content)?;
            fs::rename(&temp_path, &file_path)?;
        }
        self.recipes.write().insert(recipe.recipe_id, recipe);
        Ok(())
    }

    pub fn verify_and_get(&self, recipe_ref: &ContextRecipeRef) -> Result<ContextRecipe, String> {
        let in_memory = self.recipes.read().get(&recipe_ref.recipe_id).cloned();
        let recipe = match in_memory {
            Some(r) => r,
            None => {
                if let Some(dir) = &self.storage_dir {
                    let file_path = dir.join(format!("{}.json", recipe_ref.recipe_id));
                    if let Ok(content) = fs::read_to_string(&file_path) {
                        let r: ContextRecipe = serde_json::from_str(&content)
                            .map_err(|e| format!("Failed to parse recipe file: {}", e))?;
                        self.recipes.write().insert(r.recipe_id, r.clone());
                        r
                    } else {
                        return Err(format!("Recipe {} not found in store or disk", recipe_ref.recipe_id));
                    }
                } else {
                    return Err(format!("Recipe {} not found in store", recipe_ref.recipe_id));
                }
            }
        };

        let actual_digest = recipe.compute_digest();
        if actual_digest != recipe_ref.recipe_digest {
            return Err(format!(
                "Recipe digest mismatch: expected {:?}, actual {:?}",
                recipe_ref.recipe_digest, actual_digest
            ));
        }

        Ok(recipe)
    }
}

impl Default for RecipeStore {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedRecipeStore = Arc<RecipeStore>;
