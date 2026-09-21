use aien_inference_protocol::ContextRecipeRef;
use aien_protocol_types::Digest32;
use aien_provenance::compute_canonical_json_digest;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextRecipe {
    pub recipe_id: Uuid,
    pub root_prompt_tokens: Vec<u32>,
    pub branch_deltas: HashMap<Uuid, Vec<u32>>,
}

impl ContextRecipe {
    pub fn new(root_prompt_tokens: Vec<u32>) -> Self {
        Self {
            recipe_id: Uuid::new_v4(),
            root_prompt_tokens,
            branch_deltas: HashMap::new(),
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
    recipes: RwLock<HashMap<Uuid, ContextRecipe>>,
}

impl RecipeStore {
    pub fn new() -> Self {
        Self {
            recipes: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(&self, recipe: ContextRecipe) {
        self.recipes.write().insert(recipe.recipe_id, recipe);
    }

    pub fn verify_and_get(&self, recipe_ref: &ContextRecipeRef) -> Result<ContextRecipe, String> {
        let recipes = self.recipes.read();
        let recipe = recipes
            .get(&recipe_ref.recipe_id)
            .ok_or_else(|| format!("Recipe {} not found in store", recipe_ref.recipe_id))?;

        let actual_digest = recipe.compute_digest();
        if actual_digest != recipe_ref.recipe_digest {
            return Err(format!(
                "Recipe digest mismatch: expected {:?}, actual {:?}",
                recipe_ref.recipe_digest, actual_digest
            ));
        }

        Ok(recipe.clone())
    }
}

impl Default for RecipeStore {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedRecipeStore = Arc<RecipeStore>;
