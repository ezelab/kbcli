use async_trait::async_trait;

use kbcli_core::{Error, Result};

/// Common configuration for an embedding runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Optional override for model assets directory. Falls back to
    /// `~/.kbcli/models/embeddinggemma-300m/<runtime>/` (or `KBCLI_MODEL_PATH`).
    pub model_path: Option<std::path::PathBuf>,

    /// Matryoshka truncation; pass `None` to keep the model's native dim.
    pub matryoshka_dim: Option<usize>,

    /// Maximum batch size the runtime is allowed to process at once.
    pub max_batch: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            matryoshka_dim: None,
            max_batch: 32,
        }
    }
}

impl RuntimeConfig {
    /// Resolve the model directory for a given runtime name, honoring
    /// `KBCLI_MODEL_PATH` and `--path`-style overrides.
    pub fn resolve_model_dir(&self, runtime: &str) -> Result<std::path::PathBuf> {
        if let Some(p) = &self.model_path {
            return Ok(p.clone());
        }
        if let Ok(env) = std::env::var("KBCLI_MODEL_PATH") {
            return Ok(std::path::PathBuf::from(env));
        }
        let home = dirs::home_dir()
            .ok_or_else(|| Error::other("could not resolve home directory for model cache"))?;
        Ok(home
            .join(".kbcli")
            .join("models")
            .join("embeddinggemma-300m")
            .join(runtime))
    }
}

/// Pluggable embedding runtime.
///
/// All impls produce **mean-pooled, L2-normalized** embeddings so cosine
/// similarity is comparable across runtimes.
#[async_trait]
pub trait EmbeddingRuntime: Send + Sync {
    /// Stable name (e.g. `"hash"`, `"llama"`).
    fn name(&self) -> &'static str;

    /// Embedding dimensionality after Matryoshka truncation.
    fn dim(&self) -> usize;

    /// Maximum tokens any single input may have.
    fn max_input_tokens(&self) -> usize;

    /// Embed a batch of texts. The output `Vec` has the same length as the
    /// input slice, and each inner `Vec<f32>` has length [`Self::dim`].
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Convenience: embed a single text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_batch(&[text]).await?;
        v.pop()
            .ok_or_else(|| Error::embed("empty embedding result"))
    }
}

/// L2-normalize a vector in place. `eps` guards against zero vectors.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Truncate a vector to `dim` and re-normalize (Matryoshka).
pub fn matryoshka_truncate(v: Vec<f32>, dim: usize) -> Vec<f32> {
    if v.len() <= dim {
        return v;
    }
    let mut t = v[..dim].to_vec();
    l2_normalize(&mut t);
    t
}

/// Cosine similarity for two L2-normalized vectors. Returns 0 on length
/// mismatch instead of panicking.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
