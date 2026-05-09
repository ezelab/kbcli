//! Factory for embedding runtimes.

use std::sync::Arc;

use kbcli_core::{Error, Result};
use kbcli_embed::{EmbeddingRuntime, HashRuntime, RuntimeConfig};

/// Build a runtime by name. The two known runtimes are:
/// * `hash` — deterministic, dependency-free baseline (always available).
/// * `llama` — real EmbeddingGemma via `llama-cpp-2`. Requires the
///   `model-llama` feature; otherwise the underlying crate returns
///   `FeatureDisabled` at construction time.
pub async fn build(name: &str, dim: Option<usize>) -> Result<Arc<dyn EmbeddingRuntime>> {
    let cfg = RuntimeConfig {
        matryoshka_dim: dim,
        ..RuntimeConfig::default()
    };

    match name {
        "hash" => Ok(Arc::new(HashRuntime::new(dim.unwrap_or(128)))),

        "llama" => match kbcli_embed_llama::LlamaRuntime::new(cfg).await {
            Ok(rt) => Ok(Arc::new(rt)),
            Err(e) => Err(e),
        },

        other => Err(Error::invalid(format!(
            "unknown runtime: {other} (known: hash, llama)"
        ))),
    }
}

/// Default runtime preference when one was not previously persisted.
///
/// We prefer `llama` when the binary was built with `--features model-llama`
/// (real model loading wired up); otherwise we fall back to the
/// always-available `hash` runtime so the CLI works out of the box.
pub fn default_runtime_name() -> &'static str {
    #[cfg(feature = "model-llama")]
    {
        "llama"
    }
    #[cfg(not(feature = "model-llama"))]
    {
        "hash"
    }
}
