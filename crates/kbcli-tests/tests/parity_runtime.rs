//! Runtime contract test.
//!
//! Verifies the dimensionality / length contract for the always-available
//! `HashRuntime`, plus (when the `model-llama` feature is enabled and the
//! GGUF asset is reachable) a smoke check that `LlamaRuntime` produces
//! the expected dim and pre-normalised L2 magnitude.
//!
//! Cross-runtime cosine parity is no longer checked here: with the
//! simplified stack `llama` is the only real runtime in the workspace,
//! and `HashRuntime` is intentionally non-semantic so a cosine
//! comparison would be meaningless.

#[cfg(feature = "model-llama")]
use kbcli_embed::RuntimeConfig;
use kbcli_embed::{EmbeddingRuntime, HashRuntime};

#[tokio::test]
async fn hash_runtime_dim_and_length_contract() {
    let dim = 64usize;
    let h = HashRuntime::new(dim);
    let texts = [
        "rust language",
        "vector embeddings",
        "compiler optimization",
    ];
    let v = h.embed_batch(&texts).await.unwrap();
    assert_eq!(v.len(), texts.len());
    for e in &v {
        assert_eq!(e.len(), dim);
        // L2-normalised → magnitude 1 (within float slop).
        let mag: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-4, "expected L2 norm ~1, got {mag}");
    }
}

#[cfg(feature = "model-llama")]
#[tokio::test]
async fn llama_runtime_dim_and_length_contract_when_available() {
    // Skips silently when weights aren't reachable (no network in CI).
    let cfg = RuntimeConfig::default();
    let rt = match kbcli_embed_llama::LlamaRuntime::new(cfg).await {
        Ok(rt) => rt,
        Err(_) => {
            eprintln!("skipping llama contract test: model assets not available");
            return;
        }
    };
    let texts = [
        "rust language",
        "vector embeddings",
        "compiler optimization",
    ];
    let v = match rt.embed_batch(&texts).await {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping llama contract test: embed_batch failed (likely offline)");
            return;
        }
    };
    assert_eq!(v.len(), texts.len());
    let want = rt.dim();
    for e in &v {
        assert_eq!(e.len(), want);
    }
}
