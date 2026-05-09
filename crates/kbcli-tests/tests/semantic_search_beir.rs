//! Semantic-search retrieval-quality bench against BEIR/SciFact.
//!
//! Real EmbeddingGemma (via llama-cpp-2) indexes the SciFact corpus
//! (~5,200 docs) into sqlite-vec, runs the 300-query test set, and
//! computes mean NDCG@10, MRR@10, Recall@10 against the published qrels.
//!
//! Marked `#[ignore]` so it doesn't run by default. To execute:
//!
//! ```sh
//! cargo test -p kbcli-tests --features model-llama --release \
//!     --test semantic_search_beir -- --ignored --nocapture
//! ```
//!
//! Skips silently if the GGUF asset cannot be downloaded (e.g. no
//! network in the sandbox). On a green run with weights cached, expect
//! mean NDCG@10 ≈ 0.74 (matches the EmbeddingGemma model card for
//! SciFact at dim=768). The assertion threshold is set conservatively
//! at 0.55 to absorb day-to-day variance from chunker / RRF tuning.

#![cfg(feature = "model-llama")]

use std::sync::Arc;

use kbcli_core::QueryMode;
use kbcli_embed::{ChunkConfig, Chunker, EmbeddingRuntime, RuntimeConfig};
use kbcli_store::{StoreConfig, VectorStore};
use kbcli_tests::{beir, eval, runners::Harness};
use tempfile::TempDir;

const NDCG_FLOOR: f32 = 0.55;
const QUERY_TOPK: usize = 10;

#[tokio::test]
#[ignore]
async fn beir_scifact_ndcg10_meets_floor() {
    eprintln!("[beir] downloading scifact (cached after first run)");
    let dataset = match beir::load_scifact().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[beir] skipping: dataset unavailable ({e})");
            return;
        }
    };
    eprintln!(
        "[beir] loaded scifact: {} docs, {} labelled queries",
        dataset.doc_count(),
        dataset.query_count()
    );

    eprintln!("[beir] loading EmbeddingGemma via llama-cpp");
    let runtime: Arc<dyn EmbeddingRuntime> =
        match kbcli_embed_llama::LlamaRuntime::new(RuntimeConfig::default()).await {
            Ok(rt) => Arc::new(rt),
            Err(e) => {
                eprintln!("[beir] skipping: llama runtime not available ({e})");
                return;
            }
        };
    let dim = runtime.dim();
    eprintln!("[beir] runtime dim={dim}");

    // Build harness manually (custom dim, real runtime).
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("beir.db");
    let cfg = StoreConfig {
        embed_dim: dim,
        chunk_size: 512,
        chunk_overlap: 64,
        runtime_name: "llama".into(),
        model_id: "google/embeddinggemma-300m".into(),
    };
    let store: Arc<dyn VectorStore> = Arc::new(
        kbcli_store_sqlite::SqliteStore::open(path.clone(), &cfg)
            .await
            .unwrap(),
    );
    store.migrate().await.unwrap();
    store.put_config(&cfg).await.unwrap();
    let chunker = Chunker::new(ChunkConfig {
        size: cfg.chunk_size,
        overlap: cfg.chunk_overlap,
    })
    .unwrap();
    let h = Harness {
        _dir: dir,
        path,
        store,
        runtime,
        chunker,
        config: cfg,
    };

    eprintln!("[beir] indexing {} docs", dataset.corpus.len());
    let t_idx = std::time::Instant::now();
    eval::index_corpus(&h, &dataset.corpus).await.unwrap();
    eprintln!("[beir] indexed in {:.1}s", t_idx.elapsed().as_secs_f32());

    eprintln!("[beir] evaluating {} queries", dataset.queries.len());
    let t_q = std::time::Instant::now();
    let report = eval::evaluate_with_mode(&h, &dataset.queries, QUERY_TOPK, QueryMode::Hybrid)
        .await
        .unwrap();
    eprintln!("[beir] evaluated in {:.1}s", t_q.elapsed().as_secs_f32());

    eprintln!("[beir] report: {:?}", report);
    assert!(
        report.mean_ndcg_at_k >= NDCG_FLOOR,
        "BEIR/SciFact mean NDCG@10 below floor {NDCG_FLOOR}: got {}",
        report.mean_ndcg_at_k,
    );
}
