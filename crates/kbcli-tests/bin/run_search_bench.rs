//! Headless retrieval-quality bench (BEIR/SciFact + EmbeddingGemma).
//!
//! Same workflow as the `semantic_search_beir` test, but emits a JSON
//! report to stdout and never asserts. Useful for `docs/perf-report.md`
//! and for tracking quality drift over time.

use std::sync::Arc;
use std::time::Instant;

use kbcli_core::QueryMode;
use kbcli_embed::{ChunkConfig, Chunker, EmbeddingRuntime, RuntimeConfig};
use kbcli_store::{StoreConfig, VectorStore};
use kbcli_tests::{beir, eval, runners::Harness};
use tempfile::TempDir;

#[tokio::main]
async fn main() {
    let dataset = match beir::load_scifact().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[search-bench] dataset unavailable: {e}");
            std::process::exit(2);
        }
    };
    eprintln!(
        "[search-bench] scifact: {} docs, {} labelled queries",
        dataset.doc_count(),
        dataset.query_count()
    );

    let runtime: Arc<dyn EmbeddingRuntime> =
        match kbcli_embed_llama::LlamaRuntime::new(RuntimeConfig::default()).await {
            Ok(rt) => Arc::new(rt),
            Err(e) => {
                eprintln!("[search-bench] llama runtime unavailable: {e}");
                std::process::exit(3);
            }
        };
    let dim = runtime.dim();

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("search-bench.db");
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

    let t_idx = Instant::now();
    eval::index_corpus(&h, &dataset.corpus).await.unwrap();
    let index_ms = t_idx.elapsed().as_millis() as u64;

    let modes = [
        ("hybrid", QueryMode::Hybrid),
        ("semantic", QueryMode::Semantic),
        ("lexical", QueryMode::Lexical),
    ];
    let mut per_mode = Vec::new();
    for (label, mode) in modes {
        let t_q = Instant::now();
        let report = eval::evaluate_with_mode(&h, &dataset.queries, 10, mode)
            .await
            .unwrap();
        per_mode.push(serde_json::json!({
            "mode": label,
            "query_ms": t_q.elapsed().as_millis() as u64,
            "metrics": report.to_json(),
        }));
    }

    let out = serde_json::json!({
        "kind": "search_bench",
        "dataset": "BeIR/scifact (test)",
        "runtime": "llama (EmbeddingGemma 300m, Q8_0)",
        "backend": "sqlite-vec",
        "dim": dim,
        "doc_count": dataset.doc_count(),
        "query_count": dataset.query_count(),
        "index_ms": index_ms,
        "results": per_mode,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
