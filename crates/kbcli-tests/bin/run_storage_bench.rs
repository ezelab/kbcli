//! Storage benchmark: measures index/search latency for the storage backend.
//!
//! With the simplified stack the workspace ships a single backend
//! (`sqlite-vec`); the harness is kept generic so adding another impl
//! later is a one-line change.
//!
//! Uses `HashRuntime` to keep the embedding step constant.

use std::sync::Arc;
use std::time::Instant;

use kbcli_core::{DocId, Document, Filter, QueryMode, QueryRequest};
use kbcli_embed::{ChunkConfig, Chunker, EmbeddingRuntime, HashRuntime};
use kbcli_store::{StoreConfig, VectorStore};

#[tokio::main]
async fn main() {
    let n = std::env::var("KBCLI_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000usize);
    let dim = std::env::var("KBCLI_BENCH_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128usize);
    let q_count = std::env::var("KBCLI_BENCH_Q")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50usize);

    let mut results = Vec::new();

    results.push(
        run_one("sqlite-vec", n, dim, q_count, |path, cfg| {
            let path = path.to_path_buf();
            let cfg = cfg.clone();
            Box::pin(async move {
                let s = kbcli_store_sqlite::SqliteStore::open(path, &cfg).await?;
                let arc: Arc<dyn VectorStore> = Arc::new(s);
                Ok(arc)
            })
        })
        .await,
    );

    let out = serde_json::json!({
        "kind": "storage_bench",
        "n": n,
        "dim": dim,
        "queries": q_count,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

type StoreFut =
    std::pin::Pin<Box<dyn std::future::Future<Output = kbcli_core::Result<Arc<dyn VectorStore>>>>>;

async fn run_one<F>(name: &str, n: usize, dim: usize, q_count: usize, build: F) -> serde_json::Value
where
    F: FnOnce(&std::path::Path, &StoreConfig) -> StoreFut,
{
    let dir = match tempfile::TempDir::new() {
        Ok(d) => d,
        Err(e) => return err_json(name, format!("tempdir: {e}")),
    };
    let path = dir.path().join("bench.db");
    let cfg = StoreConfig {
        embed_dim: dim,
        chunk_size: 64,
        chunk_overlap: 8,
        runtime_name: "hash".into(),
        model_id: "hash".into(),
    };

    let store = match build(&path, &cfg).await {
        Ok(s) => s,
        Err(e) => return err_json(name, format!("open: {e}")),
    };
    if let Err(e) = store.migrate().await {
        return err_json(name, format!("migrate: {e}"));
    }
    if let Err(e) = store.put_config(&cfg).await {
        return err_json(name, format!("put_config: {e}"));
    }

    let runtime = HashRuntime::new(dim);
    let chunker = Chunker::new(ChunkConfig {
        size: 64,
        overlap: 8,
    })
    .unwrap();
    let docs = kbcli_tests::corpus::seeded_corpus(n, 13);

    // Index phase.
    let t_idx = Instant::now();
    for d in &docs {
        let mut chunks = chunker.chunk(d.id.as_str(), &d.text);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embs = match runtime.embed_batch(&texts).await {
            Ok(v) => v,
            Err(e) => return err_json(name, format!("embed: {e}")),
        };
        for (c, e) in chunks.iter_mut().zip(embs.into_iter()) {
            c.embedding = Some(e);
        }
        let doc = Document {
            id: DocId::new(d.id.clone()),
            text: d.text.clone(),
            metadata: Default::default(),
            created_at: 0,
            updated_at: 0,
        };
        if let Err(e) = store.upsert_doc(&doc, &chunks, true).await {
            return err_json(name, format!("upsert: {e}"));
        }
    }
    let index_ms = t_idx.elapsed().as_millis() as u64;

    // Query phase.
    let q_docs = kbcli_tests::corpus::seeded_corpus(q_count, 99);
    let t_q = Instant::now();
    for d in &q_docs {
        let qe = runtime.embed(&d.text).await.unwrap();
        let req = QueryRequest {
            text: d.text.clone(),
            mode: QueryMode::Hybrid,
            top_k: 10,
            filter: Filter::All,
            rrf_k: 60,
            weight_lex: 1.0,
            weight_sem: 1.0,
            embedding: Some(qe),
        };
        if let Err(e) = store.search(&req).await {
            return err_json(name, format!("search: {e}"));
        }
    }
    let query_ms = t_q.elapsed().as_millis() as u64;

    let info = store.info().await.ok();
    let size_bytes = info.as_ref().map(|i| i.size_bytes).unwrap_or(0);
    drop(store);
    drop(dir);

    serde_json::json!({
        "backend": name,
        "ok": true,
        "docs": n,
        "query_count": q_count,
        "index_ms": index_ms,
        "query_ms": query_ms,
        "qps": (q_count as f64) / (query_ms as f64 / 1000.0).max(1e-6),
        "size_bytes": size_bytes,
    })
}

fn err_json(name: &str, msg: String) -> serde_json::Value {
    serde_json::json!({
        "backend": name,
        "ok": false,
        "error": msg,
    })
}
