//! End-to-end benchmark for the chosen runtime + backend combo.
//!
//! Walks the full pipeline: chunk -> embed -> upsert -> hybrid query, then
//! reports per-stage latency.

use std::time::Instant;

use kbcli_core::{DocId, Document, Filter, QueryMode, QueryRequest};
use kbcli_embed::{ChunkConfig, Chunker, EmbeddingRuntime, HashRuntime};

#[tokio::main]
async fn main() {
    let n = std::env::var("KBCLI_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500usize);
    let dim = 128usize;

    let docs = kbcli_tests::corpus::seeded_corpus(n, 1);
    let runtime = HashRuntime::new(dim);
    let chunker = Chunker::new(ChunkConfig {
        size: 64,
        overlap: 8,
    })
    .unwrap();

    let t0 = Instant::now();
    let dir = tempfile::TempDir::new().unwrap();
    let cfg = kbcli_store::StoreConfig {
        embed_dim: dim,
        chunk_size: 64,
        chunk_overlap: 8,
        runtime_name: "hash".into(),
        model_id: "hash".into(),
    };

    let store: std::sync::Arc<dyn kbcli_store::VectorStore> = std::sync::Arc::new(
        kbcli_store_sqlite::SqliteStore::open(dir.path().join("e2e.db"), &cfg)
            .await
            .unwrap(),
    );

    store.migrate().await.unwrap();
    store.put_config(&cfg).await.unwrap();
    let t_open = t0.elapsed();

    let t_idx = Instant::now();
    for d in &docs {
        let mut chunks = chunker.chunk(d.id.as_str(), &d.text);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embs = runtime.embed_batch(&texts).await.unwrap();
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
        store.upsert_doc(&doc, &chunks, true).await.unwrap();
    }
    let index_ms = t_idx.elapsed().as_millis() as u64;

    let queries: Vec<&str> = vec![
        "rust language fast",
        "embedding vector search",
        "compiler memory",
    ];
    let t_q = Instant::now();
    for q in &queries {
        let qe = runtime.embed(q).await.unwrap();
        let req = QueryRequest {
            text: (*q).into(),
            mode: QueryMode::Hybrid,
            top_k: 10,
            filter: Filter::All,
            rrf_k: 60,
            weight_lex: 1.0,
            weight_sem: 1.0,
            embedding: Some(qe),
        };
        let _ = store.search(&req).await.unwrap();
    }
    let query_ms = t_q.elapsed().as_millis() as u64;

    let info = store.info().await.unwrap();
    let out = serde_json::json!({
        "kind": "e2e_bench",
        "n": n,
        "dim": dim,
        "open_ms": t_open.as_millis() as u64,
        "index_ms": index_ms,
        "query_ms": query_ms,
        "queries": queries.len(),
        "size_bytes": info.size_bytes,
        "doc_count": info.doc_count,
        "chunk_count": info.chunk_count,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
