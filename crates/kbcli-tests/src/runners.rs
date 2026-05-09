//! Build helpers that compose `(EmbeddingRuntime, VectorStore)` for tests.

use std::path::PathBuf;
use std::sync::Arc;

use tempfile::TempDir;

use kbcli_core::{Chunk, DocId, Document, MetaValue, Result};
use kbcli_embed::{ChunkConfig, Chunker, EmbeddingRuntime, HashRuntime};
use kbcli_store::{StoreConfig, VectorStore};

/// A test harness owning a temporary directory plus a (runtime, store) pair.
pub struct Harness {
    pub _dir: TempDir,
    pub path: PathBuf,
    pub store: Arc<dyn VectorStore>,
    pub runtime: Arc<dyn EmbeddingRuntime>,
    pub chunker: Chunker,
    pub config: StoreConfig,
}

impl Harness {
    pub fn db_path(&self) -> &std::path::Path {
        &self.path
    }
}

pub async fn sqlite_with_hash(dim: usize) -> Result<Harness> {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("t.db");
    let cfg = StoreConfig {
        embed_dim: dim,
        chunk_size: 64,
        chunk_overlap: 8,
        runtime_name: "hash".into(),
        model_id: "hash".into(),
    };
    let store: Arc<dyn VectorStore> =
        Arc::new(kbcli_store_sqlite::SqliteStore::open(path.clone(), &cfg).await?);
    store.migrate().await?;
    store.put_config(&cfg).await?;
    let runtime: Arc<dyn EmbeddingRuntime> = Arc::new(HashRuntime::new(dim));
    let chunker = Chunker::new(ChunkConfig {
        size: cfg.chunk_size,
        overlap: cfg.chunk_overlap,
    })?;
    Ok(Harness {
        _dir: dir,
        path,
        store,
        runtime,
        chunker,
        config: cfg,
    })
}

/// Insert a single doc with the given metadata; embeds with `runtime`.
pub async fn ingest_text(
    h: &Harness,
    id: impl Into<DocId>,
    text: &str,
    meta: &[(&str, MetaValue)],
) -> Result<usize> {
    let id = id.into();
    let mut chunks: Vec<Chunk> = h.chunker.chunk(id.clone(), text);
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embs = h.runtime.embed_batch(&texts).await?;
    for (c, e) in chunks.iter_mut().zip(embs.into_iter()) {
        c.embedding = Some(e);
    }
    let doc = Document {
        id: id.clone(),
        text: text.to_string(),
        metadata: meta
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
        created_at: 0,
        updated_at: 0,
    };
    h.store.upsert_doc(&doc, &chunks, true).await?;
    Ok(chunks.len())
}
