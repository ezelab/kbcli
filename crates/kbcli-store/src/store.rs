use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use kbcli_core::{Chunk, DocId, DocSummary, Document, Hit, QueryRequest, Result};

/// Per-database configuration persisted alongside the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// Embedding dimensionality (post-Matryoshka).
    pub embed_dim: usize,
    /// Default chunk size (tokens).
    pub chunk_size: u32,
    /// Default chunk overlap (tokens).
    pub chunk_overlap: u32,
    /// Name of the embedding runtime that produced the vectors.
    pub runtime_name: String,
    /// Embedding model identifier (e.g. `"google/embeddinggemma-300m"`).
    pub model_id: String,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            chunk_size: 512,
            chunk_overlap: 64,
            runtime_name: String::new(),
            model_id: "google/embeddinggemma-300m".to_string(),
        }
    }
}

/// Coarse info reported by `db info`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreInfo {
    pub backend: &'static str,
    pub config: StoreConfig,
    pub doc_count: u64,
    pub chunk_count: u64,
    pub size_bytes: u64,
}

/// Result of an upsert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertResult {
    Inserted,
    Updated,
}

/// Pluggable storage backend for documents + chunk vectors.
#[async_trait]
pub trait VectorStore: Send + Sync {
    fn backend_name(&self) -> &'static str;

    /// Run idempotent migrations to bring the schema up to date.
    async fn migrate(&self) -> Result<()>;

    /// Persist (or refresh) the per-database config; merges with any
    /// existing config row.
    async fn put_config(&self, cfg: &StoreConfig) -> Result<()>;
    async fn get_config(&self) -> Result<Option<StoreConfig>>;

    /// Insert or update a document plus its chunks. Caller must have already
    /// computed embeddings and attached them to each chunk.
    async fn upsert_doc(
        &self,
        doc: &Document,
        chunks: &[Chunk],
        upsert: bool,
    ) -> Result<UpsertResult>;

    async fn get_doc(&self, id: &DocId) -> Result<Option<Document>>;
    async fn delete_doc(&self, id: &DocId) -> Result<bool>;

    async fn list_docs(
        &self,
        filter: &kbcli_core::Filter,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<DocSummary>>;

    async fn search(&self, q: &QueryRequest) -> Result<Vec<Hit>>;

    async fn info(&self) -> Result<StoreInfo>;
}
