//! Shared helpers for opening a DB, instantiating its runtime, and chunking.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kbcli_core::{Error, Result};
use kbcli_embed::{ChunkConfig, Chunker, EmbeddingRuntime};
use kbcli_store::{StoreConfig, VectorStore};

use crate::{paths, runtime_factory, store_factory};

pub struct OpenedDb {
    #[allow(dead_code)]
    pub path: PathBuf,
    pub store: Arc<dyn VectorStore>,
    pub runtime: Arc<dyn EmbeddingRuntime>,
    #[allow(dead_code)]
    pub config: StoreConfig,
    pub chunker: Chunker,
}

pub struct OpenOpts<'a> {
    pub backend: Option<&'a str>,
    pub runtime: Option<&'a str>,
    pub chunk_size: Option<u32>,
    pub chunk_overlap: Option<u32>,
}

impl<'a> Default for OpenOpts<'a> {
    fn default() -> Self {
        Self {
            backend: None,
            runtime: None,
            chunk_size: None,
            chunk_overlap: None,
        }
    }
}

pub async fn open(
    name: &str,
    explicit_path: Option<&PathBuf>,
    opts: OpenOpts<'_>,
) -> Result<OpenedDb> {
    let path = paths::resolve_db(name, explicit_path)?;
    if !path.exists() {
        return Err(Error::not_found(format!(
            "db `{}` ({}). Run `kbcli db create {}`.",
            name,
            path.display(),
            name
        )));
    }
    let backend = opts
        .backend
        .unwrap_or(store_factory::default_backend_name());
    open_path(
        &path,
        backend,
        opts.runtime,
        opts.chunk_size,
        opts.chunk_overlap,
    )
    .await
}

pub async fn open_path(
    path: &Path,
    backend: &str,
    runtime_override: Option<&str>,
    chunk_size: Option<u32>,
    chunk_overlap: Option<u32>,
) -> Result<OpenedDb> {
    let mut cfg = StoreConfig::default();
    let store = store_factory::build(backend, path, &cfg).await?;
    if let Some(persisted) = store.get_config().await? {
        cfg = persisted;
    }

    let runtime_name = runtime_override.map(str::to_string).unwrap_or_else(|| {
        if cfg.runtime_name.is_empty() {
            runtime_factory::default_runtime_name().to_string()
        } else {
            cfg.runtime_name.clone()
        }
    });

    let runtime = runtime_factory::build(&runtime_name, Some(cfg.embed_dim)).await?;
    if runtime.dim() != cfg.embed_dim {
        return Err(Error::invalid(format!(
            "runtime `{}` produces dim {} but db expects dim {}",
            runtime_name,
            runtime.dim(),
            cfg.embed_dim
        )));
    }

    let chunker = Chunker::new(ChunkConfig {
        size: chunk_size.unwrap_or(cfg.chunk_size),
        overlap: chunk_overlap.unwrap_or(cfg.chunk_overlap),
    })?;

    Ok(OpenedDb {
        path: path.to_path_buf(),
        store,
        runtime,
        config: cfg,
        chunker,
    })
}
