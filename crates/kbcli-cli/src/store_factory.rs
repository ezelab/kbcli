//! Factory for storage backends.
//!
//! The simplified stack ships exactly one backend (`sqlite-vec`); this
//! module exists so the rest of the CLI keeps a stable seam for future
//! backends without sprinkling backend-specific types through the command
//! handlers.

use std::path::Path;
use std::sync::Arc;

use kbcli_core::{Error, Result};
use kbcli_store::{StoreConfig, VectorStore};

/// Build a backend by name. Currently only `sqlite-vec` (alias `sqlite`)
/// is supported.
pub async fn build(name: &str, path: &Path, cfg: &StoreConfig) -> Result<Arc<dyn VectorStore>> {
    match name {
        "sqlite-vec" | "sqlite" => {
            let s = kbcli_store_sqlite::SqliteStore::open(path.to_path_buf(), cfg).await?;
            Ok(Arc::new(s))
        }
        other => Err(Error::invalid(format!(
            "unknown backend: {other} (known: sqlite-vec)"
        ))),
    }
}

pub fn default_backend_name() -> &'static str {
    "sqlite-vec"
}
