//! `kbcli db ...` — create, list, info, delete local databases.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde::Serialize;

use kbcli_core::{Error, Result};
use kbcli_store::StoreConfig;

use crate::{paths, runtime_factory, store_factory};

#[derive(Subcommand, Debug)]
pub enum DbCommand {
    /// Create a new local database.
    Create(CreateArgs),
    /// List databases under the default `~/.kbcli` directory.
    List(ListArgs),
    /// Show stats about a database.
    Info(InfoArgs),
    /// Delete a database file.
    Delete(DeleteArgs),
}

#[derive(Args, Debug)]
pub struct CreateArgs {
    /// Database name (used for the file `~/.kbcli/<name>.db`).
    pub name: String,

    /// Override the on-disk file path.
    #[arg(long)]
    pub path: Option<PathBuf>,

    /// Storage backend.
    #[arg(long, default_value_t = store_factory::default_backend_name().to_string())]
    pub backend: String,

    /// Embedding runtime to associate with the DB. Determines the default
    /// dim and which model is invoked at ingest/query time.
    #[arg(long, default_value_t = runtime_factory::default_runtime_name().to_string())]
    pub runtime: String,

    /// Embedding dimensionality. Defaults to the runtime's native size.
    #[arg(long)]
    pub dim: Option<usize>,

    /// Default chunk size in approx-tokens.
    #[arg(long, default_value_t = 512)]
    pub chunk_size: u32,

    /// Default chunk overlap.
    #[arg(long, default_value_t = 64)]
    pub chunk_overlap: u32,

    /// Replace any existing DB at this location.
    #[arg(long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct ListArgs {
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    pub name: String,
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct DeleteArgs {
    pub name: String,
    #[arg(long)]
    pub path: Option<PathBuf>,
    /// Skip confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

pub async fn run(cmd: DbCommand, json: bool) -> Result<()> {
    match cmd {
        DbCommand::Create(a) => create(a, json).await,
        DbCommand::List(a) => list(a, json).await,
        DbCommand::Info(a) => info(a, json).await,
        DbCommand::Delete(a) => delete(a, json).await,
    }
}

#[derive(Serialize)]
struct CreateOut {
    ok: bool,
    name: String,
    path: String,
    backend: String,
    runtime: String,
    embed_dim: usize,
}

async fn create(a: CreateArgs, json: bool) -> Result<()> {
    let path = paths::resolve_db(&a.name, a.path.as_ref())?;
    paths::ensure_parent(&path)?;
    if path.exists() {
        if !a.force {
            return Err(Error::conflict(format!(
                "{} already exists; pass --force to replace",
                path.display()
            )));
        }
        std::fs::remove_file(&path).map_err(Error::Io)?;
    }

    // Resolve embed dim: --dim wins; else runtime native dim (probed via build()).
    let runtime_for_dim = runtime_factory::build(&a.runtime, a.dim).await?;
    let embed_dim = a.dim.unwrap_or_else(|| runtime_for_dim.dim());

    let cfg = StoreConfig {
        embed_dim,
        chunk_size: a.chunk_size,
        chunk_overlap: a.chunk_overlap,
        runtime_name: a.runtime.clone(),
        model_id: "google/embeddinggemma-300m".into(),
    };

    let store = store_factory::build(&a.backend, &path, &cfg).await?;
    store.migrate().await?;
    store.put_config(&cfg).await?;

    let out = CreateOut {
        ok: true,
        name: a.name.clone(),
        path: path.display().to_string(),
        backend: a.backend.clone(),
        runtime: a.runtime,
        embed_dim,
    };
    crate::io::render(json, &out, || {
        println!(
            "created {} at {} (backend={}, runtime={}, dim={})",
            a.name,
            path.display(),
            a.backend,
            out.runtime,
            embed_dim
        );
    })
}

#[derive(Serialize)]
struct DbEntry {
    name: String,
    path: String,
    size_bytes: u64,
}

async fn list(a: ListArgs, json: bool) -> Result<()> {
    let dir = a.path.unwrap_or(paths::default_dir()?);
    let mut out = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(&dir).map_err(Error::Io)? {
            let entry = entry.map_err(Error::Io)?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("db") {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let name = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                out.push(DbEntry {
                    name,
                    path: p.display().to_string(),
                    size_bytes: size,
                });
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));

    crate::io::render(json, &out, || {
        if out.is_empty() {
            println!("(no databases in {})", dir.display());
        } else {
            println!("{:<24} {:>12}  PATH", "NAME", "SIZE");
            for e in &out {
                println!("{:<24} {:>12}  {}", e.name, e.size_bytes, e.path);
            }
        }
    })
}

async fn info(a: InfoArgs, json: bool) -> Result<()> {
    let path = paths::resolve_db(&a.name, a.path.as_ref())?;
    if !path.exists() {
        return Err(Error::not_found(format!("db `{}`", a.name)));
    }
    // We don't know the backend up front; try the default.
    let cfg = StoreConfig::default();
    let store = store_factory::build(store_factory::default_backend_name(), &path, &cfg).await?;
    let info = store.info().await?;
    crate::io::render(json, &info, || {
        println!("name:        {}", a.name);
        println!("path:        {}", path.display());
        println!("backend:     {}", info.backend);
        println!("runtime:     {}", info.config.runtime_name);
        println!("embed_dim:   {}", info.config.embed_dim);
        println!("chunk_size:  {}", info.config.chunk_size);
        println!("chunk_over:  {}", info.config.chunk_overlap);
        println!("docs:        {}", info.doc_count);
        println!("chunks:      {}", info.chunk_count);
        println!("size_bytes:  {}", info.size_bytes);
    })
}

async fn delete(a: DeleteArgs, json: bool) -> Result<()> {
    let path = paths::resolve_db(&a.name, a.path.as_ref())?;
    if !path.exists() {
        return Err(Error::not_found(format!("db `{}`", a.name)));
    }
    if !a.yes && !json {
        eprint!("Delete {}? [y/N] ", path.display());
        use std::io::Write;
        std::io::stderr().flush().ok();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).map_err(Error::Io)?;
        let buf = buf.trim().to_lowercase();
        if buf != "y" && buf != "yes" {
            return Err(Error::other("aborted"));
        }
    }
    std::fs::remove_file(&path).map_err(Error::Io)?;
    for ext in ["-wal", "-shm", "-journal"] {
        let p = path.with_file_name(format!(
            "{}{}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            ext
        ));
        let _ = std::fs::remove_file(p);
    }
    crate::io::render(
        json,
        &serde_json::json!({"ok": true, "deleted": path.display().to_string()}),
        || {
            println!("deleted {}", path.display());
        },
    )
}
