//! `kbcli doc ...` — add, get, list, update, delete documents.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde::Serialize;

use kbcli_core::{Chunk, DocId, Document, Error, Filter, MetaValue, Result};
use kbcli_embed::EmbeddingRuntime;

use crate::{
    io::{self, DocSource, JsonlEntry},
    pipeline::{self, OpenOpts},
};

#[derive(Subcommand, Debug)]
pub enum DocCommand {
    Add(AddArgs),
    Get(GetArgs),
    List(ListArgs),
    Update(UpdateArgs),
    Delete(DeleteArgs),
}

#[derive(Args, Debug)]
pub struct AddArgs {
    pub db: String,

    /// Inline document text.
    #[arg(long, conflicts_with_all = ["file", "stdin", "from_dir", "jsonl"])]
    pub text: Option<String>,

    /// Read text from a file. `-` means stdin.
    #[arg(long, conflicts_with_all = ["text", "stdin", "from_dir", "jsonl"])]
    pub file: Option<PathBuf>,

    /// Read text from stdin until EOF.
    #[arg(long, conflicts_with_all = ["text", "file", "from_dir", "jsonl"])]
    pub stdin: bool,

    /// Recursively ingest every file under DIR; relative path becomes id.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["text", "file", "stdin", "jsonl"])]
    pub from_dir: Option<PathBuf>,

    /// Read JSON-lines (one doc per line) from stdin or `--file`.
    #[arg(long, conflicts_with_all = ["text", "stdin", "from_dir"])]
    pub jsonl: bool,

    /// Document id (otherwise UUIDv7 is generated, or the relative path for --from-dir).
    #[arg(long)]
    pub id: Option<String>,

    /// Allow updating an existing document with the same id.
    #[arg(long)]
    pub upsert: bool,

    /// `key=value` metadata; repeatable.
    #[arg(long = "meta", value_name = "K=V")]
    pub meta: Vec<String>,

    /// Override the chunker's chunk size for this ingest.
    #[arg(long)]
    pub chunk_size: Option<u32>,

    /// Override the chunker's chunk overlap for this ingest.
    #[arg(long)]
    pub chunk_overlap: Option<u32>,

    /// Override the embedding runtime for this ingest (must match the DB's dim).
    #[arg(long)]
    pub runtime: Option<String>,

    /// Override the on-disk DB path.
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct GetArgs {
    pub db: String,
    pub id: String,
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct ListArgs {
    pub db: String,
    #[arg(long = "filter", value_name = "EXPR")]
    pub filters: Vec<String>,
    #[arg(long, default_value_t = 50)]
    pub limit: u32,
    #[arg(long, default_value_t = 0)]
    pub offset: u32,
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct UpdateArgs {
    pub db: String,
    pub id: String,

    /// New body text (replaces). Mutually exclusive with --file.
    #[arg(long, conflicts_with = "file")]
    pub text: Option<String>,

    /// Read new body text from this file (`-` for stdin).
    #[arg(long)]
    pub file: Option<PathBuf>,

    /// `key=value` metadata to set / overwrite.
    #[arg(long = "meta", value_name = "K=V")]
    pub meta: Vec<String>,

    /// Metadata keys to remove.
    #[arg(long = "unset", value_name = "KEY")]
    pub unset: Vec<String>,

    #[arg(long)]
    pub chunk_size: Option<u32>,
    #[arg(long)]
    pub chunk_overlap: Option<u32>,
    #[arg(long)]
    pub runtime: Option<String>,
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct DeleteArgs {
    pub db: String,
    pub id: String,
    #[arg(long)]
    pub path: Option<PathBuf>,
}

pub async fn run(cmd: DocCommand, json: bool) -> Result<()> {
    match cmd {
        DocCommand::Add(a) => add(a, json).await,
        DocCommand::Get(a) => get(a, json).await,
        DocCommand::List(a) => list(a, json).await,
        DocCommand::Update(a) => update(a, json).await,
        DocCommand::Delete(a) => delete(a, json).await,
    }
}

fn parse_meta_args(items: &[String]) -> Result<Vec<(String, MetaValue)>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| Error::invalid(format!("expected `key=value`, got `{item}`")))?;
        if k.is_empty() {
            return Err(Error::invalid(format!("meta key is empty in `{item}`")));
        }
        out.push((k.to_string(), MetaValue::parse_cli(v)));
    }
    Ok(out)
}

#[derive(Serialize)]
struct AddOut {
    ok: bool,
    id: String,
    chunks: usize,
    upserted: bool,
}

#[derive(Serialize)]
struct AddManyOut {
    ok: bool,
    added: usize,
    failed: usize,
    docs: Vec<AddOut>,
}

async fn add(a: AddArgs, json: bool) -> Result<()> {
    let opts = OpenOpts {
        backend: None,
        runtime: a.runtime.as_deref(),
        chunk_size: a.chunk_size,
        chunk_overlap: a.chunk_overlap,
    };
    let db = pipeline::open(&a.db, a.path.as_ref(), opts).await?;
    let cli_meta = parse_meta_args(&a.meta)?;

    // Decide source. Auto-detect piped stdin when no source flag is given.
    let source = if let Some(t) = a.text.clone() {
        Some(SourceKind::Inline(t))
    } else if let Some(p) = a.file.clone() {
        if a.jsonl {
            Some(SourceKind::Jsonl(Some(p)))
        } else {
            Some(SourceKind::File(p))
        }
    } else if a.stdin {
        Some(SourceKind::Stdin)
    } else if a.jsonl {
        Some(SourceKind::Jsonl(None))
    } else if let Some(d) = a.from_dir.clone() {
        Some(SourceKind::Dir(d))
    } else if io::stdin_is_piped() {
        Some(SourceKind::Stdin)
    } else {
        None
    };
    let source = source.ok_or_else(|| {
        Error::invalid(
            "no input source: pass --text, --file, --stdin, --from-dir, or pipe via stdin",
        )
    })?;

    match source {
        SourceKind::Inline(t) => {
            let id = a.id.clone().map(DocId::from).unwrap_or_else(DocId::fresh);
            let out = ingest_one(&db, id, t, &cli_meta, a.upsert).await?;
            crate::io::render(json, &out, || {
                println!(
                    "{} {} ({} chunks)",
                    if out.upserted { "updated" } else { "added" },
                    out.id,
                    out.chunks
                );
            })
        }
        SourceKind::Stdin => {
            let text = io::read_stdin_to_string().await?;
            let id = a.id.clone().map(DocId::from).unwrap_or_else(DocId::fresh);
            let out = ingest_one(&db, id, text, &cli_meta, a.upsert).await?;
            crate::io::render(json, &out, || {
                println!(
                    "{} {} ({} chunks)",
                    if out.upserted { "updated" } else { "added" },
                    out.id,
                    out.chunks
                );
            })
        }
        SourceKind::File(p) => {
            let text = if p == PathBuf::from("-") {
                io::read_stdin_to_string().await?
            } else {
                std::fs::read_to_string(&p)?
            };
            let id =
                a.id.clone()
                    .map(DocId::from)
                    .unwrap_or_else(|| DocId::new(p.to_string_lossy().to_string()));
            let mut meta = cli_meta.clone();
            // Capture path metadata when reading from a real file.
            if p != PathBuf::from("-") {
                if let Ok(md) = std::fs::metadata(&p) {
                    meta.push(("size".into(), MetaValue::Int(md.len() as i64)));
                }
                meta.push((
                    "path".into(),
                    MetaValue::Str(p.to_string_lossy().to_string()),
                ));
            }
            let out = ingest_one(&db, id, text, &meta, a.upsert).await?;
            crate::io::render(json, &out, || {
                println!(
                    "{} {} ({} chunks)",
                    if out.upserted { "updated" } else { "added" },
                    out.id,
                    out.chunks
                );
            })
        }
        SourceKind::Dir(d) => {
            let mut docs: Vec<AddOut> = Vec::new();
            let mut failed = 0;
            for entry in walkdir::WalkDir::new(&d)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                let Ok(text) = std::fs::read_to_string(path) else {
                    failed += 1;
                    continue;
                };
                let rel = path.strip_prefix(&d).unwrap_or(path);
                let id = DocId::new(rel.to_string_lossy().to_string());
                let mut meta = cli_meta.clone();
                meta.push((
                    "path".into(),
                    MetaValue::Str(rel.to_string_lossy().to_string()),
                ));
                if let Ok(md) = std::fs::metadata(path) {
                    meta.push(("size".into(), MetaValue::Int(md.len() as i64)));
                }
                match ingest_one(&db, id, text, &meta, a.upsert).await {
                    Ok(o) => docs.push(o),
                    Err(e) => {
                        failed += 1;
                        tracing::warn!("failed to ingest {}: {e}", rel.display());
                    }
                }
            }
            let out = AddManyOut {
                ok: failed == 0,
                added: docs.len(),
                failed,
                docs,
            };
            crate::io::render(json, &out, || {
                println!(
                    "ingested {} docs ({} failed) from {}",
                    out.added,
                    out.failed,
                    d.display()
                );
            })
        }
        SourceKind::Jsonl(p) => {
            let entries = io::read_jsonl_lines(DocSource::Jsonl(p))?;
            let mut docs: Vec<AddOut> = Vec::new();
            let mut failed = 0;
            for JsonlEntry { id, text, meta } in entries {
                let id = id.map(DocId::from).unwrap_or_else(DocId::fresh);
                let mut entry_meta = cli_meta.clone();
                if let serde_json::Value::Object(map) = meta {
                    for (k, v) in map {
                        entry_meta.push((k, MetaValue::from(v)));
                    }
                }
                match ingest_one(&db, id, text, &entry_meta, true).await {
                    Ok(o) => docs.push(o),
                    Err(e) => {
                        failed += 1;
                        tracing::warn!("jsonl: {e}");
                    }
                }
            }
            let out = AddManyOut {
                ok: failed == 0,
                added: docs.len(),
                failed,
                docs,
            };
            crate::io::render(json, &out, || {
                println!(
                    "ingested {} docs from jsonl ({} failed)",
                    out.added, out.failed
                );
            })
        }
    }
}

enum SourceKind {
    Inline(String),
    File(PathBuf),
    Stdin,
    Dir(PathBuf),
    Jsonl(Option<PathBuf>),
}

async fn ingest_one(
    db: &pipeline::OpenedDb,
    id: DocId,
    text: String,
    cli_meta: &[(String, MetaValue)],
    upsert: bool,
) -> Result<AddOut> {
    let mut chunks = db.chunker.chunk(id.clone(), &text);
    if chunks.is_empty() {
        return Err(Error::invalid("document yielded zero chunks"));
    }

    // Embed all chunk texts.
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = db.runtime.embed_batch(&texts).await?;
    for (chunk, emb) in chunks.iter_mut().zip(embeddings.into_iter()) {
        chunk.embedding = Some(emb);
    }

    let mut metadata = std::collections::BTreeMap::new();
    for (k, v) in cli_meta {
        metadata.insert(k.clone(), v.clone());
    }
    let doc = Document {
        id: id.clone(),
        text,
        metadata,
        created_at: 0,
        updated_at: 0,
    };
    let res = db.store.upsert_doc(&doc, &chunks, upsert).await?;
    Ok(AddOut {
        ok: true,
        id: id.to_string(),
        chunks: chunks.len(),
        upserted: matches!(res, kbcli_store::UpsertResult::Updated),
    })
}

async fn get(a: GetArgs, json: bool) -> Result<()> {
    let opts = OpenOpts::default();
    let db = pipeline::open(&a.db, a.path.as_ref(), opts).await?;
    let id = DocId::new(a.id);
    let doc = db
        .store
        .get_doc(&id)
        .await?
        .ok_or_else(|| Error::not_found(format!("doc `{id}`")))?;
    crate::io::render(json, &doc, || {
        println!("id: {}", doc.id);
        if !doc.metadata.is_empty() {
            println!("meta:");
            for (k, v) in &doc.metadata {
                println!("  {k} = {}", serde_json::Value::from(v.clone()));
            }
        }
        println!("---");
        println!("{}", doc.text);
    })
}

async fn list(a: ListArgs, json: bool) -> Result<()> {
    let opts = OpenOpts::default();
    let db = pipeline::open(&a.db, a.path.as_ref(), opts).await?;
    let filter = Filter::parse_cli_many(&a.filters)?;
    let docs = db.store.list_docs(&filter, a.limit, a.offset).await?;
    crate::io::render(json, &docs, || {
        if docs.is_empty() {
            println!("(no matching docs)");
        } else {
            println!("{:<36} {:>6}  PREVIEW", "ID", "CHUNKS");
            for d in &docs {
                let preview: String = d.text_preview.chars().take(80).collect();
                println!("{:<36} {:>6}  {}", d.id, d.chunk_count, preview);
            }
        }
    })
}

async fn update(a: UpdateArgs, json: bool) -> Result<()> {
    let opts = OpenOpts {
        backend: None,
        runtime: a.runtime.as_deref(),
        chunk_size: a.chunk_size,
        chunk_overlap: a.chunk_overlap,
    };
    let db = pipeline::open(&a.db, a.path.as_ref(), opts).await?;
    let id = DocId::new(a.id);
    let mut existing = db
        .store
        .get_doc(&id)
        .await?
        .ok_or_else(|| Error::not_found(format!("doc `{id}`")))?;

    let new_text = if let Some(t) = a.text {
        Some(t)
    } else if let Some(p) = a.file {
        Some(if p == PathBuf::from("-") {
            io::read_stdin_to_string().await?
        } else {
            std::fs::read_to_string(p)?
        })
    } else {
        None
    };

    if let Some(t) = new_text.clone() {
        existing.text = t;
    }

    for (k, v) in parse_meta_args(&a.meta)? {
        existing.metadata.insert(k, v);
    }
    for k in &a.unset {
        existing.metadata.remove(k);
    }

    if new_text.is_some() {
        // Re-chunk and re-embed when the body changed.
        let mut chunks = db.chunker.chunk(id.clone(), &existing.text);
        if chunks.is_empty() {
            return Err(Error::invalid("updated document yielded zero chunks"));
        }
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = db.runtime.embed_batch(&texts).await?;
        for (c, e) in chunks.iter_mut().zip(embeddings.into_iter()) {
            c.embedding = Some(e);
        }
        db.store.upsert_doc(&existing, &chunks, true).await?;
    } else {
        // Metadata-only update — fetch current chunks and re-upsert.
        // The trait does not expose chunk listing, so the simplest correct
        // path is to re-chunk and re-embed the unchanged text.
        let mut chunks = db.chunker.chunk(id.clone(), &existing.text);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = db.runtime.embed_batch(&texts).await?;
        for (c, e) in chunks.iter_mut().zip(embeddings.into_iter()) {
            c.embedding = Some(e);
        }
        db.store.upsert_doc(&existing, &chunks, true).await?;
    }
    crate::io::render(
        json,
        &serde_json::json!({"ok": true, "id": id.to_string()}),
        || {
            println!("updated {id}");
        },
    )
}

async fn delete(a: DeleteArgs, json: bool) -> Result<()> {
    let opts = OpenOpts::default();
    let db = pipeline::open(&a.db, a.path.as_ref(), opts).await?;
    let id = DocId::new(a.id);
    let removed = db.store.delete_doc(&id).await?;
    if !removed {
        return Err(Error::not_found(format!("doc `{id}`")));
    }
    crate::io::render(
        json,
        &serde_json::json!({"ok": true, "deleted": id.to_string()}),
        || {
            println!("deleted {id}");
        },
    )
}

// Suppress unused-import false positive: `EmbeddingRuntime` is a trait used
// indirectly through `db.runtime.embed_batch`.
#[allow(dead_code)]
fn _imports() {
    fn _e<T: EmbeddingRuntime + ?Sized>(_: &T) {}
}

// Expose `Chunk` so docs of the public types reachable from here are
// documented even if unused directly in this file.
#[allow(dead_code)]
type _Chunk = Chunk;
