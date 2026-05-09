use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use rusqlite::{params, params_from_iter, Connection, OpenFlags, OptionalExtension};

use kbcli_core::{
    Chunk, DocId, DocSummary, Document, Error, Filter, Hit, HitComponents, Metadata, QueryMode,
    QueryRequest, Result,
};
use kbcli_store::{filter_to_sql, StoreConfig, StoreInfo, UpsertResult, VectorStore};

use crate::schema::run_migrations;

/// SQLite-based storage backend with sqlite-vec for KNN and FTS5 for lexical.
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
    config: Arc<Mutex<Option<StoreConfig>>>,
}

static AUTO_EXT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_vec_extension_registered() {
    AUTO_EXT_INIT.get_or_init(|| {
        unsafe {
            // Auto-load sqlite-vec into every new connection in this process.
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

impl SqliteStore {
    /// Open or create a SQLite database at `path` with the supplied config.
    /// The vec0 virtual table is created with `cfg.embed_dim` on first run.
    pub async fn open(path: impl Into<PathBuf>, cfg: &StoreConfig) -> Result<Self> {
        ensure_vec_extension_registered();

        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let cfg = cfg.clone();
        let path_clone = path.clone();
        let (conn, persisted_cfg) = tokio::task::spawn_blocking(move || -> Result<_> {
            let conn = Connection::open_with_flags(
                &path_clone,
                OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
            )
            .map_err(|e| Error::store(format!("open db: {e}")))?;

            // Read previously persisted config (if any) BEFORE migrations,
            // so we can use the previously-stored embed_dim for vec_chunks.
            let prior_cfg = read_config(&conn)?;
            let dim = prior_cfg
                .as_ref()
                .map(|c| c.embed_dim)
                .unwrap_or(cfg.embed_dim);
            run_migrations(&conn, dim)?;
            Ok((conn, prior_cfg))
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))??;

        Ok(SqliteStore {
            conn: Arc::new(Mutex::new(conn)),
            path,
            config: Arc::new(Mutex::new(persisted_cfg)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    #[allow(dead_code)]
    fn with_conn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Connection) -> Result<R>,
    {
        let mut g = self
            .conn
            .lock()
            .map_err(|_| Error::store("conn lock poisoned"))?;
        f(&mut g)
    }
}

fn read_config(conn: &Connection) -> Result<Option<StoreConfig>> {
    let row: Option<String> = conn
        .query_row("SELECT value FROM kv WHERE key = 'store_config'", [], |r| {
            r.get::<_, String>(0)
        })
        .optional()
        .map_err(|e| {
            // The kv table may not exist on the first open; treat as absent.
            if e.to_string().contains("no such table") {
                Error::store("kv missing")
            } else {
                Error::store(format!("read config: {e}"))
            }
        })
        .ok()
        .flatten();
    if let Some(v) = row {
        let cfg: StoreConfig = serde_json::from_str(&v)?;
        Ok(Some(cfg))
    } else {
        Ok(None)
    }
}

#[async_trait]
impl VectorStore for SqliteStore {
    fn backend_name(&self) -> &'static str {
        "sqlite-vec"
    }

    async fn migrate(&self) -> Result<()> {
        let dim = self
            .config
            .lock()
            .map_err(|_| Error::store("config lock"))?
            .as_ref()
            .map(|c| c.embed_dim)
            .unwrap_or(768);
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<()> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            run_migrations(&g, dim)
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn put_config(&self, cfg: &StoreConfig) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let cfg_owned = cfg.clone();
        let cached = Arc::clone(&self.config);
        tokio::task::spawn_blocking(move || -> Result<()> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            // Re-run migrations with the new dim (in case this is the first
            // put_config and the table was created with the placeholder dim).
            run_migrations(&g, cfg_owned.embed_dim)?;
            let json = serde_json::to_string(&cfg_owned)?;
            g.execute(
                "INSERT INTO kv(key, value) VALUES('store_config', ?)
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [&json],
            )
            .map_err(|e| Error::store(format!("put_config: {e}")))?;
            *cached.lock().map_err(|_| Error::store("config lock"))? = Some(cfg_owned);
            Ok(())
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn get_config(&self) -> Result<Option<StoreConfig>> {
        if let Some(c) = self
            .config
            .lock()
            .map_err(|_| Error::store("config lock"))?
            .clone()
        {
            return Ok(Some(c));
        }
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> Result<Option<StoreConfig>> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            read_config(&g)
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn upsert_doc(
        &self,
        doc: &Document,
        chunks: &[Chunk],
        upsert: bool,
    ) -> Result<UpsertResult> {
        let dim = self
            .config
            .lock()
            .map_err(|_| Error::store("config lock"))?
            .as_ref()
            .map(|c| c.embed_dim)
            .unwrap_or(0);
        if dim == 0 {
            return Err(Error::Schema(
                "store_config not set; call put_config before upsert".into(),
            ));
        }
        for c in chunks {
            if let Some(v) = &c.embedding {
                if v.len() != dim {
                    return Err(Error::invalid(format!(
                        "chunk embedding has dim {}, expected {dim}",
                        v.len()
                    )));
                }
            } else {
                return Err(Error::invalid("chunk is missing embedding"));
            }
        }

        let conn = Arc::clone(&self.conn);
        let doc_owned = doc.clone();
        let chunks_owned = chunks.to_vec();

        tokio::task::spawn_blocking(move || -> Result<UpsertResult> {
            let mut g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            let tx = g
                .transaction()
                .map_err(|e| Error::store(format!("begin: {e}")))?;

            let existed: bool = tx
                .query_row(
                    "SELECT 1 FROM documents WHERE id = ?",
                    [doc_owned.id.as_str()],
                    |_| Ok(true),
                )
                .optional()
                .map_err(|e| Error::store(format!("exists: {e}")))?
                .unwrap_or(false);
            if existed && !upsert {
                return Err(Error::conflict(format!(
                    "doc id `{}` already exists",
                    doc_owned.id
                )));
            }

            let now = unix_millis();
            let meta_json = serde_json::to_string(
                &doc_owned
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::from(v.clone())))
                    .collect::<serde_json::Map<_, _>>(),
            )?;

            if existed {
                tx.execute(
                    "UPDATE documents SET text = ?, meta = ?, updated_at = ? WHERE id = ?",
                    params![doc_owned.text, meta_json, now, doc_owned.id.as_str()],
                )
                .map_err(|e| Error::store(format!("update doc: {e}")))?;
                tx.execute(
                    "DELETE FROM chunks WHERE doc_id = ?",
                    [doc_owned.id.as_str()],
                )
                .map_err(|e| Error::store(format!("clear chunks: {e}")))?;
                // Note: vec_chunks rows referencing those chunk ids are
                // orphaned by this DELETE (no FK). We clean them below.
            } else {
                tx.execute(
                    "INSERT INTO documents(id, text, meta, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?)",
                    params![doc_owned.id.as_str(), doc_owned.text, meta_json, now, now],
                )
                .map_err(|e| Error::store(format!("insert doc: {e}")))?;
            }

            // Insert chunks.
            for ch in &chunks_owned {
                tx.execute(
                    "INSERT INTO chunks(doc_id, ord, text, token_count) VALUES (?, ?, ?, ?)",
                    params![
                        ch.doc_id.as_str(),
                        ch.ord as i64,
                        ch.text,
                        ch.token_count as i64
                    ],
                )
                .map_err(|e| Error::store(format!("insert chunk: {e}")))?;
                let chunk_id = tx.last_insert_rowid();

                let emb = ch.embedding.as_ref().expect("checked above");
                let bytes: &[u8] = bytemuck_cast(emb);
                tx.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    params![chunk_id, bytes],
                )
                .map_err(|e| Error::store(format!("insert vec: {e}")))?;
            }

            // Best-effort cleanup of orphan vec rows from the old version.
            tx.execute(
                "DELETE FROM vec_chunks WHERE rowid NOT IN (SELECT id FROM chunks)",
                [],
            )
            .ok();

            tx.commit()
                .map_err(|e| Error::store(format!("commit: {e}")))?;
            Ok(if existed {
                UpsertResult::Updated
            } else {
                UpsertResult::Inserted
            })
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn get_doc(&self, id: &DocId) -> Result<Option<Document>> {
        let conn = Arc::clone(&self.conn);
        let id = id.clone();
        tokio::task::spawn_blocking(move || -> Result<Option<Document>> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            let row = g
                .query_row(
                    "SELECT id, text, meta, created_at, updated_at FROM documents WHERE id = ?",
                    [id.as_str()],
                    |r| {
                        Ok((
                            r.get::<_, String>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, String>(2)?,
                            r.get::<_, i64>(3)?,
                            r.get::<_, i64>(4)?,
                        ))
                    },
                )
                .optional()
                .map_err(|e| Error::store(format!("get_doc: {e}")))?;
            match row {
                None => Ok(None),
                Some((id, text, meta, created_at, updated_at)) => {
                    let metadata = parse_meta(&meta)?;
                    Ok(Some(Document {
                        id: DocId::new(id),
                        text,
                        metadata,
                        created_at,
                        updated_at,
                    }))
                }
            }
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn delete_doc(&self, id: &DocId) -> Result<bool> {
        let conn = Arc::clone(&self.conn);
        let id = id.clone();
        tokio::task::spawn_blocking(move || -> Result<bool> {
            let mut g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            let tx = g
                .transaction()
                .map_err(|e| Error::store(format!("begin: {e}")))?;
            let n = tx
                .execute("DELETE FROM documents WHERE id = ?", [id.as_str()])
                .map_err(|e| Error::store(format!("delete doc: {e}")))?;
            tx.execute(
                "DELETE FROM vec_chunks WHERE rowid NOT IN (SELECT id FROM chunks)",
                [],
            )
            .ok();
            tx.commit()
                .map_err(|e| Error::store(format!("commit: {e}")))?;
            Ok(n > 0)
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn list_docs(&self, filter: &Filter, limit: u32, offset: u32) -> Result<Vec<DocSummary>> {
        let conn = Arc::clone(&self.conn);
        let f = filter.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<DocSummary>> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            let fsql = filter_to_sql(&f);
            let q = format!(
                "SELECT documents.id,
                        substr(documents.text, 1, 200) AS preview,
                        documents.meta, documents.created_at, documents.updated_at,
                        (SELECT COUNT(*) FROM chunks WHERE chunks.doc_id = documents.id) AS chunk_count
                 FROM documents
                 WHERE {}
                 ORDER BY documents.created_at DESC, documents.id ASC
                 LIMIT ? OFFSET ?",
                fsql.sql,
            );
            let mut params_owned: Vec<rusqlite::types::Value> = fsql
                .params
                .into_iter()
                .map(|p| coerce_param(&p))
                .collect();
            params_owned.push(rusqlite::types::Value::Integer(limit as i64));
            params_owned.push(rusqlite::types::Value::Integer(offset as i64));

            let mut stmt = g.prepare(&q).map_err(|e| Error::store(format!("prepare: {e}")))?;
            let rows = stmt
                .query_map(params_from_iter(params_owned.iter()), |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, i64>(5)?,
                    ))
                })
                .map_err(|e| Error::store(format!("query: {e}")))?;

            let mut out = Vec::new();
            for row in rows {
                let (id, preview, meta, c_at, u_at, chunks) =
                    row.map_err(|e| Error::store(format!("row: {e}")))?;
                let metadata = parse_meta(&meta)?;
                out.push(DocSummary {
                    id: DocId::new(id),
                    text_preview: preview,
                    metadata,
                    created_at: c_at,
                    updated_at: u_at,
                    chunk_count: chunks as u32,
                });
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn search(&self, q: &QueryRequest) -> Result<Vec<Hit>> {
        let conn = Arc::clone(&self.conn);
        let q = q.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Hit>> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            search_blocking(&g, &q)
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }

    async fn info(&self) -> Result<StoreInfo> {
        let conn = Arc::clone(&self.conn);
        let path = self.path.clone();
        let cached = self
            .config
            .lock()
            .map_err(|_| Error::store("config lock"))?
            .clone();
        tokio::task::spawn_blocking(move || -> Result<StoreInfo> {
            let g = conn.lock().map_err(|_| Error::store("conn lock"))?;
            let cfg = cached.unwrap_or_else(|| read_config(&g).ok().flatten().unwrap_or_default());
            let doc_count: i64 = g
                .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
                .unwrap_or(0);
            let chunk_count: i64 = g
                .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
                .unwrap_or(0);
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            Ok(StoreInfo {
                backend: "sqlite-vec",
                config: cfg,
                doc_count: doc_count as u64,
                chunk_count: chunk_count as u64,
                size_bytes,
            })
        })
        .await
        .map_err(|e| Error::store(format!("blocking: {e}")))?
    }
}

fn search_blocking(conn: &Connection, q: &QueryRequest) -> Result<Vec<Hit>> {
    let top_k = q.top_k.max(1) as usize;
    // Pull a wider candidate set so post-filtering still has enough rows.
    let candidate_k = (top_k as u32).saturating_mul(5).max(50) as usize;

    // Per-chunk results from each branch: (chunk_id, raw_score, rank).
    let lex_results: Vec<(i64, f32)> = if matches!(q.mode, QueryMode::Lexical | QueryMode::Hybrid) {
        run_lexical(conn, &q.text, candidate_k)?
    } else {
        vec![]
    };

    let sem_results: Vec<(i64, f32)> = if matches!(q.mode, QueryMode::Semantic | QueryMode::Hybrid)
    {
        let emb = q
            .embedding
            .as_ref()
            .ok_or_else(|| Error::invalid("semantic/hybrid query missing embedding"))?;
        run_semantic(conn, emb, candidate_k)?
    } else {
        vec![]
    };

    // Look up parent doc + meta for every candidate chunk in one query.
    let mut chunk_ids: Vec<i64> = lex_results
        .iter()
        .chain(sem_results.iter())
        .map(|(id, _)| *id)
        .collect();
    chunk_ids.sort_unstable();
    chunk_ids.dedup();
    if chunk_ids.is_empty() {
        return Ok(vec![]);
    }

    let chunk_doc_meta = fetch_chunks_with_meta(conn, &chunk_ids)?;

    // Apply the metadata filter at the doc level.
    let pass: HashMap<i64, (String, String, Metadata)> = chunk_doc_meta
        .into_iter()
        .filter(|(_, (_, _, meta))| filter_matches(&q.filter, meta))
        .collect();

    // RRF combination, keyed by doc_id (best-chunk wins).
    let mut by_doc: BTreeMap<String, DocAccumulator> = BTreeMap::new();
    for (rank, (cid, score)) in lex_results.iter().enumerate() {
        if let Some((doc_id, snippet, _)) = pass.get(cid) {
            let entry = by_doc.entry(doc_id.clone()).or_default();
            entry.note_lex(*cid, *score, rank as u32, snippet, q.weight_lex, q.rrf_k);
        }
    }
    for (rank, (cid, score)) in sem_results.iter().enumerate() {
        if let Some((doc_id, snippet, _)) = pass.get(cid) {
            let entry = by_doc.entry(doc_id.clone()).or_default();
            entry.note_sem(*cid, *score, rank as u32, snippet, q.weight_sem, q.rrf_k);
        }
    }

    let mut hits: Vec<Hit> = by_doc
        .into_iter()
        .map(|(doc_id, acc)| acc.finalize(DocId::new(doc_id)))
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    Ok(hits)
}

#[derive(Default)]
struct DocAccumulator {
    score: f32,
    lex_score: Option<f32>,
    sem_score: Option<f32>,
    lex_rank: Option<u32>,
    sem_rank: Option<u32>,
    snippet: Option<String>,
    best_score: f32,
}

impl DocAccumulator {
    fn note_lex(&mut self, _cid: i64, score: f32, rank: u32, snippet: &str, w: f32, rrf_k: u32) {
        let rrf = w / (rrf_k as f32 + rank as f32 + 1.0);
        self.score += rrf;
        self.lex_score = Some(self.lex_score.unwrap_or(0.0).max(score));
        self.lex_rank = Some(self.lex_rank.unwrap_or(u32::MAX).min(rank));
        if rrf > self.best_score {
            self.best_score = rrf;
            self.snippet = Some(make_snippet(snippet));
        }
    }
    fn note_sem(&mut self, _cid: i64, score: f32, rank: u32, snippet: &str, w: f32, rrf_k: u32) {
        let rrf = w / (rrf_k as f32 + rank as f32 + 1.0);
        self.score += rrf;
        self.sem_score = Some(self.sem_score.unwrap_or(0.0).max(score));
        self.sem_rank = Some(self.sem_rank.unwrap_or(u32::MAX).min(rank));
        if rrf > self.best_score {
            self.best_score = rrf;
            self.snippet = Some(make_snippet(snippet));
        }
    }
    fn finalize(self, doc_id: DocId) -> Hit {
        Hit {
            doc_id,
            score: self.score,
            components: HitComponents {
                lex_score: self.lex_score,
                sem_score: self.sem_score,
                lex_rank: self.lex_rank,
                sem_rank: self.sem_rank,
            },
            snippet: self.snippet,
            document: None,
        }
    }
}

fn make_snippet(s: &str) -> String {
    const N: usize = 200;
    if s.len() <= N {
        s.to_string()
    } else {
        // Round down to the nearest char boundary so we never slice
        // through a multi-byte UTF-8 codepoint (e.g. a unicode quote).
        let mut end = N;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

fn run_lexical(conn: &Connection, text: &str, k: usize) -> Result<Vec<(i64, f32)>> {
    if text.trim().is_empty() {
        return Ok(vec![]);
    }
    let q = "SELECT rowid, bm25(fts_chunks) FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY bm25(fts_chunks) ASC LIMIT ?";
    let mut stmt = conn
        .prepare(q)
        .map_err(|e| Error::store(format!("prepare lex: {e}")))?;
    let rows = stmt
        .query_map(params![sanitize_fts_query(text), k as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)? as f32))
        })
        .map_err(|e| Error::store(format!("lex query: {e}")))?;
    let mut v = Vec::new();
    for r in rows {
        // FTS5 bm25 returns negative numbers, more negative = more relevant.
        // Convert to a positive score for diagnostics.
        let (id, raw) = r.map_err(|e| Error::store(format!("row: {e}")))?;
        v.push((id, -raw));
    }
    Ok(v)
}

/// FTS5's `MATCH` parser is unhappy with arbitrary punctuation. We strip
/// special characters and re-quote tokens for safe matching.
fn sanitize_fts_query(text: &str) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    let words: Vec<String> = text
        .split_word_bounds()
        .filter_map(|w| {
            let t = w.trim();
            if t.is_empty() || !t.chars().any(|c| c.is_alphanumeric()) {
                None
            } else {
                Some(format!("\"{}\"", t.replace('"', "")))
            }
        })
        .collect();
    if words.is_empty() {
        // FTS won't match anything, but it must be a valid query.
        "\"\"".to_string()
    } else {
        words.join(" OR ")
    }
}

fn run_semantic(conn: &Connection, emb: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
    let bytes: &[u8] = bytemuck_cast(emb);
    // sqlite-vec returns distance (lower = better). For cosine on normalized
    // vectors, distance is 1 - cosine. Convert to similarity.
    let mut stmt = conn
        .prepare(
            "SELECT rowid, distance FROM vec_chunks
             WHERE embedding MATCH ? AND k = ?
             ORDER BY distance",
        )
        .map_err(|e| Error::store(format!("prepare sem: {e}")))?;
    let rows = stmt
        .query_map(params![bytes, k as i64], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)? as f32))
        })
        .map_err(|e| Error::store(format!("sem query: {e}")))?;
    let mut v = Vec::new();
    for r in rows {
        let (id, dist) = r.map_err(|e| Error::store(format!("row: {e}")))?;
        v.push((id, 1.0 - dist));
    }
    Ok(v)
}

fn fetch_chunks_with_meta(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<HashMap<i64, (String, String, Metadata)>> {
    let placeholders = std::iter::repeat("?")
        .take(chunk_ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let q = format!(
        "SELECT chunks.id, chunks.doc_id, chunks.text, documents.meta
         FROM chunks
         JOIN documents ON documents.id = chunks.doc_id
         WHERE chunks.id IN ({placeholders})",
    );
    let mut stmt = conn
        .prepare(&q)
        .map_err(|e| Error::store(format!("prepare meta: {e}")))?;
    let params: Vec<rusqlite::types::Value> = chunk_ids
        .iter()
        .map(|i| rusqlite::types::Value::Integer(*i))
        .collect();
    let rows = stmt
        .query_map(params_from_iter(params.iter()), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| Error::store(format!("meta query: {e}")))?;
    let mut out = HashMap::new();
    for r in rows {
        let (id, doc_id, text, meta) = r.map_err(|e| Error::store(format!("row: {e}")))?;
        let m = parse_meta(&meta)?;
        out.insert(id, (doc_id, text, m));
    }
    Ok(out)
}

fn filter_matches(filter: &Filter, meta: &Metadata) -> bool {
    use kbcli_core::{MetaValue, Predicate};
    match filter {
        Filter::All => true,
        Filter::And(items) => items.iter().all(|f| filter_matches(f, meta)),
        Filter::Or(items) => items.iter().any(|f| filter_matches(f, meta)),
        Filter::Not(inner) => !filter_matches(inner, meta),
        Filter::Atom { key, predicate } => {
            let v = meta.get(key);
            match (predicate, v) {
                (Predicate::Exists, Some(_)) => true,
                (Predicate::Exists, None) => false,
                (Predicate::Missing, None) => true,
                (Predicate::Missing, Some(_)) => false,
                (_, None) => false,
                (Predicate::Eq(rhs), Some(lhs)) => meta_eq(lhs, rhs),
                (Predicate::Ne(rhs), Some(lhs)) => !meta_eq(lhs, rhs),
                (Predicate::Lt(rhs), Some(lhs)) => {
                    meta_cmp(lhs, rhs) == Some(std::cmp::Ordering::Less)
                }
                (Predicate::Le(rhs), Some(lhs)) => matches!(
                    meta_cmp(lhs, rhs),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                ),
                (Predicate::Gt(rhs), Some(lhs)) => {
                    meta_cmp(lhs, rhs) == Some(std::cmp::Ordering::Greater)
                }
                (Predicate::Ge(rhs), Some(lhs)) => matches!(
                    meta_cmp(lhs, rhs),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                ),
                (Predicate::In(set), Some(lhs)) => set.iter().any(|x| meta_eq(lhs, x)),
                (Predicate::NotIn(set), Some(lhs)) => !set.iter().any(|x| meta_eq(lhs, x)),
                (Predicate::Contains(needle), Some(MetaValue::Str(s))) => {
                    s.to_lowercase().contains(&needle.to_lowercase())
                }
                (Predicate::Contains(_), Some(_)) => false,
            }
        }
    }
}

fn meta_eq(a: &kbcli_core::MetaValue, b: &kbcli_core::MetaValue) -> bool {
    use kbcli_core::MetaValue::*;
    match (a, b) {
        (Null, Null) => true,
        (Bool(x), Bool(y)) => x == y,
        (Str(x), Str(y)) => x == y,
        (Int(x), Int(y)) => x == y,
        (Float(x), Float(y)) => x == y,
        (Int(i), Float(f)) | (Float(f), Int(i)) => (*i as f64) == *f,
        _ => false,
    }
}

fn meta_cmp(a: &kbcli_core::MetaValue, b: &kbcli_core::MetaValue) -> Option<std::cmp::Ordering> {
    use kbcli_core::MetaValue::*;
    match (a, b) {
        (Int(x), Int(y)) => x.partial_cmp(y),
        (Float(x), Float(y)) => x.partial_cmp(y),
        (Int(i), Float(f)) => (*i as f64).partial_cmp(f),
        (Float(f), Int(i)) => f.partial_cmp(&(*i as f64)),
        (Str(x), Str(y)) => x.partial_cmp(y),
        _ => None,
    }
}

fn parse_meta(s: &str) -> Result<Metadata> {
    if s.is_empty() {
        return Ok(BTreeMap::new());
    }
    let v: serde_json::Value = serde_json::from_str(s)?;
    let mut out = BTreeMap::new();
    if let serde_json::Value::Object(map) = v {
        for (k, v) in map {
            out.insert(k, v.into());
        }
    }
    Ok(out)
}

fn coerce_param(s: &str) -> rusqlite::types::Value {
    if let Ok(i) = s.parse::<i64>() {
        return rusqlite::types::Value::Integer(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return rusqlite::types::Value::Real(f);
    }
    rusqlite::types::Value::Text(s.to_string())
}

fn unix_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Tiny safe `&[f32] -> &[u8]` cast (vetted: f32 has no padding bytes).
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain old data; the resulting byte slice has the same
    // lifetime as the input. We compute the length explicitly.
    unsafe {
        std::slice::from_raw_parts(
            v.as_ptr() as *const u8,
            v.len() * std::mem::size_of::<f32>(),
        )
    }
}
