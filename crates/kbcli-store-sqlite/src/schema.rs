use rusqlite::Connection;

use kbcli_core::{Error, Result};

pub const SCHEMA_VERSION: i64 = 1;

pub fn run_migrations(conn: &Connection, embed_dim: usize) -> Result<()> {
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS kv (
            key   TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY NOT NULL,
            text        TEXT NOT NULL,
            meta        TEXT NOT NULL DEFAULT '{}',
            created_at  INTEGER NOT NULL DEFAULT 0,
            updated_at  INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ord         INTEGER NOT NULL,
            text        TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(doc_id, ord)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        "#,
    )
    .map_err(|e| Error::Schema(format!("schema init: {e}")))?;

    // Triggers to keep FTS in sync with the chunks table.
    conn.execute_batch(
        r#"
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
        END;
        "#,
    )
    .map_err(|e| Error::Schema(format!("triggers: {e}")))?;

    // sqlite-vec virtual table: created lazily because dim is fixed at
    // creation time. We only create it once we know the configured dim.
    let stmt = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{embed_dim}])",
    );
    conn.execute_batch(&stmt)
        .map_err(|e| Error::Schema(format!("vec table (dim={embed_dim}): {e}")))?;

    conn.execute(
        "INSERT OR REPLACE INTO kv(key, value) VALUES('schema_version', ?)",
        [SCHEMA_VERSION.to_string()],
    )
    .map_err(|e| Error::Schema(format!("schema_version: {e}")))?;

    Ok(())
}
