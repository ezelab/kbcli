//! Reading/streaming inputs and rendering output.

use std::io::{Read, Write};
use std::path::PathBuf;

use serde::Serialize;
use tokio::io::AsyncReadExt;

use kbcli_core::{Error, Result};

/// Source for `doc add` ingest.
#[allow(dead_code)]
pub enum DocSource {
    Inline(String),
    File(PathBuf),
    Stdin,
    /// JSON-lines streamed from stdin or a file path. Each line: `{"id"?, "text", "meta"?}`.
    Jsonl(Option<PathBuf>),
    Dir(PathBuf),
}

/// Read every byte from stdin into a String.
pub async fn read_stdin_to_string() -> Result<String> {
    let mut buf = Vec::new();
    tokio::io::stdin()
        .read_to_end(&mut buf)
        .await
        .map_err(|e| Error::Io(e))?;
    String::from_utf8(buf).map_err(|_| Error::invalid("stdin is not valid UTF-8"))
}

/// Detect whether stdin is being piped (not a terminal). Used for
/// auto-detecting `--stdin` when no source flag is provided.
pub fn stdin_is_piped() -> bool {
    use std::io::IsTerminal;
    !std::io::stdin().is_terminal()
}

#[allow(dead_code)]
pub fn read_text_from(source: DocSource) -> Result<TextStream> {
    match source {
        DocSource::Inline(s) => Ok(TextStream::Eager(vec![("inline".to_string(), s)])),
        DocSource::Stdin => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            Ok(TextStream::Eager(vec![("stdin".to_string(), buf)]))
        }
        DocSource::File(p) => {
            if p == PathBuf::from("-") {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                Ok(TextStream::Eager(vec![("stdin".to_string(), buf)]))
            } else {
                let s = std::fs::read_to_string(&p)?;
                Ok(TextStream::Eager(vec![(
                    p.to_string_lossy().to_string(),
                    s,
                )]))
            }
        }
        DocSource::Dir(p) => {
            let mut out = Vec::new();
            for entry in walkdir::WalkDir::new(&p)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                if let Ok(s) = std::fs::read_to_string(path) {
                    let rel = path
                        .strip_prefix(&p)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();
                    out.push((rel, s));
                }
            }
            Ok(TextStream::Eager(out))
        }
        DocSource::Jsonl(_) => {
            // Caller handles JSONL because items have richer shape than (id, text).
            Err(Error::invalid("use read_jsonl_lines for JSONL sources"))
        }
    }
}

/// Iterate JSONL records (id?, text, meta?) from stdin or a file.
pub fn read_jsonl_lines(source: DocSource) -> Result<Vec<JsonlEntry>> {
    let text = match source {
        DocSource::Jsonl(None) => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
        DocSource::Jsonl(Some(p)) => std::fs::read_to_string(&p)?,
        _ => return Err(Error::invalid("expected JSONL source")),
    };
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| Error::invalid(format!("jsonl line {}: {e}", n + 1)))?;
        let id = v.get("id").and_then(|x| x.as_str()).map(String::from);
        let text = v
            .get("text")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::invalid(format!("jsonl line {}: missing 'text'", n + 1)))?
            .to_string();
        let meta = v.get("meta").cloned().unwrap_or(serde_json::json!({}));
        out.push(JsonlEntry { id, text, meta });
    }
    Ok(out)
}

#[derive(Debug)]
pub struct JsonlEntry {
    pub id: Option<String>,
    pub text: String,
    pub meta: serde_json::Value,
}

#[allow(dead_code)]
pub enum TextStream {
    Eager(Vec<(String, String)>),
}

impl TextStream {
    #[allow(dead_code)]
    pub fn into_iter_named(self) -> impl Iterator<Item = (String, String)> {
        match self {
            TextStream::Eager(v) => v.into_iter(),
        }
    }
}

/// Print a value either as a human-friendly representation (passed via the
/// `human` closure) or as JSON.
pub fn render<T: Serialize>(json: bool, value: &T, human: impl FnOnce()) -> Result<()> {
    if json {
        let mut out = std::io::stdout().lock();
        serde_json::to_writer(&mut out, value)?;
        writeln!(out)?;
    } else {
        human();
    }
    Ok(())
}
