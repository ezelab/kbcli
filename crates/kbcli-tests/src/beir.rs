//! BEIR/SciFact loader (gated on `model-llama`).
//!
//! Downloads the canonical BEIR `scifact.zip` distribution
//! (`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`)
//! and extracts:
//!   - `scifact/corpus.jsonl`     — `{"_id", "title", "text"}` per line
//!   - `scifact/queries.jsonl`    — `{"_id", "text"}` per line
//!   - `scifact/qrels/test.tsv`   — header `query-id\tcorpus-id\tscore`
//!
//! Cached at `~/.cache/kbcli-tests/scifact/` after the first download
//! (~3 MB compressed, ~8 MB uncompressed).

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::eval::LabeledQuery;

const SCIFACT_URL: &str =
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip";

#[derive(Debug, Deserialize)]
struct CorpusRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// Materialised SciFact split.
pub struct ScifactDataset {
    /// `(doc_id, doc_text)` pairs. `doc_text` is `title + "\n" + text`.
    pub corpus: Vec<(String, String)>,
    /// Queries with binary-collapsed relevance labels (test split).
    pub queries: Vec<LabeledQuery>,
}

impl ScifactDataset {
    pub fn doc_count(&self) -> usize {
        self.corpus.len()
    }
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }
}

fn cache_root() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| {
                let mut p = PathBuf::from(h);
                p.push(".cache");
                p
            })
        })
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("kbcli-tests").join("scifact")
}

/// Download (or load from cache) and parse the BEIR/SciFact test split.
///
/// Network call only on first invocation; subsequent calls reuse the
/// extracted files in `~/.cache/kbcli-tests/scifact/`.
pub async fn load_scifact() -> Result<ScifactDataset> {
    // The download itself is synchronous (`reqwest::blocking`); run it on
    // a worker thread so we don't block the caller's tokio runtime.
    let dir = tokio::task::spawn_blocking(ensure_extracted)
        .await
        .map_err(|e| anyhow!("join error: {e}"))??;

    let corpus_path = dir.join("corpus.jsonl");
    let queries_path = dir.join("queries.jsonl");
    let qrels_path = dir.join("qrels").join("test.tsv");

    let corpus_rows = read_jsonl::<CorpusRow>(&corpus_path)
        .with_context(|| format!("parse {}", corpus_path.display()))?;
    let query_rows = read_jsonl::<QueryRow>(&queries_path)
        .with_context(|| format!("parse {}", queries_path.display()))?;
    let qrels =
        read_qrels_tsv(&qrels_path).with_context(|| format!("parse {}", qrels_path.display()))?;

    let corpus: Vec<(String, String)> = corpus_rows
        .into_iter()
        .map(|r| {
            let mut text = r.title;
            if !text.is_empty() && !r.text.is_empty() {
                text.push('\n');
            }
            text.push_str(&r.text);
            (r.id, text)
        })
        .collect();

    // BEIR's queries.jsonl is the union across all splits. Keep only
    // queries that have at least one positive label in the test qrels.
    let queries: Vec<LabeledQuery> = query_rows
        .into_iter()
        .filter_map(|q| {
            let relevant = qrels.get(&q.id)?.clone();
            if relevant.is_empty() {
                None
            } else {
                Some(LabeledQuery {
                    id: q.id,
                    text: q.text,
                    relevant,
                })
            }
        })
        .collect();

    if queries.is_empty() {
        return Err(anyhow!(
            "no labelled queries found after intersecting test qrels with queries.jsonl"
        ));
    }

    Ok(ScifactDataset { corpus, queries })
}

/// Ensure the SciFact archive has been downloaded and extracted; return
/// the directory containing `corpus.jsonl`, `queries.jsonl`, and `qrels/`.
fn ensure_extracted() -> Result<PathBuf> {
    let root = cache_root();
    let extracted = root.join("scifact");
    let marker = extracted.join("corpus.jsonl");
    if marker.exists() {
        return Ok(extracted);
    }

    fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;

    let zip_path = root.join("scifact.zip");
    if !zip_path.exists() {
        eprintln!("[beir] downloading {}", SCIFACT_URL);
        let mut resp = reqwest::blocking::get(SCIFACT_URL)
            .with_context(|| format!("GET {SCIFACT_URL}"))?
            .error_for_status()
            .with_context(|| format!("GET {SCIFACT_URL}"))?;
        let mut tmp =
            File::create(&zip_path).with_context(|| format!("create {}", zip_path.display()))?;
        resp.copy_to(&mut tmp)
            .with_context(|| format!("write {}", zip_path.display()))?;
        tmp.flush().ok();
    }

    eprintln!("[beir] extracting {}", zip_path.display());
    let f = File::open(&zip_path).with_context(|| format!("open {}", zip_path.display()))?;
    let mut zip =
        zip::ZipArchive::new(f).with_context(|| format!("read zip {}", zip_path.display()))?;

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let raw_name = entry.name().to_owned();
        // Strip the leading `scifact/` so files land directly under
        // <root>/scifact/{corpus.jsonl, queries.jsonl, qrels/test.tsv}.
        let stripped = raw_name
            .strip_prefix("scifact/")
            .unwrap_or(&raw_name)
            .to_owned();
        if stripped.is_empty() {
            continue;
        }
        let out_path = extracted.join(&stripped);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)
                .with_context(|| format!("mkdir {}", out_path.display()))?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let mut out =
            File::create(&out_path).with_context(|| format!("create {}", out_path.display()))?;
        std::io::copy(&mut entry, &mut out)
            .with_context(|| format!("extract {}", out_path.display()))?;
    }

    Ok(extracted)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r = BufReader::new(f);
    let mut buf = String::new();
    r.take(u64::MAX)
        .read_to_string(&mut buf)
        .with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in buf.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: T = serde_json::from_str(line)
            .with_context(|| format!("{} line {}", path.display(), i + 1))?;
        out.push(row);
    }
    Ok(out)
}

fn read_qrels_tsv(path: &Path) -> Result<HashMap<String, HashSet<String>>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r = BufReader::new(f);
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("query-id") {
            continue; // header
        }
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let q = cols
            .next()
            .ok_or_else(|| anyhow!("missing query-id at line {}", i + 1))?;
        let d = cols
            .next()
            .ok_or_else(|| anyhow!("missing corpus-id at line {}", i + 1))?;
        let score: i32 = cols
            .next()
            .ok_or_else(|| anyhow!("missing score at line {}", i + 1))?
            .trim()
            .parse()
            .with_context(|| format!("invalid score at line {}", i + 1))?;
        if score > 0 {
            out.entry(q.to_string()).or_default().insert(d.to_string());
        }
    }
    Ok(out)
}
