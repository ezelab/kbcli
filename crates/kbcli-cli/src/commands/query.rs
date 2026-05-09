//! `kbcli query <db> <text>` — run a search against the database.

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use kbcli_core::{Error, Filter, QueryMode, QueryRequest, Result};

use crate::pipeline::{self, OpenOpts};

#[derive(Args, Debug)]
pub struct QueryArgs {
    pub db: String,
    pub text: String,

    /// Search mode.
    #[arg(long, default_value = "hybrid")]
    pub mode: String,

    #[arg(long, default_value_t = 10)]
    pub top_k: u32,

    #[arg(long = "filter", value_name = "EXPR")]
    pub filters: Vec<String>,

    /// RRF k constant (typically ~60).
    #[arg(long, default_value_t = 60)]
    pub rrf_k: u32,

    #[arg(long, default_value_t = 1.0)]
    pub weight_lex: f32,

    #[arg(long, default_value_t = 1.0)]
    pub weight_sem: f32,

    /// Override the embedding runtime for this query (must match the DB's dim).
    #[arg(long)]
    pub runtime: Option<String>,

    /// Override the on-disk DB path.
    #[arg(long)]
    pub path: Option<PathBuf>,
}

#[derive(Serialize)]
struct QueryOut<'a> {
    ok: bool,
    db: &'a str,
    mode: QueryMode,
    top_k: u32,
    hits: Vec<HitOut>,
}

#[derive(Serialize)]
struct HitOut {
    rank: u32,
    doc_id: String,
    score: f32,
    lex_score: Option<f32>,
    sem_score: Option<f32>,
    snippet: Option<String>,
}

pub async fn run(args: QueryArgs, json: bool) -> Result<()> {
    let opts = OpenOpts {
        backend: None,
        runtime: args.runtime.as_deref(),
        chunk_size: None,
        chunk_overlap: None,
    };
    let db = pipeline::open(&args.db, args.path.as_ref(), opts).await?;

    let mode: QueryMode = args.mode.parse().map_err(|e: Error| e)?;
    let filter = Filter::parse_cli_many(&args.filters)?;

    let mut request = QueryRequest {
        text: args.text.clone(),
        mode,
        top_k: args.top_k,
        filter,
        rrf_k: args.rrf_k,
        weight_lex: args.weight_lex,
        weight_sem: args.weight_sem,
        embedding: None,
    };
    if request.needs_embedding() {
        let v = db.runtime.embed(&args.text).await?;
        request.embedding = Some(v);
    }
    let hits = db.store.search(&request).await?;

    let mut out_hits = Vec::with_capacity(hits.len());
    for (i, h) in hits.iter().enumerate() {
        out_hits.push(HitOut {
            rank: (i + 1) as u32,
            doc_id: h.doc_id.to_string(),
            score: h.score,
            lex_score: h.components.lex_score,
            sem_score: h.components.sem_score,
            snippet: h.snippet.clone(),
        });
    }

    let out = QueryOut {
        ok: true,
        db: &args.db,
        mode,
        top_k: args.top_k,
        hits: out_hits,
    };
    crate::io::render(json, &out, || {
        if hits.is_empty() {
            println!("(no results)");
            return;
        }
        println!("{:>4}  {:<36} {:>10}  SNIPPET", "#", "DOC_ID", "SCORE");
        for (i, h) in hits.iter().enumerate() {
            let snip: String = h
                .snippet
                .as_deref()
                .unwrap_or("")
                .chars()
                .take(80)
                .collect();
            println!("{:>4}  {:<36} {:>10.4}  {}", i + 1, h.doc_id, h.score, snip);
        }
    })
}
