//! Retrieval-quality evaluation helpers.
//!
//! Given a labelled corpus (docs + queries + relevance judgements), runs
//! each query through a `(EmbeddingRuntime, VectorStore)` pair and
//! computes the standard IR quality metrics.
//!
//! Used by both the synthetic always-on `tests/semantic_search.rs` and
//! the BEIR-backed `tests/semantic_search_beir.rs` / `bin/run_search_bench.rs`.

use std::collections::HashSet;

use kbcli_core::{Filter, QueryMode, QueryRequest, Result};

use crate::assertions::{mrr_at_k, ndcg_at_k};
use crate::runners::{ingest_text, Harness};
/// One labelled query: text + the set of relevant doc ids (binary
/// relevance — graded BEIR scores are collapsed to "any positive" in
/// the loader).
#[derive(Clone, Debug)]
pub struct LabeledQuery {
    pub id: String,
    pub text: String,
    pub relevant: HashSet<String>,
}

/// Aggregate retrieval-quality metrics over a query set.
#[derive(Clone, Debug)]
pub struct EvalReport {
    pub k: usize,
    pub n_queries: usize,
    pub mean_ndcg_at_k: f32,
    pub mean_recall_at_k: f32,
    pub mean_mrr_at_k: f32,
}

impl EvalReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "k": self.k,
            "n_queries": self.n_queries,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_mrr_at_k": self.mean_mrr_at_k,
        })
    }
}

/// Index `(doc_id, text)` pairs into the harness using its embedding
/// runtime and chunker.
pub async fn index_corpus(h: &Harness, docs: &[(String, String)]) -> Result<()> {
    for (id, text) in docs {
        ingest_text(h, id.as_str(), text, &[]).await?;
    }
    Ok(())
}

/// Run each query through the harness in the given retrieval mode and
/// aggregate retrieval metrics.
///
/// `Hybrid` and `Semantic` modes embed the query via the harness's
/// runtime; `Lexical` mode skips embedding (FTS5 BM25 only). Top-k
/// results are deduplicated to doc-level by the store before they
/// reach us.
pub async fn evaluate_with_mode(
    h: &Harness,
    queries: &[LabeledQuery],
    k: usize,
    mode: QueryMode,
) -> Result<EvalReport> {
    let mut sum_ndcg = 0.0f32;
    let mut sum_recall = 0.0f32;
    let mut sum_mrr = 0.0f32;
    let mut counted = 0usize;

    for q in queries {
        if q.relevant.is_empty() {
            continue;
        }
        let embedding = match mode {
            QueryMode::Lexical => None,
            QueryMode::Semantic | QueryMode::Hybrid => Some(h.runtime.embed(&q.text).await?),
        };
        let req = QueryRequest {
            text: q.text.clone(),
            mode,
            top_k: k as u32,
            filter: Filter::All,
            rrf_k: 60,
            weight_lex: 1.0,
            weight_sem: 1.0,
            embedding,
        };
        let hits = h.store.search(&req).await?;
        let ranking: Vec<String> = hits.iter().map(|h| h.doc_id.to_string()).collect();

        sum_ndcg += ndcg_at_k(&ranking, &q.relevant, k);
        sum_mrr += mrr_at_k(&ranking, &q.relevant, k);

        let predicted_set: std::collections::HashSet<String> =
            ranking.iter().take(k).cloned().collect();
        let hits_in_top = q
            .relevant
            .iter()
            .filter(|d| predicted_set.contains(*d))
            .count();
        sum_recall += hits_in_top as f32 / q.relevant.len() as f32;

        counted += 1;
    }

    let n = counted.max(1) as f32;
    Ok(EvalReport {
        k,
        n_queries: counted,
        mean_ndcg_at_k: sum_ndcg / n,
        mean_recall_at_k: sum_recall / n,
        mean_mrr_at_k: sum_mrr / n,
    })
}

/// Convenience wrapper around [`evaluate_with_mode`] that uses
/// `QueryMode::Hybrid` (the CLI default).
pub async fn evaluate(h: &Harness, queries: &[LabeledQuery], k: usize) -> Result<EvalReport> {
    evaluate_with_mode(h, queries, k, QueryMode::Hybrid).await
}

/// Build a small synthetic labelled retrieval task: one "anchor" doc per
/// query, keyed by a unique trigger phrase.
///
/// Each query is the trigger phrase plus a couple of generic words; the
/// anchor doc contains the trigger plus 10–20 other words from the
/// shared vocab. Returns `(corpus_docs, labelled_queries)`.
pub fn synthetic_trigger_task(
    n_anchors: usize,
    n_distractors: usize,
    seed: u64,
) -> (Vec<(String, String)>, Vec<LabeledQuery>) {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // Triggers are nonsense tokens unlikely to clash with the seeded vocab
    // or with anything BoW-hash-similar. Hex-suffixed for uniqueness.
    let triggers: Vec<String> = (0..n_anchors)
        .map(|i| format!("zorglax{:04x}", i))
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vocab: &[&str] = &[
        "rust",
        "vector",
        "query",
        "compiler",
        "memory",
        "embedding",
        "search",
        "hybrid",
        "ranking",
        "filter",
        "metadata",
        "index",
        "storage",
        "stream",
        "binary",
        "release",
        "matrix",
        "neural",
        "tokenizer",
        "benchmark",
    ];

    let mut corpus = Vec::new();
    for (i, trig) in triggers.iter().enumerate() {
        let len = rng.gen_range(10..20);
        let mut words: Vec<String> = (0..len)
            .map(|_| vocab[rng.gen_range(0..vocab.len())].to_string())
            .collect();
        // Drop the trigger somewhere in the middle.
        let pos = rng.gen_range(0..words.len());
        words[pos] = trig.clone();
        corpus.push((format!("anchor-{i}"), words.join(" ")));
    }
    for j in 0..n_distractors {
        let len = rng.gen_range(10..20);
        let words: Vec<&str> = (0..len)
            .map(|_| vocab[rng.gen_range(0..vocab.len())])
            .collect();
        corpus.push((format!("dist-{j}"), words.join(" ")));
    }

    let queries: Vec<LabeledQuery> = triggers
        .iter()
        .enumerate()
        .map(|(i, trig)| {
            let head = vocab[rng.gen_range(0..vocab.len())];
            let tail = vocab[rng.gen_range(0..vocab.len())];
            let text = format!("{head} {trig} {tail}");
            let mut relevant = HashSet::new();
            relevant.insert(format!("anchor-{i}"));
            LabeledQuery {
                id: format!("q-{i}"),
                text,
                relevant,
            }
        })
        .collect();

    (corpus, queries)
}
