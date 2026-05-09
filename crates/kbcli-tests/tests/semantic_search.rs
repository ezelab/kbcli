//! Semantic-search retrieval-quality test (synthetic, always-on).
//!
//! Validates the *pipeline* end-to-end on a labelled corpus: chunker,
//! FTS5 indexing, BM25 ranking, doc-level dedup, NDCG/MRR/Recall
//! aggregation. Real semantic-quality testing happens against
//! EmbeddingGemma in `tests/semantic_search_beir.rs`; with the
//! deterministic `HashRuntime` the embedding axis is just noise, so
//! we exercise the lexical leg here.
//!
//! Each query has exactly one "anchor" doc keyed by a unique trigger
//! phrase. BM25 strongly weights the rare trigger, so the anchor must
//! be ranked first.
//!
//! Asserts:
//!   * mean NDCG@10  = 1.0  (anchor at rank 1 every time)
//!   * mean MRR@10   = 1.0
//!   * mean Recall@10 = 1.0

use kbcli_core::QueryMode;
use kbcli_tests::{eval, runners};

#[tokio::test]
async fn synthetic_lexical_retrieval_quality() {
    let dim = 128usize;
    let h = runners::sqlite_with_hash(dim).await.unwrap();

    let (corpus, queries) = eval::synthetic_trigger_task(20, 60, 7);
    eval::index_corpus(&h, &corpus).await.unwrap();

    let report = eval::evaluate_with_mode(&h, &queries, 10, QueryMode::Lexical)
        .await
        .unwrap();
    eprintln!("synthetic lexical search report: {:?}", report);

    assert_eq!(
        report.n_queries, 20,
        "all queries should have ≥1 relevant doc"
    );
    assert!(
        report.mean_mrr_at_k >= 0.99,
        "mean MRR@10 too low: {} (expected anchor at rank 1 for every query)",
        report.mean_mrr_at_k,
    );
    assert!(
        report.mean_recall_at_k >= 0.99,
        "mean Recall@10 too low: {} (anchor must appear in top-10)",
        report.mean_recall_at_k,
    );
    assert!(
        report.mean_ndcg_at_k >= 0.99,
        "mean NDCG@10 too low: {}",
        report.mean_ndcg_at_k,
    );
}

// Hybrid mode is not exercised here: with the deterministic `HashRuntime`,
// the "semantic" leg is essentially noise that, via RRF, drowns out the
// strong lexical trigger signal — so a hybrid threshold under hash would
// be flaky and not representative. Real semantic+hybrid quality is
// tested with EmbeddingGemma in `tests/semantic_search_beir.rs`.
